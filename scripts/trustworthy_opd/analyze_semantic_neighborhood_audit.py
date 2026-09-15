from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_lcb_reliability_calibration import build_metric_summary
from common import load_config, load_run, read_jsonl, update_status
from lcb_reliability_common import aggregate_risk, lcb_trust, request_id
from run_lcb_reliability_calibration import load_score_map
from semantic_neighborhood_audit_common import (
    PRIMARY_GROUPS,
    Q_ORDER,
    clustered_precision_interval,
    cohens_kappa,
    consensus_annotations,
    latest_annotations,
    semantic_pair_id,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze semantic-neighborhood audit")
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def load_all_annotations(run_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        path.stem: latest_annotations(path)
        for path in sorted((run_dir / "annotations").glob("*.jsonl"))
    }


def annotation_agreement(
    annotations: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    rows = []
    for first, second in itertools.combinations(sorted(annotations), 2):
        overlap = sorted(annotations[first].keys() & annotations[second].keys())
        left = [annotations[first][audit_id]["label"] for audit_id in overlap]
        right = [annotations[second][audit_id]["label"] for audit_id in overlap]
        rows.append(
            {
                "annotator_1": first,
                "annotator_2": second,
                "overlap": len(overlap),
                "exact_agreement": (
                    float(np.mean(np.asarray(left) == np.asarray(right)))
                    if overlap
                    else float("nan")
                ),
                "cohens_kappa": cohens_kappa(left, right),
            }
        )
    return pd.DataFrame(rows)


def precision_summary(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    analysis = config["analysis"]
    rows = []
    arms = sorted(frame.arm.unique())
    neighborhoods = sorted(frame.neighborhood.unique())
    representations = sorted(frame.representation.unique())
    group_values: list[str | None] = [*config["sampling"]["groups"], None]
    q_values: list[str | None] = [*config["sampling"]["q_points"], None]
    for arm in arms:
        for neighborhood in neighborhoods:
            for representation in representations:
                base = frame[
                    (frame.arm == arm)
                    & (frame.neighborhood == neighborhood)
                    & (frame.representation == representation)
                ]
                for q in q_values:
                    for group in group_values:
                        subset = base
                        if q is not None:
                            subset = subset[subset.fraction_name == q]
                        if group is not None:
                            subset = subset[subset.group == group]
                        if subset.empty:
                            continue
                        estimate, low, high = clustered_precision_interval(
                            subset,
                            samples=int(analysis["bootstrap_samples"]),
                            seed=stable_seed(
                                config["experiment"]["seed"],
                                arm,
                                neighborhood,
                                representation,
                                q or "all_q",
                                group or "all_primary_groups",
                            ),
                            confidence=float(analysis["confidence_level"]),
                        )
                        decided = subset.consensus_label.isin(
                            ["comparable", "not_comparable"]
                        )
                        rows.append(
                            {
                                "arm": arm,
                                "neighborhood": neighborhood,
                                "representation": representation,
                                "fraction_name": q or "all_q",
                                "group": group or "all_primary_groups",
                                "membership_count": len(subset),
                                "unique_pairs": int(subset.audit_id.nunique()),
                                "decided": int(decided.sum()),
                                "uncertain_or_missing": int((~decided).sum()),
                                "semantic_precision": estimate,
                                "precision_ci_low": low,
                                "precision_ci_high": high,
                            }
                        )
    return pd.DataFrame(rows)


def build_oracle_metrics(
    audit_run: Path,
    audit_config: dict[str, Any],
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_run = Path(audit_config["source"]["run_dir"])
    source_config = load_config(source_run / "config.yaml")
    comparable = set(
        labels.loc[labels.consensus_label == "comparable", "audit_id"].astype(str)
    )
    if not comparable:
        return pd.DataFrame(), pd.DataFrame()
    neighbors = pd.DataFrame(read_jsonl(source_run / "artifacts" / "neighbors.jsonl"))
    neighbors["audit_id"] = [
        semantic_pair_id(state, neighbor)
        for state, neighbor in zip(neighbors.state_id, neighbors.neighbor_point_id)
    ]
    neighbors = neighbors[neighbors.audit_id.isin(comparable)]
    anchors = {
        row["state_id"]: row
        for row in read_jsonl(source_run / "artifacts" / "anchors.jsonl")
    }
    student_scores = load_score_map(source_run, "student")
    teacher_scores = load_score_map(source_run, "teacher")
    risk_config = source_config["risk"]
    rows = []
    for (state_id, neighborhood, representation), selected in neighbors.groupby(
        ["state_id", "neighborhood", "representation"]
    ):
        anchor = anchors[state_id]
        if anchor["group"] not in PRIMARY_GROUPS:
            continue
        anchor_request = request_id(
            anchor["base_trajectory_id"],
            int(anchor["position"]),
            int(anchor["student_action_token_id"]),
        )
        anchor_student = student_scores[anchor_request]
        anchor_teacher = teacher_scores[anchor_request]
        anchor_reward = anchor_teacher - anchor_student
        neighbor_teacher = np.asarray(
            [teacher_scores[item] for item in selected.neighbor_request_id], dtype=float
        )
        neighbor_student = np.asarray(
            [student_scores[item] for item in selected.neighbor_request_id], dtype=float
        )
        neighbor_rewards = neighbor_teacher - neighbor_student
        risk = aggregate_risk(
            np.abs(neighbor_rewards - anchor_reward),
            str(risk_config["aggregation"]),
            float(risk_config["temperature"]),
        )
        row = {
            **anchor,
            "neighborhood": neighborhood,
            "representation": representation,
            "neighbor_count": len(selected),
            "teacher_sensitivity": float(np.max(anchor_teacher - neighbor_teacher)),
            "student_sensitivity": float(np.max(anchor_student - neighbor_student)),
            "relative_sensitivity": float(np.max(anchor_reward - neighbor_rewards)),
            "lcb_risk": risk,
            "risk_to_abs_reward": risk
            / max(abs(anchor_reward), float(risk_config["epsilon"])),
            "student_distance_mean": float(selected.student_cosine_distance.mean()),
            "teacher_distance_mean": float(selected.teacher_cosine_distance.mean()),
        }
        for value in risk_config["lambdas"]:
            slug = str(value).replace(".", "p")
            row[f"trust_lambda_{slug}"] = lcb_trust(
                anchor_reward,
                risk,
                float(value),
                float(risk_config["epsilon"]),
            )
        rows.append(row)
    oracle = pd.DataFrame(rows)
    if oracle.empty:
        return oracle, pd.DataFrame()
    summary_config = json.loads(json.dumps(source_config))
    summary_config["experiment"]["seed"] = audit_config["experiment"]["seed"]
    summary_config["analysis"]["bootstrap_samples"] = audit_config["analysis"][
        "bootstrap_samples"
    ]
    summary, _ = build_metric_summary(oracle, summary_config)
    return oracle, summary


def plot_precision(summary: pd.DataFrame, output: Path) -> None:
    selected = summary[
        (summary.fraction_name == "all_q") & (summary.group == "all_primary_groups")
    ]
    if selected.empty or not selected.semantic_precision.notna().any():
        return
    neighborhoods = sorted(selected.neighborhood.unique())
    representations = sorted(selected.representation.unique())
    arms = [arm for arm in ("selected", "random", "far") if arm in set(selected.arm)]
    colors = {"selected": "#2563EB", "random": "#D97706", "far": "#DC2626"}
    fig, axes = plt.subplots(
        1, len(neighborhoods), figsize=(7 * len(neighborhoods), 6), sharey=True
    )
    if len(neighborhoods) == 1:
        axes = [axes]
    x = np.arange(len(representations), dtype=float)
    width = 0.22
    for axis, neighborhood in zip(axes, neighborhoods):
        for index, arm in enumerate(arms):
            rows = (
                selected[
                    (selected.neighborhood == neighborhood) & (selected.arm == arm)
                ]
                .set_index("representation")
                .reindex(representations)
            )
            values = rows.semantic_precision.to_numpy(float)
            errors = np.vstack(
                [
                    values - rows.precision_ci_low.to_numpy(float),
                    rows.precision_ci_high.to_numpy(float) - values,
                ]
            )
            axis.bar(
                x + (index - (len(arms) - 1) / 2) * width,
                values,
                width,
                yerr=errors,
                capsize=3,
                label=arm,
                color=colors[arm],
            )
        axis.set_title(neighborhood)
        axis.set_xticks(x, representations, rotation=18, ha="right")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", color="#E5E7EB")
    axes[0].set_ylabel("Blind semantic-comparability precision (95% prompt-cluster CI)")
    axes[-1].legend(frameon=False)
    fig.suptitle(
        "Are Retrieved States Genuine Semantic Neighbors?", fontsize=17, weight="bold"
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.25, wspace=0.12)
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_selected_by_q(summary: pd.DataFrame, output: Path) -> None:
    selected = summary[
        (summary.arm == "selected")
        & (summary.group == "all_primary_groups")
        & summary.fraction_name.isin(Q_ORDER)
    ]
    if selected.empty or not selected.semantic_precision.notna().any():
        return
    neighborhoods = sorted(selected.neighborhood.unique())
    representations = sorted(selected.representation.unique())
    colors = dict(zip(representations, ["#2563EB", "#7C3AED", "#059669", "#EA580C"]))
    fig, axes = plt.subplots(
        1, len(neighborhoods), figsize=(7 * len(neighborhoods), 5.8), sharey=True
    )
    if len(neighborhoods) == 1:
        axes = [axes]
    for axis, neighborhood in zip(axes, neighborhoods):
        for representation in representations:
            rows = (
                selected[
                    (selected.neighborhood == neighborhood)
                    & (selected.representation == representation)
                ]
                .set_index("fraction_name")
                .reindex(Q_ORDER)
            )
            axis.plot(
                Q_ORDER,
                rows.semantic_precision,
                marker="o",
                label=representation,
                color=colors[representation],
            )
            axis.fill_between(
                np.arange(len(Q_ORDER)),
                rows.precision_ci_low.to_numpy(float),
                rows.precision_ci_high.to_numpy(float),
                color=colors[representation],
                alpha=0.12,
            )
        axis.set_title(neighborhood)
        axis.set_xlabel("Intervention quantile")
        axis.grid(color="#E5E7EB")
    axes[0].set_ylabel("Semantic-comparability precision")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Semantic Neighborhood Quality Across Intervention Stages",
        fontsize=17,
        weight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.15, wspace=0.12)
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_outputs(
    run_dir: Path,
    config: dict[str, Any],
    items: pd.DataFrame,
    membership: pd.DataFrame,
    annotations: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    results = run_dir / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    consensus = consensus_annotations(annotations)
    labels = items[["audit_id", "secondary_required"]].merge(
        consensus, on="audit_id", how="left"
    )
    labels["consensus_label"] = labels.consensus_label.fillna("missing")
    labeled = membership.merge(labels, on="audit_id", how="left")
    summary = precision_summary(labeled, config)
    agreement = annotation_agreement(annotations)
    oracle, oracle_summary = build_oracle_metrics(run_dir, config, labels)
    labels.to_csv(results / "consensus_labels.csv", index=False)
    summary.to_csv(results / "semantic_precision.csv", index=False)
    agreement.to_csv(results / "annotator_agreement.csv", index=False)
    oracle.to_csv(results / "oracle_filtered_metrics.csv", index=False)
    oracle_summary.to_csv(results / "oracle_metric_summary.csv", index=False)
    plot_precision(summary, figures / "semantic_precision.svg")
    plot_selected_by_q(summary, figures / "semantic_precision_by_q.svg")

    total = len(items)
    annotated = int((labels.consensus_label != "missing").sum())
    decided = int(labels.consensus_label.isin(["comparable", "not_comparable"]).sum())
    uncertain = int((labels.consensus_label == "uncertain").sum())
    disputed = int((labels.consensus_label == "disputed").sum())
    selected_cells = summary[
        (summary.arm == "selected")
        & (summary.group == "all_primary_groups")
        & summary.fraction_name.isin(Q_ORDER)
    ].copy()
    selected_cells["ci_half_width"] = (
        selected_cells.precision_ci_high - selected_cells.precision_ci_low
    ) / 2
    threshold = int(config["analysis"]["minimum_decided_per_cell"])
    target = float(config["analysis"]["target_precision_ci_half_width"])
    cells_ready = int(
        (
            (selected_cells.decided >= threshold)
            & (selected_cells.ci_half_width <= target)
        ).sum()
    )
    cells_total = len(selected_cells)
    secondary_ids = set(items.loc[items.secondary_required, "audit_id"])
    secondary_counts = [
        sum(audit_id in rows for audit_id in secondary_ids)
        for rows in annotations.values()
    ]
    secondary_complete_raters = sum(
        count == len(secondary_ids) for count in secondary_counts
    )
    payload = {
        "total_pairs": total,
        "annotated_pairs": annotated,
        "decided_pairs": decided,
        "uncertain_pairs": uncertain,
        "disputed_pairs": disputed,
        "selected_cells_ready": cells_ready,
        "selected_cells_total": cells_total,
        "secondary_pairs": len(secondary_ids),
        "secondary_complete_raters": secondary_complete_raters,
        "oracle_metric_states": int(oracle.state_id.nunique())
        if not oracle.empty
        else 0,
    }
    report = [
        "# Blinded semantic-neighborhood audit",
        "",
        "This audit asks whether retrieved states share the same immediate reasoning subgoal and whether the anchor action is semantically comparable at the neighbor state. Correctness group, metric value, representation, neighborhood, and distance are hidden during annotation.",
        "",
        f"- Annotated unique pairs: {annotated}/{total}",
        f"- Decided / uncertain / disputed: {decided} / {uncertain} / {disputed}",
        f"- Selected cells meeting count and CI stopping rule: {cells_ready}/{cells_total}",
        f"- Secondary blind-review pairs: {len(secondary_ids)}; complete secondary raters: {secondary_complete_raters}",
        f"- States in oracle-filtered diagnostic: {payload['oracle_metric_states']}",
        "",
    ]
    if annotated < total:
        report.extend(
            [
                "The audit is awaiting annotations. Any current precision or oracle-filtered AUROC is incomplete and must not be interpreted as a scientific result.",
                "",
            ]
        )
    else:
        report.extend(
            [
                "All primary annotation items are complete. Interpret selected-versus-control precision, q-stage stability, inter-rater agreement, and oracle-filtered reliability jointly.",
                "",
            ]
        )
    (results / "report.md").write_text("\n".join(report), encoding="utf-8")
    table = summary[
        (summary.fraction_name == "all_q") & (summary.group == "all_primary_groups")
    ].to_html(index=False, float_format=lambda value: f"{value:.3f}", classes="data")
    agreement_table = (
        agreement.to_html(
            index=False, float_format=lambda value: f"{value:.3f}", classes="data"
        )
        if not agreement.empty
        else "<p>尚无双人重叠标注。</p>"
    )
    dashboard = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>语义邻域审计</title>
<style>body{{font-family:system-ui;margin:0;background:#f3f4f6;color:#111827}}main{{max-width:1500px;margin:auto;padding:28px}}.card{{background:#fff;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 1px 4px #0002}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px;border:1px solid #d1d5db;text-align:left}}code{{background:#eef2ff;padding:2px 5px}}</style></head><body><main>
<h1>语义邻域盲审看板</h1><p>进度：{annotated}/{total}；确定标签：{decided}；满足停止规则的 selected cells：{cells_ready}/{cells_total}。</p>
<section class="card"><h2>检索邻居与负控制</h2>{'<img src="semantic_precision.svg">' if (figures / "semantic_precision.svg").exists() else "<p>完成部分标注后生成。</p>"}<p>纵轴是盲审认定为“同一即时子目标且 action 可比较”的比例。selected 应明显优于同 prompt/进度窗内的随机与远邻，才能证明检索产生了有意义的局部邻域。</p>{table}</section>
<section class="card"><h2>不同介入阶段</h2>{'<img src="semantic_precision_by_q.svg">' if (figures / "semantic_precision_by_q.svg").exists() else "<p>完成部分标注后生成。</p>"}<p>如果某种表示只在个别 q-point 有较高 precision，就不能把它作为全阶段统一邻域。</p></section>
<section class="card"><h2>复标一致性</h2>{agreement_table}<p>双人一致性用于区分邻域定义含混与标注噪声；至少 20% 盲法复标后再作正式解释。</p></section>
<section class="card"><h2>Oracle-filtered 诊断</h2><p>只保留盲审为 comparable 的已评分 selected 边重算 risk。它仍是发现性子样本，不能替代不重叠 prompt 的确认实验。</p></section>
</main></body></html>"""
    (figures / "semantic_audit_dashboard.html").write_text(dashboard, encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    items = pd.DataFrame(read_jsonl(run_dir / "artifacts" / "blinded_pairs.jsonl"))
    membership = pd.DataFrame(
        read_jsonl(run_dir / "artifacts" / "sample_membership.jsonl")
    )
    annotations = load_all_annotations(run_dir)
    payload = write_outputs(run_dir, config, items, membership, annotations)
    complete = payload["annotated_pairs"] == payload["total_pairs"]
    update_status(
        run_dir,
        "annotation_complete" if complete else "awaiting_annotations",
        **payload,
        report=str(run_dir / "results" / "report.md"),
        dashboard=str(
            run_dir / "results" / "figures" / "semantic_audit_dashboard.html"
        ),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"REPORT={run_dir / 'results' / 'report.md'}")
    print(
        f"DASHBOARD={run_dir / 'results' / 'figures' / 'semantic_audit_dashboard.html'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
