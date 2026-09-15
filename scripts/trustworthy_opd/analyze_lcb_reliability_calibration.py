from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_run, read_jsonl, update_status

POSITIVE = "teacher_correct_student_wrong"
NEGATIVE = "both_wrong"
Q_ORDER = ["q20", "q40", "q60", "q80"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze offline LCB reliability calibration"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels == 1].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positive = int(labels.sum())
    if positive == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    boundaries = np.r_[np.flatnonzero(np.diff(ordered_scores)) + 1, len(scores)]
    cumulative_positive = np.cumsum(ordered_labels)
    previous_positive = 0
    result = 0.0
    for boundary in boundaries:
        current_positive = int(cumulative_positive[boundary - 1])
        recall_increment = (current_positive - previous_positive) / positive
        precision = current_positive / boundary
        result += recall_increment * precision
        previous_positive = current_positive
    return float(result)


def score_definitions(config: dict[str, Any]) -> list[tuple[str, str, int]]:
    definitions = [
        ("teacher_sensitivity", "− teacher sensitivity", -1),
        ("relative_sensitivity", "− relative sensitivity", -1),
        ("lcb_risk", "− absolute LCB risk", -1),
        ("risk_to_abs_reward", "− risk / |OPD reward|", -1),
    ]
    for value in config["risk"]["lambdas"]:
        slug = str(value).replace(".", "p")
        definitions.append((f"trust_lambda_{slug}", f"trust λ={value}", 1))
    return definitions


def cluster_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    direction: int,
    samples: int,
    seed: int,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    prompts = np.asarray(sorted(frame.prompt_index.unique()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(prompts), np.full(len(prompts), 1.0 / len(prompts)), size=samples
    ).astype(np.float64)
    by_q: dict[str, np.ndarray] = {}
    for q in Q_ORDER:
        subset = frame[frame.fraction_name == q]
        positive_scores = {
            int(prompt): direction
            * group.loc[group.group == POSITIVE, metric].to_numpy(float)
            for prompt, group in subset.groupby("prompt_index")
        }
        negative_scores = {
            int(prompt): direction
            * group.loc[group.group == NEGATIVE, metric].to_numpy(float)
            for prompt, group in subset.groupby("prompt_index")
        }
        positive_count = np.asarray(
            [len(positive_scores.get(int(prompt), [])) for prompt in prompts],
            dtype=np.float64,
        )
        negative_count = np.asarray(
            [len(negative_scores.get(int(prompt), [])) for prompt in prompts],
            dtype=np.float64,
        )
        pair_credit = np.zeros((len(prompts), len(prompts)), dtype=np.float64)
        for positive_index, positive_prompt in enumerate(prompts):
            positive = positive_scores.get(int(positive_prompt), np.asarray([]))
            if not len(positive):
                continue
            for negative_index, negative_prompt in enumerate(prompts):
                negative = negative_scores.get(int(negative_prompt), np.asarray([]))
                if not len(negative):
                    continue
                comparisons = positive[:, None] - negative[None, :]
                pair_credit[positive_index, negative_index] = float(
                    (comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()
                )
        numerator = np.einsum("bi,ij,bj->b", weights, pair_credit, weights)
        denominator = (weights @ positive_count) * (weights @ negative_count)
        by_q[q] = np.divide(
            numerator,
            denominator,
            out=np.full(samples, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
    stacked = np.stack([by_q[q] for q in Q_ORDER], axis=1)
    macro = np.mean(stacked, axis=1)
    interval = {
        q: tuple(
            float(value)
            for value in np.quantile(values[np.isfinite(values)], [0.025, 0.975])
        )
        if bool(np.isfinite(values).any())
        else (float("nan"), float("nan"))
        for q, values in by_q.items()
    }
    finite_macro = macro[np.isfinite(macro)]
    macro_interval = (
        tuple(float(value) for value in np.quantile(finite_macro, [0.025, 0.975]))
        if len(finite_macro)
        else (float("nan"), float("nan"))
    )
    return interval, macro_interval


def build_metric_summary(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = frame[
        frame.group.isin([POSITIVE, NEGATIVE]) & (frame.neighbor_count > 0)
    ].copy()
    bootstrap_samples = int(config["analysis"]["bootstrap_samples"])
    seed = int(config["experiment"]["seed"])
    rows = []
    fold_rows = []
    for neighborhood in sorted(primary.neighborhood.unique()):
        for representation in sorted(primary.representation.unique()):
            base = primary[
                (primary.neighborhood == neighborhood)
                & (primary.representation == representation)
            ]
            for metric_index, (metric, label, direction) in enumerate(
                score_definitions(config)
            ):
                intervals, macro_interval = cluster_bootstrap(
                    base,
                    metric,
                    direction,
                    bootstrap_samples,
                    seed + 1009 * metric_index + 37 * len(rows),
                )
                q_estimates = []
                q_ap = []
                for q in Q_ORDER:
                    subset = base[base.fraction_name == q]
                    labels = (subset.group == POSITIVE).to_numpy(np.int8)
                    scores = direction * subset[metric].to_numpy(float)
                    auc = binary_auc(labels, scores)
                    ap = average_precision(labels, scores)
                    q_estimates.append(auc)
                    q_ap.append(ap)
                    low, high = intervals[q]
                    rows.append(
                        {
                            "neighborhood": neighborhood,
                            "representation": representation,
                            "metric": metric,
                            "metric_label": label,
                            "stratum": q,
                            "n_positive": int(labels.sum()),
                            "n_negative": int(len(labels) - labels.sum()),
                            "positive_prevalence": float(labels.mean())
                            if len(labels)
                            else float("nan"),
                            "auroc": auc,
                            "auroc_ci_low": low,
                            "auroc_ci_high": high,
                            "auprc": ap,
                            "cliffs_delta": 2 * auc - 1
                            if np.isfinite(auc)
                            else float("nan"),
                        }
                    )
                macro_auc = float(np.nanmean(q_estimates))
                macro_ap = float(np.nanmean(q_ap))
                rows.append(
                    {
                        "neighborhood": neighborhood,
                        "representation": representation,
                        "metric": metric,
                        "metric_label": label,
                        "stratum": "macro_q",
                        "n_positive": int((base.group == POSITIVE).sum()),
                        "n_negative": int((base.group == NEGATIVE).sum()),
                        "positive_prevalence": float((base.group == POSITIVE).mean()),
                        "auroc": macro_auc,
                        "auroc_ci_low": macro_interval[0],
                        "auroc_ci_high": macro_interval[1],
                        "auprc": macro_ap,
                        "cliffs_delta": 2 * macro_auc - 1,
                    }
                )
                for fold in sorted(base.fold.unique()):
                    held_out = base[base.fold == fold]
                    fold_aucs = []
                    for q in Q_ORDER:
                        subset = held_out[held_out.fraction_name == q]
                        labels = (subset.group == POSITIVE).to_numpy(np.int8)
                        fold_aucs.append(
                            binary_auc(
                                labels, direction * subset[metric].to_numpy(float)
                            )
                        )
                    fold_rows.append(
                        {
                            "neighborhood": neighborhood,
                            "representation": representation,
                            "metric": metric,
                            "fold": int(fold),
                            "prompt_count": int(held_out.prompt_index.nunique()),
                            "state_count": len(held_out),
                            "macro_q_auroc": float(np.nanmean(fold_aucs)),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def build_calibration(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    analysis = config["analysis"]
    primary = frame[
        frame.group.isin([POSITIVE, NEGATIVE])
        & (frame.neighbor_count > 0)
        & (frame.neighborhood == analysis["primary_neighborhood"])
        & (frame.representation == analysis["primary_representation"])
    ].copy()
    rows = []
    bins = int(analysis["calibration_bins"])
    selected_metrics = [
        ("lcb_risk", -1),
        ("risk_to_abs_reward", -1),
    ]
    for value in (0.003, 0.01, 0.03):
        if value in [float(item) for item in config["risk"]["lambdas"]]:
            selected_metrics.append((f"trust_lambda_{str(value).replace('.', 'p')}", 1))
    for metric, direction in selected_metrics:
        for q in Q_ORDER:
            subset = primary[primary.fraction_name == q].copy()
            score = direction * subset[metric].to_numpy(float)
            percentile = (rankdata(score, method="average") - 1) / max(len(score), 1)
            subset["score_bin"] = np.minimum((percentile * bins).astype(int), bins - 1)
            subset["score"] = score
            for score_bin, group in subset.groupby("score_bin"):
                rows.append(
                    {
                        "metric": metric,
                        "fraction_name": q,
                        "score_bin": int(score_bin) + 1,
                        "n": len(group),
                        "score_mean": float(group.score.mean()),
                        "teacher_correction_rate": float(
                            (group.group == POSITIVE).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["neighborhood", "representation", "fraction_name"], as_index=False
        )
        .agg(
            states=("state_id", "size"),
            supported=("neighbor_count", lambda values: int((values > 0).sum())),
            neighbor_count_mean=("neighbor_count", "mean"),
            student_distance_mean=("student_distance_mean", "mean"),
            teacher_distance_mean=("teacher_distance_mean", "mean"),
        )
        .assign(coverage=lambda value: value.supported / value.states)
    )


def plot_forest(summary: pd.DataFrame, config: dict[str, Any], output: Path) -> None:
    analysis = config["analysis"]
    metrics = [
        "teacher_sensitivity",
        "relative_sensitivity",
        "lcb_risk",
        "risk_to_abs_reward",
    ]
    for value in (0.003, 0.01, 0.03):
        metrics.append(f"trust_lambda_{str(value).replace('.', 'p')}")
    selected = summary[
        (summary.neighborhood == analysis["primary_neighborhood"])
        & (summary.representation == analysis["primary_representation"])
        & summary.metric.isin(metrics)
    ]
    labels = [
        selected[selected.metric == metric].metric_label.iloc[0]
        for metric in metrics
        if not selected[selected.metric == metric].empty
    ]
    metrics = [
        metric for metric in metrics if not selected[selected.metric == metric].empty
    ]
    strata = Q_ORDER + ["macro_q"]
    colors = ["#2563EB", "#7C3AED", "#059669", "#EA580C", "#111827"]
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    y = np.arange(len(metrics), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(strata))
    for offset, stratum, color in zip(offsets, strata, colors):
        rows = (
            selected[selected.stratum == stratum].set_index("metric").reindex(metrics)
        )
        values = rows.auroc.to_numpy(float)
        errors = np.vstack(
            [
                values - rows.auroc_ci_low.to_numpy(float),
                rows.auroc_ci_high.to_numpy(float) - values,
            ]
        )
        ax.errorbar(
            values,
            y + offset,
            xerr=errors,
            fmt="o",
            color=color,
            capsize=3,
            label=stratum.replace("macro_q", "macro (equal-q)"),
        )
    ax.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=1.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.25, 0.8)
    ax.set_xlabel(
        "AUROC for teacher correction among student-wrong states (higher = better)"
    )
    ax.set_title(
        "Can Local Stability Identify Reliable Teacher Correction?",
        fontsize=17,
        weight="bold",
        pad=18,
    )
    ax.grid(axis="x", color="#E5E7EB")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.9, bottom=0.22)
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_representation(summary: pd.DataFrame, output: Path) -> None:
    selected = summary[(summary.stratum == "macro_q") & (summary.metric == "lcb_risk")]
    neighborhoods = sorted(selected.neighborhood.unique())
    representations = sorted(selected.representation.unique())
    x = np.arange(len(representations), dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    for index, neighborhood in enumerate(neighborhoods):
        rows = (
            selected[selected.neighborhood == neighborhood]
            .set_index("representation")
            .reindex(representations)
        )
        values = rows.auroc.to_numpy(float)
        errors = np.vstack(
            [
                values - rows.auroc_ci_low.to_numpy(float),
                rows.auroc_ci_high.to_numpy(float) - values,
            ]
        )
        ax.bar(
            x + (index - (len(neighborhoods) - 1) / 2) * width,
            values,
            width,
            yerr=errors,
            capsize=4,
            label=neighborhood,
        )
    ax.axhline(0.5, color="#6B7280", linestyle="--")
    ax.set_xticks(x, representations, rotation=15, ha="right")
    ax.set_ylim(0.3, 0.75)
    ax.set_ylabel("Equal-q macro AUROC (95% prompt-cluster bootstrap CI)")
    ax.set_title(
        "Representation and Support-Density Ablation",
        fontsize=17,
        weight="bold",
        pad=18,
    )
    ax.grid(axis="y", color="#E5E7EB")
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_calibration(calibration: pd.DataFrame, output: Path) -> None:
    metrics = list(calibration.metric.unique())
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(4.2 * len(metrics), 4.8), sharey=True
    )
    if len(metrics) == 1:
        axes = [axes]
    colors = dict(zip(Q_ORDER, ["#2563EB", "#7C3AED", "#EA580C", "#059669"]))
    for axis, metric in zip(axes, metrics):
        for q in Q_ORDER:
            rows = calibration[
                (calibration.metric == metric) & (calibration.fraction_name == q)
            ]
            axis.plot(
                rows.score_bin,
                rows.teacher_correction_rate,
                marker="o",
                color=colors[q],
                label=q,
            )
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("Reliability-score quintile (higher = predicted safer)")
        axis.set_xticks(range(1, 6))
        axis.grid(color="#E5E7EB")
    axes[0].set_ylabel("Observed teacher correction rate | S−")
    axes[-1].legend(frameon=False)
    fig.suptitle(
        "Empirical Reliability Calibration by Intervention Stage",
        fontsize=16,
        weight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.18, wspace=0.2)
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_audit(
    frame: pd.DataFrame, run_dir: Path, config: dict[str, Any]
) -> pd.DataFrame:
    from transformers import AutoTokenizer

    analysis = config["analysis"]
    primary = frame[
        frame.group.isin([POSITIVE, NEGATIVE])
        & (frame.neighbor_count > 0)
        & (frame.neighborhood == analysis["primary_neighborhood"])
        & (frame.representation == analysis["primary_representation"])
    ].copy()
    trajectories = {
        row["base_trajectory_id"]: row
        for row in read_jsonl(run_dir / "artifacts" / "trajectories.jsonl")
    }
    points = {
        row["point_id"]: row
        for row in read_jsonl(run_dir / "artifacts" / "points.jsonl")
    }
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], local_files_only=True, trust_remote_code=True
    )
    rng = np.random.default_rng(int(config["experiment"]["seed"]) + 991)
    rows = []
    per_q = int(analysis["audit_pairs_per_q"])
    for q in Q_ORDER:
        q_rows = primary[primary.fraction_name == q]
        pieces = []
        for group in (POSITIVE, NEGATIVE):
            candidates = q_rows[q_rows.group == group]
            count = min(math.ceil(per_q / 2), len(candidates))
            pieces.append(
                candidates.iloc[rng.choice(len(candidates), count, replace=False)]
            )
        for item in pd.concat(pieces).itertuples():
            neighbor_ids = json.loads(item.neighbor_point_ids)
            neighbor_id = neighbor_ids[0]
            anchor_trajectory = trajectories[item.base_trajectory_id]
            neighbor_point = points[neighbor_id]
            neighbor_trajectory = trajectories[neighbor_point["base_trajectory_id"]]
            anchor_tokens = (
                anchor_trajectory["prompt_token_ids"]
                + anchor_trajectory["generated_token_ids"][: int(item.position)]
            )
            neighbor_tokens = (
                neighbor_trajectory["prompt_token_ids"]
                + neighbor_trajectory["generated_token_ids"][
                    : int(neighbor_point["position"])
                ]
            )
            rows.append(
                {
                    "state_id": item.state_id,
                    "fraction_name": q,
                    "group": item.group,
                    "neighbor_point_id": neighbor_id,
                    "student_cosine_distance": item.student_distance_mean,
                    "teacher_cosine_distance": item.teacher_distance_mean,
                    "anchor_excerpt": tokenizer.decode(
                        anchor_tokens[-192:], skip_special_tokens=True
                    ),
                    "neighbor_excerpt": tokenizer.decode(
                        neighbor_tokens[-192:], skip_special_tokens=True
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_dashboard(
    run_dir: Path,
    config: dict[str, Any],
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    figures = run_dir / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_forest(summary, config, figures / "reliability_auroc_forest.svg")
    plot_representation(summary, figures / "representation_ablation.svg")
    calibration = pd.read_csv(run_dir / "results" / "calibration.csv")
    plot_calibration(calibration, figures / "reliability_calibration.svg")
    primary = summary[
        (summary.stratum == "macro_q")
        & (summary.neighborhood == config["analysis"]["primary_neighborhood"])
        & (summary.representation == config["analysis"]["primary_representation"])
    ].sort_values("auroc", ascending=False)
    table = primary[
        [
            "metric_label",
            "auroc",
            "auroc_ci_low",
            "auroc_ci_high",
            "auprc",
            "n_positive",
            "n_negative",
        ]
    ].to_html(index=False, float_format=lambda value: f"{value:.3f}", classes="data")
    coverage_table = coverage.to_html(
        index=False, float_format=lambda value: f"{value:.3f}", classes="data"
    )
    audit_rows = "".join(
        f"<tr><td>{html.escape(row.fraction_name)}</td><td>{html.escape(row.group)}</td>"
        f"<td><pre>{html.escape(row.anchor_excerpt)}</pre></td>"
        f"<td><pre>{html.escape(row.neighbor_excerpt)}</pre></td></tr>"
        for row in audit.itertuples()
    )
    dashboard = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LCB 可靠性校准</title>
<style>body{{font-family:system-ui;margin:0;background:#f3f4f6;color:#111827}}main{{max-width:1500px;margin:auto;padding:28px}}
.card{{background:white;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 1px 4px #0002}}img{{width:100%}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border:1px solid #d1d5db;text-align:left}}
pre{{white-space:pre-wrap;max-height:210px;overflow:auto;font-size:12px}}code{{background:#eef2ff;padding:2px 5px}}</style></head><body><main>
<h1>LCB 邻域指标能否识别教师可靠性？</h1>
<p>主终点是学生已错误时区分 <code>(T+,S−)</code> 与 <code>(T−,S−)</code>。所有区间按 prompt 聚类 bootstrap；q20/q40/q60/q80 分开，macro 对四个 q 等权。</p>
<section class="card"><h2>主结果</h2><img src="reliability_auroc_forest.svg"><p>AUROC=0.5 表示随机；大于 0.5 表示该分数把成功纠偏排在失败纠偏之前。λ 曲线不能单独证明因果训练收益。</p>{table}</section>
<section class="card"><h2>表示与支撑密度消融</h2><img src="representation_ablation.svg"><p><code>online4</code> 对齐训练每题四条 rollout；<code>dense8</code> 使用全部八条冻结 rollout。两者差异反映支撑密度敏感性。</p></section>
<section class="card"><h2>校准曲线</h2><img src="reliability_calibration.svg"><p>横轴从模型判定最不可靠到最可靠；只有纠偏成功率随分箱单调上升，指标才具有直观校准意义。</p></section>
<section class="card"><h2>邻域覆盖</h2>{coverage_table}</section>
<section class="card"><h2>邻居文本审计样本</h2><p>这是可复核的发现性样本，不等价于盲审后的语义 precision。</p><table><tr><th>q</th><th>组</th><th>anchor 尾部</th><th>最近邻尾部</th></tr>{audit_rows}</table></section>
</main></body></html>"""
    (figures / "lcb_reliability_dashboard.html").write_text(dashboard, encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    frame = pd.read_parquet(run_dir / "results" / "state_metrics.parquet")
    summary, folds = build_metric_summary(frame, config)
    calibration = build_calibration(frame, config)
    coverage = build_coverage(frame)
    audit = make_audit(frame, run_dir, config)
    summary.to_csv(run_dir / "results" / "metric_summary.csv", index=False)
    folds.to_csv(run_dir / "results" / "fold_results.csv", index=False)
    calibration.to_csv(run_dir / "results" / "calibration.csv", index=False)
    coverage.to_csv(run_dir / "results" / "coverage.csv", index=False)
    audit.to_csv(run_dir / "results" / "neighbor_audit.csv", index=False)
    write_dashboard(run_dir, config, summary, coverage, audit)

    analysis = config["analysis"]
    primary = summary[
        (summary.stratum == "macro_q")
        & (summary.neighborhood == analysis["primary_neighborhood"])
        & (summary.representation == analysis["primary_representation"])
    ].sort_values("auroc", ascending=False)
    best = primary.iloc[0]
    supported = frame[
        (frame.neighborhood == analysis["primary_neighborhood"])
        & (frame.representation == analysis["primary_representation"])
    ]
    report = [
        "# Offline LCB reliability discovery calibration",
        "",
        "The primary contrast is `(T+,S-)` versus `(T-,S-)`. This discovery run does not authorize final hyperparameter selection without an independent prompt block.",
        "",
        f"- Metric states: {frame.state_id.nunique()}",
        f"- Primary supported fraction: {(supported.neighbor_count > 0).mean():.2%}",
        f"- Best descriptive equal-q macro metric: `{best.metric}`",
        f"- AUROC: {best.auroc:.3f} (95% prompt-cluster bootstrap CI [{best.auroc_ci_low:.3f}, {best.auroc_ci_high:.3f}])",
        "",
        "## Primary metric table",
        "",
        "| metric | AUROC | 95% CI | AUPRC | n(T+,S-) | n(T-,S-) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        report.append(
            f"| {row.metric_label} | {row.auroc:.3f} | [{row.auroc_ci_low:.3f}, {row.auroc_ci_high:.3f}] | {row.auprc:.3f} | {row.n_positive} | {row.n_negative} |"
        )
    report.extend(
        [
            "",
            "Interpretation must consider per-q direction, fold stability, neighborhood coverage, and the neighbor text audit. A discovery AUROC alone is not confirmatory evidence.",
        ]
    )
    (run_dir / "results" / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    summary_payload = {
        "metric_states": int(frame.state_id.nunique()),
        "primary_supported_fraction": float((supported.neighbor_count > 0).mean()),
        "best_descriptive_metric": str(best.metric),
        "best_macro_auroc": float(best.auroc),
        "best_macro_auroc_ci": [float(best.auroc_ci_low), float(best.auroc_ci_high)],
        "confirmatory": False,
    }
    (run_dir / "results" / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    update_status(
        run_dir,
        "completed",
        result_report=str(run_dir / "results" / "report.md"),
        dashboard=str(
            run_dir / "results" / "figures" / "lcb_reliability_dashboard.html"
        ),
        **summary_payload,
    )
    print(f"REPORT={run_dir / 'results' / 'report.md'}")
    print(
        f"DASHBOARD={run_dir / 'results' / 'figures' / 'lcb_reliability_dashboard.html'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
