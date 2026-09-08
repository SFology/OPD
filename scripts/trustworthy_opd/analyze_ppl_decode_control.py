from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ppl_branch_comparison import (
    BRANCH_COLORS,
    BRANCH_LABELS,
    FRACTIONS,
    GROUP_COLORS,
    GROUP_LABELS,
    finish_four_panel,
    make_standard_trajectory_figure,
    standard_card_finding,
    standard_trajectory_summary,
)
from common import load_config, load_run, read_jsonl, update_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the equal-decoding four-group PPL control."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def load_rows(directory: Path) -> list[dict]:
    return [
        row
        for path in sorted(directory.glob("shard_*.jsonl"))
        for row in read_jsonl(path)
    ]


def load_frames(
    directory: Path, roles: list[str], primary_window: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.DataFrame(read_jsonl(directory / "score_manifest.jsonl"))
    metadata = manifest.set_index("state_id")
    expected = set(metadata.index)
    post_rows, prefix_rows = [], []
    local_key = f"local_log_ppl_w{primary_window}"
    scores: dict[tuple[str, str], dict[str, dict]] = {}
    for branch in ("teacher", "student"):
        for role in roles:
            items = {
                row["state_id"]: row
                for row in load_rows(directory / "scores" / branch / role)
            }
            missing = expected - set(items)
            if missing:
                raise RuntimeError(
                    f"Missing {len(missing)} scores for branch={branch}, role={role}"
                )
            scores[(branch, role)] = items
            for state_id in sorted(expected):
                item = items[state_id]
                meta = metadata.loc[state_id]
                for point in item["curve"]:
                    post_rows.append(
                        {
                            "state_id": state_id,
                            "role": role,
                            "branch": branch,
                            "prompt_index": int(meta.prompt_index),
                            "base_trajectory_id": meta.base_trajectory_id,
                            "fraction_name": meta.fraction_name,
                            "normalized_position": float(meta.normalized_position),
                            "group": meta.group,
                            **point,
                        }
                    )
    for role in roles:
        items = scores[("student", role)]
        for state_id in sorted(expected):
            item = items[state_id]
            meta = metadata.loc[state_id]
            for point in item["prefix_curve"]:
                prefix_rows.append(
                    {
                        "state_id": state_id,
                        "role": role,
                        "prompt_index": int(meta.prompt_index),
                        "base_trajectory_id": meta.base_trajectory_id,
                        "fraction_name": meta.fraction_name,
                        "normalized_position": float(meta.normalized_position),
                        "group": meta.group,
                        **point,
                    }
                )

    post = pd.DataFrame(post_rows)
    prefix = pd.DataFrame(prefix_rows)
    if post.empty or prefix.empty:
        raise RuntimeError("No completed equal-decoding score rows were found")
    post["trajectory_cumulative_ppl"] = np.exp(
        post.trajectory_cumulative_log_ppl.clip(upper=50)
    )
    prefix["prefix_cumulative_ppl"] = np.exp(
        prefix.prefix_cumulative_log_ppl.clip(upper=50)
    )
    for frame in (post, prefix):
        frame["local_ppl"] = np.exp(frame[local_key].clip(upper=50))
    baseline = (
        prefix[prefix.progress == 1.0]
        .set_index(["state_id", "role"])["prefix_cumulative_log_ppl"]
        .rename("intervention_cumulative_log_ppl")
    )
    prefix = prefix.join(baseline, on=["state_id", "role"])
    post = post.join(baseline, on=["state_id", "role"])
    prefix["cumulative_log_ppl_delta_from_intervention"] = (
        prefix.prefix_cumulative_log_ppl
        - prefix.intervention_cumulative_log_ppl
    )
    post["trajectory_cumulative_log_ppl_delta_from_intervention"] = (
        post.trajectory_cumulative_log_ppl
        - post.intervention_cumulative_log_ppl
    )
    return post, prefix


def make_endpoint_group_figure(
    summary: pd.DataFrame,
    role: str,
    branch: str,
    groups: list[str],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = summary[
        (summary.contrast == "endpoint_minus_intervention")
        & (summary.role == role)
        & (summary.branch == branch)
    ]
    fig, axis = plt.subplots(figsize=(15.4, 7.8))
    x = np.arange(len(FRACTIONS), dtype=float)
    width = 0.82 / max(1, len(groups))
    for index, group in enumerate(groups):
        subset = frame[frame.group == group].set_index("fraction_name").reindex(FRACTIONS)
        center = x - 0.41 + width / 2 + index * width
        values = subset.median_log_ppl_change.to_numpy(float)
        low = subset.ci_low.to_numpy(float)
        high = subset.ci_high.to_numpy(float)
        errors = np.vstack(
            [np.maximum(0, values - low), np.maximum(0, high - values)]
        )
        axis.bar(
            center,
            values,
            width=width * 0.9,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
            yerr=errors,
            capsize=3,
            alpha=0.88,
        )
    axis.axhline(0, color="#374151", linewidth=1.2, linestyle="--")
    axis.set_xticks(x, FRACTIONS)
    axis.set_xlabel("Teacher intervention fraction", labelpad=10)
    axis.set_ylabel("Endpoint cumulative log-PPL change from intervention", labelpad=10)
    axis.set_title(
        f"Equal-decoding endpoint changes | {BRANCH_LABELS[branch]} | {role} scorer",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    axis.text(
        0.5,
        1.01,
        "Bars are prompt-cluster medians; error bars are bootstrap 95% CIs; lower is better",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=min(4, len(groups)),
        frameon=False,
    )
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.23)
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_dashboard(
    directory: Path,
    cards: list[dict],
    endpoint_figures: list[dict],
    summary: pd.DataFrame,
    groups: list[str],
) -> Path:
    curve_cards = []
    for card in cards:
        curve_cards.append(
            f'<article><h2>{html.escape(card["title"])}</h2>'
            f'<a href="{html.escape(card["file"])}" target="_blank">'
            f'<img src="{html.escape(card["file"])}" alt="{html.escape(card["title"])}"></a>'
            '<div class="analysis"><strong>标准口径：</strong>每个点累计回答开始以来的全部 token；prompt 只作为条件。'
            '红线后紫色为教师续写，绿色为原学生后缀，两者均以 temperature=1.0、top-p=1.0 生成。</div>'
            f'{standard_card_finding(card, summary)}</article>'
        )
    endpoint_cards = "".join(
        f'<article><h2>{html.escape(item["title"])}</h2>'
        f'<a href="{html.escape(item["file"])}" target="_blank">'
        f'<img src="{html.escape(item["file"])}" alt="{html.escape(item["title"])}"></a>'
        '<div class="analysis">柱高是轨迹终点相对教师介入点的累计 log-PPL 变化；负值表示完整轨迹 PPL 下降。误差线为 prompt-cluster bootstrap 95% CI。</div></article>'
        for item in endpoint_figures
    )
    counts = pd.read_json(directory / "fresh_label_summary.json", typ="series")
    count_items = "".join(
        f"<li>{html.escape(GROUP_LABELS.get(group, group))}: "
        f"{int(counts['scored_state_counts'].get(group, 0))}</li>"
        for group in groups
    )
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>等解码参数四组 PPL 对照</title>
<style>body{{margin:0;background:#f3f4f6;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#111827}}main{{max-width:1580px;margin:auto;padding:26px}}header,article{{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin-bottom:22px}}header{{line-height:1.65}}article h2{{font-size:19px;margin:4px 10px 10px}}img{{display:block;width:100%;height:auto}}.analysis,.finding{{margin:10px 16px;padding:15px 19px;border-radius:10px;line-height:1.65}}.analysis{{background:#eff6ff;border:1px solid #bfdbfe}}.finding{{background:#f0fdf4;border:1px solid #bbf7d0}}.finding h3{{margin:0 0 8px}}.finding p{{margin:6px 0}}a{{color:#2563eb}}</style></head>
<body><main><header><h1>等解码参数的四组 PPL 控制实验</h1>
<p>教师和学生分支均使用 <code>temperature=1.0, top-p=1.0</code>。教师 continuation 已重新生成并由 verifier 重新分组；没有沿用旧的 T+/T− 标签。</p>
<ul>{count_items}</ul><p>核心问题：控制解码参数后，模型自偏好是否仍存在，以及这种 PPL 变化能否区分教师正确性。</p></header>
<h1>一、四组终点对比</h1>{endpoint_cards}
<h1>二、介入前后的标准全前缀 PPL 曲线</h1>{''.join(curve_cards)}
<p>数据：<a href="../trajectory_cumulative_summary.csv">trajectory_cumulative_summary.csv</a> · <a href="../branch_curve_points.csv">branch_curve_points.csv</a> · <a href="../prefix_curve_points.csv">prefix_curve_points.csv</a></p>
</main></body></html>'''
    output = directory / "results" / "figures" / "ppl_decode_control_dashboard.html"
    output.write_text(page, encoding="utf-8")
    return output


def main() -> int:
    args = parse_args()
    run_dir, _ = load_run(args.run_dir)
    directory = run_dir / "ppl_decode_control"
    config = load_config(directory / "config.yaml")
    roles = list(config["scoring"]["roles"])
    primary_window = int(config["scoring"]["primary_local_window"])
    samples = int(config["analysis"]["bootstrap_samples"])
    seed = int(config["experiment"]["seed"])
    post, prefix = load_frames(directory, roles, primary_window)
    configured_groups = list(config["analysis"]["groups"])
    observed = set(post.group.unique())
    groups = [group for group in configured_groups if group in observed]
    results = directory / "results"
    figures = results / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    post.to_parquet(results / "branch_curve_points.parquet", index=False)
    post.to_csv(results / "branch_curve_points.csv", index=False)
    prefix.to_parquet(results / "prefix_curve_points.parquet", index=False)
    prefix.to_csv(results / "prefix_curve_points.csv", index=False)

    summary = standard_trajectory_summary(post, samples, seed)
    summary.to_csv(results / "trajectory_cumulative_summary.csv", index=False)
    cards, aggregate = [], []
    metrics = (
        ("prefix_cumulative_ppl", "trajectory_cumulative_ppl", "Standard cumulative trajectory PPL"),
        (
            "cumulative_log_ppl_delta_from_intervention",
            "trajectory_cumulative_log_ppl_delta_from_intervention",
            "Cumulative log-PPL change from intervention",
        ),
    )
    for role_index, role in enumerate(roles):
        for group_index, group in enumerate(groups):
            for metric_index, (prefix_metric, post_metric, ylabel) in enumerate(metrics):
                filename = f"trajectory__{role}__{group}__{metric_index}.svg"
                aggregate.extend(
                    make_standard_trajectory_figure(
                        prefix,
                        post,
                        role,
                        group,
                        prefix_metric,
                        post_metric,
                        ylabel,
                        figures / filename,
                        samples,
                        seed + role_index * 1000 + group_index * 100 + metric_index,
                    )
                )
                cards.append(
                    {
                        "kind": "standard_trajectory",
                        "metric": prefix_metric,
                        "role": role,
                        "group": group,
                        "file": filename,
                        "title": f"{GROUP_LABELS[group]}：{ylabel}（{role} 模型打分）",
                    }
                )
    pd.DataFrame(aggregate).to_csv(results / "curve_aggregate.csv", index=False)
    endpoint_figures = []
    for role in roles:
        for branch in ("teacher", "student"):
            filename = f"endpoint_groups__{role}__{branch}.svg"
            make_endpoint_group_figure(
                summary, role, branch, groups, figures / filename
            )
            endpoint_figures.append(
                {
                    "file": filename,
                    "title": f"四组终点变化：{BRANCH_LABELS[branch]}（{role} 模型打分）",
                }
            )
    dashboard = make_dashboard(
        directory, cards, endpoint_figures, summary, groups
    )
    result = {
        "states": int(post.state_id.nunique()),
        "groups": {
            group: int(post[post.group == group].state_id.nunique())
            for group in groups
        },
        "roles": roles,
        "decoding": {"teacher": {"temperature": 1.0, "top_p": 1.0},
                     "student": {"temperature": 1.0, "top_p": 1.0}},
        "dashboard": str(dashboard),
        "trajectory_cumulative_summary": summary.to_dict(orient="records"),
    }
    (results / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    update_status(
        directory,
        "completed",
        states=result["states"],
        groups=result["groups"],
        dashboard=str(dashboard),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
