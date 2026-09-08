from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, load_run, read_jsonl, update_status


GROUPS = ["teacher_correct_student_wrong", "both_wrong"]
GROUP_LABELS = {
    "teacher_correct_student_wrong": "Teacher corrected (T+/S-)",
    "both_wrong": "Teacher failed (T-/S-)",
}
COLORS = {
    "teacher_correct_student_wrong": "#2563EB",
    "both_wrong": "#F59E0B",
}
METRICS = {
    "local_ppl": {
        "title": "Local PPL Around the Current State",
        "ylabel": "Local PPL (recent 128 tokens; lower is better)",
    },
    "local_log_ppl_delta": {
        "title": "Change in Local log-PPL Relative to Teacher Intervention",
        "ylabel": "Local log-PPL change from progress 0",
    },
    "continuation_cumulative_ppl": {
        "title": "Cumulative PPL of the Teacher Continuation",
        "ylabel": "Continuation cumulative PPL",
    },
    "trajectory_cumulative_ppl": {
        "title": "Cumulative PPL of Student Prefix + Teacher Continuation",
        "ylabel": "Trajectory cumulative PPL",
    },
}
METRIC_EXPLANATIONS = {
    "local_ppl": {
        "meaning": "每个进度点只统计最近 128 个 token 的 PPL，主要反映当前局部文本对评分模型而言是否自然。",
        "reading": "曲线下降表示最近一段文本更符合评分模型分布；蓝线低于橙线表示成功纠偏文本具有更低局部 PPL。",
    },
    "local_log_ppl_delta": {
        "meaning": "将每个状态在教师介入前（红色竖线）的局部 log-PPL 设为 0，显示后续相对变化。",
        "reading": "负值表示相对介入前下降。真正支持假设需要蓝线下降，并且下降幅度显著大于橙线。",
    },
    "continuation_cumulative_ppl": {
        "meaning": "只对教师介入后已经生成的 continuation token 累计计算 PPL，不包含原学生前缀。",
        "reading": "它衡量教师 continuation 本身是否容易被评分模型预测，但容易受到模型评价自身生成文本的自生成效应影响。",
    },
    "trajectory_cumulative_ppl": {
        "meaning": "从学生回答开始，累计计算原学生前缀和教师 continuation 的整体 PPL。",
        "reading": "该曲线保留了介入前历史，因此变化较慢；组间差异可能来自原学生前缀质量，而不一定来自教师纠偏过程。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze teacher PPL recovery curves")
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def load_scores(curve_dir: Path, roles: list[str], primary_window: int) -> pd.DataFrame:
    rows = []
    for role in roles:
        for path in sorted((curve_dir / "scores" / role).glob("shard_*.jsonl")):
            for item in read_jsonl(path):
                for point in item["curve"]:
                    row = {
                        key: item[key]
                        for key in (
                            "state_id",
                            "role",
                            "prompt_index",
                            "base_trajectory_id",
                            "fraction_name",
                            "group",
                            "prefix_tokens",
                            "teacher_continuation_tokens",
                        )
                    }
                    row.update(point)
                    rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No PPL curve scores were found")
    local = f"local_log_ppl_w{primary_window}"
    frame["local_ppl"] = np.exp(frame[local].clip(upper=50))
    frame["continuation_cumulative_ppl"] = np.exp(
        frame.continuation_cumulative_log_ppl.clip(upper=50)
    )
    frame["trajectory_cumulative_ppl"] = np.exp(
        frame.trajectory_cumulative_log_ppl.clip(upper=50)
    )
    baseline = (
        frame[frame.progress == 0.0]
        .set_index(["state_id", "role"])[local]
        .rename("local_baseline")
    )
    frame = frame.join(baseline, on=["state_id", "role"])
    frame["local_log_ppl_delta"] = frame[local] - frame.local_baseline
    return frame


def prompt_cluster_curve(
    subset: pd.DataFrame,
    metric: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for progress, point in subset.groupby("progress"):
        prompt_values = (
            point.groupby("prompt_index")[metric].median().replace([np.inf, -np.inf], np.nan).dropna()
        )
        values = prompt_values.to_numpy(float)
        if not len(values):
            continue
        sampled = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True)
        boot = np.median(sampled, axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "progress": float(progress),
                "estimate": float(np.median(values)),
                "ci_low": float(low),
                "ci_high": float(high),
                "prompt_clusters": len(values),
                "states": int(point.state_id.nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("progress")


def make_curve_figure(
    frame: pd.DataFrame,
    role: str,
    metric: str,
    output: Path,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fractions = ["q20", "q40", "q60", "q80"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), sharex=True)
    curve_rows = []
    for axis, fraction in zip(axes.flat, fractions):
        fraction_frame = frame[
            (frame.role == role) & (frame.fraction_name == fraction)
        ]
        for group in GROUPS:
            subset = fraction_frame[fraction_frame.group == group]
            curve = prompt_cluster_curve(subset, metric, bootstrap_samples, rng)
            if curve.empty:
                continue
            x = 100 * curve.progress.to_numpy()
            estimate = curve.estimate.to_numpy()
            low = curve.ci_low.to_numpy()
            high = curve.ci_high.to_numpy()
            axis.plot(
                x,
                estimate,
                marker="o",
                linewidth=2.2,
                markersize=4.5,
                color=COLORS[group],
                label=GROUP_LABELS[group],
            )
            axis.fill_between(x, low, high, color=COLORS[group], alpha=0.16)
            for row in curve.to_dict(orient="records"):
                curve_rows.append(
                    {
                        "role": role,
                        "metric": metric,
                        "fraction_name": fraction,
                        "group": group,
                        **row,
                    }
                )
        if metric == "local_log_ppl_delta":
            axis.axhline(0, color="#6B7280", linewidth=1, linestyle="--")
        axis.axvline(
            0,
            color="#DC2626",
            linewidth=2.4,
            linestyle="--",
            zorder=6,
        )
        axis.text(
            0.025,
            0.96,
            f"Teacher intervention: {fraction}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="#B91C1C",
            bbox={"facecolor": "white", "edgecolor": "#FCA5A5", "alpha": 0.88, "pad": 3},
        )
        counts = fraction_frame.groupby("group").state_id.nunique().to_dict()
        axis.set_title(
            f"{fraction} | corrected n={counts.get(GROUPS[0], 0)}, "
            f"failed n={counts.get(GROUPS[1], 0)}",
            fontsize=11.5,
            fontweight="bold",
        )
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(-7, 102)
        axis.set_xticks(
            np.arange(0, 101, 20),
            ["0\nstart", "20", "40", "60", "80", "100\nend"],
        )
        axis.tick_params(axis="x", labelbottom=True, pad=5)
    for axis in axes[:, 0]:
        axis.set_ylabel(METRICS[metric]["ylabel"])
    for axis in axes.flat:
        axis.set_xlabel("Teacher continuation progress after intervention (%)", labelpad=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10.5)
    fig.suptitle(
        f"{METRICS[metric]['title']} | scored by {role} model",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        "Lines are medians across prompt-level medians; bands are prompt-cluster bootstrap 95% CIs",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.855,
        bottom=0.105,
        hspace=0.38,
        wspace=0.2,
    )
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return curve_rows


def bootstrap_median(values: np.ndarray, samples: int, rng) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    sampled = rng.choice(values, size=(samples, len(values)), replace=True)
    medians = np.median(sampled, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(np.median(values)), float(low), float(high)


def endpoint_summary(
    frame: pd.DataFrame,
    primary_window: int,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    local = f"local_log_ppl_w{primary_window}"
    endpoints = frame[frame.progress.isin([0.0, 1.0])].pivot_table(
        index=["state_id", "role", "prompt_index", "fraction_name", "group"],
        columns="progress",
        values=local,
    )
    endpoints = endpoints.dropna().reset_index()
    endpoints["endpoint_minus_baseline_log_ppl"] = endpoints[1.0] - endpoints[0.0]
    rng = np.random.default_rng(seed + 93000)
    rows = []
    for (role, fraction, group), subset in endpoints.groupby(
        ["role", "fraction_name", "group"]
    ):
        prompt_delta = subset.groupby("prompt_index").endpoint_minus_baseline_log_ppl.median()
        estimate, low, high = bootstrap_median(prompt_delta.to_numpy(), samples, rng)
        rows.append(
            {
                "role": role,
                "fraction_name": fraction,
                "group": group,
                "contrast": "endpoint_minus_baseline_within_group",
                "states": subset.state_id.nunique(),
                "prompt_clusters": prompt_delta.size,
                "median_endpoint_minus_baseline_log_ppl": estimate,
                "cluster_bootstrap_ci_low": low,
                "cluster_bootstrap_ci_high": high,
            }
        )
    for (role, fraction), subset in endpoints.groupby(["role", "fraction_name"]):
        by_prompt = subset.pivot_table(
            index="prompt_index",
            columns="group",
            values="endpoint_minus_baseline_log_ppl",
            aggfunc="median",
        )
        if not set(GROUPS).issubset(by_prompt.columns):
            continue
        paired = by_prompt.dropna(subset=GROUPS)
        difference = (
            paired["teacher_correct_student_wrong"] - paired["both_wrong"]
        ).to_numpy(float)
        estimate, low, high = bootstrap_median(difference, samples, rng)
        rows.append(
            {
                "role": role,
                "fraction_name": fraction,
                "group": "corrected_minus_failed",
                "contrast": "difference_in_endpoint_change",
                "states": int(subset.state_id.nunique()),
                "prompt_clusters": len(difference),
                "median_endpoint_minus_baseline_log_ppl": estimate,
                "cluster_bootstrap_ci_low": low,
                "cluster_bootstrap_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def formatted_q_values(values: pd.Series) -> str:
    return "、".join(
        f"{fraction}={value:.3f}" for fraction, value in values.items()
    )


def figure_analysis_html(
    role: str,
    metric: str,
    aggregate: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> str:
    endpoint = aggregate[
        (aggregate.role == role)
        & (aggregate.metric == metric)
        & (aggregate.progress == 1.0)
    ]
    corrected = (
        endpoint[endpoint.group == GROUPS[0]]
        .set_index("fraction_name")
        .estimate.reindex(["q20", "q40", "q60", "q80"])
    )
    failed = (
        endpoint[endpoint.group == GROUPS[1]]
        .set_index("fraction_name")
        .estimate.reindex(["q20", "q40", "q60", "q80"])
    )
    role_text = "教师模型" if role == "teacher" else "学生模型"
    if metric == "local_ppl" and role == "teacher":
        finding = (
            "教师介入后两组都迅速收敛到相近的局部 PPL。终点蓝线为 "
            f"{formatted_q_values(corrected)}；橙线为 {formatted_q_values(failed)}。"
            "四个分位点没有一致的成功组优势。"
        )
        conclusion = "局部 PPL 下降主要反映教师对自身生成文本更熟悉，不能单独作为成功纠偏证据。"
    elif metric == "local_ppl":
        finding = (
            f"学生模型评分下，终点蓝线为 {formatted_q_values(corrected)}；"
            f"橙线为 {formatted_q_values(failed)}。蓝线普遍更低，但后续相对介入点并未持续下降。"
        )
        conclusion = "该组间差异更可能在教师介入前就已存在，反映可纠偏轨迹本身更连贯。"
    elif metric == "local_log_ppl_delta":
        contrast = endpoints[
            (endpoints.role == role)
            & (endpoints.contrast == "difference_in_endpoint_change")
        ]
        contains_zero = bool(
            (
                (contrast.cluster_bootstrap_ci_low <= 0)
                & (contrast.cluster_bootstrap_ci_high >= 0)
            ).all()
        )
        finding = (
            f"终点蓝线变化为 {formatted_q_values(corrected)}；"
            f"橙线变化为 {formatted_q_values(failed)}。"
        )
        if role == "teacher":
            finding += "两组均明显下降，而且失败组在 q40–q80 下降并不更少。"
        else:
            finding += "学生评分曲线总体接近 0，q60/q80 的成功组甚至有所上升。"
        conclusion = (
            "所有成功减失败的 95% CI 都包含 0，尚无证据表明成功纠偏具有更强的 PPL 下降。"
            if contains_zero
            else "至少一个分位点的成功减失败置信区间未包含 0，需要结合对应面板判断方向。"
        )
    elif metric == "continuation_cumulative_ppl":
        corrected_lower = int((corrected < failed).sum())
        finding = (
            f"终点蓝线为 {formatted_q_values(corrected)}；橙线为 {formatted_q_values(failed)}。"
            f"成功组只在 {corrected_lower}/4 个分位点更低。"
        )
        conclusion = (
            "教师模型对成功 continuation 并没有一致赋予更低累计 PPL；这进一步否定了简单的 PPL 恢复判据。"
            if role == "teacher"
            else "学生模型下组间差异较小，说明教师是否纠偏成功不能由 continuation 累计 PPL 稳定识别。"
        )
    else:
        corrected_lower = int((corrected < failed).sum())
        finding = (
            f"终点蓝线为 {formatted_q_values(corrected)}；橙线为 {formatted_q_values(failed)}。"
            f"成功组在 {corrected_lower}/4 个分位点具有更低整体 PPL。"
        )
        conclusion = (
            "整体轨迹 PPL 有区分力，但它累积了介入前学生前缀，不能证明 PPL 是在纠偏过程中下降的。"
        )
    explanation = METRIC_EXPLANATIONS[metric]
    return f'''<div class="analysis"><h3>解释与分析</h3><dl>
<dt>指标含义</dt><dd>{html.escape(explanation["meaning"])}</dd>
<dt>如何读图</dt><dd>{html.escape(explanation["reading"])}</dd>
<dt>当前发现</dt><dd>{html.escape(finding)}</dd>
<dt>结论</dt><dd><strong>{html.escape(conclusion)}</strong></dd>
</dl><p class="model-note">本图由{role_text}评分；阴影表示 prompt-cluster bootstrap 95% CI。</p></div>'''


def make_dashboard(
    curve_dir: Path,
    figure_files: list[tuple[str, str, str]],
    aggregate: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> None:
    cards = "".join(
        f'<article><h2>{html.escape(title)}</h2><a href="{html.escape(filename)}" target="_blank"><img src="{html.escape(filename)}" alt="{html.escape(title)}"></a>{figure_analysis_html(role, metric, aggregate, endpoints)}</article>'
        for role, metric, title, filename in figure_files
    )
    dashboard = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PPL 纠偏曲线看板</title>
<style>body{{margin:0;background:#f3f4f6;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#111827}}main{{max-width:1580px;margin:auto;padding:26px}}h1{{margin-bottom:8px}}.note{{line-height:1.65;color:#4b5563}}.warn{{background:#fff7ed;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:6px}}.grid{{display:grid;grid-template-columns:1fr;gap:22px;margin-top:22px}}article{{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:12px;box-shadow:0 2px 8px #0000000d}}article h2{{font-size:19px;margin:8px 12px 10px}}img{{display:block;width:100%;height:auto}}a{{color:#2563eb}}.analysis{{margin:8px 16px 16px;padding:16px 20px;background:#f8fafc;border:1px solid #dbeafe;border-radius:10px;line-height:1.65}}.analysis h3{{margin:0 0 9px;font-size:17px}}.analysis dl{{display:grid;grid-template-columns:90px 1fr;gap:7px 14px;margin:0}}.analysis dt{{font-weight:700;color:#1e3a8a}}.analysis dd{{margin:0;color:#374151}}.model-note{{margin:10px 0 0;color:#6b7280;font-size:13px}}@media(max-width:800px){{main{{padding:12px}}.analysis dl{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>PPL 与教师纠偏过程</h1>
<p class="note">每个面板左侧的红色虚线是教师介入点：q20/q40/q60/q80 表示它位于原学生轨迹的相应分位点。图中横轴从红线开始重新计为教师 continuation 的 0%–100%，两种百分比不要混淆。蓝线是教师成功纠偏，橙线是教师仍然失败；局部 PPL 使用最近 128 个 token。</p>
<p class="warn"><strong>关键判据：</strong>不能只看蓝线是否下降，还要比较蓝线是否比橙线下降得更多。否则观察到的下降可能只是“模型对自身生成文本具有较低 PPL”的普遍现象。</p>
<div class="grid">{cards}</div>
<p class="note">数据文件：<a href="../ppl_curve_points.csv">ppl_curve_points.csv</a> · <a href="../curve_aggregate.csv">curve_aggregate.csv</a> · <a href="../endpoint_change_summary.csv">endpoint_change_summary.csv</a></p>
</main></body></html>'''
    (curve_dir / "results" / "figures" / "ppl_recovery_dashboard.html").write_text(
        dashboard, encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    run_dir, _ = load_run(args.run_dir)
    curve_dir = run_dir / "ppl_recovery_curve"
    config = load_config(curve_dir / "config.yaml")
    roles = list(config["scoring"]["roles"])
    primary_window = int(config["scoring"]["primary_local_window"])
    samples = int(config["analysis"]["bootstrap_samples"])
    seed = int(config["experiment"]["seed"])
    frame = load_scores(curve_dir, roles, primary_window)
    results = curve_dir / "results"
    figures = results / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(results / "ppl_curve_points.parquet", index=False)
    frame.to_csv(results / "ppl_curve_points.csv", index=False)

    aggregate_rows, figure_files = [], []
    for role in roles:
        for metric, metadata in METRICS.items():
            filename = f"{role}__{metric}_curve.svg"
            aggregate_rows.extend(
                make_curve_figure(
                    frame,
                    role,
                    metric,
                    figures / filename,
                    samples,
                    seed + 1000 * roles.index(role) + 100 * list(METRICS).index(metric),
                )
            )
            figure_files.append(
                (role, metric, f"{metadata['title']}（{role} 模型打分）", filename)
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(results / "curve_aggregate.csv", index=False)
    endpoints = endpoint_summary(frame, primary_window, samples, seed)
    endpoints.to_csv(results / "endpoint_change_summary.csv", index=False)
    make_dashboard(curve_dir, figure_files, aggregate, endpoints)
    summary = {
        "states": int(frame.state_id.nunique()),
        "roles": roles,
        "primary_local_window": primary_window,
        "endpoint_changes": endpoints.to_dict(orient="records"),
        "dashboard": str(figures / "ppl_recovery_dashboard.html"),
    }
    (results / "ppl_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    update_status(
        curve_dir,
        "completed",
        states=summary["states"],
        roles=roles,
        dashboard=summary["dashboard"],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
