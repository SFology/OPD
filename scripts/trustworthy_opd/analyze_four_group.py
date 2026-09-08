from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from common import load_run, update_status
from scipy.stats import kruskal, mannwhitneyu

GROUP_ORDER = [
    "teacher_correct_student_wrong",
    "teacher_wrong_student_correct",
    "both_wrong",
    "both_correct",
]
GROUP_LABELS = ["T+/S−", "T−/S+", "T−/S−", "T+/S+"]
GROUP_LONG_LABELS = [
    "教师正确 / 学生错误",
    "教师错误 / 学生正确",
    "教师和学生都错误",
    "教师和学生都正确",
]
GROUP_LONG_LABELS_EN = [
    "teacher correct / student wrong",
    "teacher wrong / student correct",
    "both wrong",
    "both correct",
]
GROUP_COLORS = ["#2563EB", "#DC2626", "#F59E0B", "#059669"]
METRICS = [
    "teacher_log_ppl",
    "student_log_ppl",
    "teacher_student_log_ppl_gap",
    "teacher_stability",
    "student_stability",
    "relative_stability",
]
METRIC_DISPLAY = {
    "teacher_log_ppl": {
        "title": "教师前缀 log-PPL",
        "title_en": "Teacher Prefix log-PPL",
        "ylabel": "教师 log-PPL（越低表示前缀越熟悉）",
        "ylabel_en": "Teacher log-PPL (lower = more familiar prefix)",
        "note": "柱高为中位数；黑色误差线为四分位区间（IQR）。",
    },
    "student_log_ppl": {
        "title": "学生前缀 log-PPL",
        "title_en": "Student Prefix log-PPL",
        "ylabel": "学生 log-PPL（越低表示前缀越熟悉）",
        "ylabel_en": "Student log-PPL (lower = more familiar prefix)",
        "note": "柱高为中位数；黑色误差线为四分位区间（IQR）。",
    },
    "teacher_student_log_ppl_gap": {
        "title": "教师－学生 log-PPL 差值",
        "title_en": "Teacher - Student log-PPL Gap",
        "ylabel": "教师 log-PPL − 学生 log-PPL",
        "ylabel_en": "Teacher log-PPL - Student log-PPL",
        "note": "正值表示教师对该前缀的 PPL 高于学生。",
    },
    "teacher_stability": {
        "title": "教师邻域 log-prob 变化",
        "title_en": "Teacher Neighborhood log-prob Change",
        "ylabel": "教师单侧邻域变化（越高表示局部对比越强）",
        "ylabel_en": "One-sided teacher change (higher = stronger local contrast)",
        "note": "该量是单侧最大 log-prob 下降，不应直接解释为可信度。",
    },
    "student_stability": {
        "title": "学生邻域 log-prob 变化",
        "title_en": "Student Neighborhood log-prob Change",
        "ylabel": "学生单侧邻域变化（越高表示局部对比越强）",
        "ylabel_en": "One-sided student change (higher = stronger local contrast)",
        "note": "该量是单侧最大 log-prob 下降，不应直接解释为可信度。",
    },
    "relative_stability": {
        "title": "教师－学生相对邻域变化",
        "title_en": "Relative Neighborhood Change (Teacher - Student)",
        "ylabel": "教师邻域变化 − 学生邻域变化",
        "ylabel_en": "Teacher neighborhood change - Student neighborhood change",
        "note": "正值表示教师的局部变化大于学生；柱高为中位数。",
    },
}
REPRESENTATION_LABELS = {
    "mid_prefix_mean": "中间层：整个前缀均值",
    "mid_tail_mean_8": "中间层：最后 8 个 token 均值",
    "final_last": "最后一层：最后一个 token",
}
REPRESENTATION_LABELS_EN = {
    "mid_prefix_mean": "mid layer / full-prefix mean",
    "mid_tail_mean_8": "mid layer / last-8-token mean",
    "final_last": "final layer / last token",
}
CONTRASTS = [
    (
        "teacher_correct_with_student_wrong",
        "teacher_correct_student_wrong",
        "both_wrong",
    ),
    (
        "teacher_correct_with_student_correct",
        "both_correct",
        "teacher_wrong_student_correct",
    ),
    (
        "which_model_to_trust",
        "teacher_correct_student_wrong",
        "teacher_wrong_student_correct",
    ),
    ("general_difficulty", "both_correct", "both_wrong"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze four-group metrics by fraction"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures/dashboard without rerunning statistical tests.",
    )
    return parser.parse_args()


def holm(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(array))
    order = finite[np.argsort(array[finite])]
    adjusted = np.full(len(array), np.nan, dtype=float)
    running = 0.0
    count = len(finite)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * array[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    finite = np.flatnonzero(np.isfinite(array))
    result = np.full(len(array), np.nan, dtype=float)
    if not len(finite):
        return pd.Series(result, index=values.index)
    order = finite[np.argsort(array[finite])]
    count = len(order)
    adjusted = array[order] * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def auc_effect(first: np.ndarray, second: np.ndarray) -> float:
    if not len(first) or not len(second):
        return float("nan")
    result = mannwhitneyu(first, second, alternative="two-sided")
    return float(result.statistic / (len(first) * len(second)))


def cluster_bootstrap_auc(
    frame: pd.DataFrame,
    metric: str,
    first_group: str,
    second_group: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    prompts = frame.prompt_index.unique()
    values = []
    grouped = {prompt: group for prompt, group in frame.groupby("prompt_index")}
    for _ in range(samples):
        sampled = rng.choice(prompts, len(prompts), replace=True)
        pieces = [grouped[prompt] for prompt in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        first = boot.loc[boot.group == first_group, metric].to_numpy(float)
        second = boot.loc[boot.group == second_group, metric].to_numpy(float)
        value = auc_effect(first, second)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(item) for item in np.quantile(values, [0.025, 0.975]))


def nice_ticks(low: float, high: float, count: int = 6) -> list[float]:
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = 0.0, 1.0
    raw = (high - low) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(max(raw, 1e-12)))
    normalized = raw / magnitude
    step = next(item for item in (1, 2, 2.5, 5, 10) if item >= normalized) * magnitude
    start = math.floor(low / step) * step
    stop = math.ceil(high / step) * step
    return [start + index * step for index in range(round((stop - start) / step) + 1)]


def format_tick(value: float) -> str:
    if abs(value) >= 100 or (0 < abs(value) < 0.01):
        return f"{value:.1e}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def grouped_bar_svg(
    summary: pd.DataFrame,
    metric: str,
    representation: str,
) -> str:
    width, height = 1280, 610
    left, right, top, bottom = 115, 30, 125, 125
    plot_width, plot_height = width - left - right, height - top - bottom
    fractions = sorted(summary.fraction_name.unique(), key=lambda item: int(item[1:]))
    metric_rows = summary[summary.metric == metric]
    q_low = float(metric_rows.q25.min())
    q_high = float(metric_rows.q75.max())
    span = max(q_high - q_low, abs(q_high) * 0.08, 0.1)
    scale_low = min(0.0, q_low - 0.12 * span)
    scale_high = max(0.0, q_high + 0.18 * span)
    ticks = nice_ticks(scale_low, scale_high)
    scale_low, scale_high = ticks[0], ticks[-1]

    def y(value: float) -> float:
        return top + (scale_high - value) / (scale_high - scale_low) * plot_height

    meta = METRIC_DISPLAY[metric]
    rep_label = REPRESENTATION_LABELS.get(representation, representation)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" rx="12" fill="#FFFFFF"/>',
        f'<text x="{left}" y="32" font-family="sans-serif" font-size="23" font-weight="700" fill="#111827">{html.escape(meta["title"])}</text>',
        f'<text x="{left}" y="57" font-family="sans-serif" font-size="14" fill="#4B5563">状态表征：{html.escape(rep_label)}　·　{html.escape(meta["note"])}</text>',
    ]
    legend_x = left
    for label, long_label, color in zip(GROUP_LABELS, GROUP_LONG_LABELS, GROUP_COLORS):
        parts.extend(
            [
                f'<rect x="{legend_x}" y="76" width="16" height="16" rx="2" fill="{color}"/>',
                f'<text x="{legend_x + 23}" y="89" font-family="sans-serif" font-size="13" fill="#374151">{html.escape(label)}：{html.escape(long_label)}</text>',
            ]
        )
        legend_x += 270
    for tick in ticks:
        position = y(tick)
        stroke = "#9CA3AF" if abs(tick) < 1e-12 else "#E5E7EB"
        stroke_width = "1.5" if abs(tick) < 1e-12 else "1"
        parts.extend(
            [
                f'<line x1="{left}" y1="{position:.2f}" x2="{width-right}" y2="{position:.2f}" stroke="{stroke}" stroke-width="{stroke_width}"/>',
                f'<text x="{left-12}" y="{position+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#4B5563">{format_tick(tick)}</text>',
            ]
        )
    cluster_width = plot_width / len(fractions)
    bar_width = min(48.0, cluster_width * 0.16)
    group_gap = bar_width * 0.18
    total_bar_width = len(GROUP_ORDER) * bar_width + (len(GROUP_ORDER) - 1) * group_gap
    zero_y = y(0.0)
    for fraction_index, fraction in enumerate(fractions):
        center = left + (fraction_index + 0.5) * cluster_width
        cluster_left = center - total_bar_width / 2
        counts = []
        for group_index, (group, label, color) in enumerate(
            zip(GROUP_ORDER, GROUP_LABELS, GROUP_COLORS)
        ):
            row = metric_rows[
                (metric_rows.fraction_name == fraction) & (metric_rows.group == group)
            ]
            if row.empty:
                counts.append(0)
                continue
            item = row.iloc[0]
            median, q25, q75, n = (
                float(item["median"]),
                float(item["q25"]),
                float(item["q75"]),
                int(item["n"]),
            )
            counts.append(n)
            x = cluster_left + group_index * (bar_width + group_gap)
            median_y = y(median)
            rect_y = min(zero_y, median_y)
            rect_height = max(1.0, abs(zero_y - median_y))
            error_x = x + bar_width / 2
            value_y = median_y - 7 if median >= 0 else median_y + 17
            tooltip = (
                f"{fraction} · {label} · n={n} · median={median:.4g} · "
                f"IQR=[{q25:.4g}, {q75:.4g}]"
            )
            parts.extend(
                [
                    f'<g><title>{html.escape(tooltip)}</title>',
                    f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_width:.2f}" height="{rect_height:.2f}" rx="3" fill="{color}" fill-opacity="0.88"/>',
                    f'<line x1="{error_x:.2f}" y1="{y(q25):.2f}" x2="{error_x:.2f}" y2="{y(q75):.2f}" stroke="#111827" stroke-width="2"/>',
                    f'<line x1="{error_x-7:.2f}" y1="{y(q25):.2f}" x2="{error_x+7:.2f}" y2="{y(q25):.2f}" stroke="#111827" stroke-width="2"/>',
                    f'<line x1="{error_x-7:.2f}" y1="{y(q75):.2f}" x2="{error_x+7:.2f}" y2="{y(q75):.2f}" stroke="#111827" stroke-width="2"/>',
                    f'<text x="{error_x:.2f}" y="{value_y:.2f}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#111827">{format_tick(median)}</text>',
                    "</g>",
                ]
            )
        parts.extend(
            [
                f'<text x="{center:.2f}" y="{top+plot_height+27}" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="700" fill="#111827">{html.escape(fraction)}（轨迹 {fraction[1:]}%）</text>',
                f'<text x="{center:.2f}" y="{top+plot_height+48}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B7280">n（蓝/红/橙/绿）={"/".join(map(str, counts))}</text>',
            ]
        )
        if fraction_index:
            separator = left + fraction_index * cluster_width
            parts.append(
                f'<line x1="{separator:.2f}" y1="{top}" x2="{separator:.2f}" y2="{top+plot_height}" stroke="#D1D5DB" stroke-dasharray="4 5"/>'
            )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#374151" stroke-width="1.5"/>',
            f'<text x="{left+plot_width/2:.2f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="600" fill="#111827">学生轨迹中的状态位置（q20 / q40 / q60 / q80）</text>',
            f'<text x="25" y="{top+plot_height/2:.2f}" transform="rotate(-90 25 {top+plot_height/2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="600" fill="#111827">{html.escape(meta["ylabel"])}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def matplotlib_grouped_bar(
    summary: pd.DataFrame,
    metric: str,
    representation: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = sorted(summary.fraction_name.unique(), key=lambda item: int(item[1:]))
    metric_rows = summary[summary.metric == metric]
    x = np.arange(len(fractions), dtype=float)
    bar_width = 0.19
    offsets = (np.arange(len(GROUP_ORDER)) - 1.5) * bar_width
    fig, ax = plt.subplots(figsize=(13.5, 6.8), constrained_layout=False)
    for group_index, (group, short, long_label, color) in enumerate(
        zip(GROUP_ORDER, GROUP_LABELS, GROUP_LONG_LABELS_EN, GROUP_COLORS)
    ):
        medians, lower, upper = [], [], []
        for fraction in fractions:
            row = metric_rows[
                (metric_rows.fraction_name == fraction) & (metric_rows.group == group)
            ]
            if row.empty:
                medians.append(np.nan)
                lower.append(0.0)
                upper.append(0.0)
                continue
            item = row.iloc[0]
            median = float(item["median"])
            medians.append(median)
            lower.append(max(0.0, median - float(item["q25"])))
            upper.append(max(0.0, float(item["q75"]) - median))
        bars = ax.bar(
            x + offsets[group_index],
            medians,
            bar_width,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            label=f"{short}: {long_label}",
            yerr=np.asarray([lower, upper]),
            error_kw={"ecolor": "#111827", "elinewidth": 1.4, "capsize": 3},
            zorder=3,
        )
        labels = ["" if not np.isfinite(value) else format_tick(value) for value in medians]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8, color="#111827")

    counts_by_fraction = []
    for fraction in fractions:
        counts = []
        for group in GROUP_ORDER:
            row = metric_rows[
                (metric_rows.fraction_name == fraction) & (metric_rows.group == group)
            ]
            counts.append(0 if row.empty else int(row.iloc[0]["n"]))
        counts_by_fraction.append(counts)
    ax.set_xticks(x, [f"{fraction}\n({fraction[1:]}% of trajectory)" for fraction in fractions])
    ax.set_xlabel("State position along the student trajectory", fontsize=12, labelpad=28)
    ax.set_ylabel(METRIC_DISPLAY[metric]["ylabel_en"], fontsize=12, labelpad=10)
    ax.set_title(METRIC_DISPLAY[metric]["title_en"], fontsize=18, fontweight="bold", pad=56)
    ax.text(
        0.5,
        1.105,
        f"Representation: {REPRESENTATION_LABELS_EN.get(representation, representation)} | bar = median, error bar = IQR",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4B5563",
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handlelength=1.4,
    )
    for position, counts in zip(x, counts_by_fraction):
        ax.text(
            position,
            -0.14,
            "n (blue/red/orange/green) = " + "/".join(map(str, counts)),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            color="#6B7280",
        )
    ax.axhline(0, color="#6B7280", linewidth=1.1, zorder=2)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9CA3AF")
    ax.tick_params(axis="both", colors="#374151")
    fig.subplots_adjust(left=0.1, right=0.985, top=0.77, bottom=0.23)
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_intervention_success_figure(
    frame: pd.DataFrame,
    output: Path,
    primary_representation: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = frame[frame.representation == primary_representation].drop_duplicates(
        "state_id"
    )
    student_wrong = primary[
        primary.group.isin(["teacher_correct_student_wrong", "both_wrong"])
    ].copy()
    student_wrong["teacher_corrected"] = (
        student_wrong.group == "teacher_correct_student_wrong"
    ).astype(int)
    fractions = sorted(
        student_wrong.fraction_name.unique(), key=lambda item: int(item[1:])
    )
    prompts = np.sort(student_wrong.prompt_index.unique())
    grouped = (
        student_wrong.groupby(["prompt_index", "fraction_name"])["teacher_corrected"]
        .agg(["sum", "count"])
        .reindex(pd.MultiIndex.from_product([prompts, fractions]), fill_value=0)
    )
    success = grouped["sum"].to_numpy().reshape(len(prompts), len(fractions))
    total = grouped["count"].to_numpy().reshape(len(prompts), len(fractions))
    rng = np.random.default_rng(seed + 91000)
    sampled = rng.integers(0, len(prompts), size=(bootstrap_samples, len(prompts)))
    boot_success = success[sampled].sum(axis=1)
    boot_total = total[sampled].sum(axis=1)
    boot_rates = np.divide(
        boot_success,
        boot_total,
        out=np.full_like(boot_success, np.nan, dtype=float),
        where=boot_total > 0,
    )
    rows = []
    for index, fraction in enumerate(fractions):
        successes = int(success[:, index].sum())
        denominator = int(total[:, index].sum())
        rate = successes / denominator
        finite = boot_rates[:, index][np.isfinite(boot_rates[:, index])]
        low, high = np.quantile(finite, [0.025, 0.975])
        rows.append(
            {
                "fraction_name": fraction,
                "trajectory_fraction": int(fraction[1:]) / 100,
                "teacher_corrections": successes,
                "student_wrong_valid_states": denominator,
                "correction_rate": rate,
                "cluster_bootstrap_ci_low": float(low),
                "cluster_bootstrap_ci_high": float(high),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "teacher_intervention_success_by_fraction.csv", index=False)
    bootstrap_decline = 100 * (boot_rates[:, -1] - boot_rates[:, 0])
    decline_low, decline_high = np.quantile(bootstrap_decline, [0.025, 0.975])
    trend = {
        "q80_minus_q20_percentage_points": float(
            100 * (result.correction_rate.iloc[-1] - result.correction_rate.iloc[0])
        ),
        "cluster_bootstrap_ci_low": float(decline_low),
        "cluster_bootstrap_ci_high": float(decline_high),
        "bootstrap_probability_nonnegative": float(np.mean(bootstrap_decline >= 0)),
        "bootstrap_samples": int(bootstrap_samples),
    }
    (output / "teacher_intervention_trend_summary.json").write_text(
        json.dumps(trend, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    x = np.arange(len(result))
    rates = result.correction_rate.to_numpy()
    low_error = rates - result.cluster_bootstrap_ci_low.to_numpy()
    high_error = result.cluster_bootstrap_ci_high.to_numpy() - rates
    fig, ax = plt.subplots(figsize=(13.5, 6.3))
    bars = ax.bar(
        x,
        rates,
        width=0.58,
        color=["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD"],
        edgecolor="white",
        linewidth=1,
        zorder=2,
    )
    ax.errorbar(
        x,
        rates,
        yerr=np.asarray([low_error, high_error]),
        fmt="none",
        ecolor="#111827",
        elinewidth=1.8,
        capsize=6,
        zorder=4,
    )
    ax.plot(x, rates, color="#1E3A8A", marker="o", markersize=7, linewidth=2.5, zorder=5)
    for bar, item in zip(bars, result.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            item.correction_rate + item.cluster_bootstrap_ci_high - item.correction_rate + 0.018,
            f"{item.correction_rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#111827",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.025,
            f"{item.teacher_corrections}/{item.student_wrong_valid_states}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2937",
        )
    decline = trend["q80_minus_q20_percentage_points"]
    ax.annotate(
        f"q20 to q80: {decline:.1f} pp\n95% CI [{decline_low:.1f}, {decline_high:.1f}]",
        xy=(x[-1], rates[-1]),
        xytext=(x[-1] - 0.85, min(0.82, rates[0] + 0.16)),
        arrowprops={"arrowstyle": "->", "color": "#B91C1C", "lw": 1.8},
        fontsize=11,
        fontweight="bold",
        color="#B91C1C",
        ha="center",
    )
    ax.set_xticks(
        x,
        [
            f"{item.fraction_name}\nTeacher intervenes at {int(100*item.trajectory_fraction)}%"
            for item in result.itertuples()
        ],
    )
    ax.set_ylim(0, max(0.75, float(result.cluster_bootstrap_ci_high.max()) + 0.1))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("Teacher intervention point along the student trajectory", fontsize=12, labelpad=14)
    ax.set_ylabel("Teacher correction success rate | student is wrong", fontsize=12, labelpad=10)
    ax.set_title(
        "Later Teacher Intervention Is Less Likely to Correct the Student Trajectory",
        fontsize=17,
        fontweight="bold",
        pad=30,
    )
    ax.text(
        0.5,
        1.035,
        "Bars use valid, parseable student-wrong states; error bars are prompt-cluster bootstrap 95% CIs",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4B5563",
    )
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9CA3AF")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.18)
    fig.savefig(
        output / "teacher_intervention_success_by_fraction.svg",
        format="svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return result, trend


def make_figures(
    frame: pd.DataFrame,
    output: Path,
    primary_representation: str,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    intervention, intervention_trend = make_intervention_success_figure(
        frame, output, primary_representation, bootstrap_samples, seed
    )
    summaries = []
    for (representation, fraction, group), subset in frame.groupby(
        ["representation", "fraction_name", "group"]
    ):
        subset = subset.drop_duplicates("state_id")
        for metric in METRICS:
            values = subset[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            summaries.append(
                {
                    "representation": representation,
                    "fraction_name": fraction,
                    "group": group,
                    "metric": metric,
                    "n": len(values),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "metric_bar_data.csv", index=False)
    representations = [
        item
        for item in ("mid_prefix_mean", "mid_tail_mean_8", "final_last")
        if item in set(summary.representation)
    ]
    figure_files: dict[str, list[tuple[str, str]]] = {}
    for representation in representations:
        rep_summary = summary[summary.representation == representation]
        figure_files[representation] = []
        for metric in METRICS:
            filename = f"{representation}__{metric}_grouped_bars.svg"
            matplotlib_grouped_bar(
                rep_summary, metric, representation, output / filename
            )
            figure_files[representation].append((metric, filename))
            if representation == primary_representation:
                matplotlib_grouped_bar(
                    rep_summary,
                    metric,
                    representation,
                    output / f"{metric}_by_group_and_fraction.svg",
                )

    options = "".join(
        f'<option value="{html.escape(rep)}"{(" selected" if rep == primary_representation else "")}>{html.escape(REPRESENTATION_LABELS.get(rep, rep))}</option>'
        for rep in representations
    )
    sections = []
    for representation in representations:
        cards = "".join(
            f'<article class="card"><a href="{html.escape(filename)}" target="_blank"><img src="{html.escape(filename)}" alt="{html.escape(METRIC_DISPLAY[metric]["title"])}"></a></article>'
            for metric, filename in figure_files[representation]
        )
        hidden = "" if representation == primary_representation else " hidden"
        sections.append(
            f'<section class="grid rep-panel" data-representation="{html.escape(representation)}"{hidden}>{cards}</section>'
        )
    decline = intervention_trend["q80_minus_q20_percentage_points"]
    decline_low = intervention_trend["cluster_bootstrap_ci_low"]
    decline_high = intervention_trend["cluster_bootstrap_ci_high"]
    dashboard = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>四组教师可靠性指标看板</title>
<style>
body{{margin:0;background:#F3F4F6;color:#111827;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1680px;margin:auto;padding:28px}} h1{{margin:0 0 8px;font-size:30px}}
.intro{{color:#4B5563;line-height:1.65;margin-bottom:18px}} .toolbar{{display:flex;gap:12px;align-items:center;margin:16px 0 24px}}
select{{font:inherit;padding:9px 12px;border:1px solid #9CA3AF;border-radius:8px;background:white}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}} .hero{{margin:20px 0}}
.card{{background:white;border:1px solid #E5E7EB;border-radius:14px;box-shadow:0 2px 8px #0000000d;overflow:hidden}}
.card img{{display:block;width:100%;height:auto}} .warning{{background:#FFF7ED;border-left:4px solid #F59E0B;padding:12px 16px;border-radius:6px;line-height:1.55}}
footer{{color:#6B7280;margin:22px 0 4px;font-size:13px}} @media(max-width:1050px){{.grid{{grid-template-columns:1fr}}main{{padding:15px}}}}
</style></head><body><main>
<h1>四组教师可靠性指标看板</h1>
<p class="intro">每张图包含 4 个轨迹分位点 × 4 个结果组，共 16 根柱。柱高是中位数，黑色误差线是 IQR；柱顶数字是中位数。点击图表可单独放大。</p>
<article class="card hero"><a href="teacher_intervention_success_by_fraction.svg" target="_blank"><img src="teacher_intervention_success_by_fraction.svg" alt="教师介入时点与纠偏成功率"></a></article>
<p class="intro"><strong>介入时点趋势：</strong>在学生错误且结果可解析的状态中，教师纠偏成功率由 q20 的 {intervention.correction_rate.iloc[0]:.1%} 降至 q80 的 {intervention.correction_rate.iloc[-1]:.1%}，变化 {decline:.1f} 个百分点（prompt-cluster bootstrap 95% CI：{decline_low:.1f} 至 {decline_high:.1f} 个百分点）。</p>
<p class="warning"><strong>样本量提示：</strong>T−/S+（红色）仅有 q20=4、q40=6、q60=2、q80=1，因此红色柱只能用于描述，不能据此作稳定统计结论。</p>
<div class="toolbar"><label for="representation"><strong>状态表征：</strong></label><select id="representation">{options}</select></div>
{''.join(sections)}
<footer>原始柱状图数据：<a href="metric_bar_data.csv">metric_bar_data.csv</a>。PPL 指标不随表征变化；表征切换主要影响三个邻域指标。</footer>
</main><script>
const select=document.getElementById('representation');
function update(){{document.querySelectorAll('.rep-panel').forEach(x=>x.hidden=x.dataset.representation!==select.value)}}
select.addEventListener('change',update); update();
</script></body></html>'''
    (output / "metric_dashboard.html").write_text(dashboard, encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    frame = pd.read_parquet(run_dir / "results" / "four_group_metrics.parquet")
    primary_representation = config["analysis"].get(
        "primary_representation", "mid_prefix_mean"
    )
    if args.plots_only:
        output = run_dir / "results" / "figures"
        make_figures(
            frame,
            output,
            primary_representation,
            int(config["analysis"]["bootstrap_samples"]),
            int(config["experiment"]["seed"]),
        )
        print(f"Metric dashboard: {output / 'metric_dashboard.html'}")
        return 0
    samples = int(config["analysis"]["bootstrap_samples"])
    rng = np.random.default_rng(int(config["experiment"]["seed"]) + 88000)
    descriptions = []
    omnibus = []
    contrasts = []
    for (representation, fraction), subset in frame.groupby(
        ["representation", "fraction_name"]
    ):
        subset = subset.drop_duplicates("state_id")
        for metric in METRICS:
            arrays = []
            for group in GROUP_ORDER:
                values = (
                    subset.loc[subset.group == group, metric].dropna().to_numpy(float)
                )
                descriptions.append(
                    {
                        "representation": representation,
                        "fraction_name": fraction,
                        "group": group,
                        "metric": metric,
                        "n": len(values),
                        "mean": float(np.mean(values)) if len(values) else float("nan"),
                        "median": float(np.median(values))
                        if len(values)
                        else float("nan"),
                        "q25": float(np.quantile(values, 0.25))
                        if len(values)
                        else float("nan"),
                        "q75": float(np.quantile(values, 0.75))
                        if len(values)
                        else float("nan"),
                    }
                )
                if len(values):
                    arrays.append(values)
            try:
                statistic, pvalue = (
                    kruskal(*arrays)
                    if len(arrays) >= 2
                    else (float("nan"), float("nan"))
                )
            except ValueError:
                statistic, pvalue = float("nan"), float("nan")
            omnibus.append(
                {
                    "representation": representation,
                    "fraction_name": fraction,
                    "metric": metric,
                    "kruskal_statistic": float(statistic),
                    "kruskal_pvalue": float(pvalue),
                }
            )
            metric_contrasts = []
            for name, first_group, second_group in CONTRASTS:
                first = (
                    subset.loc[subset.group == first_group, metric]
                    .dropna()
                    .to_numpy(float)
                )
                second = (
                    subset.loc[subset.group == second_group, metric]
                    .dropna()
                    .to_numpy(float)
                )
                if len(first) and len(second):
                    test = mannwhitneyu(first, second, alternative="two-sided")
                    auc = float(test.statistic / (len(first) * len(second)))
                    low, high = cluster_bootstrap_auc(
                        subset,
                        metric,
                        first_group,
                        second_group,
                        samples,
                        rng,
                    )
                    pvalue = float(test.pvalue)
                else:
                    auc = low = high = pvalue = float("nan")
                metric_contrasts.append(
                    {
                        "representation": representation,
                        "fraction_name": fraction,
                        "metric": metric,
                        "contrast": name,
                        "first_group": first_group,
                        "second_group": second_group,
                        "n_first": len(first),
                        "n_second": len(second),
                        "auc_probability_first_higher": auc,
                        "auc_ci_low": low,
                        "auc_ci_high": high,
                        "pvalue": pvalue,
                    }
                )
            adjusted = holm([row["pvalue"] for row in metric_contrasts])
            for row, value in zip(metric_contrasts, adjusted):
                row["pvalue_holm_four_contrasts"] = value
            contrasts.extend(metric_contrasts)

    descriptions_frame = pd.DataFrame(descriptions)
    omnibus_frame = pd.DataFrame(omnibus)
    contrasts_frame = pd.DataFrame(contrasts)
    contrasts_frame["pvalue_bh_primary_global"] = np.nan
    primary_mask = contrasts_frame.representation == primary_representation
    contrasts_frame.loc[primary_mask, "pvalue_bh_primary_global"] = benjamini_hochberg(
        contrasts_frame.loc[primary_mask, "pvalue"]
    )
    descriptions_frame.to_csv(
        run_dir / "results" / "four_group_descriptions.csv", index=False
    )
    omnibus_frame.to_csv(run_dir / "results" / "four_group_omnibus.csv", index=False)
    contrasts_frame.to_csv(
        run_dir / "results" / "four_group_contrasts.csv", index=False
    )
    make_figures(
        frame,
        run_dir / "results" / "figures",
        primary_representation,
        int(config["analysis"]["bootstrap_samples"]),
        int(config["experiment"]["seed"]),
    )

    labels = pd.read_json(run_dir / "results" / "pair_labels.jsonl", lines=True)
    valid = labels[labels.eligible_pair]
    group_counts = (
        valid.groupby(["fraction_name", "group"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    frozen_path = run_dir / "artifacts" / "frozen_rounds.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        stopping = frozen["analysis_stopping_rule"]
    else:
        stopping = config["stopping"]
    stopping_mode = stopping.get("mode", "target_count")
    if stopping_mode == "fixed_rounds":
        stopping_target_reached = True
    elif stopping_mode == "target_count":
        target_group = stopping["target_group"]
        target_count = int(stopping["minimum_target_per_fraction"])
        expected_fractions = [
            f"q{round(100 * float(value)):02d}"
            for value in config["data"]["state_fractions"]
        ]
        stopping_target_reached = all(
            int(group_counts.get(name, {}).get(target_group, 0)) >= target_count
            for name in expected_fractions
        )
    else:
        raise ValueError(f"Unknown stopping mode: {stopping_mode}")
    primary = contrasts_frame[
        contrasts_frame.representation == primary_representation
    ].sort_values(["fraction_name", "metric", "contrast"])
    summary = {
        "total_anchor_states": int(labels.state_id.nunique()),
        "eligible_pair_states": int(valid.state_id.nunique()),
        "eligible_pair_fraction": float(
            valid.state_id.nunique() / labels.state_id.nunique()
        ),
        "group_counts_by_fraction": group_counts,
        "stopping_rule": stopping,
        "stopping_target_reached": stopping_target_reached,
        "primary_representation": primary_representation,
        "primary_contrasts": primary.to_dict(orient="records"),
    }
    (run_dir / "results" / "four_group_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    update_status(
        run_dir,
        "four_group_analyzed",
        total_anchor_states=summary["total_anchor_states"],
        eligible_pair_states=summary["eligible_pair_states"],
        group_counts_by_fraction=group_counts,
        stopping_target_reached=stopping_target_reached,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
