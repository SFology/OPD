from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str
    direction: float
    color: str


METRICS = [
    MetricSpec("relative_slope_max", "Local relative slope ↑", 1.0, "#176b87"),
    MetricSpec(
        "teacher_self_consistency_pairwise_coverage_adjusted",
        "Coverage-adjusted self-consistency ↑",
        1.0,
        "#1b8a5a",
    ),
    MetricSpec("teacher_valid_answer_rate", "Valid-answer rate ↑", 1.0, "#67a61f"),
    MetricSpec(
        "teacher_metric_truncated_rate",
        "Teacher truncation rate ↓",
        -1.0,
        "#d1495b",
    ),
    MetricSpec(
        "teacher_semantic_entropy_conservative",
        "Conservative semantic entropy ↓",
        -1.0,
        "#7b5ea7",
    ),
    MetricSpec("teacher_prefix_ppl", "Teacher prefix PPL ↓", -1.0, "#ef8a17"),
    MetricSpec("topk_overlap", "Teacher–student top-k overlap ↑", 1.0, "#8c6d31"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay rank-calibration curves for OPD reliability metrics."
    )
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--locality-run", required=True, type=Path)
    return parser.parse_args()


def prepare_frame(source: Path, locality: Path) -> pd.DataFrame:
    reliability = pd.read_parquet(source / "results" / "reliability.parquet")
    reliability = reliability[
        (reliability.representation == "mid_tail_mean_8")
        & (reliability.neighborhood_method == "dual_knn")
    ].drop_duplicates("state_id")
    local = pd.read_parquet(locality / "results" / "locality_metrics.parquet")
    local = local[
        (local.representation == "mid_prefix_mean")
        & (local.neighborhood == "same_progress")
    ][["state_id", "relative_slope_max"]]
    frame = reliability.merge(local, on="state_id", validate="one_to_one")
    if len(frame) != 48:
        raise RuntimeError(f"Expected 48 matched anchor states, found {len(frame)}")
    return frame


def rank_calibration(
    frame: pd.DataFrame, metric: MetricSpec, target: str
) -> pd.DataFrame:
    subset = frame[[metric.column, target]].replace([np.inf, -np.inf], np.nan).dropna()
    score = subset[metric.column] * metric.direction
    percentile = score.rank(method="average", pct=True)
    bins = pd.cut(
        percentile,
        bins=np.linspace(0.0, 1.0, 6),
        labels=False,
        include_lowest=True,
    )
    values = subset.assign(
        score=score,
        percentile=percentile,
        bin=bins,
    )
    rows = []
    for bin_index, group in values.groupby("bin", observed=True):
        count = len(group)
        target_values = group[target].to_numpy(float)
        standard_error = (
            float(np.std(target_values, ddof=1) / np.sqrt(count)) if count > 1 else 0.0
        )
        rows.append(
            {
                "metric": metric.column,
                "label": metric.label,
                "direction": metric.direction,
                "target": target,
                "bin": int(bin_index),
                "count": count,
                "score_percentile_mean": float(group.percentile.mean()),
                "target_mean": float(group[target].mean()),
                "target_standard_error": standard_error,
                "raw_spearman": float(
                    spearmanr(subset[metric.column], subset[target]).statistic
                ),
                "oriented_spearman": float(spearmanr(score, subset[target]).statistic),
            }
        )
    return pd.DataFrame(rows)


def comparison_svg(calibration: pd.DataFrame, frame: pd.DataFrame) -> str:
    width, height = 1280, 710
    left, right, top, bottom, gap = 80, 45, 128, 98, 90
    panel_width = (width - left - right - gap) / 2
    panel_height = height - top - bottom
    panels = [
        ("q_teacher", "Teacher task success", 0.0, 1.0),
        ("q_teacher_minus_student", "Teacher advantage over student", -0.45, 0.65),
    ]
    style = (
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}"
        ".title{font-size:25px;font-weight:700}.sub{font-size:13px;fill:#586174}"
        ".panel{font-size:16px;font-weight:700}.axis{font-size:11px;fill:#687386}"
        ".legend{font-size:11px}.grid{stroke:#dce2ec;stroke-width:1}"
        ".zero{stroke:#7d8798;stroke-width:1.5;stroke-dasharray:5 5}"
        ".mean{stroke:#a4adbb;stroke-width:1.5;stroke-dasharray:3 5}</style>"
    )
    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    )
    parts = [
        svg_open,
        style,
        '<text class="title" x="20" y="32">Reliability metric comparison</text>',
        '<text class="sub" x="20" y="55">All metrics are oriented left→right from predicted unreliable to predicted reliable; points are rank-quantile bins.</text>',
    ]

    legend_x, legend_y = 20, 80
    for index, metric in enumerate(METRICS):
        column = index % 4
        row = index // 4
        x = legend_x + column * 310
        y = legend_y + row * 25
        parts.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}" '
            f'stroke="{metric.color}" stroke-width="3"/>'
        )
        parts.append(f'<circle cx="{x + 11}" cy="{y}" r="4" fill="{metric.color}"/>')
        parts.append(
            f'<text class="legend" x="{x + 29}" y="{y + 4}">{html.escape(metric.label)}</text>'
        )

    for panel_index, (target, title, ymin, ymax) in enumerate(panels):
        x0 = left + panel_index * (panel_width + gap)
        y0 = top

        def px(value: float, panel_x: float = x0) -> float:
            return panel_x + value * panel_width

        def py(
            value: float,
            panel_y: float = y0,
            maximum: float = ymax,
            minimum: float = ymin,
        ) -> float:
            return panel_y + (maximum - value) / (maximum - minimum) * panel_height

        parts.append(f'<text class="panel" x="{x0}" y="{y0 - 15}">{title}</text>')
        ticks = np.linspace(ymin, ymax, 6)
        for tick in ticks:
            y = py(float(tick))
            parts.append(
                f'<line class="grid" x1="{x0:.1f}" y1="{y:.1f}" '
                f'x2="{x0 + panel_width:.1f}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text class="axis" text-anchor="end" x="{x0 - 8:.1f}" '
                f'y="{y + 4:.1f}">{tick:.2f}</text>'
            )
        if ymin < 0 < ymax:
            parts.append(
                f'<line class="zero" x1="{x0:.1f}" y1="{py(0):.1f}" '
                f'x2="{x0 + panel_width:.1f}" y2="{py(0):.1f}"/>'
            )
        overall = float(frame[target].mean())
        parts.append(
            f'<line class="mean" x1="{x0:.1f}" y1="{py(overall):.1f}" '
            f'x2="{x0 + panel_width:.1f}" y2="{py(overall):.1f}">'
            f"<title>Overall mean={overall:.3f}</title></line>"
        )
        for tick in np.linspace(0, 1, 6):
            x = px(float(tick))
            parts.append(
                f'<text class="axis" text-anchor="middle" x="{x:.1f}" '
                f'y="{y0 + panel_height + 22:.1f}">{tick:.1f}</text>'
            )

        for metric in METRICS:
            values = calibration[
                (calibration.target == target) & (calibration.metric == metric.column)
            ].sort_values("score_percentile_mean")
            if values.empty:
                continue
            points = " ".join(
                f"{px(float(row.score_percentile_mean)):.1f},{py(float(row.target_mean)):.1f}"
                for row in values.itertuples(index=False)
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{metric.color}" '
                f'stroke-width="2.5" stroke-linejoin="round"/>'
            )
            for row in values.itertuples(index=False):
                x = px(float(row.score_percentile_mean))
                y = py(float(row.target_mean))
                low = max(ymin, float(row.target_mean - row.target_standard_error))
                high = min(ymax, float(row.target_mean + row.target_standard_error))
                parts.append(
                    f'<line x1="{x:.1f}" y1="{py(low):.1f}" x2="{x:.1f}" '
                    f'y2="{py(high):.1f}" stroke="{metric.color}" opacity="0.45"/>'
                )
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                    f'fill="{metric.color}"><title>{html.escape(metric.label)}; '
                    f"bin n={int(row.count)}; target={float(row.target_mean):.3f}; "
                    f"raw rho={float(row.raw_spearman):+.3f}</title></circle>"
                )
        parts.append(
            f'<text class="axis" text-anchor="middle" x="{x0 + panel_width / 2:.1f}" '
            f'y="{height - 47}">oriented reliability-score percentile</text>'
        )
    parts.append(
        f'<text class="axis" transform="translate(17,{top + panel_height / 2:.1f}) rotate(-90)" '
        'text-anchor="middle">observed outcome</text>'
    )
    parts.append(
        '<text class="sub" x="20" y="690">Error bars: ±1 standard error across states in each bin. Dashed horizontal line: overall mean.</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    args = parse_args()
    source = args.source_run.resolve()
    locality = args.locality_run.resolve()
    frame = prepare_frame(source, locality)
    rows = []
    for target in ["q_teacher", "q_teacher_minus_student"]:
        for metric in METRICS:
            rows.append(rank_calibration(frame, metric, target))
    calibration = pd.concat(rows, ignore_index=True)

    figures = locality / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(figures / "metric_comparison_curve_data.csv", index=False)
    svg = comparison_svg(calibration, frame)
    (figures / "metric_comparison_curves.svg").write_text(svg, encoding="utf-8")
    dashboard = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>OPD metric comparison</title>
<style>body{margin:0;background:#f4f6fa;font-family:Inter,Arial,sans-serif;color:#172033}
main{max-width:1330px;margin:auto;padding:24px}.card{background:#fff;border:1px solid #dce2ec;
border-radius:12px;padding:12px;box-shadow:0 2px 8px #18223a0b}.card img{width:100%;height:auto}
p{line-height:1.6;color:#586174}</style></head><body><main><h1>OPD reliability metric comparison</h1>
<p>Curves use the same 48 anchor states. Every metric is direction-normalized so moving right means that the metric predicts a more reliable teacher.</p>
<div class="card"><img src="metric_comparison_curves.svg" alt="Reliability metric comparison curves"></div>
</main></body></html>"""
    (figures / "metric_comparison.html").write_text(dashboard, encoding="utf-8")
    print(f"Wrote comparison chart to {figures / 'metric_comparison.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
