from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_REPRESENTATION = "mid_tail_mean_8"
PRIMARY_NEIGHBORHOOD = "dual_knn"

LABELS = {
    "teacher_max": "Teacher local max",
    "relative_max": "Relative local max",
    "relative_top_mean": "Relative local top-mean",
    "teacher_entropy": "Teacher entropy",
    "teacher_max_probability": "Teacher max probability",
    "teacher_student_kl": "Teacher–student KL",
    "teacher_student_top1_agreement": "Teacher–student top-1 agreement",
    "topk_overlap": "Teacher–student top-k overlap",
    "teacher_prefix_ppl": "Teacher prefix PPL",
    "teacher_action_ppl": "Teacher action PPL",
    "teacher_self_consistency_pairwise": "Self-consistency (pairwise)",
    "teacher_self_consistency_pairwise_coverage_adjusted": (
        "Self-consistency (coverage-adjusted)"
    ),
    "teacher_semantic_entropy": "Semantic entropy",
    "teacher_semantic_entropy_conservative": "Semantic entropy (conservative)",
    "teacher_valid_answer_rate": "Valid-answer rate",
}

TEACHER_METRICS = [
    "relative_top_mean",
    "relative_max",
    "teacher_max",
    "teacher_entropy",
    "teacher_max_probability",
    "teacher_student_kl",
    "teacher_student_top1_agreement",
    "topk_overlap",
    "teacher_prefix_ppl",
    "teacher_action_ppl",
    "teacher_self_consistency_pairwise",
    "teacher_semantic_entropy",
    "teacher_self_consistency_pairwise_coverage_adjusted",
    "teacher_semantic_entropy_conservative",
    "teacher_valid_answer_rate",
]

RELATIVE_METRICS = [
    "relative_top_mean",
    "relative_max",
    "topk_overlap",
    "teacher_student_kl",
    "teacher_student_top1_agreement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dependency-free HTML/SVG reliability dashboard."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--representation", default=PRIMARY_REPRESENTATION)
    parser.add_argument("--neighborhood", default=PRIMARY_NEIGHBORHOOD)
    return parser.parse_args()


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def correlation_svg(rows: pd.DataFrame, title: str, subtitle: str) -> str:
    rows = rows.copy().sort_values("spearman")
    width = 940
    left, right, top, bottom = 310, 75, 76, 50
    row_height = 35
    height = top + bottom + row_height * len(rows)
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + (value + 1.0) * plot_width / 2.0

    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
    )
    style = (
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}"
        ".title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#586174}"
        ".label{font-size:13px}.value{font-size:12px;font-weight:700}"
        ".axis{font-size:11px;fill:#687386}.grid{stroke:#dce2ec;stroke-width:1}"
        "</style>"
    )
    parts = [
        svg_open,
        style,
        f'<text class="title" x="18" y="29">{escape(title)}</text>',
        f'<text class="sub" x="18" y="51">{escape(subtitle)}</text>',
    ]
    for tick in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        tx = x(tick)
        parts.append(
            f'<line class="grid" x1="{tx:.1f}" y1="{top - 9}" '
            f'x2="{tx:.1f}" y2="{height - bottom + 5}"/>'
        )
        parts.append(
            f'<text class="axis" text-anchor="middle" x="{tx:.1f}" '
            f'y="{height - 15}">{tick:g}</text>'
        )
    for index, row in enumerate(rows.itertuples(index=False)):
        y = top + index * row_height
        rho = float(row.spearman)
        x0, x1 = x(0), x(rho)
        color = "#137a65" if rho >= 0 else "#c84c4c"
        parts.append(
            f'<text class="label" text-anchor="end" x="{left - 12}" '
            f'y="{y + 15}">{escape(LABELS.get(row.metric, row.metric))}</text>'
        )
        parts.append(
            f'<rect x="{min(x0, x1):.1f}" y="{y + 2}" '
            f'width="{max(1, abs(x1 - x0)):.1f}" height="18" rx="3" fill="{color}">'
            f"<title>rho={rho:.3f}; p={float(row.spearman_pvalue):.4g}; "
            f"n={int(row.n)}</title></rect>"
        )
        anchor = "start" if rho >= 0 else "end"
        offset = 6 if rho >= 0 else -6
        parts.append(
            f'<text class="value" text-anchor="{anchor}" x="{x1 + offset:.1f}" '
            f'y="{y + 16}">{rho:+.3f}</text>'
        )
    parts.append(
        f'<text class="axis" text-anchor="middle" x="{left + plot_width / 2:.1f}" '
        f'y="{height - 1}">Spearman correlation (ρ)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def auc_svg(rows: pd.DataFrame) -> str:
    rows = rows.copy().sort_values("unreliable_auc")
    width = 940
    left, right, top, bottom = 310, 75, 86, 55
    row_height = 39
    height = top + bottom + row_height * len(rows)
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + value * plot_width

    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
    )
    style = (
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}"
        ".title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#586174}"
        ".label{font-size:13px}.value{font-size:11px}.axis{font-size:11px;fill:#687386}"
        ".grid{stroke:#dce2ec;stroke-width:1}.chance{stroke:#7f899c;stroke-dasharray:5 5}"
        "</style>"
    )
    parts = [
        svg_open,
        style,
        '<text class="title" x="18" y="29">Unreliable-state classification</text>',
        '<text class="sub" x="18" y="51">Circle: preregistered direction; diamond: reversed direction. AUC=0.5 is random.</text>',
    ]
    for tick in np.linspace(0, 1, 5):
        tx = x(float(tick))
        klass = "chance" if tick == 0.5 else "grid"
        parts.append(
            f'<line class="{klass}" x1="{tx:.1f}" y1="{top - 10}" '
            f'x2="{tx:.1f}" y2="{height - bottom + 5}"/>'
        )
        parts.append(
            f'<text class="axis" text-anchor="middle" x="{tx:.1f}" '
            f'y="{height - 18}">{tick:.2f}</text>'
        )
    for index, row in enumerate(rows.itertuples(index=False)):
        y = top + index * row_height + 10
        auc = float(row.unreliable_auc)
        reverse = 1.0 - auc
        parts.append(
            f'<text class="label" text-anchor="end" x="{left - 12}" '
            f'y="{y + 4}">{escape(LABELS.get(row.metric, row.metric))}</text>'
        )
        parts.append(
            f'<line x1="{x(min(auc, reverse)):.1f}" y1="{y}" '
            f'x2="{x(max(auc, reverse)):.1f}" y2="{y}" stroke="#b9c1cf" stroke-width="3"/>'
        )
        parts.append(
            f'<circle cx="{x(auc):.1f}" cy="{y}" r="6" fill="#345995">'
            f"<title>Preregistered AUC={auc:.3f}</title></circle>"
        )
        rx = x(reverse)
        parts.append(
            f'<path d="M {rx:.1f} {y - 7} L {rx + 7:.1f} {y} L {rx:.1f} {y + 7} '
            f'L {rx - 7:.1f} {y} Z" fill="#e07a2d">'
            f"<title>Reversed AUC={reverse:.3f}</title></path>"
        )
        parts.append(
            f'<text class="value" x="{x(auc) + 8:.1f}" y="{y - 8}">{auc:.2f}</text>'
        )
    parts.append(
        f'<text class="axis" text-anchor="middle" x="{left + plot_width / 2:.1f}" '
        f'y="{height - 2}">ROC AUC for q_teacher ≤ 0.25</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def calibration_svg(calibration: pd.DataFrame, metrics: list[str], target: str) -> str:
    width, height = 940, 610
    cols, rows = 2, 2
    margin_x, margin_y = 62, 78
    gap_x, gap_y = 70, 62
    panel_w = (width - 2 * margin_x - gap_x) / cols
    panel_h = (height - margin_y - 52 - gap_y) / rows
    colors = ["#137a65", "#345995", "#e07a2d", "#8e5ba6"]
    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    )
    style = (
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}"
        ".title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#586174}"
        ".ptitle{font-size:13px;font-weight:700}.axis{font-size:10px;fill:#687386}"
        ".grid{stroke:#dce2ec;stroke-width:1}</style>"
    )
    parts = [
        svg_open,
        style,
        '<text class="title" x="18" y="29">Quintile calibration</text>',
        '<text class="sub" x="18" y="51">Each point is one metric quintile; labels show states per bin.</text>',
    ]
    for index, metric in enumerate(metrics):
        col, row = index % cols, index // cols
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_y + row * (panel_h + gap_y)
        subset = calibration[calibration.metric == metric].copy()
        if subset.empty:
            continue
        xs = subset.metric_mean.to_numpy(float)
        ys = subset.target_mean.to_numpy(float)
        xmin, xmax = float(xs.min()), float(xs.max())
        pad = max((xmax - xmin) * 0.08, 1e-6)
        xmin, xmax = xmin - pad, xmax + pad

        def px(
            value: float,
            panel_x: float = x0,
            minimum: float = xmin,
            maximum: float = xmax,
        ) -> float:
            return panel_x + (value - minimum) / (maximum - minimum) * panel_w

        def py(value: float, panel_y: float = y0) -> float:
            return panel_y + panel_h - value * panel_h

        parts.append(
            f'<text class="ptitle" x="{x0:.1f}" y="{y0 - 12:.1f}">'
            f"{escape(LABELS.get(metric, metric))}</text>"
        )
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            yy = py(tick)
            parts.append(
                f'<line class="grid" x1="{x0:.1f}" y1="{yy:.1f}" '
                f'x2="{x0 + panel_w:.1f}" y2="{yy:.1f}"/>'
            )
            parts.append(
                f'<text class="axis" text-anchor="end" x="{x0 - 7:.1f}" '
                f'y="{yy + 3:.1f}">{tick:.2f}</text>'
            )
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[index]}" '
            f'stroke-width="3"/>'
        )
        for record in subset.itertuples(index=False):
            xx, yy = px(float(record.metric_mean)), py(float(record.target_mean))
            parts.append(
                f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6" fill="{colors[index]}">'
                f"<title>metric={float(record.metric_mean):.4g}; "
                f"target={float(record.target_mean):.3f}; n={int(record.count)}</title></circle>"
            )
            parts.append(
                f'<text class="axis" x="{xx + 8:.1f}" y="{yy - 7:.1f}">n={int(record.count)}</text>'
            )
        parts.append(
            f'<text class="axis" text-anchor="middle" x="{x0 + panel_w / 2:.1f}" '
            f'y="{y0 + panel_h + 18:.1f}">metric mean</text>'
        )
    target_label = (
        "Teacher success rate" if target == "q_teacher" else "Teacher − student success"
    )
    parts.append(
        f'<text class="axis" transform="translate(13,{height / 2:.1f}) rotate(-90)" '
        f'text-anchor="middle">{target_label}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def quality_svg(primary: pd.DataFrame) -> str:
    width, height = 940, 410
    categories = [
        ("Teacher metric truncation", primary.teacher_metric_truncated_rate),
        ("Teacher label truncation", primary.teacher_label_truncated_rate),
        ("Student label truncation", primary.student_label_truncated_rate),
        ("Teacher metric valid-answer", primary.teacher_valid_answer_rate),
        ("Teacher label parseable", primary.teacher_label_parseable_rate),
        ("Student label parseable", primary.student_label_parseable_rate),
    ]
    means = [float(values.mean()) for _, values in categories]
    left, right, top, bottom = 275, 70, 72, 40
    plot_width = width - left - right
    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    )
    style = (
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}"
        ".title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#586174}"
        ".label{font-size:13px}.value{font-size:12px;font-weight:700}"
        ".grid{stroke:#dce2ec;stroke-width:1}</style>"
    )
    parts = [
        svg_open,
        style,
        '<text class="title" x="18" y="29">Validation data quality</text>',
        '<text class="sub" x="18" y="51">Mean across 48 selected states.</text>',
    ]
    for tick in np.linspace(0, 1, 5):
        xx = left + tick * plot_width
        parts.append(
            f'<line class="grid" x1="{xx:.1f}" y1="{top - 8}" x2="{xx:.1f}" '
            f'y2="{height - bottom}"/>'
        )
    for index, ((label, _), value) in enumerate(zip(categories, means)):
        y = top + index * 49
        bad = "truncation" in label.lower()
        color = "#c84c4c" if bad else "#137a65"
        parts.append(
            f'<text class="label" text-anchor="end" x="{left - 12}" y="{y + 18}">'
            f"{escape(label)}</text>"
        )
        parts.append(
            f'<rect x="{left}" y="{y + 3}" width="{value * plot_width:.1f}" '
            f'height="22" rx="4" fill="{color}"/>'
        )
        parts.append(
            f'<text class="value" x="{left + value * plot_width + 8:.1f}" y="{y + 19}">'
            f"{value:.1%}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    args = parse_args()
    results = args.run_dir / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    reliability = pd.read_csv(results / "reliability.csv")
    correlations = pd.read_csv(results / "correlations.csv")
    calibration = pd.read_csv(results / "calibration.csv")
    with (results / "summary.json").open() as handle:
        summary = json.load(handle)

    primary = reliability[
        (reliability.representation == args.representation)
        & (reliability.neighborhood_method == args.neighborhood)
    ].drop_duplicates("state_id")
    corr = correlations[
        (correlations.representation == args.representation)
        & (correlations.neighborhood_method == args.neighborhood)
    ]
    cal = calibration[
        (calibration.representation == args.representation)
        & (calibration.neighborhood_method == args.neighborhood)
    ]
    if primary.empty:
        raise RuntimeError("The requested primary representation/neighborhood is empty")

    teacher_corr = corr[
        (corr.target == "q_teacher") & corr.metric.isin(TEACHER_METRICS)
    ]
    relative_corr = corr[
        (corr.target == "q_teacher_minus_student") & corr.metric.isin(RELATIVE_METRICS)
    ]
    auc_rows = teacher_corr[
        teacher_corr.metric.isin(
            [
                "relative_top_mean",
                "relative_max",
                "teacher_prefix_ppl",
                "teacher_student_kl",
                "topk_overlap",
                "teacher_self_consistency_pairwise",
                "teacher_semantic_entropy",
                "teacher_self_consistency_pairwise_coverage_adjusted",
                "teacher_semantic_entropy_conservative",
                "teacher_valid_answer_rate",
            ]
        )
    ]

    svgs = {
        "teacher_metric_correlations.svg": correlation_svg(
            teacher_corr,
            "Metrics vs teacher reliability",
            "Positive ρ means larger metric values accompany higher teacher success.",
        ),
        "relative_metric_correlations.svg": correlation_svg(
            relative_corr,
            "Metrics vs teacher advantage",
            "Target: q_teacher − q_student; n=48 for the primary dual-kNN analysis.",
        ),
        "unreliable_auc.svg": auc_svg(auc_rows),
        "calibration_teacher.svg": calibration_svg(
            cal[cal.target == "q_teacher"],
            [
                "relative_top_mean",
                "teacher_prefix_ppl",
                "teacher_self_consistency_pairwise_coverage_adjusted",
                "teacher_semantic_entropy_conservative",
            ],
            "q_teacher",
        ),
        "validation_quality.svg": quality_svg(primary),
    }
    for filename, content in svgs.items():
        (figures / filename).write_text(content, encoding="utf-8")

    delta = float(primary.q_teacher_minus_student.mean())
    wins = int((primary.q_teacher > primary.q_student).sum())
    ties = int((primary.q_teacher == primary.q_student).sum())
    losses = int((primary.q_teacher < primary.q_student).sum())
    dashboard = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Trustworthy OPD reliability dashboard</title>
<style>
:root{{--ink:#172033;--muted:#586174;--paper:#f4f6fa;--card:#fff;--line:#dce2ec}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,Arial,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:28px 24px 60px}}h1{{margin:0 0 7px;font-size:30px}}.meta{{color:var(--muted);margin-bottom:22px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}}.card,.figure,.note{{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 8px #18223a0b}}
.card{{padding:15px}}.card strong{{display:block;font-size:25px;margin-top:4px}}.card span{{font-size:12px;color:var(--muted)}}
.figure{{margin:16px 0;padding:10px;overflow:auto}}.figure img{{display:block;max-width:100%;height:auto;margin:auto}}
.note{{padding:17px 20px;border-left:5px solid #e07a2d;line-height:1.55}}code{{background:#eef1f6;padding:2px 5px;border-radius:4px}}
@media(max-width:760px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>Trustworthy OPD reliability dashboard</h1>
<div class="meta">Run: {escape(args.run_dir.name)} · Primary analysis: {escape(args.representation)} + {escape(args.neighborhood)} · n={len(primary)}</div>
<section class="cards">
 <div class="card"><span>Teacher mean success</span><strong>{float(summary["teacher_mean_success"]):.1%}</strong></div>
 <div class="card"><span>Student mean success</span><strong>{float(summary["student_mean_success"]):.1%}</strong></div>
 <div class="card"><span>Teacher advantage</span><strong>{delta:+.1%}</strong></div>
 <div class="card"><span>Teacher win / tie / loss</span><strong>{wins} / {ties} / {losses}</strong></div>
</section>
<div class="note"><b>How to read:</b> the local-change metrics have positive correlations with success, opposite to the preregistered “larger means less reliable” direction. The AUC chart therefore shows both the preregistered and reversed directions. Coverage-adjusted metrics are strong, but are partly driven by teacher non-termination.</div>
<div class="figure"><img src="validation_quality.svg" alt="Validation data quality"></div>
<div class="figure"><img src="teacher_metric_correlations.svg" alt="Metrics versus teacher reliability"></div>
<div class="figure"><img src="relative_metric_correlations.svg" alt="Metrics versus teacher advantage"></div>
<div class="figure"><img src="unreliable_auc.svg" alt="ROC AUC comparison"></div>
<div class="figure"><img src="calibration_teacher.svg" alt="Calibration curves"></div>
</main></body></html>"""
    output = figures / "reliability_dashboard.html"
    output.write_text(dashboard, encoding="utf-8")
    print(f"Wrote dashboard to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
