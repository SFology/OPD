from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_run, update_status
from objective_reliability_common import (
    Q_ORDER,
    cross_validated_predictions,
    prompt_bootstrap_improvement,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze objective reliability study")
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    results = run_dir / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(results / "objective_state_metrics.parquet")
    frame = frame[frame.complete & (frame.neighbor_count > 0)].copy()
    metrics = [
        "teacher_sensitivity",
        "student_sensitivity",
        "relative_sensitivity",
        "lcb_risk",
        "risk_to_abs_reward",
        *[
            f"trust_lambda_{str(value).replace('.', 'p')}"
            for value in config["risk"]["lambdas"]
        ],
    ]
    prediction_frames = []
    summaries = []
    analysis = config["analysis"]
    for (arm, neighborhood, representation), block in frame.groupby(
        ["arm", "neighborhood", "representation"], sort=True
    ):
        for metric in metrics:
            prediction = cross_validated_predictions(
                block,
                metric,
                folds=int(analysis["prompt_folds"]),
                l2=float(analysis["logistic_l2"]),
            )
            if prediction.empty:
                continue
            prediction["arm"] = arm
            prediction["neighborhood"] = neighborhood
            prediction["representation"] = representation
            prediction["metric"] = metric
            prediction_frames.append(prediction)
            for q in (*Q_ORDER, "all_q"):
                subset = (
                    prediction
                    if q == "all_q"
                    else prediction[prediction.fraction_name == q]
                )
                estimate, low, high = prompt_bootstrap_improvement(
                    subset,
                    baseline_column="baseline_brier_sum",
                    model_column="metric_brier_sum",
                    samples=int(analysis["bootstrap_samples"]),
                    seed=stable_seed(
                        config["experiment"]["seed"],
                        arm,
                        neighborhood,
                        representation,
                        metric,
                        q,
                    ),
                    confidence=float(analysis["confidence_level"]),
                )
                log_estimate, log_low, log_high = prompt_bootstrap_improvement(
                    subset,
                    baseline_column="baseline_log_loss_sum",
                    model_column="metric_log_loss_sum",
                    samples=int(analysis["bootstrap_samples"]),
                    seed=stable_seed(
                        config["experiment"]["seed"],
                        "log",
                        arm,
                        neighborhood,
                        representation,
                        metric,
                        q,
                    ),
                    confidence=float(analysis["confidence_level"]),
                )
                summaries.append(
                    {
                        "arm": arm,
                        "neighborhood": neighborhood,
                        "representation": representation,
                        "metric": metric,
                        "scope": q,
                        "states": int(subset.state_id.nunique()),
                        "trials": int(subset.trials.sum()),
                        "brier_improvement": estimate,
                        "brier_ci_low": low,
                        "brier_ci_high": high,
                        "log_loss_improvement": log_estimate,
                        "log_loss_ci_low": log_low,
                        "log_loss_ci_high": log_high,
                    }
                )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    predictions.to_parquet(results / "cv_predictions.parquet", index=False)
    predictions.to_csv(results / "cv_predictions.csv", index=False)
    summary.to_csv(results / "cv_metric_summary.csv", index=False)

    paired_rows = []
    selected_predictions = predictions[predictions.arm == "selected"]
    for (neighborhood, representation, metric), selected in selected_predictions.groupby(
        ["neighborhood", "representation", "metric"], sort=True
    ):
        for control_name in ("random", "far"):
            control = predictions[
                (predictions.arm == control_name)
                & (predictions.neighborhood == neighborhood)
                & (predictions.representation == representation)
                & (predictions.metric == metric)
            ]
            paired = selected.merge(
                control[
                    [
                        "state_id",
                        "metric_brier_sum",
                        "metric_log_loss_sum",
                    ]
                ],
                on="state_id",
                suffixes=("_selected", "_control"),
                validate="one_to_one",
            )
            for loss in ("brier", "log_loss"):
                estimate, low, high = prompt_bootstrap_improvement(
                    paired,
                    baseline_column=f"metric_{loss}_sum_control",
                    model_column=f"metric_{loss}_sum_selected",
                    samples=int(analysis["bootstrap_samples"]),
                    seed=stable_seed(
                        config["experiment"]["seed"],
                        "paired",
                        control_name,
                        neighborhood,
                        representation,
                        metric,
                        loss,
                    ),
                    confidence=float(analysis["confidence_level"]),
                )
                paired_rows.append(
                    {
                        "control": control_name,
                        "neighborhood": neighborhood,
                        "representation": representation,
                        "metric": metric,
                        "loss": loss,
                        "states": int(paired.state_id.nunique()),
                        "selected_advantage": estimate,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(results / "selected_vs_controls_paired.csv", index=False)

    primary_filter = (
        (summary.neighborhood == analysis["primary_neighborhood"])
        & (summary.representation == analysis["primary_representation"])
        & (summary.scope == "all_q")
    )
    overview = summary[primary_filter].copy().sort_values("brier_improvement")
    fig, axis = plt.subplots(figsize=(12, max(5, 0.28 * len(overview))))
    labels = overview.metric + " / " + overview.arm
    colors = overview.arm.map(
        {"selected": "#1967d2", "random": "#f9ab00", "far": "#d93025"}
    )
    axis.barh(labels, overview.brier_improvement, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel(
        "Held-out Brier improvement over q-only baseline (higher is better)"
    )
    axis.set_ylabel("Reliability metric / support arm")
    axis.set_title("Can neighborhood metrics predict repeated-teacher correctness?")
    save_figure(fig, figures / "metric_support_comparison.png")

    paired_plot = paired_summary[
        (paired_summary.neighborhood == analysis["primary_neighborhood"])
        & (paired_summary.representation == analysis["primary_representation"])
        & (paired_summary.loss == "brier")
    ].copy()
    paired_plot["label"] = paired_plot.metric + " vs " + paired_plot.control
    paired_plot = paired_plot.sort_values("selected_advantage")
    fig, axis = plt.subplots(figsize=(11, max(5, 0.35 * len(paired_plot))))
    axis.errorbar(
        paired_plot.selected_advantage,
        paired_plot.label,
        xerr=np.vstack(
            [
                paired_plot.selected_advantage - paired_plot.ci_low,
                paired_plot.ci_high - paired_plot.selected_advantage,
            ]
        ),
        fmt="o",
        color="#1967d2",
        ecolor="#5f6368",
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel(
        "Paired held-out Brier advantage of selected support (higher is better)"
    )
    axis.set_ylabel("Reliability metric / matched control")
    axis.set_title("Does semantic selection outperform matched support controls?")
    save_figure(fig, figures / "selected_vs_controls_paired.png")

    primary = predictions[
        (predictions.arm == analysis["primary_arm"])
        & (predictions.neighborhood == analysis["primary_neighborhood"])
        & (predictions.representation == analysis["primary_representation"])
        & (predictions.metric == analysis["primary_metric"])
    ].copy()
    calibration = []
    for q, block in primary.groupby("fraction_name"):
        block = block.assign(
            bin=pd.qcut(
                block.metric_probability, q=min(5, len(block)), duplicates="drop"
            )
        )
        for _, part in block.groupby("bin", observed=True):
            calibration.append(
                {
                    "fraction_name": q,
                    "predicted": np.average(
                        part.metric_probability, weights=part.trials
                    ),
                    "observed": part.successes.sum() / part.trials.sum(),
                    "states": len(part),
                }
            )
    calibration_frame = pd.DataFrame(calibration)
    calibration_frame.to_csv(results / "primary_calibration.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    for q in Q_ORDER:
        block = calibration_frame[calibration_frame.fraction_name == q]
        axis.plot(block.predicted, block.observed, marker="o", label=q)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    axis.set(
        xlabel="Predicted teacher success probability",
        ylabel="Observed verifier success rate",
        title="Primary metric held-out calibration",
    )
    axis.legend()
    save_figure(fig, figures / "primary_calibration.png")

    competence = pd.read_parquet(results / "teacher_competence.parquet")
    state_meta = frame[["state_id", "fraction_name"]].drop_duplicates()
    competence = competence.merge(state_meta, on="state_id", validate="one_to_one")
    by_q = competence.groupby("fraction_name", as_index=False).agg(
        successes=("successes", "sum"),
        trials=("trials", "sum"),
        states=("state_id", "nunique"),
    )
    by_q["success_rate"] = by_q.successes / by_q.trials
    by_q.to_csv(results / "teacher_success_by_q.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    ordered = by_q.set_index("fraction_name").reindex(Q_ORDER).reset_index()
    axis.bar(ordered.fraction_name, ordered.success_rate, color="#188038")
    axis.set_ylim(0, 1)
    axis.set_xlabel("Student-prefix intervention point")
    axis.set_ylabel("Verifier success rate across teacher continuations")
    axis.set_title("Teacher correction success by intervention point")
    save_figure(fig, figures / "teacher_success_by_q.png")

    primary_summary = summary[
        (summary.arm == analysis["primary_arm"])
        & (summary.neighborhood == analysis["primary_neighborhood"])
        & (summary.representation == analysis["primary_representation"])
        & (summary.metric == analysis["primary_metric"])
    ]
    table_html = primary_summary.to_html(
        index=False, float_format=lambda value: f"{value:.5f}"
    )
    dashboard = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Objective reliability calibration</title>
<style>body{{font-family:system-ui;max-width:1180px;margin:30px auto;line-height:1.55}}img{{max-width:100%;border:1px solid #ddd}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:6px;border:1px solid #ddd}}code{{background:#eee;padding:2px 4px}}</style></head>
<body><h1>Objective teacher-reliability calibration</h1>
<p>Endpoint: for each frozen student state, the teacher receives {int(config["selection"]["target_valid_teacher_repeats"])} valid stochastic continuations. The verifier success fraction is an objective estimate of teacher competence at that state. Truncated or unparsable generations are excluded by the predeclared engineering-validity rule.</p>
<h2>Metric × support-control comparison</h2><img src='metric_support_comparison.png'>
<p>Bars report prompt-grouped cross-validated Brier improvement over a baseline that only knows q20/q40/q60/q80. Positive values mean the metric adds held-out information. <code>selected</code> is the proposed dual-ball support; <code>random</code> matches its neighbor count; <code>far</code> uses the most distant candidates. A useful locality mechanism should help under selected support and outperform these controls.</p>
<h2>Paired selected-support advantage</h2><img src='selected_vs_controls_paired.png'>
<p>Each point compares selected and a matched control on the same states; whiskers are prompt-cluster bootstrap 95% intervals. Positive values favor the proposed support. An interval crossing zero is not evidence that semantic selection adds predictive value.</p>
<h2>Primary calibration</h2><img src='primary_calibration.png'>
<p>Each curve compares held-out predicted teacher success against verifier-observed success. Proximity to the diagonal indicates calibrated probabilities; separate curves prevent intervention-stage mixing.</p>
<h2>Teacher success by intervention point</h2><img src='teacher_success_by_q.png'>
<p>This is the raw repeated-continuation competence endpoint, not a neighborhood metric. It checks whether later student prefixes are objectively harder for the teacher to repair.</p>
<h2>Predeclared primary result</h2>{table_html}
<p>Run directory: <code>{html.escape(str(run_dir))}</code></p></body></html>"""
    (figures / "index.html").write_text(dashboard, encoding="utf-8")
    update_status(
        run_dir,
        "completed",
        analyzed_states=int(frame.state_id.nunique()),
        dashboard=str(figures / "index.html"),
        primary_rows=len(primary_summary),
    )
    print(f"DASHBOARD={figures / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
