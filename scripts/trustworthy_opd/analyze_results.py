from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import load_run, update_status
from scipy.stats import rankdata, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether local instability predicts real-task reliability."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels.astype(bool)
    positives = int(positive.sum())
    negatives = int((~positive).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float(
        (ranks[positive].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan"), float("nan")
    result = spearmanr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue)


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    frame = pd.read_parquet(run_dir / "results" / "reliability.parquet")
    frame = frame[frame["valid_neighborhood"]].copy()
    if frame.empty:
        raise RuntimeError("No validated states with valid neighborhoods were found")

    teacher_metrics = [
        ("teacher_max", 1.0),
        ("teacher_q95", 1.0),
        ("teacher_top_mean", 1.0),
        ("teacher_cvar", 1.0),
        ("teacher_abs_max", 1.0),
        ("relative_max", 1.0),
        ("relative_q95", 1.0),
        ("relative_top_mean", 1.0),
        ("relative_cvar", 1.0),
        ("relative_abs_excess_max", 1.0),
        ("teacher_entropy", 1.0),
        ("teacher_max_probability", -1.0),
        ("teacher_action_log_prob", -1.0),
        ("teacher_student_action_gap", -1.0),
        ("topk_overlap", -1.0),
        ("teacher_student_kl", 1.0),
        ("student_teacher_kl", 1.0),
        ("teacher_student_jsd", 1.0),
        ("teacher_student_top1_agreement", -1.0),
        ("teacher_prefix_ppl", 1.0),
        ("teacher_action_ppl", 1.0),
        ("teacher_self_consistency_majority", -1.0),
        ("teacher_self_consistency_pairwise", -1.0),
        ("teacher_semantic_entropy", 1.0),
        ("teacher_normalized_semantic_entropy", 1.0),
        ("teacher_self_consistency_majority_coverage_adjusted", -1.0),
        ("teacher_self_consistency_pairwise_coverage_adjusted", -1.0),
        ("teacher_semantic_entropy_conservative", 1.0),
        ("teacher_normalized_semantic_entropy_conservative", 1.0),
        ("teacher_valid_answer_rate", -1.0),
    ]
    relative_metrics = [
        ("relative_max", 1.0),
        ("relative_q95", 1.0),
        ("relative_top_mean", 1.0),
        ("relative_cvar", 1.0),
        ("relative_abs_excess_max", 1.0),
        ("teacher_student_action_gap", -1.0),
        ("topk_overlap", -1.0),
        ("teacher_student_kl", 1.0),
        ("student_teacher_kl", 1.0),
        ("teacher_student_jsd", 1.0),
        ("teacher_student_top1_agreement", -1.0),
    ]
    metric_specs = [
        (metric, "q_teacher", direction) for metric, direction in teacher_metrics
    ] + [
        (metric, "q_teacher_minus_student", direction)
        for metric, direction in relative_metrics
    ]
    analysis = config["analysis"]
    teacher_threshold = float(analysis["unreliable_teacher_threshold"])
    relative_threshold = float(analysis["nonpositive_teacher_advantage_threshold"])
    summary_rows = []
    calibration_rows = []

    for (representation, method), subset in frame.groupby(
        ["representation", "neighborhood_method"]
    ):
        subset = subset.drop_duplicates("state_id")
        for metric, target, auc_direction in metric_specs:
            if metric not in subset:
                continue
            correlation, pvalue = safe_spearman(subset[metric], subset[target])
            if target == "q_teacher":
                unreliable = subset[target].to_numpy(dtype=float) <= teacher_threshold
            else:
                unreliable = subset[target].to_numpy(dtype=float) <= relative_threshold
            scores = subset[metric].to_numpy(dtype=float) * auc_direction
            finite = np.isfinite(scores)
            auc = (
                roc_auc(unreliable[finite], scores[finite])
                if finite.any()
                else float("nan")
            )
            summary_rows.append(
                {
                    "representation": representation,
                    "neighborhood_method": method,
                    "metric": metric,
                    "target": target,
                    "n": len(subset),
                    "spearman": correlation,
                    "spearman_pvalue": pvalue,
                    "unreliable_auc": auc,
                }
            )

            try:
                bins = pd.qcut(
                    subset[metric],
                    q=int(analysis["calibration_bins"]),
                    duplicates="drop",
                )
            except ValueError:
                continue
            grouped = subset.assign(metric_bin=bins).groupby(
                "metric_bin", observed=True
            )
            for bin_name, group in grouped:
                calibration_rows.append(
                    {
                        "representation": representation,
                        "neighborhood_method": method,
                        "metric": metric,
                        "target": target,
                        "bin": str(bin_name),
                        "count": len(group),
                        "metric_mean": float(group[metric].mean()),
                        "target_mean": float(group[target].mean()),
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["target", "unreliable_auc"], ascending=[True, False]
    )
    calibration = pd.DataFrame(calibration_rows)
    summary.to_csv(run_dir / "results" / "correlations.csv", index=False)
    calibration.to_csv(run_dir / "results" / "calibration.csv", index=False)
    report = {
        "validated_unique_states": int(frame["state_id"].nunique()),
        "teacher_mean_success": float(
            frame.drop_duplicates("state_id")["q_teacher"].mean()
        ),
        "student_mean_success": float(
            frame.drop_duplicates("state_id")["q_student"].mean()
        ),
        "best_teacher_predictors": summary[summary["target"] == "q_teacher"]
        .head(10)
        .to_dict("records"),
        "best_relative_predictors": summary[
            summary["target"] == "q_teacher_minus_student"
        ]
        .head(10)
        .to_dict("records"),
    }
    (run_dir / "results" / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    update_status(run_dir, "analyzed", analysis_rows=len(summary))
    print(summary.head(20).to_string(index=False), flush=True)
    print(f"Results written under {run_dir / 'results'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
