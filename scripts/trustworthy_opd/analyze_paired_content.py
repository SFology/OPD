from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import load_run, read_jsonl, update_status
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze valid-only paired trajectory content reliability."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--minimum-complete-state-fraction",
        type=float,
        help=(
            "Override the configured completeness gate. Use only for explicitly "
            "labeled exploratory analyses of incomplete fixed-count samples."
        ),
    )
    return parser.parse_args()


def answer_metrics(rows: list[dict]) -> dict[str, float]:
    answers = [str(row["predicted_answer"]).strip() for row in rows]
    count = len(answers)
    frequencies = pd.Series(answers).value_counts().to_numpy(dtype=float)
    probabilities = frequencies / count
    pairwise = (
        float(np.sum(frequencies * (frequencies - 1)) / (count * (count - 1)))
        if count > 1
        else 1.0
    )
    return {
        "teacher_self_consistency_majority": float(frequencies.max() / count),
        "teacher_self_consistency_pairwise": pairwise,
        "teacher_semantic_entropy": float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
        ),
    }


def build_labels(run_dir: Path, config: dict) -> pd.DataFrame:
    anchors = read_jsonl(run_dir / "artifacts" / "paired_selected_states.jsonl")
    attempts = read_jsonl(run_dir / "results" / "paired_content_attempts.jsonl")
    paired = config["paired_content"]
    requirements = {
        "teacher_metric_student": int(paired["teacher_metric_valid_samples"]),
        "teacher_label_student": int(paired["task_label_valid_samples_per_condition"]),
        "teacher_label_self": int(paired["task_label_valid_samples_per_condition"]),
        "student_label_student": int(paired["task_label_valid_samples_per_condition"]),
    }
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in attempts:
        if row["eligible"]:
            grouped.setdefault((row["state_id"], row["phase"]), []).append(row)
    rows = []
    for state in anchors:
        selected = {}
        complete = True
        for phase, required in requirements.items():
            values = sorted(
                grouped.get((state["state_id"], phase), []),
                key=lambda row: int(row["attempt_index"]),
            )[:required]
            selected[phase] = values
            complete &= len(values) == required
        if not complete:
            continue
        teacher_student = selected["teacher_label_student"]
        teacher_self = selected["teacher_label_self"]
        student_student = selected["student_label_student"]
        metric_rows = selected["teacher_metric_student"]
        q_teacher_student = float(np.mean([row["correct"] for row in teacher_student]))
        q_teacher_self = float(np.mean([row["correct"] for row in teacher_self]))
        q_student = float(np.mean([row["correct"] for row in student_student]))
        rows.append(
            {
                "state_id": state["state_id"],
                "prompt_index": state["prompt_index"],
                "paired_stage": state["paired_stage"],
                "normalized_position": state["normalized_position"],
                "trajectory_correct": state["trajectory_correct"],
                "q_teacher_student_prefix": q_teacher_student,
                "q_teacher_self_prefix": q_teacher_self,
                "q_student_student_prefix": q_student,
                "teacher_domain_gap": q_teacher_self - q_teacher_student,
                "teacher_advantage": q_teacher_student - q_student,
                **answer_metrics(metric_rows),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = []
    for _ in range(samples):
        index = rng.integers(0, len(x), len(x))
        xb, yb = x[index], y[index]
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            continue
        value = float(spearmanr(xb, yb).statistic)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(item) for item in np.quantile(values, [0.025, 0.975]))


def correlation_table(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    metrics = [
        "teacher_entropy",
        "teacher_max_probability",
        "teacher_prefix_ppl",
        "teacher_student_kl",
        "teacher_student_jsd",
        "teacher_student_top1_agreement",
        "topk_overlap",
        "teacher_max",
        "teacher_abs_max",
        "relative_max",
        "relative_top_mean",
        "relative_abs_excess_max",
        "teacher_self_consistency_majority",
        "teacher_self_consistency_pairwise",
        "teacher_semantic_entropy",
    ]
    targets = [
        "q_teacher_student_prefix",
        "teacher_domain_gap",
        "teacher_advantage",
    ]
    bootstrap_metrics = {
        "teacher_entropy",
        "teacher_prefix_ppl",
        "teacher_student_kl",
        "topk_overlap",
        "relative_max",
        "relative_top_mean",
        "teacher_self_consistency_pairwise",
        "teacher_semantic_entropy",
    }
    samples = int(config["paired_content"]["bootstrap_samples"])
    rng = np.random.default_rng(int(config["experiment"]["seed"]) + 55000)
    results = []
    for (representation, method), subset in frame.groupby(
        ["representation", "neighborhood_method"]
    ):
        subset = subset.drop_duplicates("state_id")
        for scope, scoped in [
            ("overall", subset),
            *list(subset.groupby("paired_stage")),
        ]:
            if isinstance(scope, tuple):
                scope = str(scope[0])
            for metric in metrics:
                if metric not in scoped:
                    continue
                for target in targets:
                    valid = (
                        scoped[[metric, target]]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )
                    if (
                        len(valid) < 8
                        or valid[metric].nunique() < 2
                        or valid[target].nunique() < 2
                    ):
                        continue
                    result = spearmanr(valid[metric], valid[target])
                    low, high = float("nan"), float("nan")
                    if scope == "overall" and metric in bootstrap_metrics:
                        low, high = bootstrap_spearman(
                            valid[metric].to_numpy(float),
                            valid[target].to_numpy(float),
                            samples,
                            rng,
                        )
                    results.append(
                        {
                            "representation": representation,
                            "neighborhood_method": method,
                            "scope": scope,
                            "metric": metric,
                            "target": target,
                            "n": len(valid),
                            "spearman": float(result.statistic),
                            "spearman_pvalue": float(result.pvalue),
                            "spearman_ci_low": low,
                            "spearman_ci_high": high,
                        }
                    )
    return pd.DataFrame(results)


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    update_status(run_dir, "analyzing_paired_content")
    try:
        labels = build_labels(run_dir, config)
        expected = int(config["data"]["num_prompts"])
        complete_fraction = len(labels) / expected
        configured_minimum = float(
            config["paired_content"]["minimum_complete_state_fraction"]
        )
        minimum_complete_fraction = (
            args.minimum_complete_state_fraction
            if args.minimum_complete_state_fraction is not None
            else configured_minimum
        )
        if not 0.0 <= minimum_complete_fraction <= 1.0:
            raise ValueError("Minimum complete-state fraction must be within [0, 1]")
        exploratory_override = args.minimum_complete_state_fraction is not None
        labels.to_csv(run_dir / "results" / "paired_content_labels.csv", index=False)
        if complete_fraction < minimum_complete_fraction:
            update_status(
                run_dir,
                "paired_content_quality_rejected",
                complete_states=len(labels),
                expected_states=expected,
                complete_state_fraction=complete_fraction,
                minimum_complete_state_fraction=minimum_complete_fraction,
                exploratory_override=exploratory_override,
            )
            raise RuntimeError(
                f"Only {len(labels)}/{expected} states have fixed-count valid labels"
            )
        stability = pd.read_parquet(run_dir / "results" / "stability.parquet")
        frame = stability.merge(labels, on="state_id", validate="many_to_one")
        frame.to_parquet(
            run_dir / "results" / "paired_content_reliability.parquet", index=False
        )
        frame.drop(columns=["neighbor_state_ids"]).to_csv(
            run_dir / "results" / "paired_content_reliability.csv", index=False
        )
        correlations = correlation_table(frame, config)
        correlations.to_csv(
            run_dir / "results" / "paired_content_correlations.csv", index=False
        )
        primary = correlations[
            (correlations.representation == "mid_prefix_mean")
            & (correlations.neighborhood_method == "dual_knn")
            & (correlations.scope == "overall")
        ]
        summary = {
            "expected_states": expected,
            "complete_states": len(labels),
            "complete_state_fraction": complete_fraction,
            "configured_minimum_complete_state_fraction": configured_minimum,
            "analysis_minimum_complete_state_fraction": minimum_complete_fraction,
            "exploratory_completeness_override": exploratory_override,
            "stage_counts": labels.paired_stage.value_counts().to_dict(),
            "mean_q_teacher_student_prefix": float(
                labels.q_teacher_student_prefix.mean()
            ),
            "mean_q_teacher_self_prefix": float(labels.q_teacher_self_prefix.mean()),
            "mean_q_student_student_prefix": float(
                labels.q_student_student_prefix.mean()
            ),
            "mean_teacher_domain_gap": float(labels.teacher_domain_gap.mean()),
            "mean_teacher_advantage": float(labels.teacher_advantage.mean()),
            "primary_correlations": primary.sort_values(
                ["target", "spearman_pvalue"]
            ).to_dict(orient="records"),
        }
        (run_dir / "results" / "paired_content_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        update_status(
            run_dir,
            "paired_content_analyzed",
            complete_states=len(labels),
            expected_states=expected,
            complete_state_fraction=complete_fraction,
            minimum_complete_state_fraction=minimum_complete_fraction,
            exploratory_override=exploratory_override,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        status_path = run_dir / "status.yaml"
        if "paired_content_quality_rejected" not in status_path.read_text():
            update_status(
                run_dir,
                "failed",
                stage="paired_content_analysis",
                error=f"{type(error).__name__}: {error}",
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
