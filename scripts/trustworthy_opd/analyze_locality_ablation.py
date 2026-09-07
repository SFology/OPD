from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    aggregate_changes,
    cosine_distances,
    create_run_dir,
    load_config,
    read_jsonl,
    update_status,
)
from scipy.stats import rankdata, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute OPD reliability metrics with progress-aligned neighbors."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def dual_knn(
    teacher_distance: np.ndarray,
    student_distance: np.ndarray,
    candidates: np.ndarray,
    k: int,
) -> np.ndarray:
    count = min(k, len(candidates))
    teacher_local = set(np.argsort(teacher_distance)[:count].tolist())
    student_local = set(np.argsort(student_distance)[:count].tolist())
    intersection = np.asarray(sorted(teacher_local & student_local), dtype=np.int64)
    if not intersection.size:
        return np.asarray([], dtype=np.int64)
    joint = np.maximum(teacher_distance[intersection], student_distance[intersection])
    return candidates[intersection[np.argsort(joint)]]


def candidate_indices(
    anchor_index: int,
    states: list[dict],
    maximum_delta: float,
    cross_rollout_only: bool,
) -> np.ndarray:
    anchor = states[anchor_index]
    result = []
    for index, state in enumerate(states):
        if index == anchor_index or state["prompt_index"] != anchor["prompt_index"]:
            continue
        if cross_rollout_only and state["rollout_index"] == anchor["rollout_index"]:
            continue
        delta = abs(
            float(state["normalized_position"]) - float(anchor["normalized_position"])
        )
        if delta > maximum_delta:
            continue
        result.append(index)
    return np.asarray(result, dtype=np.int64)


def add_aggregates(
    row: dict,
    prefix: str,
    values: np.ndarray,
    top_tail_count: int,
    cvar_quantile: float,
) -> None:
    for name, value in aggregate_changes(values, top_tail_count, cvar_quantile).items():
        row[f"{prefix}_{name}"] = value


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if samples <= 0:
        return float("nan"), float("nan")
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(x), len(x))
        xb, yb = x[indices], y[indices]
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            continue
        value = float(spearmanr(xb, yb).statistic)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(item) for item in np.quantile(values, [0.025, 0.975]))


def summarize_correlations(rows: pd.DataFrame, config: dict) -> pd.DataFrame:
    metrics = [
        column
        for column in rows.columns
        if column.startswith(
            (
                "teacher_",
                "relative_",
                "teacher_abs_",
                "relative_abs_excess_",
            )
        )
        and column
        not in {
            "teacher_radius",
            "teacher_metric_truncated_rate",
            "teacher_label_truncated_rate",
            "teacher_label_parseable_rate",
        }
    ]
    targets = ["q_teacher", "q_teacher_minus_student"]
    analysis = config["analysis"]
    bootstrap_metrics = set(analysis.get("bootstrap_metrics", metrics))
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    result = []
    for (representation, neighborhood), subset in rows.groupby(
        ["representation", "neighborhood"]
    ):
        for metric in metrics:
            for target in targets:
                valid = (
                    subset[[metric, target]].replace([np.inf, -np.inf], np.nan).dropna()
                )
                if (
                    len(valid) < 3
                    or valid[metric].nunique() < 2
                    or valid[target].nunique() < 2
                ):
                    continue
                statistic = spearmanr(valid[metric], valid[target])
                if metric in bootstrap_metrics:
                    low, high = bootstrap_spearman(
                        valid[metric].to_numpy(float),
                        valid[target].to_numpy(float),
                        int(analysis["bootstrap_samples"]),
                        rng,
                    )
                else:
                    low, high = float("nan"), float("nan")
                threshold = (
                    float(analysis["unreliable_teacher_threshold"])
                    if target == "q_teacher"
                    else float(analysis["nonpositive_teacher_advantage_threshold"])
                )
                unreliable = valid[target].to_numpy(float) <= threshold
                auc_high = roc_auc(unreliable, valid[metric].to_numpy(float))
                result.append(
                    {
                        "representation": representation,
                        "neighborhood": neighborhood,
                        "metric": metric,
                        "target": target,
                        "n": len(valid),
                        "spearman": float(statistic.statistic),
                        "spearman_pvalue": float(statistic.pvalue),
                        "spearman_ci_low": low,
                        "spearman_ci_high": high,
                        "unreliable_auc_higher_metric": auc_high,
                        "unreliable_auc_lower_metric": 1.0 - auc_high,
                    }
                )
    return pd.DataFrame(result)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    source = Path(config["source_run"]).resolve()
    if not (source / "results" / "reliability.parquet").exists():
        raise FileNotFoundError(f"Source run is not fully validated: {source}")

    run_dir = create_run_dir(config, args.config)
    update_status(run_dir, "analyzing_locality", source_run=str(source))
    try:
        states = read_jsonl(source / "artifacts" / "states.jsonl")
        selected = read_jsonl(source / "artifacts" / "selected_states.jsonl")
        state_index = {state["state_id"]: index for index, state in enumerate(states)}
        selected_ids = {state["state_id"] for state in selected}

        labels = pd.read_parquet(source / "results" / "reliability.parquet")
        labels = labels[labels.state_id.isin(selected_ids)].drop_duplicates("state_id")
        label_columns = [
            "state_id",
            "q_teacher",
            "q_student",
            "q_teacher_minus_student",
            "teacher_metric_truncated_rate",
            "teacher_label_truncated_rate",
            "teacher_label_parseable_rate",
            "teacher_valid_answer_rate",
            "teacher_prefix_ppl",
            "teacher_entropy",
            "teacher_max_probability",
            "teacher_student_kl",
            "teacher_student_top1_agreement",
            "topk_overlap",
        ]
        labels = labels[label_columns].set_index("state_id")

        teacher = np.load(source / "features" / "teacher.npz")
        student = np.load(source / "features" / "student.npz")
        expected_ids = [state["state_id"] for state in states]
        if teacher["state_ids"].tolist() != expected_ids:
            raise RuntimeError(
                "Teacher feature state order does not match states.jsonl"
            )
        if student["state_ids"].tolist() != expected_ids:
            raise RuntimeError(
                "Student feature state order does not match states.jsonl"
            )

        output_rows = []
        top_count = int(config["top_tail_count"])
        cvar = float(config["cvar_quantile"])
        epsilon = float(config["distance_epsilon"])
        cross_rollout = bool(config.get("cross_rollout_only", True))
        minimum = int(config["min_neighbors"])

        for representation in config["representations"]:
            teacher_rep = teacher[f"representation__{representation}"].astype(
                np.float32
            )
            student_rep = student[f"representation__{representation}"].astype(
                np.float32
            )
            for neighborhood in config["neighborhoods"]:
                name = str(neighborhood["name"])
                maximum_delta = float(neighborhood["max_normalized_position_delta"])
                k = int(neighborhood["knn_k_per_model"])
                for state_id in sorted(selected_ids):
                    anchor_index = state_index[state_id]
                    candidates = candidate_indices(
                        anchor_index,
                        states,
                        maximum_delta,
                        cross_rollout,
                    )
                    if candidates.size:
                        teacher_distance = cosine_distances(
                            teacher_rep[anchor_index], teacher_rep[candidates]
                        )
                        student_distance = cosine_distances(
                            student_rep[anchor_index], student_rep[candidates]
                        )
                        neighbors = dual_knn(
                            teacher_distance, student_distance, candidates, k
                        )
                    else:
                        teacher_distance = np.asarray([], dtype=np.float32)
                        student_distance = np.asarray([], dtype=np.float32)
                        neighbors = np.asarray([], dtype=np.int64)

                    row = {
                        "state_id": state_id,
                        "representation": representation,
                        "neighborhood": name,
                        "max_normalized_position_delta": maximum_delta,
                        "candidate_count": len(candidates),
                        "neighbor_count": len(neighbors),
                        "valid_neighborhood": len(neighbors) >= minimum,
                    }
                    row.update(labels.loc[state_id].to_dict())
                    if len(neighbors) >= minimum:
                        neighbor_lookup = {
                            int(item): index for index, item in enumerate(candidates)
                        }
                        local = np.asarray(
                            [neighbor_lookup[int(item)] for item in neighbors]
                        )
                        joint_distance = np.maximum(
                            teacher_distance[local], student_distance[local]
                        )
                        column = int(teacher["action_columns"][anchor_index])
                        teacher_change = (
                            teacher["action_log_probs"][anchor_index, column]
                            - teacher["action_log_probs"][neighbors, column]
                        ).astype(np.float32)
                        student_change = (
                            student["action_log_probs"][anchor_index, column]
                            - student["action_log_probs"][neighbors, column]
                        ).astype(np.float32)
                        relative_change = teacher_change - student_change
                        relative_abs_excess = np.abs(teacher_change) - np.abs(
                            student_change
                        )
                        scale = np.maximum(joint_distance, epsilon)
                        add_aggregates(row, "teacher", teacher_change, top_count, cvar)
                        add_aggregates(
                            row,
                            "teacher_abs",
                            np.abs(teacher_change),
                            top_count,
                            cvar,
                        )
                        add_aggregates(
                            row, "relative", relative_change, top_count, cvar
                        )
                        add_aggregates(
                            row,
                            "relative_abs_excess",
                            relative_abs_excess,
                            top_count,
                            cvar,
                        )
                        add_aggregates(
                            row,
                            "teacher_slope_abs",
                            np.abs(teacher_change) / scale,
                            top_count,
                            cvar,
                        )
                        add_aggregates(
                            row,
                            "relative_slope",
                            relative_change / scale,
                            top_count,
                            cvar,
                        )
                        row["mean_joint_distance"] = float(joint_distance.mean())
                        row["max_joint_distance"] = float(joint_distance.max())
                        row["mean_normalized_position_delta"] = float(
                            np.mean(
                                [
                                    abs(
                                        float(states[item]["normalized_position"])
                                        - float(
                                            states[anchor_index]["normalized_position"]
                                        )
                                    )
                                    for item in neighbors
                                ]
                            )
                        )
                    output_rows.append(row)

        metrics = pd.DataFrame(output_rows)
        valid = metrics[metrics.valid_neighborhood].copy()
        correlations = summarize_correlations(valid, config)
        metrics.to_parquet(
            run_dir / "results" / "locality_metrics.parquet", index=False
        )
        metrics.to_csv(run_dir / "results" / "locality_metrics.csv", index=False)
        correlations.to_csv(
            run_dir / "results" / "locality_correlations.csv", index=False
        )

        core = correlations[
            correlations.metric.isin(
                [
                    "teacher_max",
                    "teacher_abs_max",
                    "teacher_slope_abs_max",
                    "relative_max",
                    "relative_top_mean",
                    "relative_abs_excess_max",
                    "relative_slope_max",
                ]
            )
        ].copy()
        best = {}
        for target in ["q_teacher", "q_teacher_minus_student"]:
            subset = core[core.target == target].copy()
            subset["absolute_spearman"] = subset.spearman.abs()
            best[target] = (
                subset.sort_values("absolute_spearman", ascending=False)
                .head(10)
                .drop(columns="absolute_spearman")
                .to_dict(orient="records")
            )
        report = {
            "source_run": str(source),
            "validated_anchor_states": len(labels),
            "metric_rows": len(metrics),
            "valid_metric_rows": len(valid),
            "best_core_metrics": best,
        }
        (run_dir / "results" / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        update_status(
            run_dir,
            "analyzed",
            source_run=str(source),
            validated_anchor_states=len(labels),
            metric_rows=len(metrics),
            valid_metric_rows=len(valid),
        )
        print(f"Locality ablation written to {run_dir}", flush=True)
        print(
            core.sort_values("spearman_pvalue").head(20).to_string(index=False),
            flush=True,
        )
        return 0
    except Exception as error:
        update_status(run_dir, "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
