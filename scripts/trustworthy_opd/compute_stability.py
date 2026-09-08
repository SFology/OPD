from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    aggregate_changes,
    cosine_distances,
    load_run,
    read_jsonl,
    update_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build model-specific double-ball neighborhoods and OPD trust metrics."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def candidate_indices(index: int, states: list[dict], config: dict) -> np.ndarray:
    anchor = states[index]
    candidates = []
    max_delta = config["neighborhood"].get("max_position_delta")
    max_normalized_delta = config["neighborhood"].get("max_normalized_position_delta")
    for other_index, other in enumerate(states):
        if other_index == index or other["prompt_index"] != anchor["prompt_index"]:
            continue
        if (
            config["neighborhood"].get("cross_rollout_only", True)
            and other["rollout_index"] == anchor["rollout_index"]
        ):
            continue
        if max_delta is not None and abs(
            int(other["position"]) - int(anchor["position"])
        ) > int(max_delta):
            continue
        if max_normalized_delta is not None and abs(
            float(other["normalized_position"]) - float(anchor["normalized_position"])
        ) > float(max_normalized_delta):
            continue
        if other["input_ids"] == anchor["input_ids"]:
            continue
        candidates.append(other_index)
    return np.asarray(candidates, dtype=np.int64)


def dual_knn_neighbors(
    teacher_distance: np.ndarray,
    student_distance: np.ndarray,
    candidates: np.ndarray,
    k: int,
) -> np.ndarray:
    count = min(k, len(candidates))
    teacher_local = set(np.argsort(teacher_distance)[:count].tolist())
    student_local = set(np.argsort(student_distance)[:count].tolist())
    intersection = np.asarray(sorted(teacher_local & student_local), dtype=np.int64)
    if intersection.size == 0:
        return np.asarray([], dtype=np.int64)
    joint_distance = np.maximum(
        teacher_distance[intersection], student_distance[intersection]
    )
    return candidates[intersection[np.argsort(joint_distance)]]


def collect_radius_thresholds(
    states: list[dict], teacher_rep: np.ndarray, student_rep: np.ndarray, config: dict
) -> tuple[float, float]:
    teacher_values: list[np.ndarray] = []
    student_values: list[np.ndarray] = []
    for index in range(len(states)):
        candidates = candidate_indices(index, states, config)
        if candidates.size == 0:
            continue
        teacher_values.append(
            cosine_distances(teacher_rep[index], teacher_rep[candidates])
        )
        student_values.append(
            cosine_distances(student_rep[index], student_rep[candidates])
        )
    if not teacher_values:
        return float("nan"), float("nan")
    quantile = float(config["neighborhood"]["radius_quantile"])
    return (
        float(np.quantile(np.concatenate(teacher_values), quantile)),
        float(np.quantile(np.concatenate(student_values), quantile)),
    )


def add_aggregates(row: dict, prefix: str, values: np.ndarray, config: dict) -> None:
    summary = aggregate_changes(
        values,
        int(config["stability"]["top_tail_count"]),
        float(config["stability"]["cvar_quantile"]),
    )
    for name, value in summary.items():
        row[f"{prefix}_{name}"] = value


def distribution_comparisons(
    teacher: np.lib.npyio.NpzFile, student: np.lib.npyio.NpzFile
) -> dict[str, np.ndarray]:
    if "full_log_probs" not in teacher or "full_log_probs" not in student:
        raise RuntimeError(
            "Exact KL/JSD requires models.store_full_log_probs=true during feature extraction"
        )
    count = len(teacher["state_ids"])
    result = {
        "teacher_student_kl": np.empty(count, dtype=np.float32),
        "student_teacher_kl": np.empty(count, dtype=np.float32),
        "teacher_student_jsd": np.empty(count, dtype=np.float32),
    }
    for index in range(count):
        teacher_log = teacher["full_log_probs"][index].astype(np.float32)
        student_log = student["full_log_probs"][index].astype(np.float32)
        teacher_log -= np.logaddexp.reduce(teacher_log)
        student_log -= np.logaddexp.reduce(student_log)
        teacher_prob = np.exp(teacher_log)
        student_prob = np.exp(student_log)
        midpoint_log = np.logaddexp(teacher_log, student_log) - np.log(2.0)
        result["teacher_student_kl"][index] = max(
            0.0,
            float(np.sum(teacher_prob * (teacher_log - student_log), dtype=np.float64)),
        )
        result["student_teacher_kl"][index] = max(
            0.0,
            float(np.sum(student_prob * (student_log - teacher_log), dtype=np.float64)),
        )
        result["teacher_student_jsd"][index] = max(
            0.0,
            0.5
            * (
                np.sum(teacher_prob * (teacher_log - midpoint_log), dtype=np.float64)
                + np.sum(student_prob * (student_log - midpoint_log), dtype=np.float64)
            ),
        )
    return result


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    states = read_jsonl(run_dir / "artifacts" / "states.jsonl")
    teacher = np.load(run_dir / "features" / "teacher.npz")
    student = np.load(run_dir / "features" / "student.npz")
    if teacher["state_ids"].tolist() != student["state_ids"].tolist():
        raise RuntimeError(
            "Teacher and student feature files contain different state ordering"
        )
    if teacher["state_ids"].tolist() != [state["state_id"] for state in states]:
        raise RuntimeError("Feature ordering does not match states.jsonl")
    if teacher["action_vocab"].tolist() != student["action_vocab"].tolist():
        raise RuntimeError("Teacher and student action vocabularies differ")

    update_status(run_dir, "computing_stability")
    definitions = [item["name"] for item in config["representations"]["definitions"]]
    methods = config["neighborhood"]["methods"]
    minimum = int(config["neighborhood"]["min_neighbors"])
    k = int(config["neighborhood"]["knn_k_per_model"])
    distribution_metrics = distribution_comparisons(teacher, student)
    rows: list[dict] = []

    for representation in definitions:
        teacher_rep = teacher[f"representation__{representation}"].astype(np.float32)
        student_rep = student[f"representation__{representation}"].astype(np.float32)
        teacher_radius, student_radius = collect_radius_thresholds(
            states, teacher_rep, student_rep, config
        )
        for index, state in enumerate(states):
            candidates = candidate_indices(index, states, config)
            if candidates.size:
                teacher_distance = cosine_distances(
                    teacher_rep[index], teacher_rep[candidates]
                )
                student_distance = cosine_distances(
                    student_rep[index], student_rep[candidates]
                )
            else:
                teacher_distance = np.asarray([], dtype=np.float32)
                student_distance = np.asarray([], dtype=np.float32)

            for method in methods:
                if method == "dual_knn":
                    neighbors = dual_knn_neighbors(
                        teacher_distance, student_distance, candidates, k
                    )
                elif method == "dual_radius":
                    mask = (teacher_distance <= teacher_radius) & (
                        student_distance <= student_radius
                    )
                    neighbors = candidates[mask]
                else:
                    raise ValueError(f"Unknown neighborhood method: {method}")

                row = {
                    "state_id": state["state_id"],
                    "prompt_index": state["prompt_index"],
                    "rollout_index": state["rollout_index"],
                    "position": state["position"],
                    "normalized_position": state["normalized_position"],
                    "action_token_id": state["action_token_id"],
                    "trajectory_correct": state["trajectory_correct"],
                    "representation": representation,
                    "neighborhood_method": method,
                    "candidate_count": len(candidates),
                    "neighbor_count": len(neighbors),
                    "valid_neighborhood": bool(len(neighbors) >= minimum),
                    "neighbor_state_ids": [
                        states[item]["state_id"] for item in neighbors
                    ],
                    "teacher_radius": teacher_radius,
                    "student_radius": student_radius,
                    "teacher_entropy": float(teacher["entropy"][index]),
                    "student_entropy": float(student["entropy"][index]),
                    "teacher_max_probability": float(teacher["max_probability"][index]),
                    "student_max_probability": float(student["max_probability"][index]),
                    "teacher_action_log_prob": float(
                        teacher["action_log_probs"][
                            index, teacher["action_columns"][index]
                        ]
                    ),
                    "student_action_log_prob": float(
                        student["action_log_probs"][
                            index, student["action_columns"][index]
                        ]
                    ),
                    "teacher_prefix_ppl": float(teacher["prefix_ppl"][index]),
                    "teacher_action_ppl": float(
                        np.exp(
                            np.clip(
                                -teacher["action_log_probs"][
                                    index, teacher["action_columns"][index]
                                ],
                                None,
                                50,
                            )
                        )
                    ),
                    "teacher_student_kl": float(
                        distribution_metrics["teacher_student_kl"][index]
                    ),
                    "student_teacher_kl": float(
                        distribution_metrics["student_teacher_kl"][index]
                    ),
                    "teacher_student_jsd": float(
                        distribution_metrics["teacher_student_jsd"][index]
                    ),
                    "teacher_student_top1_agreement": bool(
                        teacher["topk_ids"][index, 0] == student["topk_ids"][index, 0]
                    ),
                    "topk_overlap": float(
                        len(
                            set(teacher["topk_ids"][index].tolist())
                            & set(student["topk_ids"][index].tolist())
                        )
                        / max(1, teacher["topk_ids"].shape[1])
                    ),
                }
                row["teacher_student_action_gap"] = (
                    row["teacher_action_log_prob"] - row["student_action_log_prob"]
                )
                if row["valid_neighborhood"]:
                    column = int(teacher["action_columns"][index])
                    teacher_change = (
                        teacher["action_log_probs"][index, column]
                        - teacher["action_log_probs"][neighbors, column]
                    )
                    student_change = (
                        student["action_log_probs"][index, column]
                        - student["action_log_probs"][neighbors, column]
                    )
                    relative_change = teacher_change - student_change
                    relative_absolute_excess = np.abs(teacher_change) - np.abs(
                        student_change
                    )
                else:
                    teacher_change = np.asarray([], dtype=np.float32)
                    student_change = np.asarray([], dtype=np.float32)
                    relative_change = np.asarray([], dtype=np.float32)
                    relative_absolute_excess = np.asarray([], dtype=np.float32)
                add_aggregates(row, "teacher", teacher_change, config)
                add_aggregates(row, "student", student_change, config)
                add_aggregates(row, "relative", relative_change, config)
                add_aggregates(row, "teacher_abs", np.abs(teacher_change), config)
                add_aggregates(
                    row, "relative_abs_excess", relative_absolute_excess, config
                )
                rows.append(row)

        print(f"Computed neighborhoods for representation {representation}", flush=True)

    output = pd.DataFrame(rows)
    output_path = run_dir / "results" / "stability.parquet"
    output.to_parquet(output_path, index=False)
    output.drop(columns=["neighbor_state_ids"]).to_csv(
        run_dir / "results" / "stability.csv", index=False
    )
    update_status(
        run_dir,
        "stability_computed",
        stability_rows=len(output),
        valid_stability_rows=int(output["valid_neighborhood"].sum()),
    )
    print(f"Wrote {len(output)} rows to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
