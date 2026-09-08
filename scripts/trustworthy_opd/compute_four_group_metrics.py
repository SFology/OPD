from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from common import cosine_distances, load_run, read_jsonl, update_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PPL and dual-space neighborhood stability for four groups."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def ordinal_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    if len(values) > 1:
        ranks /= len(values) - 1
    return ranks


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    states = read_jsonl(run_dir / "artifacts" / "states.jsonl")
    labels = {
        row["state_id"]: row
        for row in read_jsonl(run_dir / "results" / "pair_labels.jsonl")
        if row["eligible_pair"]
    }
    teacher = np.load(run_dir / "features" / "teacher.npz")
    student = np.load(run_dir / "features" / "student.npz")
    expected_ids = [state["state_id"] for state in states]
    if teacher["state_ids"].tolist() != expected_ids:
        raise RuntimeError("Teacher feature ordering does not match states")
    if student["state_ids"].tolist() != expected_ids:
        raise RuntimeError("Student feature ordering does not match states")

    action_vocab = np.asarray(
        __import__("json").loads(
            (run_dir / "artifacts" / "action_vocab.json").read_text(encoding="utf-8")
        ),
        dtype=np.int64,
    )
    action_column = {int(token): index for index, token in enumerate(action_vocab)}
    candidate_groups = defaultdict(list)
    for index, state in enumerate(states):
        candidate_groups[(int(state["prompt_index"]), state["fraction_name"])].append(
            index
        )

    k = int(config["neighborhood"]["k"])
    minimum = int(config["neighborhood"]["minimum_neighbors"])
    representations = [
        item["name"] for item in config["representations"]["definitions"]
    ]
    rows = []
    for representation in representations:
        teacher_rep = teacher[f"representation__{representation}"].astype(np.float32)
        student_rep = student[f"representation__{representation}"].astype(np.float32)
        for index, state in enumerate(states):
            label = labels.get(state["state_id"])
            if label is None:
                continue
            candidates = [
                item
                for item in candidate_groups[
                    (int(state["prompt_index"]), state["fraction_name"])
                ]
                if item != index
                and (
                    not config["neighborhood"].get("cross_rollout_only", True)
                    or states[item]["rollout_index"] != state["rollout_index"]
                )
            ]
            if len(candidates) < minimum:
                continue
            candidate_array = np.asarray(candidates, dtype=np.int64)
            teacher_distance = cosine_distances(
                teacher_rep[index], teacher_rep[candidate_array]
            )
            student_distance = cosine_distances(
                student_rep[index], student_rep[candidate_array]
            )
            joint_rank = np.maximum(
                ordinal_rank(teacher_distance), ordinal_rank(student_distance)
            )
            count = min(k, len(candidate_array))
            local = np.argsort(joint_rank, kind="stable")[:count]
            neighbors = candidate_array[local]
            if len(neighbors) < minimum:
                continue

            token = int(label["student_action_token_id"])
            column = action_column[token]
            teacher_change = (
                teacher["action_log_probs"][index, column]
                - teacher["action_log_probs"][neighbors, column]
            )
            student_change = (
                student["action_log_probs"][index, column]
                - student["action_log_probs"][neighbors, column]
            )
            teacher_stability = float(np.max(teacher_change))
            student_stability = float(np.max(student_change))
            relative_stability = float(np.max(teacher_change - student_change))
            teacher_ppl = float(teacher["prefix_ppl"][index])
            student_ppl = float(student["prefix_ppl"][index])
            rows.append(
                {
                    **label,
                    "rollout_index": state["rollout_index"],
                    "representation": representation,
                    "candidate_count": len(candidates),
                    "neighbor_count": len(neighbors),
                    "neighbor_state_ids": [
                        states[item]["state_id"] for item in neighbors
                    ],
                    "teacher_log_ppl": float(np.log(max(teacher_ppl, 1e-12))),
                    "student_log_ppl": float(np.log(max(student_ppl, 1e-12))),
                    "teacher_student_log_ppl_gap": float(
                        np.log(max(teacher_ppl, 1e-12))
                        - np.log(max(student_ppl, 1e-12))
                    ),
                    "teacher_stability": teacher_stability,
                    "student_stability": student_stability,
                    "relative_stability": relative_stability,
                }
            )
        print(f"Computed metrics for {representation}", flush=True)

    output = pd.DataFrame(rows)
    output.to_parquet(run_dir / "results" / "four_group_metrics.parquet", index=False)
    output.drop(columns=["neighbor_state_ids"]).to_csv(
        run_dir / "results" / "four_group_metrics.csv", index=False
    )
    update_status(
        run_dir,
        "group_metrics_computed",
        metric_rows=len(output),
        metric_states=output.state_id.nunique(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
