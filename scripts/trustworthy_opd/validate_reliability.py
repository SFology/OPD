from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import (
    append_jsonl,
    load_math_grader,
    load_model_and_tokenizer,
    load_run,
    read_jsonl,
    set_seed,
    trim_generated_tokens,
    update_status,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate real-task teacher/student continuation success from selected states."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("student", "teacher"))
    return parser.parse_args()


def select_anchor_states(run_dir: Path, config: dict, states: list[dict]) -> list[dict]:
    selected_path = run_dir / "artifacts" / "selected_states.jsonl"
    existing = read_jsonl(selected_path)
    if existing:
        return existing

    stability = pd.read_parquet(run_dir / "results" / "stability.parquet")
    validation = config["validation"]
    subset = stability[
        (stability["representation"] == validation["primary_representation"])
        & (stability["neighborhood_method"] == validation["primary_neighborhood"])
        & stability["valid_neighborhood"]
    ].drop_duplicates("state_id")
    count = min(int(validation["num_anchor_states"]), len(subset))
    if count == 0:
        raise RuntimeError("No valid anchor states are available for task validation")
    strategy = validation.get("selection_strategy", "primary_metric_stratified")
    metric = validation.get("primary_metric")
    if strategy == "primary_metric_stratified":
        if not metric:
            raise ValueError(
                "primary_metric is required for metric-stratified selection"
            )
        subset = subset[np.isfinite(subset[metric])].sort_values(metric)
        positions = np.linspace(0, len(subset) - 1, num=count, dtype=int)
        chosen = subset.iloc[positions]
    elif strategy == "prompt_balanced_random":
        rng = np.random.default_rng(int(config["experiment"]["seed"]) + 30000)
        queues = {
            prompt_index: rng.permutation(group.index.to_numpy()).tolist()
            for prompt_index, group in subset.groupby("prompt_index")
        }
        prompt_order = rng.permutation(sorted(queues)).tolist()
        chosen_indices: list[int] = []
        while len(chosen_indices) < count:
            added = False
            for prompt_index in prompt_order:
                if queues[prompt_index]:
                    chosen_indices.append(queues[prompt_index].pop())
                    added = True
                    if len(chosen_indices) == count:
                        break
            if not added:
                break
            prompt_order = rng.permutation(prompt_order).tolist()
        chosen = subset.loc[chosen_indices]
    else:
        raise ValueError(f"Unknown validation selection strategy: {strategy}")
    state_by_id = {state["state_id"]: state for state in states}
    rows = []
    for _, item in chosen.iterrows():
        state = dict(state_by_id[item["state_id"]])
        state["selection_strategy"] = strategy
        state["selection_metric"] = (
            metric if strategy == "primary_metric_stratified" else None
        )
        state["selection_value"] = (
            float(item[metric]) if strategy == "primary_metric_stratified" else None
        )
        rows.append(state)
    write_jsonl(selected_path, rows)
    return rows


def summarize_if_complete(run_dir: Path, config: dict, selected: list[dict]) -> bool:
    records = read_jsonl(run_dir / "results" / "continuations.jsonl")
    required = int(config["validation"]["continuations_per_state"])
    metric_count = int(config["validation"]["metric_continuations_per_state"])
    if not 0 < metric_count < required:
        raise ValueError(
            "metric_continuations_per_state must be between zero and continuations_per_state"
        )
    counts: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        counts.setdefault((record["state_id"], record["role"]), []).append(record)
    if any(
        len(counts.get((state["state_id"], role), []))
        < (required if role == "teacher" else required - metric_count)
        for state in selected
        for role in ("student", "teacher")
    ):
        return False

    labels = []
    for state in selected:
        student_rows = sorted(
            counts[(state["state_id"], "student")],
            key=lambda row: int(row["continuation_index"]),
        )[:required]
        teacher_rows = sorted(
            counts[(state["state_id"], "teacher")],
            key=lambda row: int(row["continuation_index"]),
        )[:required]
        teacher_metric_rows = [
            row for row in teacher_rows if int(row["continuation_index"]) < metric_count
        ]
        student_label_rows = [
            row
            for row in student_rows
            if int(row["continuation_index"]) >= metric_count
        ]
        teacher_label_rows = [
            row
            for row in teacher_rows
            if int(row["continuation_index"]) >= metric_count
        ]
        q_student = float(np.mean([row["correct"] for row in student_label_rows]))
        q_teacher = float(np.mean([row["correct"] for row in teacher_label_rows]))

        answers = [
            str(row.get("predicted_answer", "")).strip() for row in teacher_metric_rows
        ]
        valid_answers = [
            answer
            for answer in answers
            if answer and answer not in {"None", "[INVALID]"}
        ]
        clusters: dict[str, int] = {}
        for answer in valid_answers:
            clusters[answer] = clusters.get(answer, 0) + 1
        cluster_counts = np.asarray(list(clusters.values()), dtype=np.float64)
        if len(valid_answers):
            probabilities = cluster_counts / len(valid_answers)
            semantic_entropy = float(
                -np.sum(probabilities * np.log(probabilities), dtype=np.float64)
            )
            majority = float(cluster_counts.max() / len(valid_answers))
        else:
            semantic_entropy = float("nan")
            majority = float("nan")
        if len(valid_answers) >= 2:
            pairwise = float(
                np.sum(cluster_counts * (cluster_counts - 1))
                / (len(valid_answers) * (len(valid_answers) - 1))
            )
            normalized_entropy = float(
                semantic_entropy / np.log(len(valid_answers))
                if len(valid_answers) > 1
                else 0.0
            )
        else:
            pairwise = float("nan")
            normalized_entropy = 0.0 if len(valid_answers) == 1 else float("nan")
        invalid_count = metric_count - len(valid_answers)
        majority_adjusted = (
            float(cluster_counts.max() / metric_count) if len(cluster_counts) else 0.0
        )
        pairwise_adjusted = (
            float(
                np.sum(cluster_counts * (cluster_counts - 1))
                / (metric_count * (metric_count - 1))
            )
            if metric_count >= 2
            else float("nan")
        )
        conservative_counts = np.concatenate(
            [cluster_counts, np.ones(invalid_count, dtype=np.float64)]
        )
        conservative_probabilities = conservative_counts / metric_count
        conservative_entropy = float(
            -np.sum(
                conservative_probabilities * np.log(conservative_probabilities),
                dtype=np.float64,
            )
        )
        conservative_normalized_entropy = float(
            conservative_entropy / np.log(metric_count) if metric_count > 1 else 0.0
        )
        token_limit = int(config["validation"]["max_new_tokens"])
        labels.append(
            {
                "state_id": state["state_id"],
                "q_student": q_student,
                "q_teacher": q_teacher,
                "q_teacher_minus_student": q_teacher - q_student,
                "teacher_any_correct": any(
                    row["correct"] for row in teacher_label_rows
                ),
                "student_any_correct": any(
                    row["correct"] for row in student_label_rows
                ),
                "teacher_self_consistency_majority": majority,
                "teacher_self_consistency_pairwise": pairwise,
                "teacher_semantic_entropy": semantic_entropy,
                "teacher_normalized_semantic_entropy": normalized_entropy,
                "teacher_self_consistency_majority_coverage_adjusted": majority_adjusted,
                "teacher_self_consistency_pairwise_coverage_adjusted": pairwise_adjusted,
                "teacher_semantic_entropy_conservative": conservative_entropy,
                "teacher_normalized_semantic_entropy_conservative": conservative_normalized_entropy,
                "teacher_valid_answer_rate": len(valid_answers) / metric_count,
                "teacher_metric_truncated_rate": float(
                    np.mean(
                        [
                            len(row["generated_token_ids"]) >= token_limit
                            for row in teacher_metric_rows
                        ]
                    )
                ),
                "teacher_label_truncated_rate": float(
                    np.mean(
                        [
                            len(row["generated_token_ids"]) >= token_limit
                            for row in teacher_label_rows
                        ]
                    )
                ),
                "student_label_truncated_rate": float(
                    np.mean(
                        [
                            len(row["generated_token_ids"]) >= token_limit
                            for row in student_label_rows
                        ]
                    )
                ),
                "teacher_label_parseable_rate": float(
                    np.mean(
                        [
                            row.get("predicted_answer") not in (None, "", "[INVALID]")
                            for row in teacher_label_rows
                        ]
                    )
                ),
                "student_label_parseable_rate": float(
                    np.mean(
                        [
                            row.get("predicted_answer") not in (None, "", "[INVALID]")
                            for row in student_label_rows
                        ]
                    )
                ),
                "teacher_metric_continuations": metric_count,
                "task_label_continuations_per_role": required - metric_count,
            }
        )
    labels_frame = pd.DataFrame(labels)
    stability = pd.read_parquet(run_dir / "results" / "stability.parquet")
    reliability = stability[stability["state_id"].isin(labels_frame["state_id"])].merge(
        labels_frame, on="state_id"
    )
    reliability.to_parquet(run_dir / "results" / "reliability.parquet", index=False)
    reliability.drop(columns=["neighbor_state_ids"]).to_csv(
        run_dir / "results" / "reliability.csv", index=False
    )
    update_status(run_dir, "validated", validated_states=len(labels))
    return True


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    states = read_jsonl(run_dir / "artifacts" / "states.jsonl")
    selected = select_anchor_states(run_dir, config, states)
    validation = config["validation"]
    output_path = run_dir / "results" / "continuations.jsonl"
    required = int(validation["continuations_per_state"])
    metric_count = int(validation["metric_continuations_per_state"])
    existing = read_jsonl(output_path)
    completed = {
        (row["state_id"], row["role"], int(row["continuation_index"]))
        for row in existing
    }
    update_status(run_dir, f"validating_{args.role}")

    try:
        set_seed(
            int(config["experiment"]["seed"]) + (0 if args.role == "student" else 10000)
        )
        model, tokenizer, device = load_model_and_tokenizer(
            config["models"][args.role], config
        )
        grade = load_math_grader()
        for state_index, state in enumerate(selected):
            target_indices = (
                range(required)
                if args.role == "teacher"
                else range(metric_count, required)
            )
            missing = [
                index
                for index in target_indices
                if (state["state_id"], args.role, index) not in completed
            ]
            if not missing:
                continue
            prefix = torch.tensor(
                [state["input_ids"] + [int(state["action_token_id"])]],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(prefix)
            with torch.inference_mode():
                sequences = model.generate(
                    input_ids=prefix,
                    attention_mask=attention_mask,
                    do_sample=float(validation["temperature"]) > 0,
                    temperature=float(validation["temperature"]),
                    top_p=float(validation["top_p"]),
                    max_new_tokens=int(validation["max_new_tokens"]),
                    num_return_sequences=len(missing),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            rows = []
            for sequence, continuation_index in zip(sequences, missing):
                sequence_ids = sequence.detach().cpu().tolist()
                generated_ids = trim_generated_tokens(
                    sequence_ids[prefix.shape[-1] :], tokenizer.eos_token_id
                )
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                full_text = tokenizer.decode(
                    sequence_ids[: prefix.shape[-1]] + generated_ids,
                    skip_special_tokens=True,
                )
                result = grade(full_text, str(state["ground_truth"]))
                rows.append(
                    {
                        "state_id": state["state_id"],
                        "role": args.role,
                        "continuation_index": continuation_index,
                        "generated_token_ids": generated_ids,
                        "generated_text": text,
                        "correct": bool(result["acc"]),
                        "predicted_answer": result["pred"],
                        "ground_truth": state["ground_truth"],
                    }
                )
            append_jsonl(output_path, rows)
            print(
                f"{args.role}: validated state {state_index + 1}/{len(selected)}",
                flush=True,
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        complete = summarize_if_complete(run_dir, config, selected)
        if not complete:
            update_status(run_dir, f"validated_{args.role}_partial")
            print(
                "One model role remains before reliability labels can be summarized",
                flush=True,
            )
        return 0
    except Exception as exc:
        update_status(
            run_dir,
            "failed",
            stage=f"validate_{args.role}",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
