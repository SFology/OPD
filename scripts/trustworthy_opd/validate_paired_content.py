from __future__ import annotations

import argparse
import gc
import hashlib
from collections import Counter
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

PHASES_BY_ROLE = {
    "teacher": (
        "teacher_metric_student",
        "teacher_label_student",
        "teacher_label_self",
    ),
    "student": ("student_label_student",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect valid-only paired teacher/student content labels."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("teacher", "student"))
    return parser.parse_args()


def select_anchors(run_dir: Path, config: dict) -> list[dict]:
    output = run_dir / "artifacts" / "paired_selected_states.jsonl"
    existing = read_jsonl(output)
    if existing:
        return existing
    states = read_jsonl(run_dir / "artifacts" / "states.jsonl")
    stability = pd.read_parquet(run_dir / "results" / "stability.parquet")
    valid_ids = set(
        stability[
            (stability.representation == "mid_prefix_mean")
            & (stability.neighborhood_method == "dual_knn")
            & stability.valid_neighborhood
        ].state_id
    )
    candidates = [state for state in states if state["state_id"] in valid_ids]
    by_prompt: dict[int, list[dict]] = {}
    for state in candidates:
        by_prompt.setdefault(int(state["prompt_index"]), []).append(state)
    fractions = [float(item) for item in config["paired_content"]["anchor_fractions"]]
    rng = np.random.default_rng(int(config["experiment"]["seed"]) + 51000)
    rows = []
    for prompt_order, prompt_index in enumerate(sorted(by_prompt)):
        target = fractions[prompt_order % len(fractions)]
        group = by_prompt[prompt_index]
        distance = np.asarray(
            [abs(float(state["normalized_position"]) - target) for state in group]
        )
        nearest = np.flatnonzero(np.isclose(distance, distance.min(), atol=1e-8))
        state = dict(group[int(rng.choice(nearest))])
        state["paired_stage"] = {0.25: "early", 0.5: "middle", 0.75: "late"}.get(
            target, f"fraction_{target:g}"
        )
        state["paired_target_fraction"] = target
        rows.append(state)
    expected = int(config["data"]["num_prompts"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} paired anchors, found {len(rows)}")
    write_jsonl(output, rows)
    return rows


def trajectory_lookup(run_dir: Path) -> dict[tuple[int, int], dict]:
    return {
        (int(row["prompt_index"]), int(row["rollout_index"])): row
        for row in read_jsonl(run_dir / "artifacts" / "trajectories.jsonl")
    }


def generation_kwargs(tokenizer, paired: dict, samples: int) -> dict:
    return {
        "do_sample": True,
        "temperature": float(paired["temperature"]),
        "top_p": float(paired["top_p"]),
        "max_new_tokens": int(paired["max_new_tokens"]),
        "num_return_sequences": samples,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }


def generate(model, prefix: torch.Tensor, tokenizer, paired: dict, samples: int):
    mask = torch.ones_like(prefix)
    with torch.inference_mode():
        result = model.generate(
            input_ids=prefix,
            attention_mask=mask,
            **generation_kwargs(tokenizer, paired, samples),
        )
    return [row.detach().cpu().tolist() for row in result]


def stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return base + int.from_bytes(digest[:4], "big") % 1_000_000_000


def build_teacher_references(
    run_dir: Path,
    config: dict,
    anchors: list[dict],
    trajectories: dict[tuple[int, int], dict],
    model,
    tokenizer,
    device: torch.device,
    grade,
) -> dict[str, dict]:
    path = run_dir / "artifacts" / "paired_teacher_references.jsonl"
    all_rows = read_jsonl(path)
    existing = {}
    for row in sorted(all_rows, key=lambda item: int(item["attempt_index"])):
        if row.get("eligible"):
            # Keep the first valid reference so a resumed run uses exactly the
            # same teacher prefix as the uninterrupted run.
            existing.setdefault(row["state_id"], row)
    attempt_count = Counter(row["state_id"] for row in all_rows)
    paired = config["paired_content"]
    maximum_attempts = int(paired["max_generation_attempts_per_condition"])
    batch_size = int(paired["attempt_batch_size"])
    base_seed = int(config["experiment"]["seed"]) + 52000
    for state_index, state in enumerate(anchors):
        state_id = state["state_id"]
        trajectory = trajectories[(state["prompt_index"], state["rollout_index"])]
        prompt_ids = [int(item) for item in trajectory["prompt_token_ids"]]
        while state_id not in existing and attempt_count[state_id] < maximum_attempts:
            remaining = maximum_attempts - attempt_count[state_id]
            count = min(batch_size, remaining)
            start = attempt_count[state_id]
            set_seed(stable_seed(base_seed, state_id, start))
            prefix = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            sequences = generate(model, prefix, tokenizer, paired, count)
            rows = []
            for offset, sequence in enumerate(sequences):
                raw = sequence[len(prompt_ids) :]
                generated = trim_generated_tokens(raw, tokenizer.eos_token_id)
                text = tokenizer.decode(generated, skip_special_tokens=True)
                result = grade(text, str(state["ground_truth"]))
                eligible = len(generated) < int(paired["max_new_tokens"]) and result[
                    "pred"
                ] not in (None, "", "[INVALID]")
                row = {
                    "state_id": state_id,
                    "prompt_index": state["prompt_index"],
                    "paired_stage": state["paired_stage"],
                    "attempt_index": start + offset,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": generated,
                    "generated_text": text,
                    "generated_tokens": len(generated),
                    "eligible": bool(eligible),
                    "predicted_answer": result["pred"],
                    "correct": bool(result["acc"]),
                    "ground_truth": state["ground_truth"],
                }
                rows.append(row)
                if eligible and state_id not in existing:
                    existing[state_id] = row
            append_jsonl(path, rows)
            attempt_count[state_id] += len(rows)
        print(
            f"teacher reference: {state_index + 1}/{len(anchors)} "
            f"eligible={state_id in existing}",
            flush=True,
        )
    return existing


def phase_prefix(
    phase: str,
    state: dict,
    reference: dict | None,
) -> list[int]:
    if phase in {
        "teacher_metric_student",
        "teacher_label_student",
        "student_label_student",
    }:
        return [*state["input_ids"], int(state["action_token_id"])]
    if phase == "teacher_label_self":
        if reference is None:
            raise RuntimeError(f"No valid teacher reference for {state['state_id']}")
        generated = reference["generated_token_ids"]
        position = max(
            1,
            min(
                len(generated),
                round(len(generated) * float(state["paired_target_fraction"])),
            ),
        )
        return [*reference["prompt_token_ids"], *generated[:position]]
    raise ValueError(phase)


def required_samples(phase: str, paired: dict) -> int:
    if phase == "teacher_metric_student":
        return int(paired["teacher_metric_valid_samples"])
    return int(paired["task_label_valid_samples_per_condition"])


def collect_phase(
    run_dir: Path,
    config: dict,
    role: str,
    phase: str,
    state: dict,
    reference: dict | None,
    model,
    tokenizer,
    device: torch.device,
    grade,
) -> None:
    path = run_dir / "results" / "paired_content_attempts.jsonl"
    rows = read_jsonl(path)
    relevant = [
        row
        for row in rows
        if row["state_id"] == state["state_id"] and row["phase"] == phase
    ]
    eligible = [row for row in relevant if row["eligible"]]
    paired = config["paired_content"]
    target = required_samples(phase, paired)
    maximum_attempts = int(paired["max_generation_attempts_per_condition"])
    batch_size = int(paired["attempt_batch_size"])
    prefix_ids = phase_prefix(phase, state, reference)
    base_seed = int(config["experiment"]["seed"]) + (
        53000 if role == "teacher" else 54000
    )
    while len(eligible) < target and len(relevant) < maximum_attempts:
        count = min(batch_size, maximum_attempts - len(relevant))
        start = len(relevant)
        set_seed(stable_seed(base_seed, state["state_id"], phase, start))
        prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        sequences = generate(model, prefix, tokenizer, paired, count)
        new_rows = []
        for offset, sequence in enumerate(sequences):
            raw = sequence[len(prefix_ids) :]
            generated = trim_generated_tokens(raw, tokenizer.eos_token_id)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            full_text = tokenizer.decode(
                [*prefix_ids, *generated], skip_special_tokens=True
            )
            result = grade(full_text, str(state["ground_truth"]))
            is_eligible = len(generated) < int(paired["max_new_tokens"]) and result[
                "pred"
            ] not in (None, "", "[INVALID]")
            row = {
                "state_id": state["state_id"],
                "prompt_index": state["prompt_index"],
                "paired_stage": state["paired_stage"],
                "role": role,
                "phase": phase,
                "attempt_index": start + offset,
                "prefix_tokens": len(prefix_ids),
                "generated_token_ids": generated,
                "generated_text": text,
                "generated_tokens": len(generated),
                "eligible": bool(is_eligible),
                "predicted_answer": result["pred"],
                "correct": bool(result["acc"]),
                "ground_truth": state["ground_truth"],
            }
            new_rows.append(row)
        append_jsonl(path, new_rows)
        relevant.extend(new_rows)
        eligible.extend(row for row in new_rows if row["eligible"])


def write_role_progress(run_dir: Path, config: dict, anchors: list[dict]) -> None:
    attempts = read_jsonl(run_dir / "results" / "paired_content_attempts.jsonl")
    paired = config["paired_content"]
    records = []
    for state in anchors:
        for role, phases in PHASES_BY_ROLE.items():
            for phase in phases:
                count = sum(
                    row["eligible"]
                    for row in attempts
                    if row["state_id"] == state["state_id"] and row["phase"] == phase
                )
                records.append(
                    {
                        "state_id": state["state_id"],
                        "role": role,
                        "phase": phase,
                        "required_valid": required_samples(phase, paired),
                        "collected_valid": count,
                        "complete": count >= required_samples(phase, paired),
                    }
                )
    pd.DataFrame(records).to_csv(
        run_dir / "results" / "paired_content_progress.csv", index=False
    )


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    anchors = select_anchors(run_dir, config)
    trajectories = trajectory_lookup(run_dir)
    update_status(run_dir, f"validating_paired_content_{args.role}")
    model = None
    try:
        model, tokenizer, device = load_model_and_tokenizer(
            config["models"][args.role], config
        )
        grade = load_math_grader()
        references: dict[str, dict] = {}
        if args.role == "teacher":
            references = build_teacher_references(
                run_dir,
                config,
                anchors,
                trajectories,
                model,
                tokenizer,
                device,
                grade,
            )
        else:
            references = {}
            reference_rows = read_jsonl(
                run_dir / "artifacts" / "paired_teacher_references.jsonl"
            )
            for row in sorted(
                reference_rows, key=lambda item: int(item["attempt_index"])
            ):
                if row["eligible"]:
                    references.setdefault(row["state_id"], row)
        for index, state in enumerate(anchors):
            reference = references.get(state["state_id"])
            if args.role == "teacher" and reference is None:
                print(
                    f"paired content: skipping {state['state_id']} without valid reference",
                    flush=True,
                )
                continue
            for phase in PHASES_BY_ROLE[args.role]:
                collect_phase(
                    run_dir,
                    config,
                    args.role,
                    phase,
                    state,
                    reference,
                    model,
                    tokenizer,
                    device,
                    grade,
                )
            print(
                f"paired content {args.role}: state {index + 1}/{len(anchors)}",
                flush=True,
            )
        write_role_progress(run_dir, config, anchors)
        update_status(run_dir, f"validated_paired_content_{args.role}")
        return 0
    except Exception as error:
        update_status(
            run_dir,
            "failed",
            stage=f"paired_content_{args.role}",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
