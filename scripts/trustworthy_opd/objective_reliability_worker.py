from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from common import (
    load_math_grader,
    load_model_and_tokenizer,
    load_run,
    read_jsonl,
    set_seed,
)
from four_group_common import (
    atomic_write_jsonl,
    eligible_generation,
    logical_shard,
    parse_shards,
)
from four_group_worker import generate_rows
from objective_reliability_common import stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated-teacher generation worker")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--shards", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    assigned = parse_shards(args.shards)
    shard_count = int(config["parallel"]["logical_shards"])
    states = [
        row
        for row in read_jsonl(run_dir / "artifacts" / "state_manifest.jsonl")
        if logical_shard(row["state_id"], shard_count) in assigned
    ]
    if not states:
        return 0

    model, tokenizer, device = load_model_and_tokenizer(
        config["models"]["teacher"], config
    )
    grader = load_math_grader()
    target = int(config["selection"]["target_valid_teacher_repeats"])
    maximum = int(config["selection"]["maximum_new_attempts_per_state"])
    token_limit = int(config["generation"]["max_new_tokens"])
    generation = {
        "do_sample": True,
        "temperature": float(config["generation"]["temperature"]),
        "top_p": float(config["generation"]["top_p"]),
        "max_new_tokens": token_limit,
    }
    try:
        for number, state in enumerate(states, start=1):
            output = run_dir / "repeats" / "by_state" / f"{state['state_id']}.jsonl"
            rows = read_jsonl(output) if output.exists() else []
            attempts_done = {int(row["attempt_index"]) for row in rows}
            valid = int(state["existing_repeat"]["eligible"]) + sum(
                bool(row["eligible"]) for row in rows
            )
            for attempt_index in range(1, maximum + 1):
                if valid >= target:
                    break
                if attempt_index in attempts_done:
                    continue
                set_seed(
                    stable_seed(
                        config["experiment"]["seed"],
                        "teacher-repeat",
                        state["state_id"],
                        attempt_index,
                    )
                    % (2**31)
                )
                prefix = torch.tensor(
                    [state["input_ids"]], dtype=torch.long, device=device
                )
                generated = generate_rows(
                    model, prefix, tokenizer, generation, count=1
                )[0]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                graded = grader(text, str(state["ground_truth"]))
                row = {
                    "state_id": state["state_id"],
                    "prompt_index": int(state["prompt_index"]),
                    "fraction_name": state["fraction_name"],
                    "attempt_index": attempt_index,
                    "generated_token_ids": generated,
                    "generated_text": text,
                    "generated_tokens": len(generated),
                    "predicted_answer": graded.get("pred"),
                    "correct": bool(graded["acc"]),
                }
                row["eligible"] = eligible_generation(row, token_limit)
                rows.append(row)
                atomic_write_jsonl(output, rows)
                valid += int(row["eligible"])
                print(
                    f"state={number}/{len(states)} id={state['state_id']} "
                    f"valid={valid}/{target} attempts={len(rows)}/{maximum}",
                    flush=True,
                )
            marker = run_dir / "repeats" / "done" / f"{state['state_id']}.done"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"valid={valid}\n", encoding="utf-8")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
