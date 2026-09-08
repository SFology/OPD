from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path

import torch

from common import (
    load_config,
    load_math_grader,
    load_model_and_tokenizer,
    load_run,
    read_jsonl,
    set_seed,
)
from four_group_common import atomic_write_jsonl, eligible_generation, parse_shards, stable_int
from four_group_worker import generate_rows
from ppl_branch_worker import prefix_curve_from_nll
from ppl_recovery_worker import curve_from_nll, score_token_nll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Equal-decoding PPL control worker")
    parser.add_argument("mode", choices=("generate-teacher", "score-branch"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--control-config", required=True, type=Path)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--role", choices=("teacher", "student"))
    parser.add_argument("--branch", choices=("teacher", "student"))
    return parser.parse_args()


def control_dir(run_dir: Path) -> Path:
    return run_dir / "ppl_decode_control"


def generate_teacher(args: argparse.Namespace, shards: set[int]) -> None:
    run_dir, base_config = load_run(args.run_dir)
    config = load_config(args.control_config)
    directory = control_dir(run_dir)
    manifest = read_jsonl(directory / "state_manifest.jsonl")
    tasks: dict[int, list[dict]] = defaultdict(list)
    for row in manifest:
        shard = int(row["generation_shard"])
        if shard in shards:
            tasks[shard].append(row)
    if not tasks:
        return

    model, tokenizer, device = load_model_and_tokenizer(
        base_config["models"]["teacher"], base_config
    )
    grader = load_math_grader()
    generation_config = config["generation"]["teacher"]
    token_limit = int(generation_config["max_new_tokens"])
    generation = {
        "do_sample": True,
        "temperature": float(generation_config["temperature"]),
        "top_p": float(generation_config["top_p"]),
        "max_new_tokens": token_limit,
    }
    seed = int(config["experiment"]["seed"])
    try:
        for shard, rows in sorted(tasks.items()):
            output = directory / "teacher_continuations" / f"shard_{shard:03d}.jsonl"
            if output.exists():
                continue
            results = []
            for index, row in enumerate(rows):
                set_seed(
                    stable_int(seed, "equal-decoding-teacher", row["state_id"])
                    % (2**31)
                )
                prefix = torch.tensor(
                    [row["input_ids"]], dtype=torch.long, device=device
                )
                generated = generate_rows(
                    model, prefix, tokenizer, generation, count=1
                )[0]
                generated_text = tokenizer.decode(
                    generated, skip_special_tokens=True
                )
                graded = grader(generated_text, str(row["ground_truth"]))
                result = {
                    "state_id": row["state_id"],
                    "prompt_index": row["prompt_index"],
                    "base_trajectory_id": row["base_trajectory_id"],
                    "fraction_name": row["fraction_name"],
                    "generated_token_ids": generated,
                    "generated_text": generated_text,
                    "generated_tokens": len(generated),
                    "predicted_answer": graded.get("pred"),
                    "correct": bool(graded["acc"]),
                }
                result["eligible"] = eligible_generation(result, token_limit)
                results.append(result)
                print(
                    f"decode-control generation shard={shard} "
                    f"{index + 1}/{len(rows)} state={row['state_id']} "
                    f"tokens={len(generated)} eligible={result['eligible']} "
                    f"correct={result['correct']}",
                    flush=True,
                )
            atomic_write_jsonl(output, results)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_branch(args: argparse.Namespace, shards: set[int]) -> None:
    if args.role is None or args.branch is None:
        raise ValueError("score-branch requires --role and --branch")
    run_dir, base_config = load_run(args.run_dir)
    config = load_config(args.control_config)
    directory = control_dir(run_dir)
    manifest = read_jsonl(directory / "score_manifest.jsonl")
    shard_key = f"{args.branch}_score_shard"
    tasks: dict[int, list[dict]] = defaultdict(list)
    for row in manifest:
        shard = int(row[shard_key])
        if shard in shards:
            tasks[shard].append(row)
    if not tasks:
        return

    model, _, device = load_model_and_tokenizer(
        base_config["models"][args.role], base_config
    )
    chunk_size = int(config["scoring"]["token_chunk_size"])
    try:
        for shard, rows in sorted(tasks.items()):
            output = (
                directory
                / "scores"
                / args.branch
                / args.role
                / f"shard_{shard:03d}.jsonl"
            )
            if output.exists():
                continue
            if args.branch == "student":
                grouped: dict[str, list[dict]] = defaultdict(list)
                for row in rows:
                    grouped[row["base_trajectory_id"]].append(row)
            else:
                grouped = {row["state_id"]: [row] for row in rows}

            results = []
            for index, (identifier, states) in enumerate(sorted(grouped.items())):
                reference = states[0]
                branch_tokens = reference[f"{args.branch}_generated_token_ids"]
                sequence = reference["input_ids"] + branch_tokens
                if args.branch == "student":
                    for row in states[1:]:
                        candidate = (
                            row["input_ids"]
                            + row["student_generated_token_ids"]
                        )
                        if candidate != sequence:
                            raise RuntimeError(
                                f"Inconsistent student trajectory: {identifier}"
                            )
                nll = score_token_nll(model, sequence, device, chunk_size)
                for row in sorted(states, key=lambda item: item["fraction_name"]):
                    tokens = row[f"{args.branch}_generated_token_ids"]
                    curve_row = dict(row)
                    curve_row["teacher_generated_token_ids"] = tokens
                    results.append(
                        {
                            "state_id": row["state_id"],
                            "role": args.role,
                            "branch": args.branch,
                            "prompt_index": row["prompt_index"],
                            "base_trajectory_id": row["base_trajectory_id"],
                            "fraction_name": row["fraction_name"],
                            "normalized_position": row["normalized_position"],
                            "group": row["group"],
                            "prefix_tokens": len(row["input_ids"]),
                            "continuation_tokens": len(tokens),
                            "prefix_curve": prefix_curve_from_nll(
                                row, nll, config
                            ),
                            "curve": curve_from_nll(curve_row, nll, config),
                        }
                    )
                print(
                    f"decode-control score role={args.role} branch={args.branch} "
                    f"shard={shard} {index + 1}/{len(grouped)} "
                    f"item={identifier} states={len(states)} tokens={len(sequence)}",
                    flush=True,
                )
            atomic_write_jsonl(output, results)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    shards = parse_shards(args.shards)
    if args.mode == "generate-teacher":
        generate_teacher(args, shards)
    else:
        score_branch(args, shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
