from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common import load_config, load_model_and_tokenizer, load_run, read_jsonl
from four_group_common import atomic_write_jsonl, parse_shards
from ppl_recovery_worker import curve_from_nll, mean_nll, score_token_nll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the original student prefix and student continuation."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--branch-config", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("teacher", "student"))
    parser.add_argument("--shards", required=True)
    return parser.parse_args()


def prefix_curve_from_nll(row: dict, nll: np.ndarray, config: dict) -> list[dict]:
    """Measure the original student response from its start to the intervention."""
    prompt_length = int(row["prompt_token_count"])
    prefix_length = len(row["input_ids"])
    response_prefix_length = prefix_length - prompt_length
    if response_prefix_length <= 0:
        raise ValueError(f"State {row['state_id']} has no student response prefix")

    progress_points = [float(item) for item in config["scoring"]["progress_points"]]
    windows = [int(item) for item in config["scoring"]["local_windows"]]
    curve = []
    previous_checkpoint = 0
    for progress in progress_points:
        checkpoint = (
            0
            if progress == 0.0
            else max(1, min(response_prefix_length, round(progress * response_prefix_length)))
        )
        sequence_stop = prompt_length + checkpoint
        item = {
            "progress": progress,
            "student_prefix_tokens": checkpoint,
            "student_prefix_fraction_actual": checkpoint / response_prefix_length,
            "prefix_cumulative_log_ppl": mean_nll(
                nll, prompt_length, sequence_stop
            ),
            "prefix_bin_log_ppl": mean_nll(
                nll, prompt_length + previous_checkpoint, sequence_stop
            ),
        }
        for window in windows:
            item[f"local_log_ppl_w{window}"] = mean_nll(
                nll, max(prompt_length, sequence_stop - window), sequence_stop
            )
        curve.append(item)
        previous_checkpoint = checkpoint
    return curve


def main() -> int:
    args = parse_args()
    run_dir, base_config = load_run(args.run_dir)
    branch_config = load_config(args.branch_config)
    selected_shards = parse_shards(args.shards)
    branch_dir = run_dir / "ppl_branch_comparison"
    manifest = read_jsonl(branch_dir / "manifest.jsonl")
    tasks: dict[int, list[dict]] = {}
    for row in manifest:
        shard = int(row["logical_shard"])
        if shard in selected_shards:
            tasks.setdefault(shard, []).append(row)
    if not tasks:
        return 0

    model, _, device = load_model_and_tokenizer(
        base_config["models"][args.role], base_config
    )
    chunk_size = int(branch_config["scoring"]["token_chunk_size"])
    try:
        for shard, rows in sorted(tasks.items()):
            output = (
                branch_dir
                / "scores"
                / "student_branch"
                / args.role
                / f"shard_{shard:03d}.jsonl"
            )
            if output.exists():
                continue
            results = []
            by_trajectory: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_trajectory[row["base_trajectory_id"]].append(row)
            for index, (trajectory_id, states) in enumerate(
                sorted(by_trajectory.items())
            ):
                reference = states[0]
                sequence = (
                    reference["input_ids"]
                    + reference["student_generated_token_ids"]
                )
                for row in states[1:]:
                    candidate = row["input_ids"] + row["student_generated_token_ids"]
                    if candidate != sequence:
                        raise RuntimeError(
                            f"Inconsistent frozen trajectory tokens for {trajectory_id}"
                        )
                nll = score_token_nll(model, sequence, device, chunk_size)
                for row in sorted(states, key=lambda item: item["fraction_name"]):
                    suffix = row["student_generated_token_ids"]
                    branch_row = dict(row)
                    branch_row["teacher_generated_token_ids"] = suffix
                    results.append(
                        {
                            "state_id": row["state_id"],
                            "role": args.role,
                            "branch": "student",
                            "prompt_index": row["prompt_index"],
                            "base_trajectory_id": row["base_trajectory_id"],
                            "fraction_name": row["fraction_name"],
                            "normalized_position": row["normalized_position"],
                            "group": row["group"],
                            "prefix_tokens": len(row["input_ids"]),
                            "student_response_prefix_tokens": (
                                len(row["input_ids"])
                                - int(row["prompt_token_count"])
                            ),
                            "student_continuation_tokens": len(suffix),
                            "prefix_curve": prefix_curve_from_nll(
                                row, nll, branch_config
                            ),
                            "curve": curve_from_nll(
                                branch_row, nll, branch_config
                            ),
                        }
                    )
                print(
                    f"ppl-branch role={args.role} shard={shard} "
                    f"{index + 1}/{len(by_trajectory)} trajectory={trajectory_id} "
                    f"states={len(states)} tokens={len(sequence)}",
                    flush=True,
                )
            atomic_write_jsonl(output, results)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
