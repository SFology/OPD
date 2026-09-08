from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import load_config, load_model_and_tokenizer, load_run, read_jsonl
from four_group_common import atomic_write_jsonl, parse_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score token-level PPL recovery curves")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--curve-config", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("teacher", "student"))
    parser.add_argument("--shards", required=True)
    return parser.parse_args()


def cross_entropy_rows(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    parts = []
    for start in range(0, len(targets), 64):
        stop = min(start + 64, len(targets))
        parts.append(
            F.cross_entropy(
                logits[start:stop].float(),
                targets[start:stop],
                reduction="none",
            )
        )
    return torch.cat(parts) if parts else torch.empty(0, device=logits.device)


def score_token_nll(
    model,
    token_ids: list[int],
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """Return nll[i] = -log p(token_i | token_<i) using an exact KV cache."""
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
    result = np.full(len(token_ids), np.nan, dtype=np.float32)
    past_key_values = None
    previous_last_logits = None
    with torch.inference_mode():
        for start in range(0, len(token_ids), chunk_size):
            stop = min(start + chunk_size, len(token_ids))
            chunk = tokens[start:stop].unsqueeze(0)
            attention_mask = torch.ones((1, stop), dtype=torch.long, device=device)
            outputs = model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
                logits_to_keep=0,
            )
            logits = outputs.logits[0]
            if previous_last_logits is not None:
                boundary = cross_entropy_rows(
                    previous_last_logits.unsqueeze(0), tokens[start : start + 1]
                )
                result[start] = float(boundary.item())
            if stop - start > 1:
                within = cross_entropy_rows(logits[:-1], chunk[0, 1:])
                result[start + 1 : stop] = within.float().cpu().numpy()
            previous_last_logits = logits[-1].detach()
            past_key_values = outputs.past_key_values
            del outputs, logits
    del past_key_values, previous_last_logits, tokens
    return result


def mean_nll(values: np.ndarray, start: int, stop: int) -> float | None:
    part = values[max(1, start) : stop]
    part = part[np.isfinite(part)]
    return float(part.mean()) if len(part) else None


def curve_from_nll(row: dict, nll: np.ndarray, config: dict) -> list[dict]:
    prefix_length = len(row["input_ids"])
    continuation_length = len(row["teacher_generated_token_ids"])
    prompt_length = int(row["prompt_token_count"])
    progress_points = [float(item) for item in config["scoring"]["progress_points"]]
    windows = [int(item) for item in config["scoring"]["local_windows"]]
    curve = []
    previous_checkpoint = 0
    for progress in progress_points:
        checkpoint = (
            0
            if progress == 0.0
            else max(1, min(continuation_length, round(progress * continuation_length)))
        )
        sequence_stop = prefix_length + checkpoint
        item = {
            "progress": progress,
            "continuation_tokens": checkpoint,
            "continuation_fraction_actual": checkpoint / continuation_length,
            "continuation_cumulative_log_ppl": mean_nll(
                nll, prefix_length, sequence_stop
            ),
            "continuation_bin_log_ppl": mean_nll(
                nll, prefix_length + previous_checkpoint, sequence_stop
            ),
            "trajectory_cumulative_log_ppl": mean_nll(
                nll, prompt_length, sequence_stop
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
    curve_config = load_config(args.curve_config)
    shards = parse_shards(args.shards)
    curve_dir = run_dir / "ppl_recovery_curve"
    manifest = read_jsonl(curve_dir / "manifest.jsonl")
    tasks: dict[int, list[dict]] = {}
    for row in manifest:
        shard = int(row["logical_shard"])
        if shard in shards:
            tasks.setdefault(shard, []).append(row)
    if not tasks:
        return 0

    model, _, device = load_model_and_tokenizer(
        base_config["models"][args.role], base_config
    )
    chunk_size = int(curve_config["scoring"]["token_chunk_size"])
    try:
        for shard, rows in sorted(tasks.items()):
            output = curve_dir / "scores" / args.role / f"shard_{shard:03d}.jsonl"
            if output.exists():
                continue
            results = []
            for index, row in enumerate(rows):
                sequence = row["input_ids"] + row["teacher_generated_token_ids"]
                nll = score_token_nll(model, sequence, device, chunk_size)
                results.append(
                    {
                        "state_id": row["state_id"],
                        "role": args.role,
                        "prompt_index": row["prompt_index"],
                        "base_trajectory_id": row["base_trajectory_id"],
                        "fraction_name": row["fraction_name"],
                        "group": row["group"],
                        "prefix_tokens": len(row["input_ids"]),
                        "teacher_continuation_tokens": len(
                            row["teacher_generated_token_ids"]
                        ),
                        "curve": curve_from_nll(row, nll, curve_config),
                    }
                )
                print(
                    f"ppl-curve role={args.role} shard={shard} "
                    f"{index + 1}/{len(rows)} state={row['state_id']} "
                    f"tokens={len(sequence)}",
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
