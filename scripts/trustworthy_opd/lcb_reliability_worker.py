from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "verl"))

from common import (
    load_model_and_tokenizer,
    load_run,
    read_jsonl,
    resolve_hidden_indices,
)
from four_group_common import (
    atomic_write_jsonl,
    logical_shard,
    parse_shards,
)
from lcb_reliability_common import point_id
from verl.trainer.ppo.robust_opd import load_projected_embedding_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LCB reliability calibration GPU worker"
    )
    parser.add_argument("mode", choices=("features", "scores"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("teacher", "student"))
    parser.add_argument("--shards", required=True)
    return parser.parse_args()


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def assigned_trajectories(
    run_dir: Path, shards: set[int], shard_count: int
) -> list[dict]:
    return [
        row
        for row in read_jsonl(run_dir / "artifacts" / "trajectories.jsonl")
        if logical_shard(f"prompt:{row['prompt_index']}", shard_count) in shards
    ]


def projected_features(
    input_ids: torch.Tensor,
    table: torch.Tensor,
    definition: dict,
    ends: list[int],
) -> torch.Tensor:
    tail = int(definition.get("tail_tokens", 16))
    vectors = table.index_select(0, input_ids).float()
    cumulative = torch.cat(
        [torch.zeros(1, vectors.shape[1]), vectors.cumsum(dim=0)], dim=0
    )
    pooled = []
    for end in ends:
        if definition["pooling"] == "tail_mean":
            start = max(0, end - tail)
        elif definition["pooling"] == "prefix_mean":
            start = 0
        else:
            raise ValueError(f"Unsupported projected pooling: {definition['pooling']}")
        pooled.append((cumulative[end] - cumulative[start]) / max(end - start, 1))
    return F.normalize(torch.stack(pooled), p=2, dim=-1)


def contextual_features(
    hidden_states: tuple[torch.Tensor, ...],
    definition: dict,
    resolved_layers: list[int],
    ends: list[int],
) -> torch.Tensor:
    hidden = torch.stack([hidden_states[layer][0] for layer in resolved_layers]).mean(0)
    pooling = definition["pooling"]
    pooled = []
    if pooling == "prefix_mean":
        cumulative = torch.cat(
            [
                torch.zeros(
                    1, hidden.shape[1], dtype=torch.float32, device=hidden.device
                ),
                hidden.float().cumsum(dim=0),
            ],
            dim=0,
        )
        pooled = [cumulative[end] / max(end, 1) for end in ends]
    elif pooling == "tail_mean":
        tail = int(definition.get("tail_tokens", 8))
        pooled = [hidden[max(0, end - tail) : end].float().mean(dim=0) for end in ends]
    elif pooling == "last_token":
        pooled = [hidden[end - 1].float() for end in ends]
    else:
        raise ValueError(f"Unsupported contextual pooling: {pooling}")
    return F.normalize(torch.stack(pooled), p=2, dim=-1)


def extract_features(args: argparse.Namespace, shards: set[int]) -> None:
    run_dir, config = load_run(args.run_dir)
    shard_count = int(config["parallel"]["logical_shards"])
    trajectories = assigned_trajectories(run_dir, shards, shard_count)
    points = read_jsonl(run_dir / "artifacts" / "points.jsonl")
    points_by_trajectory: dict[str, list[int]] = defaultdict(list)
    for row in points:
        shard = logical_shard(f"prompt:{row['prompt_index']}", shard_count)
        if shard in shards:
            points_by_trajectory[row["base_trajectory_id"]].append(int(row["position"]))
    if not trajectories:
        return

    model, _, device = load_model_and_tokenizer(config["models"][args.role], config)
    definitions = config["representations"]["definitions"]
    contextual = [item for item in definitions if item["kind"] == "contextual"]
    projected = [
        item for item in definitions if item["kind"] == "projected_token_embedding"
    ]
    layer_count = int(model.config.num_hidden_layers)
    resolved = {
        item["name"]: resolve_hidden_indices(item["layers"], layer_count)
        for item in contextual
    }
    projection = config["representations"]
    table = load_projected_embedding_table(
        config["models"][args.role],
        projection["embedding_cache_dir"],
        int(projection["projection_dim"]),
        int(projection["projection_seed"]),
        projection["projection_method"],
    )
    by_shard: dict[int, list[dict]] = defaultdict(list)
    for trajectory in trajectories:
        by_shard[
            logical_shard(f"prompt:{trajectory['prompt_index']}", shard_count)
        ].append(trajectory)

    try:
        for shard, shard_trajectories in sorted(by_shard.items()):
            output = run_dir / "features" / args.role / f"shard_{shard:03d}.npz"
            if output.exists():
                continue
            feature_rows: dict[str, list[np.ndarray]] = {
                item["name"]: [] for item in definitions
            }
            point_ids: list[str] = []
            for trajectory_index, trajectory in enumerate(shard_trajectories):
                base_id = trajectory["base_trajectory_id"]
                positions = sorted(set(points_by_trajectory[base_id]))
                if not positions:
                    continue
                maximum = max(positions)
                prompt_count = len(trajectory["prompt_token_ids"])
                context = (
                    trajectory["prompt_token_ids"]
                    + trajectory["generated_token_ids"][:maximum]
                )
                input_ids_cpu = torch.tensor(context, dtype=torch.long)
                input_ids = input_ids_cpu.to(device)
                ends = [prompt_count + position for position in positions]
                outputs = None
                if contextual:
                    with torch.inference_mode():
                        outputs = model.model(
                            input_ids=input_ids.unsqueeze(0),
                            use_cache=False,
                            output_hidden_states=True,
                        )
                for definition in projected:
                    feature_rows[definition["name"]].append(
                        projected_features(input_ids_cpu, table, definition, ends)
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                if contextual:
                    assert outputs is not None
                    for definition in contextual:
                        feature_rows[definition["name"]].append(
                            contextual_features(
                                outputs.hidden_states,
                                definition,
                                resolved[definition["name"]],
                                ends,
                            )
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                point_ids.extend(point_id(base_id, position) for position in positions)
                if outputs is not None:
                    del outputs
                print(
                    f"features role={args.role} shard={shard} "
                    f"trajectory={trajectory_index + 1}/{len(shard_trajectories)} "
                    f"points={len(positions)}",
                    flush=True,
                )
            arrays: dict[str, np.ndarray] = {"point_ids": np.asarray(point_ids)}
            for name, values in feature_rows.items():
                arrays[f"representation__{name}"] = np.concatenate(values, axis=0)
            save_npz_atomic(output, arrays)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_requests(args: argparse.Namespace, shards: set[int]) -> None:
    run_dir, config = load_run(args.run_dir)
    shard_count = int(config["parallel"]["logical_shards"])
    trajectories = assigned_trajectories(run_dir, shards, shard_count)
    requests = read_jsonl(run_dir / "artifacts" / "requests.jsonl")
    requests_by_trajectory: dict[str, list[dict]] = defaultdict(list)
    for row in requests:
        shard = logical_shard(f"prompt:{row['prompt_index']}", shard_count)
        if shard in shards:
            requests_by_trajectory[row["base_trajectory_id"]].append(row)
    if not trajectories:
        return

    model, _, device = load_model_and_tokenizer(config["models"][args.role], config)
    by_shard: dict[int, list[dict]] = defaultdict(list)
    for trajectory in trajectories:
        by_shard[
            logical_shard(f"prompt:{trajectory['prompt_index']}", shard_count)
        ].append(trajectory)
    try:
        for shard, shard_trajectories in sorted(by_shard.items()):
            output = run_dir / "scores" / args.role / f"shard_{shard:03d}.jsonl"
            if output.exists():
                continue
            rows: list[dict] = []
            for trajectory_index, trajectory in enumerate(shard_trajectories):
                base_id = trajectory["base_trajectory_id"]
                trajectory_requests = requests_by_trajectory.get(base_id, [])
                if not trajectory_requests:
                    continue
                prompt_count = len(trajectory["prompt_token_ids"])
                positions = sorted(
                    {int(item["position"]) for item in trajectory_requests}
                )
                maximum = max(positions)
                context = (
                    trajectory["prompt_token_ids"]
                    + trajectory["generated_token_ids"][:maximum]
                )
                input_ids = torch.tensor([context], dtype=torch.long, device=device)
                logit_indices = torch.tensor(
                    [prompt_count + position - 1 for position in positions],
                    dtype=torch.long,
                    device=device,
                )
                with torch.inference_mode():
                    logits = model(
                        input_ids=input_ids,
                        use_cache=False,
                        output_hidden_states=False,
                        logits_to_keep=logit_indices,
                    ).logits[0]
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                position_row = {
                    position: index for index, position in enumerate(positions)
                }
                for item in trajectory_requests:
                    value = log_probs[
                        position_row[int(item["position"])],
                        int(item["action_token_id"]),
                    ]
                    rows.append(
                        {
                            "request_id": item["request_id"],
                            "role": args.role,
                            "log_prob": float(value),
                        }
                    )
                del logits, log_probs
                print(
                    f"scores role={args.role} shard={shard} "
                    f"trajectory={trajectory_index + 1}/{len(shard_trajectories)} "
                    f"requests={len(trajectory_requests)}",
                    flush=True,
                )
            atomic_write_jsonl(output, rows)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    shards = parse_shards(args.shards)
    if args.mode == "features":
        extract_features(args, shards)
    else:
        score_requests(args, shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
