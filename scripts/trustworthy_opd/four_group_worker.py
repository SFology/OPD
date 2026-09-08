from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import (
    load_math_grader,
    load_model_and_tokenizer,
    load_run,
    pool_hidden,
    read_jsonl,
    resolve_hidden_indices,
    set_seed,
    trim_generated_tokens,
)
from four_group_common import (
    atomic_write_jsonl,
    eligible_generation,
    logical_shard,
    parse_shards,
    read_json,
    stable_int,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU four-group experiment worker"
    )
    parser.add_argument("mode", choices=("collect-base", "rollout", "extract-features"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--round", type=int)
    parser.add_argument("--role", choices=("student", "teacher"))
    return parser.parse_args()


def generate_rows(model, prefix: torch.Tensor, tokenizer, kwargs: dict, count: int):
    attention_mask = torch.ones_like(prefix)
    with torch.inference_mode():
        sequences = model.generate(
            input_ids=prefix,
            attention_mask=attention_mask,
            num_return_sequences=count,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            **kwargs,
        )
    prompt_length = prefix.shape[-1]
    return [
        trim_generated_tokens(
            row[prompt_length:].detach().cpu().tolist(), tokenizer.eos_token_id
        )
        for row in sequences
    ]


def collect_base(args: argparse.Namespace, shards: set[int]) -> None:
    if args.round is None:
        raise ValueError("collect-base requires --round")
    run_dir, config = load_run(args.run_dir)
    round_dir = run_dir / "rounds" / f"round_{args.round:03d}"
    manifest = read_jsonl(round_dir / "prompt_manifest.jsonl")
    shard_count = int(config["parallel"]["logical_shards"])
    tasks = defaultdict(list)
    for row in manifest:
        shard = logical_shard(f"prompt:{row['prompt_index']}", shard_count)
        if shard in shards:
            tasks[shard].append(row)
    if not tasks:
        return

    model, tokenizer, device = load_model_and_tokenizer(
        config["models"]["student"], config
    )
    grader = load_math_grader()
    frame = pd.read_parquet(config["data"]["parquet"])
    seed = int(config["experiment"]["seed"])
    target = int(config["data"]["rollouts_per_prompt"])
    maximum = int(config["data"]["base_max_attempts_per_prompt"])
    batch_size = int(config["data"]["base_attempt_batch_size"])
    token_limit = int(config["data"]["base_max_new_tokens"])
    generation = {
        "do_sample": True,
        "temperature": float(config["data"]["base_temperature"]),
        "top_p": float(config["data"]["base_top_p"]),
        "max_new_tokens": token_limit,
    }
    try:
        for shard, prompt_rows in sorted(tasks.items()):
            output = round_dir / "base" / f"shard_{shard:03d}.jsonl"
            attempts_output = round_dir / "base_attempts" / f"shard_{shard:03d}.jsonl"
            if output.exists() and attempts_output.exists():
                continue
            accepted_rows = []
            attempt_rows = []
            for prompt_row in prompt_rows:
                prompt_index = int(prompt_row["prompt_index"])
                item = frame.iloc[prompt_index]
                messages = item["prompt"]
                if hasattr(messages, "tolist"):
                    messages = messages.tolist()
                messages = [dict(message) for message in messages]
                prompt_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if prompt_ids.shape[-1] > int(config["data"]["max_prompt_tokens"]):
                    continue
                prompt_list = prompt_ids[0].tolist()
                ground_truth = str(item["reward_model"]["ground_truth"])
                valid_for_prompt = []
                attempt_index = 0
                while len(valid_for_prompt) < target and attempt_index < maximum:
                    count = min(batch_size, maximum - attempt_index)
                    set_seed(
                        stable_int(
                            seed, "base", args.round, prompt_index, attempt_index
                        )
                        % (2**31)
                    )
                    generated_batch = generate_rows(
                        model, prompt_ids.to(device), tokenizer, generation, count
                    )
                    for generated in generated_batch:
                        text = tokenizer.decode(generated, skip_special_tokens=True)
                        graded = grader(text, ground_truth)
                        row = {
                            "round": args.round,
                            "prompt_index": prompt_index,
                            "attempt_index": attempt_index,
                            "messages": messages,
                            "prompt_token_ids": prompt_list,
                            "generated_token_ids": generated,
                            "generated_text": text,
                            "ground_truth": ground_truth,
                            "predicted_answer": graded.get("pred"),
                            "correct": bool(graded["acc"]),
                        }
                        row["eligible"] = eligible_generation(row, token_limit)
                        attempt_rows.append(row)
                        if row["eligible"] and len(valid_for_prompt) < target:
                            valid_for_prompt.append(row)
                        attempt_index += 1
                    print(
                        f"base round={args.round} prompt={prompt_index} "
                        f"eligible={len(valid_for_prompt)}/{target} "
                        f"attempts={attempt_index}/{maximum}",
                        flush=True,
                    )
                for rollout_index, row in enumerate(valid_for_prompt):
                    accepted_rows.append(
                        {
                            **row,
                            "rollout_index": rollout_index,
                            "base_trajectory_id": (
                                f"r{args.round:03d}_p{prompt_index}_b{rollout_index}"
                            ),
                        }
                    )
            atomic_write_jsonl(output, accepted_rows)
            atomic_write_jsonl(attempts_output, attempt_rows)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rollout(args: argparse.Namespace, shards: set[int]) -> None:
    if args.round is None or args.role is None:
        raise ValueError("rollout requires --round and --role")
    run_dir, config = load_run(args.run_dir)
    round_dir = run_dir / "rounds" / f"round_{args.round:03d}"
    states = read_jsonl(round_dir / "states.jsonl")
    shard_count = int(config["parallel"]["logical_shards"])
    tasks = defaultdict(list)
    for state in states:
        shard = logical_shard(state["state_id"], shard_count)
        if shard in shards:
            tasks[shard].append(state)
    if not tasks:
        return

    model, tokenizer, device = load_model_and_tokenizer(
        config["models"][args.role], config
    )
    grader = load_math_grader()
    seed = int(config["experiment"]["seed"])
    continuation = config["continuation"]
    token_limit = int(continuation["max_new_tokens"])
    generation = {
        "do_sample": True,
        "temperature": float(continuation["temperature"]),
        "top_p": float(continuation["top_p"]),
        "max_new_tokens": token_limit,
    }
    try:
        for shard, shard_states in sorted(tasks.items()):
            output = (
                round_dir / "continuations" / args.role / f"shard_{shard:03d}.jsonl"
            )
            if output.exists():
                continue
            rows = []
            for index, state in enumerate(shard_states):
                set_seed(
                    stable_int(seed, "continuation", args.role, state["state_id"])
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
                    "round": args.round,
                    "role": args.role,
                    "generated_token_ids": generated,
                    "generated_text": text,
                    "generated_tokens": len(generated),
                    "first_token_id": int(generated[0]) if generated else None,
                    "predicted_answer": graded.get("pred"),
                    "correct": bool(graded["acc"]),
                }
                row["eligible"] = eligible_generation(row, token_limit)
                rows.append(row)
                print(
                    f"continuation role={args.role} round={args.round} "
                    f"shard={shard} {index + 1}/{len(shard_states)}",
                    flush=True,
                )
            atomic_write_jsonl(output, rows)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def prefix_ppl_for_groups(
    model,
    states: list[dict],
    grouped_indices: dict[str, list[int]],
    device: torch.device,
) -> np.ndarray:
    result = np.full(len(states), np.nan, dtype=np.float32)
    for group_index, indices in enumerate(grouped_indices.values()):
        longest = max(indices, key=lambda item: len(states[item]["input_ids"]))
        token_ids = torch.tensor(
            [states[longest]["input_ids"]], dtype=torch.long, device=device
        )
        prompt_count = int(states[longest]["prompt_token_count"])
        with torch.inference_mode():
            outputs = model(
                input_ids=token_ids,
                output_hidden_states=False,
                use_cache=False,
                logits_to_keep=0,
            )
            logits = outputs.logits[0, :-1]
            targets = token_ids[0, 1:]
            generated_start = max(0, prompt_count - 1)
            parts = []
            for start in range(generated_start, len(targets), 64):
                stop = min(start + 64, len(targets))
                parts.append(
                    F.cross_entropy(
                        logits[start:stop].float(),
                        targets[start:stop],
                        reduction="none",
                    )
                )
            nll = torch.cat(parts) if parts else None
        if nll is not None:
            cumulative = nll.cumsum(0)
            for state_index in indices:
                count = len(states[state_index]["input_ids"]) - prompt_count
                if count > 0:
                    result[state_index] = float(
                        torch.exp((cumulative[count - 1] / count).clamp(max=50))
                    )
        del outputs, logits, targets, nll
        print(
            f"prefix-ppl trajectory {group_index + 1}/{len(grouped_indices)}",
            flush=True,
        )
    return result


def extract_features(args: argparse.Namespace, shards: set[int]) -> None:
    if args.role is None:
        raise ValueError("extract-features requires --role")
    run_dir, config = load_run(args.run_dir)
    states = read_jsonl(run_dir / "artifacts" / "states.jsonl")
    action_vocab = np.asarray(read_json(run_dir / "artifacts" / "action_vocab.json"))
    shard_count = int(config["parallel"]["logical_shards"])
    tasks = defaultdict(list)
    for state in states:
        shard = logical_shard(state["base_trajectory_id"], shard_count)
        if shard in shards:
            tasks[shard].append(state)
    if not tasks:
        return

    model, tokenizer, device = load_model_and_tokenizer(
        config["models"][args.role], config
    )
    definitions = config["representations"]["definitions"]
    layer_count = int(model.config.num_hidden_layers)
    resolved = {
        item["name"]: resolve_hidden_indices(item["layers"], layer_count)
        for item in definitions
    }
    action_tensor = torch.tensor(action_vocab, dtype=torch.long, device=device)
    batch_size = int(config["models"]["feature_batch_size"])
    try:
        for shard, shard_states in sorted(tasks.items()):
            output = run_dir / "features" / args.role / f"shard_{shard:03d}.npz"
            if output.exists():
                continue
            representations: dict[str, list[np.ndarray]] = {
                item["name"]: [] for item in definitions
            }
            action_rows = []
            for start in range(0, len(shard_states), batch_size):
                batch = shard_states[start : start + batch_size]
                encoded = tokenizer.pad(
                    {"input_ids": [state["input_ids"] for state in batch]},
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                mask = encoded["attention_mask"].to(device)
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=mask,
                        output_hidden_states=True,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    log_probs = torch.log_softmax(outputs.logits[:, -1, :].float(), -1)
                    action_rows.append(
                        log_probs.index_select(-1, action_tensor).cpu().numpy()
                    )
                    for definition in definitions:
                        hidden = torch.stack(
                            [
                                outputs.hidden_states[layer]
                                for layer in resolved[definition["name"]]
                            ]
                        ).mean(0)
                        pooled = pool_hidden(hidden, mask, definition).float()
                        if config["representations"].get("normalize", True):
                            pooled = F.normalize(pooled, p=2, dim=-1)
                        representations[definition["name"]].append(
                            pooled.cpu().numpy().astype(np.float16)
                        )
                print(
                    f"features role={args.role} shard={shard} "
                    f"{min(start + batch_size, len(shard_states))}/{len(shard_states)}",
                    flush=True,
                )
            grouped = defaultdict(list)
            for index, state in enumerate(shard_states):
                grouped[state["base_trajectory_id"]].append(index)
            prefix_ppl = prefix_ppl_for_groups(model, shard_states, grouped, device)
            arrays = {
                "state_ids": np.asarray([state["state_id"] for state in shard_states]),
                "action_log_probs": np.concatenate(action_rows).astype(np.float32),
                "prefix_ppl": prefix_ppl,
            }
            for name, values in representations.items():
                arrays[f"representation__{name}"] = np.concatenate(values)
            save_npz_atomic(output, arrays)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    shards = parse_shards(args.shards)
    if args.mode == "collect-base":
        collect_base(args, shards)
    elif args.mode == "rollout":
        rollout(args, shards)
    else:
        extract_features(args, shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
