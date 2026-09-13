from __future__ import annotations

import argparse
import gc
import os
from collections import defaultdict
from pathlib import Path

import torch
from formal_eval_common import (
    expected_seed,
    generation_path,
    load_yaml,
    parse_shard,
    read_jsonl,
    validate_generation_shard,
    write_jsonl,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resumable formal-evaluation shards.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shards", required=True, help="Comma-separated DATASET:ROLLOUT pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = load_yaml(run_dir / "config.yaml")
    model_paths = load_yaml(run_dir / "artifacts" / "model_paths.yaml")
    if args.model not in model_paths:
        raise KeyError(f"Unknown resolved model {args.model!r}")
    model_path = str(model_paths[args.model]["inference_path"])
    requested = [parse_shard(item) for item in args.shards.split(",") if item]

    prompt_rows = read_jsonl(run_dir / "artifacts" / "prompt_manifest.jsonl")
    prompts_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in prompt_rows:
        prompts_by_dataset[row["dataset"]].append(row)
    for rows in prompts_by_dataset.values():
        rows.sort(key=lambda row: int(row["example_id"]))

    missing: list[tuple[str, int]] = []
    for dataset, rollout in requested:
        path = generation_path(run_dir, args.model, dataset, rollout)
        if not validate_generation_shard(
            path,
            model=args.model,
            dataset=dataset,
            rollout=rollout,
            expected_prompts=len(prompts_by_dataset[dataset]),
            expected_prompt_ids={row["prompt_id"] for row in prompts_by_dataset[dataset]},
        ):
            missing.append((dataset, rollout))
    if not missing:
        print("All assigned generation shards are already complete", flush=True)
        return

    generation = config["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype=str(generation["dtype"]),
        max_model_len=int(generation["max_model_len"]),
        gpu_memory_utilization=float(generation["gpu_memory_utilization"]),
        max_num_seqs=int(generation["max_num_seqs"]),
        enable_prefix_caching=bool(generation["enable_prefix_caching"]),
        tensor_parallel_size=1,
        seed=int(generation["seed_base"]),
    )

    dataset_indices = {
        item["name"]: index for index, item in enumerate(config["datasets"])
    }
    try:
        for dataset, rollout in missing:
            rows = prompts_by_dataset[dataset]
            formatted_prompts: list[str] = []
            prompt_token_counts: list[int] = []
            sampling_params: list[SamplingParams] = []
            request_seeds: list[int] = []
            for row in rows:
                messages = row["messages"]
                prompt_token_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                if len(prompt_token_ids) > int(generation["max_prompt_tokens"]):
                    raise ValueError(
                        f"{row['prompt_id']} has {len(prompt_token_ids)} prompt tokens; "
                        f"limit={generation['max_prompt_tokens']}"
                    )
                formatted_prompts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
                prompt_token_counts.append(len(prompt_token_ids))
                request_seed = expected_seed(
                    config,
                    dataset_indices[dataset],
                    int(row["example_id"]),
                    rollout,
                )
                request_seeds.append(request_seed)
                sampling_params.append(
                    SamplingParams(
                        temperature=float(generation["temperature"]),
                        top_p=float(generation["top_p"]),
                        repetition_penalty=float(generation["repetition_penalty"]),
                        max_tokens=int(generation["max_new_tokens"]),
                        seed=request_seed,
                    )
                )

            print(
                f"[{args.model}] generating {dataset} rollout={rollout} "
                f"prompts={len(rows)} on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
                flush=True,
            )
            outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=False)
            generated_rows = []
            for row, output, prompt_tokens, request_seed in zip(
                rows, outputs, prompt_token_counts, request_seeds, strict=True
            ):
                candidate = output.outputs[0]
                generated_rows.append(
                    {
                        "model": args.model,
                        "dataset": dataset,
                        "prompt_id": row["prompt_id"],
                        "example_id": int(row["example_id"]),
                        "rollout": rollout,
                        "seed": request_seed,
                        "ground_truth": row["ground_truth"],
                        "response": candidate.text,
                        "prompt_tokens": prompt_tokens,
                        "response_tokens": len(candidate.token_ids),
                        "finish_reason": candidate.finish_reason,
                        "stop_reason": candidate.stop_reason,
                        "at_token_limit": candidate.finish_reason == "length",
                    }
                )
            write_jsonl(
                generation_path(run_dir, args.model, dataset, rollout), generated_rows
            )
            print(
                f"[{args.model}] completed {dataset} rollout={rollout}", flush=True
            )
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
