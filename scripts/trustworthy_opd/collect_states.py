from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from common import (
    append_jsonl,
    choose_state_positions,
    create_run_dir,
    load_config,
    load_math_grader,
    load_model_and_tokenizer,
    read_jsonl,
    save_yaml,
    set_seed,
    tokenizer_fingerprint,
    update_status,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect frozen student rollouts and intermediate states without training."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Resume collection in an existing pilot run directory.",
    )
    return parser.parse_args()


def build_states(trajectories: list[dict], config: dict) -> list[dict]:
    rows: list[dict] = []
    data_config = config["data"]
    for trajectory in trajectories:
        generated = trajectory["generated_token_ids"]
        positions = choose_state_positions(
            len(generated),
            data_config["state_fractions"],
            data_config["min_generated_tokens"],
        )
        for position in positions:
            rows.append(
                {
                    "state_id": f"p{trajectory['prompt_index']}_r{trajectory['rollout_index']}_t{position}",
                    "prompt_index": trajectory["prompt_index"],
                    "rollout_index": trajectory["rollout_index"],
                    "position": position,
                    "normalized_position": position / max(1, len(generated)),
                    "input_ids": trajectory["prompt_token_ids"] + generated[:position],
                    "prompt_token_count": len(trajectory["prompt_token_ids"]),
                    "action_token_id": generated[position],
                    "ground_truth": trajectory["ground_truth"],
                    "trajectory_correct": trajectory["trajectory_correct"],
                    "trajectory_id": trajectory["trajectory_id"],
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_dir = (
        args.run_dir.resolve() if args.run_dir else create_run_dir(config, args.config)
    )
    if args.run_dir:
        config = load_config(run_dir / "config.yaml")
    print(f"RUN_DIR={run_dir}", flush=True)

    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    update_status(run_dir, "collecting")
    trajectories_path = run_dir / "artifacts" / "trajectories.jsonl"

    try:
        model, tokenizer, device = load_model_and_tokenizer(
            config["models"]["student"], config
        )
        fingerprint = tokenizer_fingerprint(tokenizer)
        save_yaml(
            run_dir / "artifacts" / "tokenizer.yaml",
            {
                "student_path": config["models"]["student"],
                "fingerprint": fingerprint,
                "vocab_size": len(tokenizer),
            },
        )
        frame = pd.read_parquet(config["data"]["parquet"])
        completed = {
            (int(row["prompt_index"]), int(row["rollout_index"]))
            for row in read_jsonl(trajectories_path)
        }
        grade = load_math_grader()
        target_prompts = int(config["data"]["num_prompts"])
        offset = int(config["data"].get("prompt_offset", 0))
        rollouts = int(config["data"]["rollouts_per_prompt"])
        accepted_prompts = 0

        for prompt_index in range(offset, len(frame)):
            if accepted_prompts >= target_prompts:
                break
            item = frame.iloc[prompt_index]
            messages = item["prompt"]
            if hasattr(messages, "tolist"):
                messages = messages.tolist()
            messages = [dict(message) for message in messages]
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            if prompt_ids.shape[-1] > int(config["data"]["max_prompt_tokens"]):
                continue
            accepted_prompts += 1
            missing = [
                index
                for index in range(rollouts)
                if (prompt_index, index) not in completed
            ]
            if not missing:
                continue

            prompt_ids = prompt_ids.to(device)
            with torch.inference_mode():
                sequences = model.generate(
                    input_ids=prompt_ids,
                    do_sample=True,
                    temperature=float(config["data"]["temperature"]),
                    top_p=float(config["data"]["top_p"]),
                    max_new_tokens=int(config["data"]["max_new_tokens"]),
                    num_return_sequences=len(missing),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            prompt_list = prompt_ids[0].detach().cpu().tolist()
            new_rows = []
            for sequence, rollout_index in zip(sequences, missing):
                generated_ids = sequence[prompt_ids.shape[-1] :].detach().cpu().tolist()
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                ground_truth = str(item["reward_model"]["ground_truth"])
                graded = grade(text, ground_truth)
                new_rows.append(
                    {
                        "trajectory_id": f"p{prompt_index}_r{rollout_index}",
                        "prompt_index": prompt_index,
                        "rollout_index": rollout_index,
                        "messages": messages,
                        "prompt_token_ids": prompt_list,
                        "generated_token_ids": generated_ids,
                        "generated_text": text,
                        "ground_truth": ground_truth,
                        "trajectory_correct": bool(graded["acc"]),
                    }
                )
            append_jsonl(trajectories_path, new_rows)
            print(
                f"Collected prompt {accepted_prompts}/{target_prompts} (dataset row {prompt_index}, {len(new_rows)} rollouts)",
                flush=True,
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trajectories = read_jsonl(trajectories_path)
        states = build_states(trajectories, config)
        write_jsonl(run_dir / "artifacts" / "states.jsonl", states)
        update_status(
            run_dir,
            "collected",
            trajectories=len(trajectories),
            states=len(states),
            tokenizer_fingerprint=fingerprint,
        )
        print(
            f"Collected {len(trajectories)} trajectories and {len(states)} states",
            flush=True,
        )
        return 0
    except Exception as exc:
        update_status(
            run_dir, "failed", stage="collect", error=f"{type(exc).__name__}: {exc}"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
