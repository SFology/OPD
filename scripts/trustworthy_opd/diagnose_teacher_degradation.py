from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import (
    append_jsonl,
    create_run_dir,
    load_config,
    load_math_grader,
    load_model_and_tokenizer,
    read_jsonl,
    set_seed,
    tokenizer_fingerprint,
    trim_generated_tokens,
    update_status,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose teacher degeneration across prompt provenance and decoding."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def load_or_create_run(args: argparse.Namespace) -> tuple[Path, dict]:
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        config = load_config(run_dir / "config.yaml")
        return run_dir, config
    config = load_config(args.config)
    run_dir = create_run_dir(config, args.config)
    return run_dir, config


def select_states(run_dir: Path, config: dict, source: Path) -> list[dict]:
    output = run_dir / "artifacts" / "diagnostic_states.jsonl"
    existing = read_jsonl(output)
    if existing:
        return existing

    selected = {
        row["state_id"]: row
        for row in read_jsonl(source / "artifacts" / "selected_states.jsonl")
    }
    reliability = pd.read_parquet(source / "results" / "reliability.parquet")
    reliability = reliability[
        (reliability.representation == "mid_tail_mean_8")
        & (reliability.neighborhood_method == "dual_knn")
    ].drop_duplicates("state_id")
    selection = config["selection"]
    rng = np.random.default_rng(int(config["experiment"]["seed"]) + 41000)

    strata = [
        (
            "clean",
            reliability.teacher_metric_truncated_rate
            <= float(selection["clean_max_metric_truncated_rate"]),
            int(selection["clean_states"]),
        ),
        (
            "degraded",
            reliability.teacher_metric_truncated_rate
            >= float(selection["degraded_min_metric_truncated_rate"]),
            int(selection["degraded_states"]),
        ),
    ]
    rows = []
    for name, mask, count in strata:
        candidates = reliability[mask].copy()
        if len(candidates) < count:
            raise RuntimeError(
                f"Only {len(candidates)} states are available for stratum {name}; "
                f"requested {count}"
            )
        chosen = candidates.iloc[rng.choice(len(candidates), count, replace=False)]
        for item in chosen.itertuples(index=False):
            state = dict(selected[item.state_id])
            state.update(
                {
                    "diagnostic_stratum": name,
                    "source_q_teacher": float(item.q_teacher),
                    "source_q_student": float(item.q_student),
                    "source_teacher_metric_truncated_rate": float(
                        item.teacher_metric_truncated_rate
                    ),
                    "source_teacher_label_truncated_rate": float(
                        item.teacher_label_truncated_rate
                    ),
                }
            )
            rows.append(state)
    rows.sort(key=lambda row: (row["diagnostic_stratum"], row["prompt_index"]))
    write_jsonl(output, rows)
    return rows


def trajectory_lookup(source: Path) -> dict[tuple[int, int], dict]:
    return {
        (int(row["prompt_index"]), int(row["rollout_index"])): row
        for row in read_jsonl(source / "artifacts" / "trajectories.jsonl")
    }


def generation_kwargs(
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    samples: int,
) -> dict:
    result = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": samples,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0:
        result.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        result["do_sample"] = False
    return result


def generate_sequences(model, prefix: torch.Tensor, kwargs: dict) -> list[list[int]]:
    attention_mask = torch.ones_like(prefix)
    with torch.inference_mode():
        sequences = model.generate(
            input_ids=prefix,
            attention_mask=attention_mask,
            **kwargs,
        )
    return [sequence.detach().cpu().tolist() for sequence in sequences]


def token_diagnostics(token_ids: list[int]) -> dict[str, float]:
    if not token_ids:
        return {
            "unique_token_rate": 0.0,
            "most_common_token_rate": 0.0,
        }
    counts = Counter(token_ids)
    return {
        "unique_token_rate": len(counts) / len(token_ids),
        "most_common_token_rate": counts.most_common(1)[0][1] / len(token_ids),
    }


def build_reference_trajectories(
    run_dir: Path,
    config: dict,
    states: list[dict],
    trajectories: dict[tuple[int, int], dict],
    model,
    tokenizer,
    device: torch.device,
    grade,
) -> dict[str, dict]:
    output = run_dir / "artifacts" / "teacher_reference_trajectories.jsonl"
    existing = {row["state_id"]: row for row in read_jsonl(output)}
    maximum = int(config["reference_generation"]["max_new_tokens"])
    kwargs = generation_kwargs(tokenizer, maximum, 0.0, 1.0, 1)
    for index, state in enumerate(states):
        if state["state_id"] in existing:
            continue
        set_seed(int(config["experiment"]["seed"]) + 43000 + index)
        trajectory = trajectories[(state["prompt_index"], state["rollout_index"])]
        prompt_ids = [int(item) for item in trajectory["prompt_token_ids"]]
        prefix = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        sequence = generate_sequences(model, prefix, kwargs)[0]
        raw_generated = sequence[len(prompt_ids) :]
        generated = trim_generated_tokens(raw_generated, tokenizer.eos_token_id)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        result = grade(text, str(state["ground_truth"]))
        row = {
            "state_id": state["state_id"],
            "diagnostic_stratum": state["diagnostic_stratum"],
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated,
            "generated_text": text,
            "hit_token_limit": len(generated) >= maximum,
            "predicted_answer": result["pred"],
            "correct": bool(result["acc"]),
            "ground_truth": state["ground_truth"],
            **token_diagnostics(generated),
        }
        append_jsonl(output, [row])
        existing[state["state_id"]] = row
        print(f"reference: generated state {index + 1}/{len(states)}", flush=True)
    return existing


def prefix_for_condition(
    condition: str,
    state: dict,
    trajectory: dict,
    reference: dict,
) -> list[int]:
    if condition == "prompt_only":
        return [int(item) for item in trajectory["prompt_token_ids"]]
    if condition == "student_prefix":
        return [*state["input_ids"], int(state["action_token_id"])]
    if condition == "teacher_self_prefix":
        generated = reference["generated_token_ids"]
        if not generated:
            return [int(item) for item in trajectory["prompt_token_ids"]]
        position = max(
            1,
            min(
                len(generated),
                round(len(generated) * float(state["normalized_position"])),
            ),
        )
        return [*reference["prompt_token_ids"], *generated[:position]]
    raise ValueError(f"Unknown prefix source: {condition}")


def run_conditions(
    run_dir: Path,
    config: dict,
    states: list[dict],
    trajectories: dict[tuple[int, int], dict],
    references: dict[str, dict],
    model,
    tokenizer,
    device: torch.device,
    grade,
) -> list[dict]:
    output = run_dir / "results" / "continuations.jsonl"
    existing_rows = read_jsonl(output)
    completed = {
        (
            row["state_id"],
            row["prefix_source"],
            row["decoding"],
            int(row["sample_index"]),
        )
        for row in existing_rows
    }
    maximum = int(config["conditions"]["max_new_tokens"])
    total_states = len(states)
    for state_index, state in enumerate(states):
        trajectory = trajectories[(state["prompt_index"], state["rollout_index"])]
        for condition_index, condition in enumerate(
            config["conditions"]["prefix_sources"]
        ):
            prefix_ids = prefix_for_condition(
                condition,
                state,
                trajectory,
                references[state["state_id"]],
            )
            prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            for decoding_index, decoding in enumerate(
                config["conditions"]["decodings"]
            ):
                sample_count = int(decoding["samples"])
                missing = [
                    sample_index
                    for sample_index in range(sample_count)
                    if (
                        state["state_id"],
                        condition,
                        decoding["name"],
                        sample_index,
                    )
                    not in completed
                ]
                if not missing:
                    continue
                set_seed(
                    int(config["experiment"]["seed"])
                    + 44000
                    + state_index * 1000
                    + condition_index * 100
                    + decoding_index * 10
                )
                kwargs = generation_kwargs(
                    tokenizer,
                    maximum,
                    float(decoding["temperature"]),
                    float(decoding["top_p"]),
                    sample_count,
                )
                sequences = generate_sequences(model, prefix, kwargs)
                rows = []
                for sample_index, sequence in enumerate(sequences):
                    if sample_index not in missing:
                        continue
                    raw_generated = sequence[len(prefix_ids) :]
                    generated = trim_generated_tokens(
                        raw_generated, tokenizer.eos_token_id
                    )
                    generated_text = tokenizer.decode(
                        generated, skip_special_tokens=True
                    )
                    if condition == "prompt_only":
                        grade_text = generated_text
                    else:
                        grade_text = tokenizer.decode(
                            [*prefix_ids, *generated], skip_special_tokens=True
                        )
                    result = grade(grade_text, str(state["ground_truth"]))
                    rows.append(
                        {
                            "state_id": state["state_id"],
                            "diagnostic_stratum": state["diagnostic_stratum"],
                            "prefix_source": condition,
                            "decoding": decoding["name"],
                            "temperature": float(decoding["temperature"]),
                            "sample_index": sample_index,
                            "prefix_tokens": len(prefix_ids),
                            "generated_token_ids": generated,
                            "generated_text": generated_text,
                            "generated_tokens": len(generated),
                            "hit_token_limit": len(generated) >= maximum,
                            "predicted_answer": result["pred"],
                            "parseable": result["pred"] not in (None, "", "[INVALID]"),
                            "correct": bool(result["acc"]),
                            "ground_truth": state["ground_truth"],
                            **token_diagnostics(generated),
                        }
                    )
                append_jsonl(output, rows)
                for row in rows:
                    completed.add(
                        (
                            row["state_id"],
                            row["prefix_source"],
                            row["decoding"],
                            int(row["sample_index"]),
                        )
                    )
        print(
            f"diagnostic: completed state {state_index + 1}/{total_states}",
            flush=True,
        )
    return read_jsonl(output)


def summarize(run_dir: Path, references: dict[str, dict], rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(
            ["diagnostic_stratum", "prefix_source", "decoding"], as_index=False
        )
        .agg(
            continuations=("state_id", "size"),
            states=("state_id", "nunique"),
            truncated_rate=("hit_token_limit", "mean"),
            parseable_rate=("parseable", "mean"),
            accuracy=("correct", "mean"),
            median_generated_tokens=("generated_tokens", "median"),
            mean_unique_token_rate=("unique_token_rate", "mean"),
            mean_most_common_token_rate=("most_common_token_rate", "mean"),
        )
        .sort_values(["diagnostic_stratum", "prefix_source", "decoding"])
    )
    summary.to_csv(run_dir / "results" / "condition_summary.csv", index=False)
    reference_frame = pd.DataFrame(references.values())
    reference_summary = (
        reference_frame.groupby("diagnostic_stratum", as_index=False)
        .agg(
            states=("state_id", "nunique"),
            truncated_rate=("hit_token_limit", "mean"),
            accuracy=("correct", "mean"),
        )
        .to_dict(orient="records")
    )
    report = {
        "condition_summary": summary.to_dict(orient="records"),
        "teacher_reference_summary": reference_summary,
    }
    (run_dir / "results" / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


def main() -> int:
    args = parse_args()
    run_dir, config = load_or_create_run(args)
    source = Path(config["source_run"]).resolve()
    print(f"RUN_DIR={run_dir}", flush=True)
    update_status(run_dir, "running_teacher_degradation_diagnostic")
    try:
        states = select_states(run_dir, config, source)
        trajectories = trajectory_lookup(source)
        set_seed(int(config["experiment"]["seed"]) + 42000)
        model, tokenizer, device = load_model_and_tokenizer(
            config["models"]["teacher"], config
        )
        source_teacher = load_config(source / "features" / "teacher.yaml")
        fingerprint = tokenizer_fingerprint(tokenizer)
        if fingerprint != source_teacher["tokenizer_fingerprint"]:
            raise RuntimeError("Teacher tokenizer does not match the source run")
        grade = load_math_grader()
        references = build_reference_trajectories(
            run_dir,
            config,
            states,
            trajectories,
            model,
            tokenizer,
            device,
            grade,
        )
        rows = run_conditions(
            run_dir,
            config,
            states,
            trajectories,
            references,
            model,
            tokenizer,
            device,
            grade,
        )
        summarize(run_dir, references, rows)
        update_status(
            run_dir,
            "analyzed",
            diagnostic_states=len(states),
            continuations=len(rows),
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0
    except Exception as error:
        update_status(run_dir, "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
