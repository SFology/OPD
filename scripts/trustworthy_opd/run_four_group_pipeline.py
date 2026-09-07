from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from common import (
    choose_state_positions,
    create_run_dir,
    load_config,
    load_run,
    read_jsonl,
    update_status,
    write_jsonl,
)
from four_group_common import (
    GROUP_NAMES,
    fraction_name,
    load_sharded,
    logical_shard,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the adaptive four-group PPL/stability experiment."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Reuse frozen rounds and run feature extraction/analysis without collection.",
    )
    parser.add_argument(
        "--completed-rounds",
        type=int,
        help="Number of already-complete rounds to include with --analysis-only.",
    )
    args = parser.parse_args()
    if args.run_dir is None and args.config is None:
        parser.error("--config is required when --run-dir is not supplied")
    if args.analysis_only and args.run_dir is None:
        parser.error("--analysis-only requires --run-dir")
    if args.analysis_only and not args.completed_rounds:
        parser.error("--analysis-only requires a positive --completed-rounds")
    return args


def query_idle_gpus(config: dict) -> list[int]:
    parallel = config["parallel"]
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = []
    for line in result.stdout.splitlines():
        index, memory, utilization = [int(item.strip()) for item in line.split(",")]
        if memory >= int(parallel["min_free_mb"]) and utilization <= int(
            parallel["max_utilization"]
        ):
            candidates.append((index, memory, utilization))
    candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
    maximum = parallel.get("max_gpus")
    if maximum is not None:
        candidates = candidates[: int(maximum)]
    return [item[0] for item in candidates]


def wait_for_idle_gpus(config: dict) -> list[int]:
    parallel = config["parallel"]
    deadline = time.monotonic() + int(parallel["gpu_wait_seconds"])
    while True:
        selected = query_idle_gpus(config)
        if selected:
            time.sleep(int(parallel["gpu_stability_seconds"]))
            confirmed = set(query_idle_gpus(config))
            stable = [item for item in selected if item in confirmed]
            if stable:
                print(f"Selected idle physical GPUs: {stable}", flush=True)
                return stable
        if time.monotonic() >= deadline:
            raise RuntimeError("No GPU satisfied the configured idle thresholds")
        print("Waiting for at least one idle GPU", flush=True)
        time.sleep(int(parallel["gpu_poll_seconds"]))


def assign_shards(shards: list[int], gpus: list[int]) -> list[tuple[int, list[int]]]:
    return [
        (gpu, shards[index :: len(gpus)])
        for index, gpu in enumerate(gpus)
        if shards[index :: len(gpus)]
    ]


def run_workers(
    run_dir: Path,
    config: dict,
    stage: str,
    mode: str,
    shards: list[int],
    output_for_shard,
    *,
    round_index: int | None = None,
    role: str | None = None,
) -> None:
    worker = Path(__file__).with_name("four_group_worker.py")
    for retry in range(3):
        missing = sorted(
            shard for shard in shards if not output_for_shard(shard).exists()
        )
        if not missing:
            return
        gpus = wait_for_idle_gpus(config)
        assignments = assign_shards(missing, gpus)
        processes = []
        for worker_index, (gpu, assigned) in enumerate(assignments):
            command = [
                sys.executable,
                "-u",
                str(worker),
                mode,
                "--run-dir",
                str(run_dir),
                "--shards",
                ",".join(map(str, assigned)),
            ]
            if round_index is not None:
                command.extend(["--round", str(round_index)])
            if role is not None:
                command.extend(["--role", role])
            log_path = (
                run_dir
                / "logs"
                / (f"{stage}_retry{retry}_worker{worker_index}_gpu{gpu}.log")
            )
            log_handle = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[{stage}] GPU {gpu}: logical shards {assigned}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, log_handle, log_path))
        failed = []
        for process, handle, log_path in processes:
            return_code = process.wait()
            handle.close()
            if return_code:
                failed.append((return_code, log_path))
        remaining = [shard for shard in missing if not output_for_shard(shard).exists()]
        if not remaining:
            return
        print(
            f"[{stage}] retry {retry + 1}/3; remaining shards={remaining}; "
            f"failed_workers={failed}",
            flush=True,
        )
    raise RuntimeError(f"Stage {stage} still has incomplete shards after retries")


def prepare_round(run_dir: Path, config: dict, round_index: int) -> Path:
    round_dir = run_dir / "rounds" / f"round_{round_index:03d}"
    for name in (
        "base",
        "base_attempts",
        "continuations/student",
        "continuations/teacher",
    ):
        (round_dir / name).mkdir(parents=True, exist_ok=True)
    manifest_path = round_dir / "prompt_manifest.jsonl"
    if not manifest_path.exists():
        count = int(config["data"]["prompts_per_round"])
        offset = int(config["data"]["prompt_offset"]) + round_index * count
        frame_rows = pq.ParquetFile(config["data"]["parquet"]).metadata.num_rows
        if offset + count > frame_rows:
            raise RuntimeError("The configured prompt range exceeds the dataset")
        write_jsonl(
            manifest_path,
            [
                {"round": round_index, "prompt_index": index}
                for index in range(offset, offset + count)
            ],
        )
    return round_dir


def required_shards(rows: list[dict], identifier, count: int) -> list[int]:
    return sorted({logical_shard(identifier(row), count) for row in rows})


def build_round_states(round_dir: Path, config: dict, round_index: int) -> list[dict]:
    output = round_dir / "states.jsonl"
    if output.exists():
        return read_jsonl(output)
    base = load_sharded(round_dir / "base")
    by_prompt = defaultdict(list)
    for row in base:
        by_prompt[int(row["prompt_index"])].append(row)
    required = int(config["data"]["rollouts_per_prompt"])
    fractions = [float(item) for item in config["data"]["state_fractions"]]
    minimum = int(config["data"]["min_generated_tokens"])
    states = []
    dropped = []
    for prompt_index, trajectories in sorted(by_prompt.items()):
        trajectories.sort(key=lambda row: int(row["rollout_index"]))
        if len(trajectories) != required:
            dropped.append(prompt_index)
            continue
        for trajectory in trajectories:
            generated = trajectory["generated_token_ids"]
            positions = choose_state_positions(len(generated), fractions, minimum)
            if len(positions) != len(fractions):
                continue
            for fraction, position in zip(fractions, positions):
                name = fraction_name(fraction)
                states.append(
                    {
                        "state_id": f"{trajectory['base_trajectory_id']}_{name}",
                        "round": round_index,
                        "prompt_index": prompt_index,
                        "rollout_index": trajectory["rollout_index"],
                        "base_trajectory_id": trajectory["base_trajectory_id"],
                        "fraction": fraction,
                        "fraction_name": name,
                        "position": position,
                        "normalized_position": position / len(generated),
                        "input_ids": trajectory["prompt_token_ids"]
                        + generated[:position],
                        "prompt_token_count": len(trajectory["prompt_token_ids"]),
                        "student_action_token_id": int(generated[position]),
                        "student_correct": bool(trajectory["correct"]),
                        "ground_truth": trajectory["ground_truth"],
                        "base_trajectory_correct": trajectory["correct"],
                    }
                )
    write_jsonl(output, states)
    write_json(
        round_dir / "state_build_summary.json",
        {"states": len(states), "dropped_prompts": dropped},
    )
    return states


def build_round_labels(round_dir: Path) -> list[dict]:
    output = round_dir / "pair_labels.jsonl"
    if output.exists():
        return read_jsonl(output)
    states = {row["state_id"]: row for row in read_jsonl(round_dir / "states.jsonl")}
    teacher = {
        row["state_id"]: row
        for row in load_sharded(round_dir / "continuations" / "teacher")
    }
    rows = []
    for state_id, state in states.items():
        teacher_row = teacher.get(state_id)
        eligible = bool(
            teacher_row
            and teacher_row["eligible"]
            and state["student_action_token_id"] is not None
        )
        group = (
            GROUP_NAMES[(bool(teacher_row["correct"]), bool(state["student_correct"]))]
            if eligible
            else "invalid_pair"
        )
        rows.append(
            {
                "state_id": state_id,
                "round": state["round"],
                "prompt_index": state["prompt_index"],
                "base_trajectory_id": state["base_trajectory_id"],
                "fraction": state["fraction"],
                "fraction_name": state["fraction_name"],
                "eligible_pair": eligible,
                "group": group,
                "teacher_correct": bool(teacher_row["correct"])
                if teacher_row
                else None,
                "student_correct": bool(state["student_correct"]),
                "student_action_token_id": state["student_action_token_id"],
            }
        )
    write_jsonl(output, rows)
    return rows


def aggregate_rounds(run_dir: Path, rounds: int) -> tuple[list[dict], list[dict]]:
    states = []
    labels = []
    for round_index in range(rounds):
        round_dir = run_dir / "rounds" / f"round_{round_index:03d}"
        states.extend(read_jsonl(round_dir / "states.jsonl"))
        labels.extend(read_jsonl(round_dir / "pair_labels.jsonl"))
    write_jsonl(run_dir / "artifacts" / "states.jsonl", states)
    write_jsonl(run_dir / "results" / "pair_labels.jsonl", labels)
    valid_actions = sorted(
        {int(row["student_action_token_id"]) for row in labels if row["eligible_pair"]}
    )
    write_json(run_dir / "artifacts" / "action_vocab.json", valid_actions)
    return states, labels


def stopping_counts(labels: list[dict], config: dict) -> dict[str, Counter]:
    fractions = [
        fraction_name(float(item)) for item in config["data"]["state_fractions"]
    ]
    counts = {name: Counter() for name in fractions}
    for row in labels:
        if row["eligible_pair"]:
            counts[row["fraction_name"]][row["group"]] += 1
    return counts


def should_stop(counts: dict[str, Counter], rounds: int, config: dict) -> bool:
    mode = config["stopping"].get("mode", "target_count")
    if mode == "fixed_rounds":
        return rounds >= int(config["stopping"]["fixed_rounds"])
    if mode != "target_count":
        raise ValueError(f"Unknown stopping mode: {mode}")
    if rounds < int(config["stopping"]["minimum_rounds"]):
        return False
    group = config["stopping"]["target_group"]
    target = int(config["stopping"]["minimum_target_per_fraction"])
    return all(counter[group] >= target for counter in counts.values())


def merge_feature_shards(
    run_dir: Path, role: str, states: list[dict], config: dict
) -> None:
    output = run_dir / "features" / f"{role}.npz"
    if output.exists():
        return
    arrays = defaultdict(list)
    for path in sorted((run_dir / "features" / role).glob("shard_*.npz")):
        shard = np.load(path)
        for key in shard.files:
            arrays[key].append(shard[key])
    merged = {key: np.concatenate(values) for key, values in arrays.items()}
    order = {
        state_id: index for index, state_id in enumerate(merged["state_ids"].tolist())
    }
    indices = np.asarray([order[state["state_id"]] for state in states])
    merged = {key: value[indices] for key, value in merged.items()}
    with output.open("wb") as handle:
        np.savez(handle, **merged)


def run_analysis(
    run_dir: Path, config: dict, completed_rounds: int
) -> None:
    frozen_path = run_dir / "artifacts" / "frozen_rounds.json"
    if not frozen_path.exists():
        raise RuntimeError(
            "Analysis-only mode requires artifacts/frozen_rounds.json; "
            "run freeze_four_group_rounds.py first"
        )
    frozen = read_json(frozen_path)
    frozen_count = len(frozen["rounds"])
    if completed_rounds != frozen_count:
        raise RuntimeError(
            f"Requested {completed_rounds} rounds, but the snapshot contains "
            f"{frozen_count}"
        )
    states, labels = aggregate_rounds(run_dir, completed_rounds)
    counts = stopping_counts(labels, config)
    serializable = {name: dict(value) for name, value in counts.items()}
    write_json(run_dir / "results" / "stopping_counts.json", serializable)
    update_status(
        run_dir,
        "extracting_metrics",
        analysis_only=True,
        completed_rounds=completed_rounds,
        total_states=len(states),
        eligible_pairs=sum(row["eligible_pair"] for row in labels),
        stopping_counts=serializable,
    )
    feature_shards = required_shards(
        states,
        lambda row: row["base_trajectory_id"],
        int(config["parallel"]["logical_shards"]),
    )
    for role in ("teacher", "student"):
        (run_dir / "features" / role).mkdir(parents=True, exist_ok=True)
        run_workers(
            run_dir,
            config,
            f"features_{role}",
            "extract-features",
            feature_shards,
            lambda shard, role=role: (
                run_dir / "features" / role / f"shard_{shard:03d}.npz"
            ),
            role=role,
        )
        merge_feature_shards(run_dir, role, states, config)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for script, status in (
        ("compute_four_group_metrics.py", "computing_group_metrics"),
        ("analyze_four_group.py", "analyzing_groups"),
    ):
        update_status(run_dir, status, analysis_only=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name(script)),
                "--run-dir",
                str(run_dir),
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
        )


def main() -> int:
    args = parse_args()
    if args.run_dir:
        run_dir, config = load_run(args.run_dir)
    else:
        config = load_config(args.config)
        run_dir = create_run_dir(config, args.config)
    print(f"FOUR_GROUP_RUN={run_dir}", flush=True)
    shard_count = int(config["parallel"]["logical_shards"])
    try:
        if args.analysis_only:
            run_analysis(run_dir, config, int(args.completed_rounds))
            return 0
        all_labels = []
        completed_rounds = 0
        maximum_rounds = int(config["data"]["maximum_rounds"])
        for round_index in range(maximum_rounds):
            round_dir = prepare_round(run_dir, config, round_index)
            manifest = read_jsonl(round_dir / "prompt_manifest.jsonl")
            base_shards = required_shards(
                manifest, lambda row: f"prompt:{row['prompt_index']}", shard_count
            )
            update_status(run_dir, "collecting_base", round=round_index)
            run_workers(
                run_dir,
                config,
                f"round{round_index:03d}_base",
                "collect-base",
                base_shards,
                lambda shard, rd=round_dir: rd / "base" / f"shard_{shard:03d}.jsonl",
                round_index=round_index,
            )
            states = build_round_states(round_dir, config, round_index)
            state_shards = required_shards(
                states, lambda row: row["state_id"], shard_count
            )
            for role in config["continuation"].get("generated_roles", ["teacher"]):
                update_status(
                    run_dir, "collecting_continuations", round=round_index, role=role
                )
                run_workers(
                    run_dir,
                    config,
                    f"round{round_index:03d}_{role}_rollout",
                    "rollout",
                    state_shards,
                    lambda shard, role=role, rd=round_dir: (
                        rd / "continuations" / role / f"shard_{shard:03d}.jsonl"
                    ),
                    round_index=round_index,
                    role=role,
                )
            labels = build_round_labels(round_dir)
            all_labels.extend(labels)
            completed_rounds = round_index + 1
            counts = stopping_counts(all_labels, config)
            serializable = {name: dict(value) for name, value in counts.items()}
            write_json(run_dir / "results" / "stopping_counts.json", serializable)
            update_status(
                run_dir,
                "checking_stopping_rule",
                completed_rounds=completed_rounds,
                stopping_counts=serializable,
            )
            print(f"Stopping counts after round {completed_rounds}: {serializable}")
            if should_stop(counts, completed_rounds, config):
                break

        run_analysis(run_dir, config, completed_rounds)
        return 0
    except Exception as error:
        update_status(
            run_dir,
            "failed",
            stage="four_group_pipeline",
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
