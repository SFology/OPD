from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from common import (
    load_config,
    load_run,
    read_jsonl,
    save_yaml,
    update_status,
    write_jsonl,
)
from four_group_common import GROUP_NAMES, load_sharded, logical_shard, read_json, write_json
from run_four_group_pipeline import assign_shards, wait_for_idle_gpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the equal-decoding teacher/student PPL control."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def prepare_directory(run_dir: Path, source_config: Path) -> tuple[Path, dict]:
    directory = run_dir / "ppl_decode_control"
    for name in (
        "logs",
        "teacher_continuations",
        "scores/teacher/teacher",
        "scores/teacher/student",
        "scores/student/teacher",
        "scores/student/student",
        "results/figures",
    ):
        (directory / name).mkdir(parents=True, exist_ok=True)
    config = load_config(source_config)
    snapshot = directory / "config.yaml"
    if snapshot.exists():
        if load_config(snapshot) != config:
            raise RuntimeError(f"Config differs from existing snapshot: {snapshot}")
    else:
        save_yaml(snapshot, config)
    if not (directory / "status.yaml").exists():
        save_yaml(
            directory / "status.yaml",
            {
                "status": "prepared",
                "source_run": str(run_dir),
                "source_config": str(source_config.resolve()),
            },
        )
    return directory, config


def prepare_state_manifest(
    run_dir: Path, directory: Path, config: dict
) -> list[dict]:
    output = directory / "state_manifest.jsonl"
    if output.exists():
        return read_jsonl(output)
    frozen = read_json(run_dir / "artifacts" / "frozen_rounds.json")
    expected_rounds = int(config["selection"]["completed_rounds"])
    if len(frozen["rounds"]) != expected_rounds:
        raise RuntimeError(
            f"Frozen snapshot has {len(frozen['rounds'])} rounds, expected "
            f"{expected_rounds}"
        )
    states = read_jsonl(run_dir / "artifacts" / "states.jsonl")
    base = {}
    for item in frozen["rounds"]:
        round_dir = run_dir / "rounds" / f"round_{int(item['round']):03d}"
        for row in load_sharded(round_dir / "base"):
            base[row["base_trajectory_id"]] = row
    shard_count = int(config["parallel"]["logical_shards"])
    rows = []
    for state in sorted(states, key=lambda item: item["state_id"]):
        trajectory = base[state["base_trajectory_id"]]
        position = int(state["position"])
        expected_prefix = (
            trajectory["prompt_token_ids"]
            + trajectory["generated_token_ids"][:position]
        )
        if state["input_ids"] != expected_prefix:
            raise RuntimeError(f"Prefix mismatch for {state['state_id']}")
        student_suffix = trajectory["generated_token_ids"][position:]
        if not student_suffix:
            raise RuntimeError(f"Empty student suffix for {state['state_id']}")
        rows.append(
            {
                "state_id": state["state_id"],
                "round": state["round"],
                "prompt_index": state["prompt_index"],
                "base_trajectory_id": state["base_trajectory_id"],
                "fraction_name": state["fraction_name"],
                "normalized_position": state["normalized_position"],
                "prompt_token_count": state["prompt_token_count"],
                "input_ids": state["input_ids"],
                "student_generated_token_ids": student_suffix,
                "student_correct": bool(state["student_correct"]),
                "ground_truth": state["ground_truth"],
                "generation_shard": logical_shard(
                    state["state_id"], shard_count
                ),
            }
        )
    write_jsonl(output, rows)
    write_json(
        directory / "state_manifest_summary.json",
        {
            "states": len(rows),
            "base_trajectories": len(
                {row["base_trajectory_id"] for row in rows}
            ),
            "student_correctness": dict(
                Counter("correct" if row["student_correct"] else "wrong" for row in rows)
            ),
            "selection_policy": (
                "all frozen states; no selection using old teacher continuations"
            ),
        },
    )
    return rows


def run_workers(
    run_dir: Path,
    directory: Path,
    config: dict,
    *,
    stage: str,
    mode: str,
    shards: list[int],
    output_for_shard,
    role: str | None = None,
    branch: str | None = None,
) -> None:
    worker = Path(__file__).with_name("ppl_decode_control_worker.py")
    config_path = directory / "config.yaml"
    for retry in range(3):
        missing = [shard for shard in shards if not output_for_shard(shard).exists()]
        if not missing:
            return
        assignments = assign_shards(missing, wait_for_idle_gpus(config))
        processes = []
        for worker_index, (gpu, assigned) in enumerate(assignments):
            command = [
                sys.executable,
                "-u",
                str(worker),
                mode,
                "--run-dir",
                str(run_dir),
                "--control-config",
                str(config_path),
                "--shards",
                ",".join(map(str, assigned)),
            ]
            if role is not None:
                command.extend(["--role", role])
            if branch is not None:
                command.extend(["--branch", branch])
            log = (
                directory
                / "logs"
                / f"{stage}_retry{retry}_worker{worker_index}_gpu{gpu}.log"
            )
            handle = log.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[{stage}] GPU {gpu}: shards {assigned}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, handle, log))
        failures = []
        for process, handle, log in processes:
            return_code = process.wait()
            handle.close()
            if return_code:
                failures.append((return_code, str(log)))
        remaining = [
            shard for shard in missing if not output_for_shard(shard).exists()
        ]
        if not remaining:
            return
        print(
            f"[{stage}] retry {retry + 1}/3; remaining={remaining}; "
            f"failures={failures}",
            flush=True,
        )
    raise RuntimeError(f"Stage {stage} remains incomplete")


def build_fresh_labels_and_score_manifest(
    directory: Path, state_manifest: list[dict], config: dict
) -> list[dict]:
    score_output = directory / "score_manifest.jsonl"
    if score_output.exists():
        return read_jsonl(score_output)
    teacher = {
        row["state_id"]: row
        for row in load_sharded(directory / "teacher_continuations")
    }
    if len(teacher) != len(state_manifest):
        raise RuntimeError(
            f"Teacher continuation count {len(teacher)} != state count "
            f"{len(state_manifest)}"
        )
    minimum = int(config["selection"]["minimum_branch_tokens"])
    maximum = int(config["selection"]["maximum_sequence_tokens"])
    shard_count = int(config["parallel"]["logical_shards"])
    labels, score_rows, dropped = [], [], []
    for state in state_manifest:
        continuation = teacher[state["state_id"]]
        eligible = bool(continuation["eligible"])
        group = (
            GROUP_NAMES[(bool(continuation["correct"]), state["student_correct"])]
            if eligible
            else "invalid_pair"
        )
        labels.append(
            {
                "state_id": state["state_id"],
                "prompt_index": state["prompt_index"],
                "base_trajectory_id": state["base_trajectory_id"],
                "fraction_name": state["fraction_name"],
                "eligible_pair": eligible,
                "group": group,
                "teacher_correct": bool(continuation["correct"]),
                "student_correct": bool(state["student_correct"]),
            }
        )
        if not eligible:
            dropped.append({"state_id": state["state_id"], "reason": "teacher_invalid"})
            continue
        teacher_tokens = continuation["generated_token_ids"]
        student_tokens = state["student_generated_token_ids"]
        if min(len(teacher_tokens), len(student_tokens)) < minimum:
            dropped.append({"state_id": state["state_id"], "reason": "branch_too_short"})
            continue
        if max(
            len(state["input_ids"]) + len(teacher_tokens),
            len(state["input_ids"]) + len(student_tokens),
        ) > maximum:
            dropped.append({"state_id": state["state_id"], "reason": "sequence_too_long"})
            continue
        score_rows.append(
            {
                **state,
                "group": group,
                "teacher_correct": bool(continuation["correct"]),
                "teacher_generated_token_ids": teacher_tokens,
                "teacher_score_shard": logical_shard(
                    state["state_id"], shard_count
                ),
                "student_score_shard": logical_shard(
                    state["base_trajectory_id"], shard_count
                ),
            }
        )
    write_jsonl(directory / "fresh_pair_labels.jsonl", labels)
    write_jsonl(score_output, score_rows)
    label_counts = Counter(row["group"] for row in labels)
    scored_counts = Counter(row["group"] for row in score_rows)
    write_json(
        directory / "fresh_label_summary.json",
        {
            "generated_states": len(state_manifest),
            "fresh_label_counts": dict(label_counts),
            "scored_state_counts": dict(scored_counts),
            "dropped_from_scoring": dropped,
            "teacher_generation": config["generation"]["teacher"],
            "student_generation": {"temperature": 1.0, "top_p": 1.0},
        },
    )
    return score_rows


def run_analysis(run_dir: Path) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("analyze_ppl_decode_control.py")),
            "--run-dir",
            str(run_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
    )


def main() -> int:
    args = parse_args()
    run_dir, _ = load_run(args.run_dir)
    directory, config = prepare_directory(run_dir, args.config)
    try:
        states = prepare_state_manifest(run_dir, directory, config)
        generation_shards = sorted({int(row["generation_shard"]) for row in states})
        update_status(
            directory,
            "prepared" if args.prepare_only else "generating_teacher",
            states=len(states),
            generation_shards=len(generation_shards),
            decoding={"teacher": {"temperature": 1.0, "top_p": 1.0},
                      "student": {"temperature": 1.0, "top_p": 1.0}},
        )
        if args.prepare_only:
            print(
                f"Prepared {len(states)} states across "
                f"{len(generation_shards)} generation shards"
            )
            return 0
        if not args.analysis_only:
            run_workers(
                run_dir,
                directory,
                config,
                stage="generate_teacher_t1_p1",
                mode="generate-teacher",
                shards=generation_shards,
                output_for_shard=lambda shard: (
                    directory / "teacher_continuations" / f"shard_{shard:03d}.jsonl"
                ),
            )
        score_manifest = build_fresh_labels_and_score_manifest(
            directory, states, config
        )
        if not args.analysis_only:
            for branch in config["scoring"]["branches"]:
                shard_key = f"{branch}_score_shard"
                shards = sorted({int(row[shard_key]) for row in score_manifest})
                for role in config["scoring"]["roles"]:
                    stage = f"score_{branch}_branch_by_{role}"
                    update_status(directory, "scoring", branch=branch, role=role)
                    run_workers(
                        run_dir,
                        directory,
                        config,
                        stage=stage,
                        mode="score-branch",
                        shards=shards,
                        role=role,
                        branch=branch,
                        output_for_shard=lambda shard, b=branch, r=role: (
                            directory
                            / "scores"
                            / b
                            / r
                            / f"shard_{shard:03d}.jsonl"
                        ),
                    )
        update_status(directory, "analyzing")
        run_analysis(run_dir)
        return 0
    except Exception as error:
        update_status(
            directory,
            "failed",
            stage="ppl_decode_control",
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
