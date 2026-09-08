from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import (
    load_config,
    load_run,
    read_jsonl,
    save_yaml,
    update_status,
    write_jsonl,
)
from four_group_common import load_sharded, logical_shard, read_json, write_json
from run_four_group_pipeline import assign_shards, wait_for_idle_gpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare student-prefix and paired teacher/student branch PPL."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate the manifest without loading either model.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Analyze already completed branch scores without loading models.",
    )
    return parser.parse_args()


def prepare_branch_dir(run_dir: Path, source_config: Path) -> tuple[Path, dict]:
    branch_dir = run_dir / "ppl_branch_comparison"
    for name in (
        "logs",
        "scores/student_branch/teacher",
        "scores/student_branch/student",
        "results/figures",
    ):
        (branch_dir / name).mkdir(parents=True, exist_ok=True)
    config = load_config(source_config)
    snapshot = branch_dir / "config.yaml"
    if snapshot.exists():
        if load_config(snapshot) != config:
            raise RuntimeError(f"Config differs from existing snapshot: {snapshot}")
    else:
        save_yaml(snapshot, config)
    if not (branch_dir / "status.yaml").exists():
        save_yaml(
            branch_dir / "status.yaml",
            {
                "status": "prepared",
                "source_run": str(run_dir),
                "source_config": str(source_config.resolve()),
            },
        )
    return branch_dir, config


def prepare_manifest(run_dir: Path, branch_dir: Path, config: dict) -> list[dict]:
    output = branch_dir / "manifest.jsonl"
    if output.exists():
        existing = read_jsonl(output)
        if existing and all(
            int(row.get("manifest_schema_version", 0)) == 2 for row in existing
        ):
            return existing

    frozen = read_json(run_dir / "artifacts" / "frozen_rounds.json")
    expected_rounds = int(config["selection"]["completed_rounds"])
    if len(frozen["rounds"]) != expected_rounds:
        raise RuntimeError(
            f"Frozen snapshot has {len(frozen['rounds'])} rounds, expected "
            f"{expected_rounds}"
        )
    states = {
        row["state_id"]: row
        for row in read_jsonl(run_dir / "artifacts" / "states.jsonl")
    }
    labels = {
        row["state_id"]: row
        for row in read_jsonl(run_dir / "results" / "pair_labels.jsonl")
    }
    base = {}
    for item in frozen["rounds"]:
        round_dir = run_dir / "rounds" / f"round_{int(item['round']):03d}"
        for row in load_sharded(round_dir / "base"):
            identifier = row["base_trajectory_id"]
            if identifier in base:
                raise RuntimeError(f"Duplicate base trajectory: {identifier}")
            base[identifier] = row

    groups = set(config["selection"]["groups"])
    minimum = int(config["selection"]["minimum_branch_tokens"])
    maximum = int(config["selection"]["maximum_sequence_tokens"])
    shard_count = int(config["parallel"]["logical_shards"])
    rows, dropped = [], []
    for state_id, label in sorted(labels.items()):
        if not label["eligible_pair"] or label["group"] not in groups:
            continue
        state = states[state_id]
        trajectory = base.get(state["base_trajectory_id"])
        if trajectory is None:
            dropped.append({"state_id": state_id, "reason": "missing_base_trajectory"})
            continue
        prompt = trajectory["prompt_token_ids"]
        generated = trajectory["generated_token_ids"]
        position = int(state["position"])
        expected_prefix = prompt + generated[:position]
        if state["input_ids"] != expected_prefix:
            raise RuntimeError(f"Prefix token mismatch for {state_id}")
        if position >= len(generated):
            dropped.append({"state_id": state_id, "reason": "empty_student_branch"})
            continue
        suffix = generated[position:]
        if int(suffix[0]) != int(state["student_action_token_id"]):
            raise RuntimeError(f"Student action token mismatch for {state_id}")
        total = len(state["input_ids"]) + len(suffix)
        if len(suffix) < minimum:
            dropped.append({"state_id": state_id, "reason": "branch_too_short"})
            continue
        if total > maximum:
            dropped.append({"state_id": state_id, "reason": "sequence_too_long"})
            continue
        rows.append(
            {
                "manifest_schema_version": 2,
                "state_id": state_id,
                "round": state["round"],
                "prompt_index": state["prompt_index"],
                "base_trajectory_id": state["base_trajectory_id"],
                "fraction_name": state["fraction_name"],
                "normalized_position": state["normalized_position"],
                "group": label["group"],
                "prompt_token_count": state["prompt_token_count"],
                "input_ids": state["input_ids"],
                "student_generated_token_ids": suffix,
                # All intervention points from one frozen rollout share a shard so
                # the worker can score the full student trajectory only once.
                "logical_shard": logical_shard(
                    state["base_trajectory_id"], shard_count
                ),
            }
        )
    write_jsonl(output, rows)
    fractions = sorted({row["fraction_name"] for row in rows})
    write_json(
        branch_dir / "manifest_summary.json",
        {
            "selected_states": len(rows),
            "dropped_states": dropped,
            "groups": {
                group: sum(row["group"] == group for row in rows)
                for group in sorted(groups)
            },
            "fractions": {
                fraction: sum(row["fraction_name"] == fraction for row in rows)
                for fraction in fractions
            },
            "student_branch_source": "original frozen base rollout suffix",
            "unique_base_trajectories": len(
                {row["base_trajectory_id"] for row in rows}
            ),
            "scoring_reuse": "one causal forward pass per base trajectory and role",
        },
    )
    return rows


def verify_teacher_branch_scores(
    run_dir: Path, manifest: list[dict], roles: list[str]
) -> None:
    expected = {row["state_id"] for row in manifest}
    recovery = run_dir / "ppl_recovery_curve" / "scores"
    for role in roles:
        observed = {
            row["state_id"]
            for path in sorted((recovery / role).glob("shard_*.jsonl"))
            for row in read_jsonl(path)
        }
        missing = expected - observed
        if missing:
            raise RuntimeError(
                f"Existing teacher-branch {role} scores miss {len(missing)} states"
            )


def run_role(
    run_dir: Path,
    branch_dir: Path,
    config: dict,
    role: str,
    shards: list[int],
) -> None:
    worker = Path(__file__).with_name("ppl_branch_worker.py")
    config_path = branch_dir / "config.yaml"
    for retry in range(3):
        missing = [
            shard
            for shard in shards
            if not (
                branch_dir
                / "scores"
                / "student_branch"
                / role
                / f"shard_{shard:03d}.jsonl"
            ).exists()
        ]
        if not missing:
            return
        assignments = assign_shards(missing, wait_for_idle_gpus(config))
        processes = []
        for worker_index, (gpu, assigned) in enumerate(assignments):
            log = (
                branch_dir
                / "logs"
                / f"student_branch_{role}_retry{retry}_worker{worker_index}_gpu{gpu}.log"
            )
            handle = log.open("w", encoding="utf-8")
            command = [
                sys.executable,
                "-u",
                str(worker),
                "--run-dir",
                str(run_dir),
                "--branch-config",
                str(config_path),
                "--role",
                role,
                "--shards",
                ",".join(map(str, assigned)),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[student-branch-{role}] GPU {gpu}: shards {assigned}", flush=True)
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
            shard
            for shard in missing
            if not (
                branch_dir
                / "scores"
                / "student_branch"
                / role
                / f"shard_{shard:03d}.jsonl"
            ).exists()
        ]
        if not remaining:
            return
        print(
            f"Retry {retry + 1}/3 for {role}; remaining={remaining}; "
            f"failures={failures}",
            flush=True,
        )
    raise RuntimeError(f"Student-branch PPL scoring for {role} remains incomplete")


def run_analysis(run_dir: Path) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("analyze_ppl_branch_comparison.py")),
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
    branch_dir, config = prepare_branch_dir(run_dir, args.config)
    try:
        manifest = prepare_manifest(run_dir, branch_dir, config)
        roles = list(config["scoring"]["roles"])
        verify_teacher_branch_scores(run_dir, manifest, roles)
        shards = sorted({int(row["logical_shard"]) for row in manifest})
        update_status(
            branch_dir,
            "prepared" if args.prepare_only else "scoring",
            selected_states=len(manifest),
            required_shards=len(shards),
            reused_teacher_branch_scores=True,
        )
        if args.prepare_only:
            print(f"Prepared {len(manifest)} states across {len(shards)} shards")
            return 0
        if not args.analysis_only:
            for role in roles:
                update_status(branch_dir, "scoring", role=role)
                run_role(run_dir, branch_dir, config, role, shards)
        update_status(branch_dir, "analyzing")
        run_analysis(run_dir)
        return 0
    except Exception as error:
        update_status(
            branch_dir,
            "failed",
            stage="ppl_branch_comparison",
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
