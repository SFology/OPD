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
        description="Score and analyze PPL recovery after teacher intervention."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def prepare_curve_dir(run_dir: Path, source_config: Path) -> tuple[Path, dict]:
    curve_dir = run_dir / "ppl_recovery_curve"
    for name in ("logs", "scores/teacher", "scores/student", "results/figures"):
        (curve_dir / name).mkdir(parents=True, exist_ok=True)
    config = load_config(source_config)
    snapshot = curve_dir / "config.yaml"
    if snapshot.exists():
        existing = load_config(snapshot)
        if existing != config:
            raise RuntimeError(
                f"Curve config differs from the existing snapshot: {snapshot}"
            )
    else:
        save_yaml(snapshot, config)
    if not (curve_dir / "status.yaml").exists():
        save_yaml(
            curve_dir / "status.yaml",
            {
                "status": "prepared",
                "source_run": str(run_dir),
                "source_config": str(source_config.resolve()),
            },
        )
    return curve_dir, config


def prepare_manifest(run_dir: Path, curve_dir: Path, config: dict) -> list[dict]:
    output = curve_dir / "manifest.jsonl"
    if output.exists():
        return read_jsonl(output)
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
    continuations = {}
    for item in frozen["rounds"]:
        round_dir = run_dir / "rounds" / f"round_{int(item['round']):03d}"
        for row in load_sharded(round_dir / "continuations" / "teacher"):
            continuations[row["state_id"]] = row

    groups = set(config["selection"]["groups"])
    minimum = int(config["selection"]["minimum_continuation_tokens"])
    maximum = int(config["selection"]["maximum_sequence_tokens"])
    shard_count = int(config["parallel"]["logical_shards"])
    rows, dropped = [], []
    for state_id, label in sorted(labels.items()):
        if not label["eligible_pair"] or label["group"] not in groups:
            continue
        state = states[state_id]
        continuation = continuations.get(state_id)
        if continuation is None:
            dropped.append({"state_id": state_id, "reason": "missing_continuation"})
            continue
        generated = continuation["generated_token_ids"]
        total = len(state["input_ids"]) + len(generated)
        if len(generated) < minimum:
            dropped.append({"state_id": state_id, "reason": "continuation_too_short"})
            continue
        if total > maximum:
            dropped.append({"state_id": state_id, "reason": "sequence_too_long"})
            continue
        rows.append(
            {
                "state_id": state_id,
                "round": state["round"],
                "prompt_index": state["prompt_index"],
                "base_trajectory_id": state["base_trajectory_id"],
                "fraction_name": state["fraction_name"],
                "group": label["group"],
                "prompt_token_count": state["prompt_token_count"],
                "input_ids": state["input_ids"],
                "teacher_generated_token_ids": generated,
                "logical_shard": logical_shard(state_id, shard_count),
            }
        )
    write_jsonl(output, rows)
    write_json(
        curve_dir / "manifest_summary.json",
        {
            "selected_states": len(rows),
            "dropped_states": dropped,
            "groups": {
                group: sum(row["group"] == group for row in rows) for group in groups
            },
            "fractions": {
                fraction: sum(row["fraction_name"] == fraction for row in rows)
                for fraction in sorted({row["fraction_name"] for row in rows})
            },
        },
    )
    return rows


def run_role(
    run_dir: Path,
    curve_dir: Path,
    curve_config: dict,
    role: str,
    shards: list[int],
) -> None:
    worker = Path(__file__).with_name("ppl_recovery_worker.py")
    config_path = curve_dir / "config.yaml"
    for retry in range(3):
        missing = [
            shard
            for shard in shards
            if not (curve_dir / "scores" / role / f"shard_{shard:03d}.jsonl").exists()
        ]
        if not missing:
            return
        gpus = wait_for_idle_gpus(curve_config)
        assignments = assign_shards(missing, gpus)
        processes = []
        for worker_index, (gpu, assigned) in enumerate(assignments):
            log = (
                curve_dir
                / "logs"
                / f"score_{role}_retry{retry}_worker{worker_index}_gpu{gpu}.log"
            )
            handle = log.open("w", encoding="utf-8")
            command = [
                sys.executable,
                "-u",
                str(worker),
                "--run-dir",
                str(run_dir),
                "--curve-config",
                str(config_path),
                "--role",
                role,
                "--shards",
                ",".join(map(str, assigned)),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[ppl-curve-{role}] GPU {gpu}: shards {assigned}", flush=True)
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
            if not (curve_dir / "scores" / role / f"shard_{shard:03d}.jsonl").exists()
        ]
        if not remaining:
            return
        print(
            f"Retry {retry + 1}/3 for {role}; remaining={remaining}; "
            f"failures={failures}",
            flush=True,
        )
    raise RuntimeError(f"PPL curve scoring for {role} remains incomplete")


def main() -> int:
    args = parse_args()
    run_dir, _ = load_run(args.run_dir)
    curve_dir, config = prepare_curve_dir(run_dir, args.config)
    try:
        manifest = prepare_manifest(run_dir, curve_dir, config)
        shards = sorted({int(row["logical_shard"]) for row in manifest})
        update_status(
            curve_dir,
            "scoring",
            selected_states=len(manifest),
            required_shards=len(shards),
        )
        for role in config["scoring"]["roles"]:
            update_status(curve_dir, "scoring", role=role)
            run_role(run_dir, curve_dir, config, role, shards)
        update_status(curve_dir, "analyzing")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("analyze_ppl_recovery_curve.py")),
                "--run-dir",
                str(run_dir),
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
        )
        return 0
    except Exception as error:
        update_status(
            curve_dir,
            "failed",
            stage="ppl_recovery_curve",
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
