from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from common import load_run, read_jsonl, update_status
from four_group_common import load_sharded, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and checksum completed four-group collection rounds."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--completed-rounds", required=True, type=int)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the existing checksum manifest without rewriting it.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_count(rows: list[dict], key: str) -> int:
    return len({row[key] for row in rows})


def validate_round(round_dir: Path, config: dict) -> dict:
    manifest = read_jsonl(round_dir / "prompt_manifest.jsonl")
    base = load_sharded(round_dir / "base")
    states = read_jsonl(round_dir / "states.jsonl")
    teacher = load_sharded(round_dir / "continuations" / "teacher")
    labels = read_jsonl(round_dir / "pair_labels.jsonl")

    expected_prompts = int(config["data"]["prompts_per_round"])
    expected_base = expected_prompts * int(config["data"]["rollouts_per_prompt"])
    expected_states = expected_base * len(config["data"]["state_fractions"])
    checks = {
        "prompts": (len(manifest), expected_prompts),
        "base_trajectories": (len(base), expected_base),
        "unique_base_trajectories": (
            unique_count(base, "base_trajectory_id"),
            expected_base,
        ),
        "states": (len(states), expected_states),
        "unique_states": (unique_count(states, "state_id"), expected_states),
        "teacher_continuations": (len(teacher), expected_states),
        "unique_teacher_continuations": (
            unique_count(teacher, "state_id"),
            expected_states,
        ),
        "pair_labels": (len(labels), expected_states),
        "unique_pair_labels": (unique_count(labels, "state_id"), expected_states),
    }
    failures = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    state_ids = {row["state_id"] for row in states}
    if {row["state_id"] for row in teacher} != state_ids:
        failures["teacher_state_id_set"] = "does not match states"
    if {row["state_id"] for row in labels} != state_ids:
        failures["label_state_id_set"] = "does not match states"
    if failures:
        raise RuntimeError(f"Round validation failed for {round_dir}: {failures}")

    eligible = sum(bool(row["eligible_pair"]) for row in labels)
    return {
        name: actual for name, (actual, _) in checks.items()
    } | {"eligible_pairs": eligible}


def main() -> int:
    args = parse_args()
    if args.completed_rounds <= 0:
        raise ValueError("--completed-rounds must be positive")
    run_dir, config = load_run(args.run_dir)
    output = run_dir / "artifacts" / "frozen_rounds.json"
    if args.verify_only:
        if not output.exists():
            raise FileNotFoundError(output)
        manifest = read_json(output)
        if len(manifest["rounds"]) != args.completed_rounds:
            raise RuntimeError("Frozen round count does not match the request")
        failures = []
        for item in manifest["files"]:
            path = run_dir / item["path"]
            if not path.exists():
                failures.append(f"missing: {item['path']}")
            elif path.stat().st_size != int(item["bytes"]):
                failures.append(f"size changed: {item['path']}")
            elif sha256(path) != item["sha256"]:
                failures.append(f"checksum changed: {item['path']}")
        if failures:
            raise RuntimeError(f"Frozen snapshot verification failed: {failures}")
        print(
            f"Verified {len(manifest['files'])} frozen files for "
            f"{args.completed_rounds} round(s)"
        )
        return 0
    rounds = []
    frozen_files = [run_dir / "config.yaml"]
    for index in range(args.completed_rounds):
        round_dir = run_dir / "rounds" / f"round_{index:03d}"
        counts = validate_round(round_dir, config)
        rounds.append({"round": index, "counts": counts})
        frozen_files.extend(
            path
            for path in round_dir.rglob("*")
            if path.is_file() and ".tmp" not in path.suffixes
        )

    files = []
    for path in sorted(set(frozen_files)):
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "rounds": rounds,
        "analysis_stopping_rule": {
            "mode": "fixed_rounds",
            "fixed_rounds": args.completed_rounds,
            "reason": "Frozen completed collection; no adaptive target-count extension.",
        },
        "files": files,
    }
    write_json(output, manifest)
    update_status(
        run_dir,
        "rounds_frozen",
        completed_rounds=args.completed_rounds,
        frozen_manifest=str(output),
        analysis_stopping_rule=manifest["analysis_stopping_rule"],
    )
    print(f"Frozen manifest: {output}")
    for item in rounds:
        print(f"round {item['round']}: {item['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
