from __future__ import annotations

import argparse
from pathlib import Path

from common import load_run, read_jsonl, update_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject trajectory collections dominated by token-limit truncation."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-truncated-fraction", type=float, default=0.9)
    parser.add_argument("--min-parseable-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    rows = read_jsonl(run_dir / "artifacts" / "trajectories.jsonl")
    if not rows:
        raise RuntimeError("No collected trajectories were found")

    token_limit = int(config["data"]["max_new_tokens"])
    truncated = sum(len(row["generated_token_ids"]) >= token_limit for row in rows)
    parseable = sum(
        row.get("trajectory_predicted_answer") not in (None, "", "[INVALID]")
        for row in rows
    )
    correct = sum(bool(row["trajectory_correct"]) for row in rows)
    truncated_fraction = truncated / len(rows)
    parseable_fraction = parseable / len(rows)
    correct_fraction = correct / len(rows)

    print(f"trajectories={len(rows)}")
    print(f"token_limit={token_limit}")
    print(f"truncated={truncated} ({truncated_fraction:.1%})")
    print(f"parseable={parseable} ({parseable_fraction:.1%})")
    print(f"correct={correct} ({correct_fraction:.1%})")

    if truncated_fraction > args.max_truncated_fraction:
        update_status(
            run_dir,
            "collection_rejected",
            rejection_reason="excessive_truncation",
            truncated_fraction=truncated_fraction,
            parseable_fraction=parseable_fraction,
        )
        raise RuntimeError(
            "Collection rejected: too many trajectories reached the token limit"
        )
    if parseable_fraction < args.min_parseable_fraction:
        update_status(
            run_dir,
            "collection_rejected",
            rejection_reason="insufficient_parseable_answers",
            truncated_fraction=truncated_fraction,
            parseable_fraction=parseable_fraction,
        )
        raise RuntimeError(
            "Collection rejected: too few trajectories contain parseable final answers"
        )
    update_status(
        run_dir,
        "collection_checked",
        truncated_fraction=truncated_fraction,
        parseable_fraction=parseable_fraction,
        trajectory_correct_fraction=correct_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
