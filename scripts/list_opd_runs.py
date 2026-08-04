#!/usr/bin/env python3
"""List managed OPD runs without depending on a tracking service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.path.join(os.environ.get("OPD_STORAGE_ROOT", ""), "experiments"),
    )
    args = parser.parse_args()
    root = Path(args.root)
    print(f"{'RUN_ID':72} {'STATUS':10} {'EXIT':4} UPDATED")
    if not root.is_dir():
        return 0
    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        status_path = run_dir / "status.yaml"
        status = yaml.safe_load(status_path.read_text()) if status_path.is_file() else {}
        print(
            f"{run_dir.name[:72]:72} "
            f"{str(status.get('status', 'unknown'))[:10]:10} "
            f"{str(status.get('exit_code', '-')):4} "
            f"{status.get('updated_at_utc', '-')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

