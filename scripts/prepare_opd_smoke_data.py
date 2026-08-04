#!/usr/bin/env python3
"""Create the deterministic OPD smoke-test dataset on attached storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    opd_root = Path(os.environ.get("OPD_ROOT", Path(__file__).resolve().parents[1]))
    data_root = Path(os.environ.get("OPD_DATA_DIR", opd_root / "datasets"))
    source = opd_root / "datasets" / "dapo-math-17k.parquet"
    output_dir = data_root / "smoke"
    output = output_dir / f"dapo-math-17k-{args.rows}.parquet"
    manifest = output.with_suffix(".manifest.json")

    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_parquet(source)
    if args.rows > len(frame):
        raise ValueError(f"requested {args.rows} rows from a {len(frame)}-row dataset")

    subset = frame.sample(n=args.rows, random_state=args.seed).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output, index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "rows": args.rows,
        "seed": args.seed,
    }
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

