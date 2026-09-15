#!/usr/bin/env python3
"""Freeze a label-free scaled-OPD coefficient from FP32 calibration metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from verl.trainer.ppo.robust_opd import calibrate_fixed_opd_scale


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-steps", type=int, default=5)
    parser.add_argument("--maximum-scale", type=float, default=20.0)
    args = parser.parse_args()

    metrics_path = args.metrics.resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(f"calibration metrics not found: {metrics_path}")
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = calibrate_fixed_opd_scale(
        records,
        minimum_steps=args.minimum_steps,
        maximum_scale=args.maximum_scale,
    )
    result.update(
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_metrics": str(metrics_path),
            "source_metrics_sha256": file_sha256(metrics_path),
            "scientific_constraint": "uses training reward RMS only; no verifier or test labels",
        }
    )
    atomic_write_json(args.output.resolve(), result)
    print(f"FIXED_OPD_SCALE={result['fixed_opd_scale']:.12g}")
    print(f"CALIBRATION_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
