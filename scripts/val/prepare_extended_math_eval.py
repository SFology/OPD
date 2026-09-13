#!/usr/bin/env python3
"""Convert the upstream JSON math suites into the local verl parquet schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/attached/remote-home1/liufengkai/opd/datasets/extended_math_eval")
PROMPT_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
SOURCES = {
    "MATH-500": REPO_ROOT / "datasets/test_data/MATH-500/test.json",
    "Minerva": REPO_ROOT / "datasets/test_data/Minerva/test.json",
    "Olympiad-Bench": REPO_ROOT / "datasets/test_data/Olympiad-Bench/test.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    manifest = {"schema_version": 1, "datasets": []}
    for name, source in SOURCES.items():
        records = json.loads(source.read_text(encoding="utf-8"))
        rows = []
        for index, record in enumerate(records):
            problem = str(record["prompt"]).strip()
            if "put your final answer within \\boxed{}" not in problem:
                problem += PROMPT_SUFFIX
            rows.append(
                {
                    "data_source": name,
                    "prompt": [{"role": "user", "content": problem}],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": str(record["answer"]).strip(),
                    },
                    "extra_info": {
                        "index": index,
                        "source_id": str(record.get("id", index)),
                    },
                }
            )
        target = output_root / name / "test.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(target)
        manifest["datasets"].append(
            {
                "name": name,
                "rows": len(rows),
                "source": str(source.resolve()),
                "source_sha256": sha256(source),
                "parquet": str(target),
                "parquet_sha256": sha256(target),
            }
        )
        print(f"Prepared {name}: {len(rows)} prompts -> {target}")
    manifest_path = output_root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
