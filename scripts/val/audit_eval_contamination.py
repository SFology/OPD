#!/usr/bin/env python3
"""Audit exact normalized prompt overlap between training and evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from formal_eval_common import load_yaml, sha256_file, write_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "datasets" / "dapo-math-17k.parquet"
DEFAULT_OUTPUT = Path(
    "/attached/remote-home1/liufengkai/opd/datasets/extended_math_eval/contamination_audit.yaml"
)
PROMPT_SUFFIXES = (
    "please reason step by step, and put your final answer within \\boxed{}.",
    "please reason step by step and put your final answer within \\boxed{}.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def prompt_content(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        user_parts = [
            str(item.get("content", ""))
            for item in value
            if isinstance(item, dict) and str(item.get("role", "")) == "user"
        ]
        if user_parts:
            return "\n".join(user_parts)
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(value)


def canonical_prompt(value: Any) -> str:
    text = prompt_content(value).strip().lower()
    for suffix in PROMPT_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    # This catches formatting-only differences without pretending to detect
    # semantic near-duplicates.
    return re.sub(r"\s+", " ", text)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    train_path = args.train_path.resolve()
    train = pd.read_parquet(train_path)
    train_hashes: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(train["prompt"]):
        train_hashes[digest(canonical_prompt(value))].append(index)

    eval_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dataset_records = []
    for dataset in config["datasets"]:
        path = Path(dataset["path"]).resolve()
        frame = pd.read_parquet(path)
        for index, value in enumerate(frame["prompt"]):
            eval_hashes[digest(canonical_prompt(value))].append(
                {"dataset": dataset["name"], "row": index}
            )
        dataset_records.append(
            {
                "name": dataset["name"],
                "path": str(path),
                "rows": len(frame),
                "sha256": sha256_file(path),
            }
        )

    overlaps = []
    for prompt_hash in sorted(set(train_hashes) & set(eval_hashes)):
        overlaps.append(
            {
                "prompt_sha256": prompt_hash,
                "train_rows": train_hashes[prompt_hash],
                "evaluation_rows": eval_hashes[prompt_hash],
            }
        )
    duplicate_eval = [
        {"prompt_sha256": prompt_hash, "evaluation_rows": rows}
        for prompt_hash, rows in sorted(eval_hashes.items())
        if len(rows) > 1
    ]
    report = {
        "schema_version": 1,
        "method": "lowercase + whitespace normalization + known answer-suffix removal + exact SHA-256",
        "scope_note": "Exact normalized overlap audit; this does not rule out paraphrased or semantic contamination.",
        "train": {
            "path": str(train_path),
            "rows": len(train),
            "sha256": sha256_file(train_path),
            "unique_normalized_prompts": len(train_hashes),
        },
        "evaluation": {
            "datasets": dataset_records,
            "rows": sum(record["rows"] for record in dataset_records),
            "unique_normalized_prompts": len(eval_hashes),
        },
        "exact_train_overlap_unique_prompts": len(overlaps),
        "exact_train_overlap_rows": sum(len(item["evaluation_rows"]) for item in overlaps),
        "duplicate_evaluation_unique_prompts": len(duplicate_eval),
        "overlaps": overlaps,
        "evaluation_duplicates": duplicate_eval,
    }
    output = args.output.resolve()
    write_yaml(output, report)
    print(
        f"Audited {report['evaluation']['rows']} evaluation prompts: "
        f"train_overlap={len(overlaps)}, eval_duplicates={len(duplicate_eval)}"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
