#!/usr/bin/env python3
"""Summarize reward, gradient, and real checkpoint updates for a paired probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from audit_opd_reproduction import (
    checkpoint_precision_audit,
    latest_checkpoint,
    parse_training_metrics,
)

EXPECTED_ROLES = ("opd", "scaled_opd", "ropd", "normalized_ropd")
MEAN_METRICS = (
    "ropd/training_opd_token_rms",
    "ropd/training_ropd_token_rms",
    "ropd/training_selected_token_rms",
    "ropd/training_reward_scale",
    "ropd/trust_mean",
    "ropd/zero_trust_fraction",
    "ropd/zero_neighbor_fraction",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_runs(path: Path) -> dict[str, Path]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    runs = {row["role"]: Path(row["run_dir"]).resolve() for row in rows}
    missing = [role for role in EXPECTED_ROLES if role not in runs]
    if missing:
        raise ValueError(f"run manifest is missing roles: {missing}")
    return runs


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def summarize_run(role: str, run_dir: Path) -> dict[str, Any]:
    status = yaml.safe_load((run_dir / "status.yaml").read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("exit_code") != 0:
        raise ValueError(f"{role} is not a completed successful run: {run_dir}")
    metric_rows = [
        json.loads(line)
        for line in (run_dir / "metrics" / "ropd_step_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    training_rows = parse_training_metrics(run_dir / "logs" / "train.log")
    gradients = [row["actor/grad_norm"] for row in training_rows if "actor/grad_norm" in row]
    if not metric_rows or not gradients:
        raise ValueError(f"{role} is missing reward or gradient metrics")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "role": role,
        "run_dir": str(run_dir),
        "step_count": len(metric_rows),
        "actor_grad_norm_mean": statistics.fmean(gradients),
        "actor_grad_norm_min": min(gradients),
        "actor_grad_norm_max": max(gradients),
        "initial_model": str(Path(config["models"]["actor_path"]).resolve()),
    }
    summary.update({key: mean(metric_rows, key) for key in MEAN_METRICS})
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force-checkpoint-audit", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    output_dir = (args.output_dir or manifest.parent / "probe_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(manifest)
    summaries = [summarize_run(role, runs[role]) for role in EXPECTED_ROLES]

    for summary in summaries:
        role = summary["role"]
        cache = output_dir / "parameter_audits" / f"{role}.json"
        if cache.is_file() and not args.force_checkpoint_audit:
            audit = json.loads(cache.read_text(encoding="utf-8"))
            if audit.get("actor_dir") != str(latest_checkpoint(runs[role])):
                raise ValueError(f"stale parameter-audit cache: {cache}")
            print(f"Reusing parameter audit: {role}", flush=True)
        else:
            print(f"Auditing representative checkpoint updates: {role}", flush=True)
            audit = checkpoint_precision_audit(
                latest_checkpoint(runs[role]),
                Path(summary["initial_model"]),
                include_optimizer=False,
            )
            atomic_json(cache, audit)
        aggregate = audit["representative_parameter_aggregate"]
        summary.update(
            {
                "representative_changed_fraction": aggregate["changed_fraction"],
                "representative_delta_rms": aggregate["delta_rms"],
                "representative_delta_l2": aggregate["delta_l2"],
                "representative_relative_delta_l2": aggregate["relative_delta_l2"],
            }
        )

    baseline = next(row for row in summaries if row["role"] == "opd")
    for summary in summaries:
        for metric in (
            "actor_grad_norm_mean",
            "ropd/training_selected_token_rms",
            "representative_delta_rms",
            "representative_relative_delta_l2",
        ):
            denominator = float(baseline[metric])
            summary[f"{metric}_relative_to_opd"] = (
                float(summary[metric]) / denominator if denominator else math.nan
            )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "scope": "five-step probe; representative checkpoint tensors, not held-out correctness",
        "runs": summaries,
    }
    atomic_json(output_dir / "summary.json", report)
    write_csv(output_dir / "summary.csv", summaries)

    lines = [
        "# FP32 update-matched probe",
        "",
        "This five-step diagnostic does not measure final task effectiveness.",
        "",
        "| arm | selected RMS | grad norm | grad / OPD | parameter delta RMS | delta / OPD | zero trust |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['role']} | {row['ropd/training_selected_token_rms']:.6g} | "
            f"{row['actor_grad_norm_mean']:.6g} | "
            f"{row['actor_grad_norm_mean_relative_to_opd']:.3f} | "
            f"{row['representative_delta_rms']:.6g} | "
            f"{row['representative_delta_rms_relative_to_opd']:.3f} | "
            f"{row['ropd/zero_trust_fraction']:.2%} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REPORT={output_dir / 'report.md'}")
    print(f"SUMMARY_JSON={output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
