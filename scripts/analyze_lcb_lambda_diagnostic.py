#!/usr/bin/env python3
"""Aggregate full-state counterfactual LCB lambda diagnostics from a managed run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def slug(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p").replace("+", "")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def summarize_lambda_grid(
    metric_rows: list[dict[str, Any]], lambdas: list[float]
) -> tuple[float, list[dict[str, Any]]]:
    if not metric_rows:
        raise ValueError("diagnostic run has no step metrics")
    fields = (
        "trust_mean",
        "zero_trust_fraction",
        "supported_trust_mean",
        "supported_zero_trust_fraction",
        "selected_token_rms",
        "effective_abs_reward_mass_fraction",
        "unsupported_selected_abs_mass_fraction",
    )
    summaries = []
    opd_rms = mean(metric_rows, "ropd/training_opd_token_rms")
    for value in lambdas:
        prefix = f"ropd/cf_lambda_{slug(value)}"
        item: dict[str, Any] = {"lambda": value, "step_count": len(metric_rows)}
        for field in fields:
            values = [float(row[f"{prefix}/{field}"]) for row in metric_rows]
            item[field] = statistics.fmean(values)
            item[f"{field}_step_std"] = statistics.pstdev(values)
        item["selected_token_rms_relative_to_opd"] = item["selected_token_rms"] / opd_rms
        summaries.append(item)
    return opd_rms, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "results" / "lcb_lambda_diagnostic").resolve()
    status = yaml.safe_load((run_dir / "status.yaml").read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("exit_code") != 0:
        raise ValueError(f"diagnostic run is not complete: {run_dir}")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    lambdas = [float(value) for value in config["distillation"]["robust_opd"]["diagnostic_lambdas"]]
    metric_rows = [
        json.loads(line)
        for line in (run_dir / "metrics" / "ropd_step_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    opd_rms, summaries = summarize_lambda_grid(metric_rows, lambdas)

    if 1.0 in lambdas:
        one = summaries[lambdas.index(1.0)]
        observed_rms = mean(metric_rows, "ropd/training_ropd_token_rms")
        observed_trust = mean(metric_rows, "ropd/trust_mean")
        if abs(one["selected_token_rms"] - observed_rms) > 1e-5:
            raise AssertionError("lambda=1 counterfactual RMS does not reproduce the configured LCB")
        if abs(one["trust_mean"] - observed_trust) > 1e-5:
            raise AssertionError("lambda=1 counterfactual trust does not reproduce the configured LCB")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "step_count": len(metric_rows),
        "opd_token_rms_mean": opd_rms,
        "warning": "The lambda grid measures signal retention only; it does not validate teacher reliability.",
        "lambda_summaries": summaries,
    }
    atomic_json(output_dir / "summary.json", report)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Full-state LCB lambda diagnostic",
        "",
        "This table measures signal retention on an OPD trajectory. It does not select lambda by correctness.",
        "",
        "| lambda | supported trust | supported zero | selected RMS / OPD | effective abs mass | unsupported share |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['lambda']:.6g} | {row['supported_trust_mean']:.2%} | "
            f"{row['supported_zero_trust_fraction']:.2%} | "
            f"{row['selected_token_rms_relative_to_opd']:.3f} | "
            f"{row['effective_abs_reward_mass_fraction']:.2%} | "
            f"{row['unsupported_selected_abs_mass_fraction']:.2%} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REPORT={output_dir / 'report.md'}")
    print(f"SUMMARY_JSON={output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
