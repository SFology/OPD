from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from formal_eval_common import load_yaml, read_jsonl, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "verl"))
from verl.utils.reward_score.ttrl_math import compute_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze formal OPD/ROPD evaluation outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def pass_at_k(correct: int, total: int, k: int) -> float:
    if total < k:
        return float("nan")
    if correct <= 0:
        return 0.0
    if total - correct < k:
        return 1.0
    return 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def percentile_ci(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_mean(
    values_by_dataset: dict[str, np.ndarray],
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = []
        for values in values_by_dataset.values():
            chosen = rng.integers(0, len(values), size=len(values))
            selected.append(values[chosen])
        draws[index] = float(np.concatenate(selected).mean())
    return percentile_ci(draws, confidence)


def bootstrap_delta(
    delta_by_dataset: dict[str, np.ndarray],
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    return bootstrap_mean(
        delta_by_dataset,
        samples=samples,
        confidence=confidence,
        rng=rng,
    )


def named_rng(base_seed: int, *parts: object) -> np.random.Generator:
    """Return a stable RNG stream that does not depend on analysis loop order."""

    label = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([base_seed, derived]))


def bootstrap_gap_recovery(
    values_by_dataset: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
    minimum_teacher_gap: float = 1e-8,
) -> tuple[float, float, float]:
    """Bootstrap (trained - student) / (teacher - student) over paired prompts.

    Replicates with a non-positive teacher gap are not interpretable as recovery
    of a teacher advantage and are excluded while their fraction is reported.
    """

    draws = []
    for _ in range(samples):
        trained_parts = []
        teacher_parts = []
        student_parts = []
        for trained, teacher, student in values_by_dataset.values():
            chosen = rng.integers(0, len(student), size=len(student))
            trained_parts.append(trained[chosen])
            teacher_parts.append(teacher[chosen])
            student_parts.append(student[chosen])
        trained_mean = float(np.concatenate(trained_parts).mean())
        teacher_mean = float(np.concatenate(teacher_parts).mean())
        student_mean = float(np.concatenate(student_parts).mean())
        teacher_gap = teacher_mean - student_mean
        if teacher_gap > minimum_teacher_gap:
            draws.append((trained_mean - student_mean) / teacher_gap)
    valid_fraction = len(draws) / samples
    if not draws:
        return float("nan"), float("nan"), valid_fraction
    low, high = percentile_ci(np.asarray(draws, dtype=np.float64), confidence)
    return low, high, valid_fraction


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fmt_percent(value: float) -> str:
    return "—" if not math.isfinite(value) else f"{100.0 * value:.2f}%"


def html_table(rows: list[dict], columns: list[tuple[str, str]], percent: set[str]) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            if key in percent:
                rendered = fmt_percent(float(value))
            elif isinstance(value, float):
                rendered = f"{value:.3f}"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def load_outputs(run_dir: Path, config: dict) -> list[dict]:
    rows = []
    expected_n = int(config["generation"]["rollouts_per_prompt"])
    for model in config["models"]:
        for dataset in config["datasets"]:
            for rollout in range(expected_n):
                path = (
                    run_dir
                    / "generations"
                    / model["name"]
                    / dataset["name"]
                    / f"rollout_{rollout:03d}.jsonl"
                )
                rows.extend(read_jsonl(path))
    return rows


def grade_output(row: dict, fast: bool) -> dict:
    grade = compute_score(row["response"], row["ground_truth"], fast=fast)
    return {
        **row,
        "parsed": bool(grade["format_score"]),
        "correct": bool(grade["acc"]),
        "valid_complete": bool(grade["format_score"])
        and not bool(row["at_token_limit"]),
        "prediction": grade["pred"],
        "extraction_method": grade.get("extraction_method"),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = load_yaml(run_dir / "config.yaml")
    analysis = config["analysis"]
    confidence = float(analysis["confidence_level"])
    bootstrap_samples = int(analysis["bootstrap_samples"])
    bootstrap_seed = int(analysis["bootstrap_seed"])
    model_names = [item["name"] for item in config["models"]]
    dataset_names = [item["name"] for item in config["datasets"]]
    pass_ks = [int(item) for item in analysis["pass_k"]]
    rollouts_per_prompt = int(config["generation"]["rollouts_per_prompt"])

    raw_rows = load_outputs(run_dir, config)
    grader_fast = bool(analysis.get("grader_fast", True))
    grader_workers = int(analysis.get("grader_workers", 1))
    if grader_workers < 1:
        raise ValueError("analysis.grader_workers must be positive")
    if grader_workers == 1:
        graded_rows = [grade_output(row, grader_fast) for row in raw_rows]
    else:
        print(
            f"Grading {len(raw_rows)} outputs with {grader_workers} processes "
            f"(fast={grader_fast})",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=grader_workers) as executor:
            graded_rows = list(
                executor.map(grade_output, raw_rows, repeat(grader_fast), chunksize=16)
            )
    graded_rows.sort(
        key=lambda row: (row["model"], row["dataset"], row["example_id"], row["rollout"])
    )
    results_dir = run_dir / "results"
    write_jsonl(results_dir / "graded_outputs.jsonl", graded_rows)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in graded_rows:
        grouped[(row["model"], row["dataset"], row["prompt_id"])].append(row)

    prompt_rows = []
    for (model, dataset, prompt_id), rows in sorted(grouped.items()):
        correct = sum(bool(row["correct"]) for row in rows)
        parsed = sum(bool(row["parsed"]) for row in rows)
        valid_complete = sum(bool(row["valid_complete"]) for row in rows)
        item = {
            "model": model,
            "dataset": dataset,
            "prompt_id": prompt_id,
            "samples": len(rows),
            "correct": correct,
            "parsed": parsed,
            "accuracy": correct / len(rows),
            "parse_rate": parsed / len(rows),
            "valid_complete_rate": valid_complete / len(rows),
            "at_limit_rate": sum(bool(row["at_token_limit"]) for row in rows) / len(rows),
        }
        for k in pass_ks:
            item[f"pass_at_{k}"] = pass_at_k(correct, len(rows), k)
        prompt_rows.append(item)
    write_csv(results_dir / "prompt_summary.csv", prompt_rows)

    summaries = []
    for model in model_names:
        for dataset in dataset_names + ["ALL"]:
            selected_prompts = [
                row
                for row in prompt_rows
                if row["model"] == model and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            selected_outputs = [
                row
                for row in graded_rows
                if row["model"] == model and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            values_by_dataset = {
                name: np.asarray(
                    [row["accuracy"] for row in selected_prompts if row["dataset"] == name],
                    dtype=np.float64,
                )
                for name in dataset_names
                if any(row["dataset"] == name for row in selected_prompts)
            }
            accuracy = float(np.mean([row["accuracy"] for row in selected_prompts]))
            accuracy_low, accuracy_high = bootstrap_mean(
                values_by_dataset,
                samples=bootstrap_samples,
                confidence=confidence,
                rng=named_rng(bootstrap_seed, "summary", model, dataset, "accuracy"),
            )
            parsed = sum(bool(row["parsed"]) for row in selected_outputs)
            correct = sum(bool(row["correct"]) for row in selected_outputs)
            complete_outputs = [row for row in selected_outputs if row["valid_complete"]]
            complete_correct = sum(bool(row["correct"]) for row in complete_outputs)
            summary = {
                "model": model,
                "dataset": dataset,
                "prompts": len(selected_prompts),
                "samples": len(selected_outputs),
                "parsed": parsed,
                "correct": correct,
                "parse_rate": parsed / len(selected_outputs),
                "accuracy": accuracy,
                "accuracy_ci_low": accuracy_low,
                "accuracy_ci_high": accuracy_high,
                "accuracy_given_parsed": correct / parsed if parsed else float("nan"),
                "valid_complete_rate": len(complete_outputs) / len(selected_outputs),
                "accuracy_given_valid_complete": (
                    complete_correct / len(complete_outputs)
                    if complete_outputs
                    else float("nan")
                ),
                "at_limit_rate": float(np.mean([row["at_token_limit"] for row in selected_outputs])),
                "mean_response_tokens": float(np.mean([row["response_tokens"] for row in selected_outputs])),
            }
            for k in pass_ks:
                pass_values = {
                    name: np.asarray(
                        [
                            row[f"pass_at_{k}"]
                            for row in selected_prompts
                            if row["dataset"] == name
                        ],
                        dtype=np.float64,
                    )
                    for name in dataset_names
                    if any(row["dataset"] == name for row in selected_prompts)
                }
                pass_low, pass_high = bootstrap_mean(
                    pass_values,
                    samples=bootstrap_samples,
                    confidence=confidence,
                    rng=named_rng(bootstrap_seed, "summary", model, dataset, f"pass_at_{k}"),
                )
                summary[f"pass_at_{k}"] = float(
                    np.mean([row[f"pass_at_{k}"] for row in selected_prompts])
                )
                summary[f"pass_at_{k}_ci_low"] = pass_low
                summary[f"pass_at_{k}_ci_high"] = pass_high
            summaries.append(summary)
    write_csv(results_dir / "dataset_summary.csv", summaries)

    by_prompt = {
        (row["model"], row["dataset"], row["prompt_id"]): row for row in prompt_rows
    }
    comparisons = []
    for treatment, control in analysis["comparisons"]:
        for dataset in dataset_names + ["ALL"]:
            included = dataset_names if dataset == "ALL" else [dataset]
            comparison = {
                "treatment": treatment,
                "control": control,
                "dataset": dataset,
            }
            metric_names = ["accuracy", *(f"pass_at_{k}" for k in pass_ks)]
            prompt_count = 0
            for metric in metric_names:
                delta_by_dataset: dict[str, np.ndarray] = {}
                point_values = []
                treatment_better = 0
                equal = 0
                control_better = 0
                for name in included:
                    treatment_ids = {
                        key[2]
                        for key in by_prompt
                        if key[0] == treatment and key[1] == name
                    }
                    control_ids = {
                        key[2]
                        for key in by_prompt
                        if key[0] == control and key[1] == name
                    }
                    if treatment_ids != control_ids:
                        raise RuntimeError(
                            f"Unpaired prompt sets for {treatment} vs {control} on {name}"
                        )
                    prompt_ids = sorted(treatment_ids)
                    deltas = np.asarray(
                        [
                            by_prompt[(treatment, name, prompt_id)][metric]
                            - by_prompt[(control, name, prompt_id)][metric]
                            for prompt_id in prompt_ids
                        ],
                        dtype=np.float64,
                    )
                    delta_by_dataset[name] = deltas
                    point_values.extend(deltas.tolist())
                    treatment_better += int(np.sum(deltas > 1e-12))
                    equal += int(np.sum(np.abs(deltas) <= 1e-12))
                    control_better += int(np.sum(deltas < -1e-12))
                low, high = bootstrap_delta(
                    delta_by_dataset,
                    samples=bootstrap_samples,
                    confidence=confidence,
                    rng=named_rng(
                        bootstrap_seed,
                        "paired_delta",
                        treatment,
                        control,
                        dataset,
                        metric,
                    ),
                )
                comparison[f"{metric}_delta"] = float(np.mean(point_values))
                comparison[f"{metric}_delta_ci_low"] = low
                comparison[f"{metric}_delta_ci_high"] = high
                comparison[f"{metric}_treatment_better_prompts"] = treatment_better
                comparison[f"{metric}_equal_prompts"] = equal
                comparison[f"{metric}_control_better_prompts"] = control_better
                prompt_count = len(point_values)
            comparison["prompts"] = prompt_count
            comparisons.append(comparison)
    write_csv(results_dir / "paired_comparisons.csv", comparisons)

    gap_rows = []
    gap_config = analysis.get("gap_recovery") or {}
    if gap_config:
        student = str(gap_config["student"])
        teacher = str(gap_config["teacher"])
        for trained in gap_config["trained_models"]:
            for dataset in dataset_names + ["ALL"]:
                included = dataset_names if dataset == "ALL" else [dataset]
                values_by_dataset = {}
                for name in included:
                    prompt_ids = sorted(
                        key[2] for key in by_prompt if key[0] == student and key[1] == name
                    )
                    expected_ids = set(prompt_ids)
                    for model in (teacher, trained):
                        available = {
                            key[2] for key in by_prompt if key[0] == model and key[1] == name
                        }
                        if available != expected_ids:
                            raise RuntimeError(
                                f"Unpaired prompt sets for gap recovery: {model} on {name}"
                            )
                    values_by_dataset[name] = tuple(
                        np.asarray(
                            [by_prompt[(model, name, prompt_id)]["accuracy"] for prompt_id in prompt_ids],
                            dtype=np.float64,
                        )
                        for model in (trained, teacher, student)
                    )
                trained_values = np.concatenate([values[0] for values in values_by_dataset.values()])
                teacher_values = np.concatenate([values[1] for values in values_by_dataset.values()])
                student_values = np.concatenate([values[2] for values in values_by_dataset.values()])
                trained_delta = float((trained_values - student_values).mean())
                teacher_gap = float((teacher_values - student_values).mean())
                recovery = trained_delta / teacher_gap if teacher_gap > 1e-8 else float("nan")
                low, high, valid_fraction = bootstrap_gap_recovery(
                    values_by_dataset,
                    samples=bootstrap_samples,
                    confidence=confidence,
                    rng=named_rng(bootstrap_seed, "gap_recovery", trained, dataset),
                )
                gap_rows.append(
                    {
                        "trained_model": trained,
                        "student": student,
                        "teacher": teacher,
                        "dataset": dataset,
                        "prompts": len(student_values),
                        "trained_minus_student": trained_delta,
                        "teacher_minus_student": teacher_gap,
                        "gap_recovery": recovery,
                        "gap_recovery_ci_low": low,
                        "gap_recovery_ci_high": high,
                        "valid_bootstrap_fraction": valid_fraction,
                    }
                )
        write_csv(results_dir / "gap_recovery.csv", gap_rows)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    unique_prompt_count = len({row["prompt_id"] for row in graded_rows})
    width = min(0.18, 0.80 / max(len(model_names), 1))
    x = np.arange(len(dataset_names) + 1)
    offsets = (np.arange(len(model_names)) - (len(model_names) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(12, 6.8))
    for model_index, model in enumerate(model_names):
        values = [
            next(row for row in summaries if row["model"] == model and row["dataset"] == dataset)
            for dataset in dataset_names + ["ALL"]
        ]
        means = np.asarray([row["accuracy"] for row in values]) * 100.0
        errors = np.vstack(
            [
                means - np.asarray([row["accuracy_ci_low"] for row in values]) * 100.0,
                np.asarray([row["accuracy_ci_high"] for row in values]) * 100.0 - means,
            ]
        )
        ax.bar(x + offsets[model_index], means, width, yerr=errors, capsize=3, label=model)
    ax.set_xticks(x, dataset_names + [f"All {unique_prompt_count} prompts"])
    ax.set_ylabel(f"Average correctness across {rollouts_per_prompt} rollouts (%)")
    ax.set_xlabel("Held-out dataset")
    ax.set_title("Held-out average correctness\nPrompt-bootstrap 95% confidence intervals")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(4, len(model_names)))
    fig.subplots_adjust(bottom=0.25, top=0.86)
    fig.savefig(figures_dir / "accuracy_comparison.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for model in model_names:
        row = next(item for item in summaries if item["model"] == model and item["dataset"] == "ALL")
        means = np.asarray([100.0 * row[f"pass_at_{k}"] for k in pass_ks])
        lows = np.asarray([100.0 * row[f"pass_at_{k}_ci_low"] for k in pass_ks])
        highs = np.asarray([100.0 * row[f"pass_at_{k}_ci_high"] for k in pass_ks])
        line = ax.plot(pass_ks, means, marker="o", label=model)[0]
        ax.fill_between(pass_ks, lows, highs, color=line.get_color(), alpha=0.14)
    ax.set_xticks(pass_ks)
    ax.set_xlabel("k sampled continuations per prompt")
    ax.set_ylabel(f"pass@k across {unique_prompt_count} prompts (%)")
    ax.set_title("Held-out pass@k comparison\nPrompt-bootstrap 95% confidence bands")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "pass_at_k.svg", format="svg")
    plt.close(fig)

    plotted_pairs = {
        tuple(pair) for pair in analysis.get("plot_comparisons", analysis["comparisons"])
    }
    all_comparisons = [
        row
        for row in comparisons
        if row["dataset"] == "ALL"
        and (row["treatment"], row["control"]) in plotted_pairs
    ]
    fig, ax = plt.subplots(figsize=(10.5, max(5.8, 0.72 * len(all_comparisons) + 2.6)))
    labels = [f"{row['treatment']} − {row['control']}" for row in all_comparisons]
    points = np.asarray([row["accuracy_delta"] for row in all_comparisons]) * 100.0
    lows = np.asarray([row["accuracy_delta_ci_low"] for row in all_comparisons]) * 100.0
    highs = np.asarray([row["accuracy_delta_ci_high"] for row in all_comparisons]) * 100.0
    positions = np.arange(len(labels))
    ax.errorbar(points, positions, xerr=np.vstack([points - lows, highs - points]), fmt="o", capsize=4)
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Paired change in average correctness (percentage points)")
    ax.set_title(
        f"Method effects over all {unique_prompt_count} prompts\n"
        "Prompt-bootstrap 95% confidence intervals"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.36, bottom=0.16, top=0.84)
    fig.savefig(figures_dir / "paired_accuracy_deltas.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for row in all_comparisons:
        means = np.asarray([100.0 * row[f"pass_at_{k}_delta"] for k in pass_ks])
        lows = np.asarray([100.0 * row[f"pass_at_{k}_delta_ci_low"] for k in pass_ks])
        highs = np.asarray([100.0 * row[f"pass_at_{k}_delta_ci_high"] for k in pass_ks])
        label = f"{row['treatment']} − {row['control']}"
        line = ax.plot(pass_ks, means, marker="o", label=label)[0]
        ax.fill_between(pass_ks, lows, highs, color=line.get_color(), alpha=0.12)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(pass_ks)
    ax.set_xlabel("k sampled continuations per prompt")
    ax.set_ylabel("Paired pass@k change (percentage points)")
    ax.set_title("Paired pass@k effects\nPrompt-bootstrap 95% confidence intervals")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "paired_pass_at_k_deltas.svg", format="svg")
    plt.close(fig)

    if gap_rows:
        overall_gap_rows = [row for row in gap_rows if row["dataset"] == "ALL"]
        fig, ax = plt.subplots(figsize=(9.5, max(5.4, 0.8 * len(overall_gap_rows) + 2.8)))
        positions = np.arange(len(overall_gap_rows))
        points = np.asarray([100.0 * row["gap_recovery"] for row in overall_gap_rows])
        lows = np.asarray([100.0 * row["gap_recovery_ci_low"] for row in overall_gap_rows])
        highs = np.asarray([100.0 * row["gap_recovery_ci_high"] for row in overall_gap_rows])
        ax.errorbar(
            points,
            positions,
            xerr=np.vstack([points - lows, highs - points]),
            fmt="o",
            capsize=4,
        )
        ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
        ax.axvline(100.0, color="#4c78a8", linewidth=1, linestyle=":")
        ax.set_yticks(positions, [row["trained_model"] for row in overall_gap_rows])
        ax.set_xlabel("Recovered teacher–student accuracy gap (%)")
        ax.set_title("Teacher–student gap recovery\nPaired prompt-bootstrap 95% confidence intervals")
        ax.grid(axis="x", alpha=0.25)
        fig.subplots_adjust(left=0.28, bottom=0.16, top=0.84)
        fig.savefig(figures_dir / "gap_recovery.svg", format="svg")
        plt.close(fig)

    overall_rows = [row for row in summaries if row["dataset"] == "ALL"]
    dashboard = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>OPD / ROPD 正式评测</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;line-height:1.55;color:#202124}}
table{{border-collapse:collapse;width:100%;margin:14px 0 28px}}th,td{{border:1px solid #ddd;padding:7px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{width:100%;border:1px solid #eee;margin:8px 0 20px}}
.note{{background:#f5f7fa;padding:14px 18px;border-left:4px solid #4c78a8}}</style></head><body>
<h1>{html.escape(str(config.get('experiment', {}).get('name', 'OPD / ROPD formal evaluation')))}</h1>
<div class="note">Average correctness 是每题 {rollouts_per_prompt} 次采样正确率的题目平均；置信区间按独立题目 bootstrap。
pass@k 衡量每题采样 k 次至少一次正确的概率估计。解析失败保留在总样本 accuracy 的分母中；达到 token
上限的输出也保留在固定预算主指标中，同时单独报告“可解析且未达到上限”子集，后者只用于工程有效性敏感性检查。</div>
<h2>总体结果</h2>
{html_table(overall_rows, [('model','Model'),('prompts','Prompts'),('samples','Generations'),('parse_rate','Parse rate'),('accuracy','Avg correctness'),('accuracy_ci_low','95% CI low'),('accuracy_ci_high','95% CI high'),('accuracy_given_parsed','Accuracy | parsed'),('valid_complete_rate','Parsed & non-limit'),('accuracy_given_valid_complete','Accuracy | parsed & non-limit'),('at_limit_rate','At-limit rate'),('mean_response_tokens','Mean response tokens')], {'parse_rate','accuracy','accuracy_ci_low','accuracy_ci_high','accuracy_given_parsed','valid_complete_rate','accuracy_given_valid_complete','at_limit_rate'})}
<img src="accuracy_comparison.svg" alt="Accuracy comparison">
<p>该图用于比较 {len(model_names)} 个模型/checkpoint 在每个数据集以及全部 {unique_prompt_count} 道题上的平均正确率。误差线反映题目之间的不确定性，而不是把同题 {rollouts_per_prompt} 次 rollout 当成独立题目。</p>
<img src="paired_accuracy_deltas.svg" alt="Paired deltas">
<p>差值图是主要方法比较：正值表示前一个模型更好；只有 ROPD−OPD 的置信区间稳定远离 0，才能支持本轮 ROPD 相对 OPD 的性能提升。</p>
<img src="pass_at_k.svg" alt="Pass at k">
<p>pass@k 展示增加采样预算后的解题覆盖率，不应与单次回答正确率混淆。</p>
<img src="paired_pass_at_k_deltas.svg" alt="Paired pass at k deltas">
<p>配对 pass@k 曲线直接展示方法差值及其区间；某个 k 的点估计为正但区间跨过 0 时，只能视为探索性趋势。</p>
<h2>逐数据集结果</h2>
{html_table(summaries, [('model','Model'),('dataset','Dataset'),('prompts','Prompts'),('parse_rate','Parse rate'),('valid_complete_rate','Parsed & non-limit'),('accuracy','Avg correctness'),('accuracy_ci_low','95% CI low'),('accuracy_ci_high','95% CI high'), *[(f'pass_at_{k}',f'pass@{k}') for k in pass_ks]], {'parse_rate','valid_complete_rate','accuracy','accuracy_ci_low','accuracy_ci_high', *(f'pass_at_{k}' for k in pass_ks)})}
<h2>配对差值</h2>
{html_table(comparisons, [('treatment','Treatment'),('control','Control'),('dataset','Dataset'),('prompts','Prompts'),('accuracy_delta','Accuracy delta'),('accuracy_delta_ci_low','95% CI low'),('accuracy_delta_ci_high','95% CI high')], {'accuracy_delta','accuracy_delta_ci_low','accuracy_delta_ci_high'})}
<h2>配对 pass@k 差值</h2>
{html_table(all_comparisons, [('treatment','Treatment'),('control','Control'),('prompts','Prompts'), *[(f'pass_at_{k}_delta',f'pass@{k} delta') for k in pass_ks]], {f'pass_at_{k}_delta' for k in pass_ks})}
{('<h2>教师—学生差距恢复率</h2><img src="gap_recovery.svg" alt="Gap recovery"><p>恢复率定义为 (训练后−初始学生)/(教师−初始学生)。教师优势接近零或为负时该比值不可解释，因此同时报告有效 bootstrap 比例。</p>' + html_table(gap_rows, [('trained_model','Trained model'),('dataset','Dataset'),('prompts','Prompts'),('trained_minus_student','Trained−student'),('teacher_minus_student','Teacher−student'),('gap_recovery','Gap recovery'),('gap_recovery_ci_low','95% CI low'),('gap_recovery_ci_high','95% CI high'),('valid_bootstrap_fraction','Valid bootstrap')], {'trained_minus_student','teacher_minus_student','gap_recovery','gap_recovery_ci_low','gap_recovery_ci_high','valid_bootstrap_fraction'})) if gap_rows else ''}
</body></html>"""
    (figures_dir / "dashboard.html").write_text(dashboard, encoding="utf-8")
    write_json(
        results_dir / "summary.json",
        {
            "models": model_names,
            "datasets": dataset_names,
            "unique_prompts": unique_prompt_count,
            "generations": len(graded_rows),
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": confidence,
            "grader_fast": grader_fast,
            "grader_workers": grader_workers,
            "overall": overall_rows,
            "paired_comparisons": all_comparisons,
            "paired_comparisons_by_dataset": comparisons,
            "gap_recovery": gap_rows,
        },
    )
    print(f"Analysis complete: {figures_dir / 'dashboard.html'}", flush=True)


if __name__ == "__main__":
    main()
