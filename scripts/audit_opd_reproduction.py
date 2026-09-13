#!/usr/bin/env python3
"""Audit an OPD managed run against the paper recipe and detect inert updates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

TRACKED_METRICS = (
    "val-topk/overlap_ratio",
    "ropd/opd_token_reward_mean",
    "actor/entropy",
    "teacher/entropy",
    "actor/grad_norm",
    "actor/pg_loss",
    "critic/true_reward/mean",
    "response_length/clip_ratio",
)

REPRESENTATIVE_PARAMETERS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.10.mlp.down_proj.weight",
    "model.layers.27.self_attn.o_proj.weight",
    "model.norm.weight",
)


def nested(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def paper_contract(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return configuration deviations, separating scientific and execution parity."""

    expected = (
        ("models.dtype", "fp32", "scientific"),
        ("data.shuffle", False, "scientific"),
        ("data.max_prompt_length", 1024, "scientific"),
        ("data.max_response_length", 7168, "scientific"),
        ("optimization.learning_rate", 1e-6, "scientific"),
        ("optimization.mini_batch_size", 64, "scientific"),
        ("optimization.loss_agg_mode", "token-mean", "scientific"),
        ("rollout.n", 4, "scientific"),
        ("rollout.temperature", 1.0, "scientific"),
        ("distillation.advantage_estimator", "token_reward_direct", "scientific"),
        ("distillation.log_prob_top_k", 16, "scientific"),
        ("distillation.top_k_strategy", "only_stu", "scientific"),
        ("distillation.reward_weight_mode", "student_p", "scientific"),
        ("distillation.teacher_temperature", 1.0, "scientific"),
        ("distillation.use_kl", False, "scientific"),
        ("trainer.total_epochs", 1, "scientific"),
        ("trainer.n_gpus_per_node", 8, "execution"),
        ("optimization.max_tokens_per_gpu", None, "execution"),
        ("optimization.param_offload", False, "execution"),
        ("optimization.optimizer_offload", False, "execution"),
        ("reward_model.micro_batch_size_per_gpu", 24, "execution"),
        ("reward_model.param_offload", False, "execution"),
        ("rollout.gpu_memory_utilization", 0.8, "execution"),
    )
    deviations = []
    for path, target, category in expected:
        actual = nested(config, path)
        if actual != target:
            deviations.append(
                {"path": path, "expected": target, "actual": actual, "category": category}
            )

    robust = nested(config, "distillation.robust_opd")
    if isinstance(robust, dict) and robust.get("enabled", False):
        deviations.append(
            {
                "path": "distillation.robust_opd.enabled",
                "expected": False,
                "actual": True,
                "category": "instrumentation",
                "note": "Even apply_to_training=false adds a non-upstream measurement path.",
            }
        )
    return deviations


def parse_training_metrics(path: Path) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            match = re.search(r"(?:^|\s)step:(\d+)\s+-\s+", line)
            if not match:
                continue
            row: dict[str, float] = {"step": float(match.group(1))}
            for field in line[match.end() :].split(" - "):
                key, separator, raw_value = field.partition(":")
                raw_value = raw_value.strip()
                if separator and key in TRACKED_METRICS and FLOAT_RE.match(raw_value):
                    row[key] = float(raw_value)
            if len(row) > 1:
                by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def summarize_metrics(rows: list[dict[str, float]], window: int = 20) -> dict[str, Any]:
    result: dict[str, Any] = {
        "steps": len(rows),
        "first_step": int(rows[0]["step"]) if rows else None,
        "last_step": int(rows[-1]["step"]) if rows else None,
        "metrics": {},
    }
    for key in TRACKED_METRICS:
        pairs = [(row["step"], row[key]) for row in rows if key in row]
        if not pairs:
            continue
        size = min(window, len(pairs))
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)
        denominator = sum((value - x_mean) ** 2 for value in x_values)
        slope = (
            sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator
            if denominator
            else math.nan
        )
        early = sum(y_values[:size]) / size
        late = sum(y_values[-size:]) / size
        result["metrics"][key] = {
            "first": y_values[0],
            "last": y_values[-1],
            "early_window_mean": early,
            "late_window_mean": late,
            "late_minus_early": late - early,
            "linear_slope_per_100_steps": slope * 100,
            "minimum": min(y_values),
            "maximum": max(y_values),
            "window": size,
        }
    return result


def latest_checkpoint(run_dir: Path) -> Path:
    pointer = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    if not pointer.is_file():
        raise FileNotFoundError(f"checkpoint pointer not found: {pointer}")
    step = int(pointer.read_text(encoding="utf-8").strip())
    path = run_dir / "checkpoints" / f"global_step_{step}" / "actor"
    if not path.is_dir():
        raise FileNotFoundError(f"actor checkpoint not found: {path}")
    return path


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local().detach().cpu() if callable(to_local) else value.detach().cpu()


def tensor_inventory(value: Any, counter: Counter[str]) -> None:
    if isinstance(value, torch.Tensor):
        counter[str(value.dtype)] += value.numel()
    elif isinstance(value, dict):
        for child in value.values():
            tensor_inventory(child, counter)
    elif isinstance(value, (list, tuple)):
        for child in value:
            tensor_inventory(child, counter)


def safetensor_file_for(model_dir: Path, name: str) -> Path:
    single = model_dir / "model.safetensors"
    if single.is_file():
        return single
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no safetensors model found in {model_dir}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return model_dir / index["weight_map"][name]


def checkpoint_precision_audit(actor_dir: Path, initial_model: Path) -> dict[str, Any]:
    model_files = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    optimizer_files = sorted(actor_dir.glob("optim_world_size_*_rank_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"model rank files not found in {actor_dir}")

    selected_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in REPRESENTATIVE_PARAMETERS
    }
    model_numel_by_dtype: Counter[str] = Counter()
    for rank_index, path in enumerate(model_files):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if rank_index == 0:
            tensor_inventory(state, model_numel_by_dtype)
        for name in REPRESENTATIVE_PARAMETERS:
            if name in state:
                selected_parts[name].append(local_tensor(state[name]).clone())
        del state
        gc.collect()

    changes: dict[str, Any] = {}
    for name, parts in selected_parts.items():
        if not parts:
            continue
        checkpoint_tensor = torch.cat(parts, dim=0)
        source_path = safetensor_file_for(initial_model, name)
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            source_tensor = handle.get_tensor(name)
        checkpoint_tensor = checkpoint_tensor[: source_tensor.shape[0]]
        if checkpoint_tensor.shape != source_tensor.shape:
            changes[name] = {
                "error": f"shape mismatch: {tuple(checkpoint_tensor.shape)} vs {tuple(source_tensor.shape)}"
            }
            continue
        changed = checkpoint_tensor != source_tensor
        delta = (checkpoint_tensor.float() - source_tensor.float()).abs()
        changes[name] = {
            "numel": source_tensor.numel(),
            "changed_elements": int(changed.sum().item()),
            "changed_fraction": changed.float().mean().item(),
            "absolute_delta_mean": delta.mean().item(),
            "absolute_delta_max": delta.max().item(),
        }

    optimizer_numel_by_dtype: Counter[str] = Counter()
    if optimizer_files:
        optimizer = torch.load(optimizer_files[0], map_location="cpu", weights_only=False)
        tensor_inventory(optimizer, optimizer_numel_by_dtype)
        del optimizer
        gc.collect()

    return {
        "actor_dir": str(actor_dir),
        "model_rank_files": len(model_files),
        "optimizer_rank_files": len(optimizer_files),
        "rank0_model_numel_by_dtype": dict(model_numel_by_dtype),
        "rank0_optimizer_numel_by_dtype": dict(optimizer_numel_by_dtype),
        "representative_parameter_changes": changes,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Run: {report['run_dir']}")
    print(f"Status: {report['status'].get('status', 'unknown')}")
    print("\nPaper-contract deviations:")
    deviations = report["paper_contract_deviations"]
    if not deviations:
        print("  none")
    for item in deviations:
        print(
            f"  [{item['category']}] {item['path']}: "
            f"actual={item['actual']!r}, expected={item['expected']!r}"
        )

    dynamics = report["training_dynamics"]
    print(f"\nTraining dynamics: {dynamics['steps']} steps")
    for key, item in dynamics["metrics"].items():
        print(
            f"  {key}: first={item['first']:.6g}, last={item['last']:.6g}, "
            f"late-early={item['late_minus_early']:+.6g}, "
            f"slope/100={item['linear_slope_per_100_steps']:+.6g}"
        )

    precision = report.get("checkpoint_precision")
    if precision:
        print("\nCheckpoint precision:")
        print(f"  model numel by dtype (rank 0): {precision['rank0_model_numel_by_dtype']}")
        print(f"  optimizer numel by dtype (rank 0): {precision['rank0_optimizer_numel_by_dtype']}")
        for name, item in precision["representative_parameter_changes"].items():
            if "error" in item:
                print(f"  {name}: {item['error']}")
            else:
                print(
                    f"  {name}: changed={item['changed_fraction']:.4%}, "
                    f"mean_abs_delta={item['absolute_delta_mean']:.3g}, "
                    f"max_abs_delta={item['absolute_delta_max']:.3g}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--checkpoint-audit", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    status_path = run_dir / "status.yaml"
    status = yaml.safe_load(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    rows = parse_training_metrics(run_dir / "logs" / "train.log")
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "status": status,
        "paper_contract_deviations": paper_contract(config),
        "training_dynamics": summarize_metrics(rows),
    }
    if args.checkpoint_audit:
        initial_model = args.initial_model or Path(config["models"]["actor_path"])
        report["checkpoint_precision"] = checkpoint_precision_audit(
            latest_checkpoint(run_dir), initial_model.resolve()
        )

    print_report(report)
    if args.json_out:
        output = args.json_out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(f"\nJSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
