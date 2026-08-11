from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(
                f"Environment variable {name} is required by the configuration"
            )
        return os.environ[name]

    return ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    return expand_env(config)


def save_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(config: dict[str, Any], config_path: Path) -> Path:
    root = Path(config["experiment"]["output_root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{stamp}_{config['experiment']['name']}"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    for name in ("artifacts", "features", "logs", "results"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "config.yaml", config)
    save_yaml(
        run_dir / "status.yaml",
        {
            "status": "collecting",
            "created_at_utc": utc_now(),
            "source_config": str(config_path.resolve()),
        },
    )
    return run_dir


def load_run(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(run_dir).resolve()
    return path, load_config(path / "config.yaml")


def update_status(run_dir: Path, status: str, **extra: Any) -> None:
    current_path = run_dir / "status.yaml"
    current = (
        yaml.safe_load(current_path.read_text(encoding="utf-8"))
        if current_path.exists()
        else {}
    )
    current.update({"status": status, "updated_at_utc": utc_now(), **extra})
    save_yaml(current_path, current)


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def tokenizer_fingerprint(tokenizer: Any) -> str:
    digest = hashlib.sha256()
    for token, token_id in sorted(
        tokenizer.get_vocab().items(), key=lambda item: item[1]
    ):
        digest.update(str(token_id).encode())
        digest.update(b"\0")
        digest.update(token.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_model_and_tokenizer(model_path: str, config: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype(config["models"]["dtype"]),
        attn_implementation=config["models"].get("attention_implementation", "sdpa"),
    )
    device = torch.device(config["models"]["device"])
    model.to(device)
    model.eval()
    return model, tokenizer, device


def load_math_grader():
    """Accept both DAPO ``Answer:`` and boxed-answer output conventions."""
    sys.path.insert(0, str(REPO_ROOT / "verl"))
    from verl.utils.reward_score.math_dapo import compute_score as grade_dapo
    from verl.utils.reward_score.ttrl_math import compute_score as grade_boxed

    def grade(response: str, ground_truth: str) -> dict[str, Any]:
        dapo_result = grade_dapo(response, ground_truth)
        if dapo_result["acc"]:
            return dapo_result
        boxed_result = grade_boxed(response, ground_truth, fast=True)
        if boxed_result["acc"] or boxed_result.get("pred"):
            return boxed_result
        return dapo_result

    return grade


def resolve_hidden_indices(layer_specs: list[Any], num_hidden_layers: int) -> list[int]:
    indices: list[int] = []
    for spec in layer_specs:
        if spec == "embedding":
            index = 0
        elif isinstance(spec, float) and 0.0 <= spec <= 1.0:
            index = max(1, min(num_hidden_layers, round(spec * num_hidden_layers)))
        elif isinstance(spec, int):
            index = spec if spec >= 0 else num_hidden_layers + 1 + spec
        else:
            raise ValueError(f"Invalid layer specification: {spec!r}")
        if not 0 <= index <= num_hidden_layers:
            raise ValueError(
                f"Resolved hidden-state index {index} is outside [0, {num_hidden_layers}]"
            )
        indices.append(index)
    return indices


def pool_hidden(
    hidden: torch.Tensor, attention_mask: torch.Tensor, definition: dict[str, Any]
) -> torch.Tensor:
    pooling = definition["pooling"]
    if pooling == "last_token":
        return hidden[:, -1, :]
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    if pooling == "prefix_mean":
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    if pooling == "tail_mean":
        tail = int(definition.get("tail_tokens", 8))
        tail_hidden = hidden[:, -tail:, :]
        tail_mask = mask[:, -tail:, :]
        return (tail_hidden * tail_mask).sum(dim=1) / tail_mask.sum(dim=1).clamp_min(1)
    raise ValueError(f"Unsupported pooling: {pooling}")


def representation_names(config: dict[str, Any]) -> list[str]:
    return [item["name"] for item in config["representations"]["definitions"]]


def choose_state_positions(
    length: int, fractions: list[float], minimum: int
) -> list[int]:
    if length <= minimum:
        return []
    positions = {
        min(length - 1, max(minimum, round(length * fraction)))
        for fraction in fractions
    }
    return sorted(position for position in positions if 0 <= position < length)


def cosine_distances(anchor: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    anchor_norm = anchor / max(float(np.linalg.norm(anchor)), 1e-12)
    candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    normalized = candidates / np.maximum(candidate_norms, 1e-12)
    return 1.0 - normalized @ anchor_norm


def aggregate_changes(
    values: np.ndarray, top_tail_count: int, cvar_quantile: float
) -> dict[str, float]:
    if values.size == 0:
        return {
            "max": float("nan"),
            "q95": float("nan"),
            "top_mean": float("nan"),
            "cvar": float("nan"),
        }
    descending = np.sort(values)[::-1]
    cutoff = float(np.quantile(values, cvar_quantile))
    tail = values[values >= cutoff]
    return {
        "max": float(descending[0]),
        "q95": float(np.quantile(values, 0.95)),
        "top_mean": float(
            descending[: max(1, min(top_tail_count, len(descending)))].mean()
        ),
        "cvar": float(tail.mean()),
    }
