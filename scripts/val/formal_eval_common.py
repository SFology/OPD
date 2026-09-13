from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

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
            raise ValueError(f"Environment variable {name} is required")
        return os.environ[name]

    return ENV_PATTERN.sub(replace, value)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return expand_env(value)


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_status(
    run_dir: Path,
    status: str,
    *,
    clear_fields: Iterable[str] = (),
    **extra: Any,
) -> None:
    path = run_dir / "status.yaml"
    current = load_yaml(path) if path.exists() else {}
    for field in clear_fields:
        current.pop(field, None)
    current.update({"status": status, "updated_at_utc": utc_now(), **extra})
    write_yaml(path, current)


def generation_path(run_dir: Path, model: str, dataset: str, rollout: int) -> Path:
    return (
        run_dir
        / "generations"
        / model
        / dataset
        / f"rollout_{rollout:03d}.jsonl"
    )


def parse_shard(value: str) -> tuple[str, int]:
    dataset, rollout = value.rsplit(":", 1)
    return dataset, int(rollout)


def expected_seed(config: dict[str, Any], dataset_index: int, example_id: int, rollout: int) -> int:
    base = int(config["generation"]["seed_base"])
    return base + dataset_index * 1_000_000 + rollout * 10_000 + example_id


def validate_generation_shard(
    path: Path,
    *,
    model: str,
    dataset: str,
    rollout: int,
    expected_prompts: int,
    expected_prompt_ids: set[str] | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return False
    if len(rows) != expected_prompts:
        return False
    keys = {(row.get("prompt_id"), row.get("rollout")) for row in rows}
    return (
        len(keys) == expected_prompts
        and (
            expected_prompt_ids is None
            or {str(row.get("prompt_id")) for row in rows} == expected_prompt_ids
        )
        and all(row.get("model") == model for row in rows)
        and all(row.get("dataset") == dataset for row in rows)
        and all(int(row.get("rollout", -1)) == rollout for row in rows)
    )
