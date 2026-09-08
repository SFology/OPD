from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from common import read_jsonl, write_jsonl

GROUP_NAMES = {
    (True, False): "teacher_correct_student_wrong",
    (False, True): "teacher_wrong_student_correct",
    (False, False): "both_wrong",
    (True, True): "both_correct",
}


def stable_int(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def logical_shard(identifier: str, count: int) -> int:
    return stable_int(identifier) % count


def parse_shards(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def load_sharded(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("shard_*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def eligible_generation(row: dict[str, Any], token_limit: int) -> bool:
    return len(row.get("generated_token_ids", [])) < token_limit and row.get(
        "predicted_answer"
    ) not in (None, "", "[INVALID]")


def fraction_name(value: float) -> str:
    return f"q{round(100 * value):02d}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
