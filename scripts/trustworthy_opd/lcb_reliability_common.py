from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def point_id(base_trajectory_id: str, position: int) -> str:
    return f"{base_trajectory_id}@{int(position)}"


def request_id(base_trajectory_id: str, position: int, action_token_id: int) -> str:
    payload = f"{base_trajectory_id}|{int(position)}|{int(action_token_id)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def candidate_positions(
    *,
    anchor_position: int,
    anchor_length: int,
    target_length: int,
    stride: int,
    offsets: int,
    progress_window: float,
) -> list[int]:
    """Mirror the online dense-discrete candidate geometry for one anchor."""

    if min(anchor_length, target_length, stride) <= 0:
        return []
    anchor_progress = anchor_position / max(anchor_length - 1, 1)
    sampled = np.arange(0, target_length, stride, dtype=np.int64)
    center = round(anchor_progress * max(target_length - 1, 1) / stride)
    result: list[int] = []
    for local_index in range(center - offsets, center + offsets + 1):
        if not 0 <= local_index < len(sampled):
            continue
        position = int(sampled[local_index])
        candidate_progress = position / max(target_length - 1, 1)
        if abs(candidate_progress - anchor_progress) <= progress_window:
            result.append(position)
    return result


def rollout_is_available(
    anchor_rollout: int, candidate_rollout: int, rollout_block_size: int | None
) -> bool:
    if anchor_rollout == candidate_rollout:
        return False
    if rollout_block_size is None:
        return True
    return (
        anchor_rollout // rollout_block_size == candidate_rollout // rollout_block_size
    )


def ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks


def select_dual_ball_neighbors(
    *,
    anchor_student: np.ndarray,
    anchor_teacher: np.ndarray,
    candidate_student: np.ndarray,
    candidate_teacher: np.ndarray,
    radius_quantile: float,
    neighbor_k: int,
    minimum_neighbors: int,
    max_student_distance: float | None,
    max_teacher_distance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(candidate_student):
        empty = np.asarray([], dtype=np.int64)
        return empty, empty.astype(np.float32), empty.astype(np.float32)
    anchor_student = anchor_student.astype(np.float32)
    anchor_teacher = anchor_teacher.astype(np.float32)
    candidate_student = candidate_student.astype(np.float32)
    candidate_teacher = candidate_teacher.astype(np.float32)
    anchor_student /= max(float(np.linalg.norm(anchor_student)), 1e-12)
    anchor_teacher /= max(float(np.linalg.norm(anchor_teacher)), 1e-12)
    candidate_student /= np.maximum(
        np.linalg.norm(candidate_student, axis=1, keepdims=True), 1e-12
    )
    candidate_teacher /= np.maximum(
        np.linalg.norm(candidate_teacher, axis=1, keepdims=True), 1e-12
    )
    student_distance = 1.0 - candidate_student @ anchor_student
    teacher_distance = 1.0 - candidate_teacher @ anchor_teacher
    keep = max(1, math.ceil(len(student_distance) * radius_quantile))
    valid = (ordinal_ranks(student_distance) < keep) & (
        ordinal_ranks(teacher_distance) < keep
    )
    if max_student_distance is not None:
        valid &= student_distance <= max_student_distance
    if max_teacher_distance is not None:
        valid &= teacher_distance <= max_teacher_distance
    candidates = np.flatnonzero(valid)
    if len(candidates) < minimum_neighbors:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty.astype(np.float32), empty.astype(np.float32)
    joint = np.maximum(student_distance[candidates], teacher_distance[candidates])
    chosen = candidates[np.argsort(joint, kind="stable")[:neighbor_k]]
    return chosen, student_distance[chosen], teacher_distance[chosen]


def aggregate_risk(
    deviations: Iterable[float], method: str, temperature: float
) -> float:
    values = np.asarray(list(deviations), dtype=np.float64)
    if not len(values):
        return float("nan")
    if method == "max":
        return float(values.max())
    if method != "softmax":
        raise ValueError(f"Unsupported risk aggregation: {method}")
    shifted = values / temperature
    shifted -= shifted.max()
    weights = np.exp(shifted)
    weights /= weights.sum()
    return float(np.dot(weights, values))


def lcb_trust(anchor_reward: float, risk: float, value: float, epsilon: float) -> float:
    if not np.isfinite(risk):
        return 1.0
    magnitude = abs(anchor_reward)
    if magnitude <= epsilon:
        return 0.0
    return float(np.clip((magnitude - value * risk) / magnitude, 0.0, 1.0))


def stable_fold(prompt_index: int, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}|{prompt_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def source_path(source_run: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else source_run / path


def validate_config(config: dict[str, Any]) -> None:
    definitions = config["representations"]["definitions"]
    names = [item["name"] for item in definitions]
    if len(names) != len(set(names)):
        raise ValueError("Representation names must be unique")
    supported = {"projected_token_embedding", "contextual"}
    if any(item["kind"] not in supported for item in definitions):
        raise ValueError("Unsupported representation kind")
    neighborhood = config["neighborhood"]
    if int(neighborhood["minimum_neighbors"]) > int(neighborhood["neighbor_k"]):
        raise ValueError("minimum_neighbors must not exceed neighbor_k")
    variants = neighborhood["variants"]
    variant_names = [item["name"] for item in variants]
    if len(variant_names) != len(set(variant_names)):
        raise ValueError("Neighborhood variant names must be unique")
    lambdas = [float(value) for value in config["risk"]["lambdas"]]
    if len(lambdas) != len(set(lambdas)) or any(value < 0 for value in lambdas):
        raise ValueError("risk.lambdas must be unique and non-negative")
