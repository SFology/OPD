# Copyright 2026 Individual Contributor: SFology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dense discrete trajectory neighborhoods for robust on-policy distillation.

The implementation deliberately uses real student-generated prefix states. It
does not create continuous embedding perturbations. The legacy hard-min method
compares actions present in both Top-K sets. The LCB method instead requests the
exact log-probability of every anchor's sampled action at its neighbor states,
then uses neighborhood instability only to shrink the anchor OPD reward.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


@dataclass
class DenseDiscreteROPDResult:
    """Result returned by :func:`compute_dense_discrete_ropd`."""

    ropd_scores: torch.Tensor
    metrics: dict[str, float]
    samples: list[dict[str, Any]]


@dataclass
class DenseDiscreteLCBSupport:
    """Neighbor geometry and reverse-scattered sampled-action requests."""

    request_ids: torch.Tensor
    neighbor_rows: torch.Tensor
    neighbor_positions: torch.Tensor
    request_slots: torch.Tensor
    neighbor_valid: torch.Tensor
    student_distances: torch.Tensor
    teacher_distances: torch.Tensor
    metrics: dict[str, float]


def validate_dense_discrete_config(config: dict[str, Any]) -> None:
    if config.get("method", "dense_discrete") != "dense_discrete":
        raise ValueError("robust_opd.method must be 'dense_discrete'")
    aggregation = config.get("aggregation", "hard_min")
    if aggregation not in {"hard_min", "lcb_gate"}:
        raise ValueError("robust_opd.aggregation must be 'hard_min' or 'lcb_gate'")
    if config.get("projection_method", "count_sketch") != "count_sketch":
        raise ValueError("Only projection_method='count_sketch' is implemented")
    if config.get("representation", "token_embedding_tail_mean") not in {
        "token_embedding_tail_mean",
        "token_embedding_prefix_mean",
    }:
        raise ValueError("Unsupported dense ROPD representation")
    integer_options = {
        "projection_dim": 32,
        "tail_tokens": 16,
        "candidate_stride": 16,
        "candidate_offsets": 4,
        "neighbor_k": 3,
        "minimum_neighbors": 1,
        "anchor_chunk_size": 256,
        "sample_records_per_step": 64,
    }
    for name, default in integer_options.items():
        value = int(config.get(name, default))
        minimum = 0 if name == "sample_records_per_step" else 1
        if value < minimum:
            raise ValueError(f"robust_opd.{name} must be >= {minimum}")
    if int(config.get("minimum_neighbors", 1)) > int(config.get("neighbor_k", 3)):
        raise ValueError("robust_opd.minimum_neighbors must not exceed neighbor_k")
    progress_window = float(config.get("progress_window", 0.03))
    if not 0.0 <= progress_window <= 1.0:
        raise ValueError("robust_opd.progress_window must be in [0, 1]")
    radius_quantile = float(config.get("radius_quantile", 0.25))
    if not 0.0 < radius_quantile <= 1.0:
        raise ValueError("robust_opd.radius_quantile must be in (0, 1]")
    for name in ("max_teacher_cosine_distance", "max_student_cosine_distance"):
        value = config.get(name)
        if value is not None and not 0.0 <= float(value) <= 2.0:
            raise ValueError(f"robust_opd.{name} must be null or in [0, 2]")
    if aggregation == "lcb_gate":
        if config.get("action_evaluation", "sampled_token_exact") != "sampled_token_exact":
            raise ValueError("LCB ROPD requires action_evaluation='sampled_token_exact'")
        if config.get("risk_aggregation", "softmax") not in {"softmax", "max"}:
            raise ValueError("robust_opd.risk_aggregation must be 'softmax' or 'max'")
        if float(config.get("risk_temperature", 1.0)) <= 0:
            raise ValueError("robust_opd.risk_temperature must be positive")
        if float(config.get("lcb_lambda", 1.0)) < 0:
            raise ValueError("robust_opd.lcb_lambda must be non-negative")
        if float(config.get("lcb_epsilon", 1e-6)) <= 0:
            raise ValueError("robust_opd.lcb_epsilon must be positive")
        if int(config.get("max_request_slots", 128)) < 1:
            raise ValueError("robust_opd.max_request_slots must be positive")


def _model_fingerprint(model_path: Path, projection_dim: int, seed: int, projection_method: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(model_path.resolve()).encode())
    for name in ("config.json", "model.safetensors", "model.safetensors.index.json"):
        path = model_path / name
        if path.is_file():
            stat = path.stat()
            digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(f"projection_dim={projection_dim};seed={seed};projection_method={projection_method}".encode())
    return digest.hexdigest()[:20]


def _embedding_tensor_location(model_path: Path) -> tuple[Path, str]:
    single = model_path / "model.safetensors"
    key = "model.embed_tokens.weight"
    if single.is_file():
        return single, key
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard = index["weight_map"].get(key)
        if shard is None:
            raise KeyError(f"{key} is absent from {index_path}")
        return model_path / shard, key
    raise FileNotFoundError(f"No safetensors weights found under {model_path}")


def load_projected_embedding_table(
    model_path: str | Path,
    cache_root: str | Path,
    projection_dim: int,
    seed: int,
    projection_method: str = "count_sketch",
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Load or build a deterministic projection of input embeddings.

    The returned CPU float16 table is small enough for fast prefix feature
    construction.  The full model is never instantiated on the trainer driver.
    CountSketch is linear in the original table size, unlike a dense Gaussian
    matrix multiplication, and preserves inner products in expectation.
    """

    model_path = Path(model_path)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    if projection_method != "count_sketch":
        raise ValueError("Only projection_method='count_sketch' is implemented")
    fingerprint = _model_fingerprint(model_path, projection_dim, seed, projection_method)
    cache_path = cache_root / f"{model_path.name}-{fingerprint}.pt"
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        table = payload["table"]
        if table.ndim != 2 or table.shape[1] != projection_dim:
            raise RuntimeError(f"Invalid projected embedding cache: {cache_path}")
        return table

    weights_path, key = _embedding_tensor_location(model_path)
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor(key)
    hidden_size = int(embedding.shape[1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    buckets = torch.randint(projection_dim, (hidden_size,), generator=generator)
    signs = torch.randint(0, 2, (hidden_size,), generator=generator).float().mul_(2).sub_(1)
    rows = []
    for start in range(0, len(embedding), chunk_size):
        chunk = embedding[start : start + chunk_size].float()
        projected = torch.zeros(len(chunk), projection_dim, dtype=torch.float32)
        projected.scatter_add_(1, buckets.expand(len(chunk), -1), chunk * signs)
        rows.append(projected.to(torch.float16))
    table = torch.cat(rows, dim=0)
    payload = {
        "table": table,
        "model_path": str(model_path.resolve()),
        "fingerprint": fingerprint,
        "projection_dim": projection_dim,
        "seed": seed,
        "projection_method": projection_method,
    }
    temporary = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    return table


def build_prefix_state_features(
    prompts: torch.Tensor,
    responses: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    representation: str,
    tail_tokens: int,
) -> torch.Tensor:
    """Build a model-specific feature for every response action state.

    State t contains the prompt and response tokens strictly before response[t].
    Padding is excluded.  Features are L2-normalized for cosine distance.
    """

    prompts = prompts.detach().cpu().long()
    responses = responses.detach().cpu().long()
    attention_mask = attention_mask.detach().cpu().bool()
    response_mask = response_mask.detach().cpu().bool()
    batch_size, response_length = responses.shape
    prompt_length = prompts.shape[1]
    projection_dim = embedding_table.shape[1]
    result = torch.zeros(batch_size, response_length, projection_dim, dtype=torch.float32)
    prompt_mask = attention_mask[:, :prompt_length]

    for row in range(batch_size):
        prompt_ids = prompts[row][prompt_mask[row]]
        valid_length = int(response_mask[row].sum().item())
        if valid_length == 0:
            continue
        context_ids = torch.cat([prompt_ids, responses[row, :valid_length]])
        vectors = embedding_table.index_select(0, context_ids).float()
        cumulative = torch.cat([torch.zeros(1, projection_dim, dtype=torch.float32), vectors.cumsum(dim=0)], dim=0)
        prompt_count = len(prompt_ids)
        end = prompt_count + torch.arange(valid_length)
        if representation == "token_embedding_prefix_mean":
            start = torch.zeros_like(end)
        else:
            start = (end - tail_tokens).clamp_min(0)
        counts = (end - start).clamp_min(1).unsqueeze(-1)
        pooled = (cumulative[end] - cumulative[start]) / counts
        result[row, :valid_length] = F.normalize(pooled, p=2, dim=-1)
    return result


def _rank_with_invalid_last(distances: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = distances.masked_fill(~valid, float("inf"))
    order = torch.argsort(masked, dim=1, stable=True)
    ranks = torch.empty_like(order)
    ordinal = torch.arange(order.shape[1], device=order.device).expand_as(order)
    ranks.scatter_(1, order, ordinal)
    return ranks


def _evenly_spaced_indices(num_values: int, sample_count: int, device: torch.device) -> torch.Tensor:
    """Return endpoint-inclusive spaced indices using integer arithmetic only."""
    if sample_count == 1:
        return torch.tensor([num_values // 2], device=device, dtype=torch.int64)
    positions = torch.arange(sample_count, device=device, dtype=torch.int64)
    return torch.div(
        positions * (num_values - 1),
        sample_count - 1,
        rounding_mode="floor",
    )


def _quantiles(values: torch.Tensor, max_quantile_samples: int = 1_000_000) -> dict[str, float]:
    """Summarize a tensor without exceeding CUDA's quantile input limit.

    Mean and standard deviation remain exact.  Quantiles are diagnostic-only,
    so large tensors use a deterministic, evenly spaced sample.  In
    particular, dense ROPD's action-level tensor can contain tens of millions
    of values (batch x response x top-k), while CUDA ``torch.quantile`` rejects
    inputs larger than 2**24 elements.
    """
    if values.numel() == 0:
        return {name: float("nan") for name in ("mean", "std", "p05", "p50", "p95")}
    values = values.float().reshape(-1)
    quantile_values = values
    if values.numel() > max_quantile_samples:
        # linspace avoids allocating a permutation as large as ``values`` and
        # makes diagnostic metrics reproducible across resumed runs.
        indices = _evenly_spaced_indices(values.numel(), max_quantile_samples, values.device)
        quantile_values = values[indices]
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p05": float(torch.quantile(quantile_values, 0.05)),
        "p50": float(torch.quantile(quantile_values, 0.50)),
        "p95": float(torch.quantile(quantile_values, 0.95)),
    }


def prepare_dense_discrete_lcb_support(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    uids: np.ndarray | list[Any],
    config: dict[str, Any],
) -> DenseDiscreteLCBSupport:
    """Select dual-ball neighbors and build exact sampled-action requests.

    A directed anchor-to-neighbor edge asks both models to evaluate the
    anchor's sampled action at the neighbor state. Requests are reverse
    scattered to target states and deduplicated by ``(state, token_id)`` so a
    single additional student forward and the existing teacher forward can
    score all selected edges.
    """

    validate_dense_discrete_config(config)
    if config.get("aggregation") != "lcb_gate":
        raise ValueError("Exact sampled-action support is only used by aggregation='lcb_gate'")
    responses = responses.detach().cpu().long()
    mask = response_mask.detach().cpu().bool()
    student_features = F.normalize(student_features.detach().cpu().float(), p=2, dim=-1)
    teacher_features = F.normalize(teacher_features.detach().cpu().float(), p=2, dim=-1)
    if responses.shape != mask.shape:
        raise ValueError("responses and response_mask must have identical shapes")
    if student_features.shape[:2] != mask.shape or teacher_features.shape[:2] != mask.shape:
        raise ValueError("State features do not match response_mask")

    batch_size, response_length = mask.shape
    neighbor_k = int(config.get("neighbor_k", 3))
    neighbor_rows = torch.zeros(batch_size, response_length, neighbor_k, dtype=torch.long)
    neighbor_positions = torch.zeros_like(neighbor_rows)
    neighbor_valid = torch.zeros_like(neighbor_rows, dtype=torch.bool)
    student_distances = torch.full_like(neighbor_rows, float("nan"), dtype=torch.float32)
    teacher_distances = torch.full_like(neighbor_rows, float("nan"), dtype=torch.float32)

    group_rows: dict[str, list[int]] = {}
    for row, uid in enumerate(uids):
        group_rows.setdefault(str(uid), []).append(row)

    stride = int(config.get("candidate_stride", 16))
    offsets = int(config.get("candidate_offsets", 4))
    progress_window = float(config.get("progress_window", 0.03))
    radius_quantile = float(config.get("radius_quantile", 0.25))
    minimum_neighbors = int(config.get("minimum_neighbors", 1))
    chunk_size = int(config.get("anchor_chunk_size", 256))
    max_teacher_distance = config.get("max_teacher_cosine_distance")
    max_student_distance = config.get("max_student_cosine_distance")
    max_teacher_distance = None if max_teacher_distance is None else float(max_teacher_distance)
    max_student_distance = None if max_student_distance is None else float(max_student_distance)
    candidate_count = 0
    dual_ball_count = 0

    for rows in group_rows.values():
        if len(rows) < 2:
            continue
        lengths = {row: int(mask[row].sum()) for row in rows}
        for anchor_row in rows:
            anchor_length = lengths[anchor_row]
            for chunk_start in range(0, anchor_length, chunk_size):
                anchor_positions = torch.arange(
                    chunk_start, min(chunk_start + chunk_size, anchor_length), dtype=torch.long
                )
                anchor_progress = anchor_positions.float() / max(anchor_length - 1, 1)
                candidate_rows_parts = []
                candidate_positions_parts = []
                candidate_valid_parts = []
                for other_row in rows:
                    other_length = lengths[other_row]
                    if other_row == anchor_row or other_length == 0:
                        continue
                    sampled_positions = torch.arange(0, other_length, stride, dtype=torch.long)
                    centers = torch.round(anchor_progress * max(other_length - 1, 1) / stride).long()
                    local_indices = centers[:, None] + torch.arange(-offsets, offsets + 1)[None, :]
                    index_valid = (local_indices >= 0) & (local_indices < len(sampled_positions))
                    safe_indices = local_indices.clamp(0, max(len(sampled_positions) - 1, 0))
                    positions = sampled_positions[safe_indices]
                    candidate_progress = positions.float() / max(other_length - 1, 1)
                    progress_valid = (candidate_progress - anchor_progress[:, None]).abs() <= progress_window
                    candidate_rows_parts.append(torch.full_like(positions, other_row))
                    candidate_positions_parts.append(positions)
                    candidate_valid_parts.append(index_valid & progress_valid)
                if not candidate_rows_parts:
                    continue

                candidate_rows = torch.cat(candidate_rows_parts, dim=1)
                candidate_positions = torch.cat(candidate_positions_parts, dim=1)
                candidate_valid = torch.cat(candidate_valid_parts, dim=1)
                candidate_student = student_features[candidate_rows, candidate_positions]
                candidate_teacher = teacher_features[candidate_rows, candidate_positions]
                anchor_student = student_features[anchor_row, anchor_positions]
                anchor_teacher = teacher_features[anchor_row, anchor_positions]
                student_distance = 1.0 - (anchor_student[:, None, :] * candidate_student).sum(dim=-1)
                teacher_distance = 1.0 - (anchor_teacher[:, None, :] * candidate_teacher).sum(dim=-1)

                valid_counts = candidate_valid.sum(dim=1)
                keep_count = torch.ceil(valid_counts.float() * radius_quantile).long().clamp_min(1)
                dual_valid = (
                    candidate_valid
                    & (_rank_with_invalid_last(student_distance, candidate_valid) < keep_count[:, None])
                    & (_rank_with_invalid_last(teacher_distance, candidate_valid) < keep_count[:, None])
                )
                if max_student_distance is not None:
                    dual_valid &= student_distance <= max_student_distance
                if max_teacher_distance is not None:
                    dual_valid &= teacher_distance <= max_teacher_distance
                candidate_count += int(candidate_valid.sum())
                dual_ball_count += int(dual_valid.sum())

                joint_distance = torch.maximum(student_distance, teacher_distance).masked_fill(
                    ~dual_valid, float("inf")
                )
                selected_k = min(neighbor_k, joint_distance.shape[1])
                selected_distance, selected_index = torch.topk(
                    joint_distance, k=selected_k, dim=1, largest=False, sorted=True
                )
                selected_valid = torch.isfinite(selected_distance)
                selected_valid &= (selected_valid.sum(dim=1) >= minimum_neighbors)[:, None]
                destination = (anchor_row, anchor_positions, slice(0, selected_k))
                neighbor_rows[destination] = candidate_rows.gather(1, selected_index)
                neighbor_positions[destination] = candidate_positions.gather(1, selected_index)
                neighbor_valid[destination] = selected_valid
                selected_student = student_distance.gather(1, selected_index)
                selected_teacher = teacher_distance.gather(1, selected_index)
                student_distances[destination] = selected_student.masked_fill(~selected_valid, float("nan"))
                teacher_distances[destination] = selected_teacher.masked_fill(~selected_valid, float("nan"))

    valid_edges = neighbor_valid.nonzero(as_tuple=False)
    request_slots = torch.zeros_like(neighbor_rows)
    if len(valid_edges) == 0:
        request_ids = torch.zeros(batch_size, response_length, 1, dtype=torch.long)
        unique_request_count = 0
        max_slots = 1
    else:
        anchor_rows = valid_edges[:, 0]
        anchor_positions = valid_edges[:, 1]
        target_rows = neighbor_rows[neighbor_valid]
        target_positions = neighbor_positions[neighbor_valid]
        action_ids = responses[anchor_rows, anchor_positions]
        target_flat = target_rows * response_length + target_positions
        vocabulary_size = int(max(int(responses.max()) + 1, 1))
        keys = target_flat * vocabulary_size + action_ids
        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        unique_targets = torch.div(unique_keys, vocabulary_size, rounding_mode="floor")
        unique_actions = torch.remainder(unique_keys, vocabulary_size)
        request_counts = torch.bincount(unique_targets, minlength=batch_size * response_length)
        max_slots = max(int(request_counts.max()), 1)
        configured_max = int(config.get("max_request_slots", 128))
        if max_slots > configured_max:
            raise RuntimeError(
                f"Exact neighbor requests need {max_slots} slots at one state, exceeding "
                f"robust_opd.max_request_slots={configured_max}; increase the explicit limit"
            )
        group_starts = torch.cumsum(request_counts, dim=0) - request_counts
        unique_slots = torch.arange(len(unique_keys)) - group_starts[unique_targets]
        request_ids = torch.zeros(batch_size * response_length, max_slots, dtype=torch.long)
        request_ids[unique_targets, unique_slots] = unique_actions
        request_ids = request_ids.view(batch_size, response_length, max_slots)
        request_slots[neighbor_valid] = unique_slots[inverse]
        unique_request_count = len(unique_keys)

    valid_state_count = int(mask.sum())
    edge_count = int(neighbor_valid.sum())
    valid_student_distances = student_distances[neighbor_valid]
    valid_teacher_distances = teacher_distances[neighbor_valid]
    metrics = {
        "ropd/states": float(valid_state_count),
        "ropd/groups": float(len(group_rows)),
        "ropd/groups_with_multiple_rollouts": float(sum(len(rows) > 1 for rows in group_rows.values())),
        "ropd/neighbor_count_mean": float(neighbor_valid.sum(dim=-1)[mask].float().mean())
        if valid_state_count
        else 0.0,
        "ropd/zero_neighbor_fraction": float((neighbor_valid.sum(dim=-1)[mask] == 0).float().mean())
        if valid_state_count
        else 1.0,
        "ropd/action_neighbor_weighted_coverage": float((neighbor_valid.any(dim=-1)[mask]).float().mean())
        if valid_state_count
        else 0.0,
        "ropd/dual_ball_candidate_fraction": dual_ball_count / max(candidate_count, 1),
        "ropd/exact_request_edges": float(edge_count),
        "ropd/exact_request_unique": float(unique_request_count),
        "ropd/exact_request_dedup_fraction": 1.0 - unique_request_count / max(edge_count, 1),
        "ropd/exact_request_max_slots": float(max_slots),
        "ropd/selected_student_distance_mean": float(valid_student_distances.mean()) if edge_count else float("nan"),
        "ropd/selected_teacher_distance_mean": float(valid_teacher_distances.mean()) if edge_count else float("nan"),
        "ropd/selected_student_distance_std": float(valid_student_distances.std(unbiased=False))
        if edge_count
        else float("nan"),
        "ropd/selected_teacher_distance_std": float(valid_teacher_distances.std(unbiased=False))
        if edge_count
        else float("nan"),
    }
    return DenseDiscreteLCBSupport(
        request_ids=request_ids,
        neighbor_rows=neighbor_rows,
        neighbor_positions=neighbor_positions,
        request_slots=request_slots,
        neighbor_valid=neighbor_valid,
        student_distances=student_distances,
        teacher_distances=teacher_distances,
        metrics=metrics,
    )


def compute_dense_discrete_lcb(
    *,
    sampled_student_log_probs: torch.Tensor,
    sampled_teacher_log_probs: torch.Tensor,
    neighbor_student_log_probs: torch.Tensor,
    neighbor_teacher_log_probs: torch.Tensor,
    opd_raw_rewards: torch.Tensor,
    opd_reward_weights: torch.Tensor,
    response_mask: torch.Tensor,
    responses: torch.Tensor,
    uids: np.ndarray | list[Any],
    support: DenseDiscreteLCBSupport,
    config: dict[str, Any],
    step: int = 0,
) -> DenseDiscreteROPDResult:
    """Apply sign-preserving LCB shrinkage using exact sampled-action risk."""

    validate_dense_discrete_config(config)
    sampled_student = sampled_student_log_probs.detach().cpu().float()
    sampled_teacher = sampled_teacher_log_probs.detach().cpu().float()
    neighbor_student = neighbor_student_log_probs.detach().cpu().float()
    neighbor_teacher = neighbor_teacher_log_probs.detach().cpu().float()
    raw_opd = opd_raw_rewards.detach().cpu().float()
    weights = opd_reward_weights.detach().cpu().float()
    mask = response_mask.detach().cpu().bool()
    responses = responses.detach().cpu().long()
    if sampled_student.shape != mask.shape or sampled_teacher.shape != mask.shape:
        raise ValueError("Sampled-action log-probs must match response_mask")
    if raw_opd.shape != weights.shape or raw_opd.shape[:2] != mask.shape:
        raise ValueError("Top-K OPD rewards and weights have inconsistent shapes")
    if neighbor_student.shape != support.request_ids.shape or neighbor_teacher.shape != support.request_ids.shape:
        raise ValueError("Exact neighbor log-probs do not match prepared request IDs")

    rows = support.neighbor_rows
    positions = support.neighbor_positions
    slots = support.request_slots
    valid = support.neighbor_valid
    neighbor_student_selected = neighbor_student[rows, positions, slots]
    neighbor_teacher_selected = neighbor_teacher[rows, positions, slots]
    neighbor_rewards = neighbor_teacher_selected - neighbor_student_selected
    anchor_reward = sampled_teacher - sampled_student
    deviations = (neighbor_rewards - anchor_reward.unsqueeze(-1)).abs()
    deviations = deviations.masked_fill(~valid, float("-inf"))
    has_neighbor = valid.any(dim=-1) & mask

    risk_method = config.get("risk_aggregation", "softmax")
    if risk_method == "max":
        risk = deviations.max(dim=-1).values
    else:
        temperature = float(config.get("risk_temperature", 1.0))
        risk_weights = torch.softmax(deviations / temperature, dim=-1)
        risk_weights = torch.nan_to_num(risk_weights, nan=0.0)
        risk = (risk_weights * deviations.masked_fill(~valid, 0.0)).sum(dim=-1)
    risk = torch.where(has_neighbor, risk, torch.zeros_like(risk))

    lcb_lambda = float(config.get("lcb_lambda", 1.0))
    robust_magnitude = (anchor_reward.abs() - lcb_lambda * risk).clamp_min(0.0)
    sampled_lcb = anchor_reward.sign() * robust_magnitude
    epsilon = float(config.get("lcb_epsilon", 1e-6))
    trust = torch.where(anchor_reward.abs() > epsilon, robust_magnitude / anchor_reward.abs(), torch.zeros_like(risk))
    trust = torch.where(has_neighbor, trust, torch.ones_like(trust)).clamp(0.0, 1.0)
    robust_scores = raw_opd * weights * trust.unsqueeze(-1)
    original_scores = raw_opd * weights

    token_opd = original_scores.sum(dim=-1)[mask]
    token_lcb = robust_scores.sum(dim=-1)[mask]
    valid_risk = risk[mask]
    valid_trust = trust[mask]
    sampled_anchor = anchor_reward[mask]
    sampled_robust = sampled_lcb[mask]
    metrics = dict(support.metrics)
    metrics.update(
        {
            "ropd/changed_action_fraction": float((valid_trust < 1.0 - 1e-8).float().mean()),
            "ropd/changed_token_fraction": float((valid_trust < 1.0 - 1e-8).float().mean()),
            "ropd/zero_trust_fraction": float((valid_trust <= 1e-8).float().mean()),
            "ropd/weighted_action_mass": float(weights[mask.unsqueeze(-1).expand_as(weights)].sum()),
            "ropd/lcb_lambda": lcb_lambda,
        }
    )
    for prefix, values in (
        ("opd_token_reward", token_opd),
        ("ropd_token_reward", token_lcb),
        ("sampled_opd_reward", sampled_anchor),
        ("sampled_lcb_reward", sampled_robust),
        ("neighborhood_risk", valid_risk),
        ("trust", valid_trust),
        ("absolute_reward_reduction", token_opd.abs() - token_lcb.abs()),
    ):
        metrics.update({f"ropd/{prefix}_{name}": value for name, value in _quantiles(values).items()})
    if torch.any(robust_scores[mask].abs() > original_scores[mask].abs() + 1e-6):
        raise AssertionError("LCB shrinkage increased an OPD reward magnitude")

    sample_limit = int(config.get("sample_records_per_step", 64))
    valid_states = mask.nonzero(as_tuple=False)
    if len(valid_states) > sample_limit > 0:
        sample_indices = _evenly_spaced_indices(len(valid_states), sample_limit, valid_states.device)
        valid_states = valid_states[sample_indices]
    samples = []
    for row, position in valid_states.tolist() if sample_limit > 0 else []:
        edge_valid = valid[row, position]
        edge_rewards = neighbor_rewards[row, position]
        edge_student_logp = neighbor_student_selected[row, position]
        edge_teacher_logp = neighbor_teacher_selected[row, position]
        worst_index = int(torch.argmax(deviations[row, position])) if bool(edge_valid.any()) else -1
        record = {
            "step": step,
            "uid": str(uids[row]),
            "anchor_row": row,
            "anchor_position": position,
            "action_token_id": int(responses[row, position]),
            "anchor_student_log_prob": float(sampled_student[row, position]),
            "anchor_teacher_log_prob": float(sampled_teacher[row, position]),
            "sampled_opd_reward": float(anchor_reward[row, position]),
            "sampled_lcb_reward": float(sampled_lcb[row, position]),
            "opd_token_reward": float(original_scores[row, position].sum()),
            "lcb_token_reward": float(robust_scores[row, position].sum()),
            "neighborhood_risk": float(risk[row, position]),
            "trust": float(trust[row, position]),
            "neighbor_count": int(edge_valid.sum()),
            "neighbor_student_log_probs": edge_student_logp[edge_valid].tolist(),
            "neighbor_teacher_log_probs": edge_teacher_logp[edge_valid].tolist(),
            "neighbor_rewards": edge_rewards[edge_valid].tolist(),
            "worst_neighbor_row": -1,
            "worst_neighbor_position": -1,
            "teacher_cosine_distance": float("nan"),
            "student_cosine_distance": float("nan"),
        }
        if worst_index >= 0:
            record.update(
                {
                    "worst_neighbor_row": int(rows[row, position, worst_index]),
                    "worst_neighbor_position": int(positions[row, position, worst_index]),
                    "teacher_cosine_distance": float(support.teacher_distances[row, position, worst_index]),
                    "student_cosine_distance": float(support.student_distances[row, position, worst_index]),
                }
            )
        samples.append(record)
    return DenseDiscreteROPDResult(ropd_scores=robust_scores, metrics=metrics, samples=samples)


def compute_dense_discrete_ropd(
    *,
    student_top_k_ids: torch.Tensor,
    student_top_k_log_probs: torch.Tensor,
    teacher_on_student_log_probs: torch.Tensor,
    opd_reward_weights: torch.Tensor,
    response_mask: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    uids: np.ndarray | list[Any],
    config: dict[str, Any],
    step: int = 0,
) -> DenseDiscreteROPDResult:
    """Compute hard-min ROPD over dense cross-rollout trajectory states."""

    validate_dense_discrete_config(config)
    ids = student_top_k_ids.detach().cpu().long()
    student_logp = student_top_k_log_probs.detach().cpu().float()
    teacher_logp = teacher_on_student_log_probs.detach().cpu().float()
    weights = opd_reward_weights.detach().cpu().float()
    mask = response_mask.detach().cpu().bool()
    student_features = F.normalize(student_features.detach().cpu().float(), p=2, dim=-1)
    teacher_features = F.normalize(teacher_features.detach().cpu().float(), p=2, dim=-1)
    if ids.shape != student_logp.shape or ids.shape != teacher_logp.shape or ids.shape != weights.shape:
        raise ValueError("Top-K ids, log-probs, and weights must have identical shapes")
    if ids.shape[:2] != mask.shape:
        raise ValueError("response_mask does not match Top-K tensors")
    if student_features.shape[:2] != mask.shape or teacher_features.shape[:2] != mask.shape:
        raise ValueError("State features do not match response_mask")

    raw_opd = teacher_logp - student_logp
    ropd_raw = raw_opd.clone()
    neighbor_count = torch.zeros_like(mask, dtype=torch.int16)
    action_has_neighbor = torch.zeros_like(ids, dtype=torch.bool)
    group_rows: dict[str, list[int]] = {}
    for row, uid in enumerate(uids):
        group_rows.setdefault(str(uid), []).append(row)

    stride = int(config.get("candidate_stride", 16))
    offsets = int(config.get("candidate_offsets", 4))
    progress_window = float(config.get("progress_window", 0.03))
    radius_quantile = float(config.get("radius_quantile", 0.25))
    neighbor_k = int(config.get("neighbor_k", 3))
    minimum_neighbors = int(config.get("minimum_neighbors", 1))
    chunk_size = int(config.get("anchor_chunk_size", 256))
    max_teacher_distance = config.get("max_teacher_cosine_distance")
    max_student_distance = config.get("max_student_cosine_distance")
    max_teacher_distance = None if max_teacher_distance is None else float(max_teacher_distance)
    max_student_distance = None if max_student_distance is None else float(max_student_distance)
    sample_limit = int(config.get("sample_records_per_step", 64))
    total_valid_states = int(mask.sum())
    sample_interval = max(total_valid_states // max(sample_limit, 1), 1)
    visited_states = 0
    samples: list[dict[str, Any]] = []
    candidate_count = 0
    dual_ball_count = 0
    selected_distance_count = 0
    student_distance_sum = 0.0
    teacher_distance_sum = 0.0
    student_distance_sq_sum = 0.0
    teacher_distance_sq_sum = 0.0

    for uid, rows in group_rows.items():
        if len(rows) < 2:
            visited_states += sum(int(mask[row].sum()) for row in rows)
            continue
        lengths = {row: int(mask[row].sum()) for row in rows}
        for anchor_row in rows:
            anchor_length = lengths[anchor_row]
            if anchor_length == 0:
                continue
            for chunk_start in range(0, anchor_length, chunk_size):
                anchor_positions = torch.arange(
                    chunk_start, min(chunk_start + chunk_size, anchor_length), dtype=torch.long
                )
                anchor_progress = anchor_positions.float() / max(anchor_length - 1, 1)
                candidate_rows_parts = []
                candidate_positions_parts = []
                candidate_valid_parts = []
                for other_row in rows:
                    if other_row == anchor_row or lengths[other_row] == 0:
                        continue
                    other_length = lengths[other_row]
                    candidate_positions = torch.arange(0, other_length, stride, dtype=torch.long)
                    center = torch.round(anchor_progress * max(other_length - 1, 1) / stride).long()
                    local_indices = center[:, None] + torch.arange(-offsets, offsets + 1)[None, :]
                    index_valid = (local_indices >= 0) & (local_indices < len(candidate_positions))
                    safe_indices = local_indices.clamp(0, max(len(candidate_positions) - 1, 0))
                    positions = candidate_positions[safe_indices]
                    candidate_progress = positions.float() / max(other_length - 1, 1)
                    progress_valid = (candidate_progress - anchor_progress[:, None]).abs() <= progress_window
                    valid = index_valid & progress_valid
                    candidate_rows_parts.append(torch.full_like(positions, other_row))
                    candidate_positions_parts.append(positions)
                    candidate_valid_parts.append(valid)
                if not candidate_rows_parts:
                    visited_states += len(anchor_positions)
                    continue

                candidate_rows = torch.cat(candidate_rows_parts, dim=1)
                candidate_positions = torch.cat(candidate_positions_parts, dim=1)
                candidate_valid = torch.cat(candidate_valid_parts, dim=1)
                anchor_student = student_features[anchor_row, anchor_positions]
                anchor_teacher = teacher_features[anchor_row, anchor_positions]
                candidate_student = student_features[candidate_rows, candidate_positions]
                candidate_teacher = teacher_features[candidate_rows, candidate_positions]
                student_distance = 1.0 - (anchor_student[:, None, :] * candidate_student).sum(dim=-1)
                teacher_distance = 1.0 - (anchor_teacher[:, None, :] * candidate_teacher).sum(dim=-1)

                valid_counts = candidate_valid.sum(dim=1)
                keep_count = torch.ceil(valid_counts.float() * radius_quantile).long().clamp_min(1)
                student_rank = _rank_with_invalid_last(student_distance, candidate_valid)
                teacher_rank = _rank_with_invalid_last(teacher_distance, candidate_valid)
                dual_valid = (
                    candidate_valid & (student_rank < keep_count[:, None]) & (teacher_rank < keep_count[:, None])
                )
                if max_student_distance is not None:
                    dual_valid &= student_distance <= max_student_distance
                if max_teacher_distance is not None:
                    dual_valid &= teacher_distance <= max_teacher_distance
                candidate_count += int(candidate_valid.sum())
                dual_ball_count += int(dual_valid.sum())
                joint_distance = torch.maximum(student_distance, teacher_distance).masked_fill(
                    ~dual_valid, float("inf")
                )
                selected_k = min(neighbor_k, joint_distance.shape[1])
                selected_distance, selected_index = torch.topk(
                    joint_distance, k=selected_k, dim=1, largest=False, sorted=True
                )
                selected_valid = torch.isfinite(selected_distance)
                enough = selected_valid.sum(dim=1) >= minimum_neighbors
                selected_valid &= enough[:, None]
                selected_rows = candidate_rows.gather(1, selected_index)
                selected_positions = candidate_positions.gather(1, selected_index)
                selected_student_distance = student_distance.gather(1, selected_index)
                selected_teacher_distance = teacher_distance.gather(1, selected_index)
                valid_student_distances = selected_student_distance[selected_valid]
                valid_teacher_distances = selected_teacher_distance[selected_valid]
                selected_distance_count += int(selected_valid.sum())
                student_distance_sum += float(valid_student_distances.sum())
                teacher_distance_sum += float(valid_teacher_distances.sum())
                student_distance_sq_sum += float(valid_student_distances.square().sum())
                teacher_distance_sq_sum += float(valid_teacher_distances.square().sum())
                neighbor_count[anchor_row, anchor_positions] = selected_valid.sum(dim=1).to(torch.int16)

                anchor_ids = ids[anchor_row, anchor_positions]
                anchor_raw = raw_opd[anchor_row, anchor_positions]
                neighbor_ids = ids[selected_rows, selected_positions]
                neighbor_raw = raw_opd[selected_rows, selected_positions]
                matches = neighbor_ids[:, :, None, :] == anchor_ids[:, None, :, None]
                matches &= selected_valid[:, :, None, None]
                candidate_action_rewards = neighbor_raw[:, :, None, :].expand_as(matches)
                candidate_action_rewards = candidate_action_rewards.masked_fill(~matches, float("inf"))
                per_neighbor_reward = candidate_action_rewards.min(dim=-1).values
                per_neighbor_valid = matches.any(dim=-1)
                neighbor_min, worst_neighbor = per_neighbor_reward.min(dim=1)
                has_neighbor = per_neighbor_valid.any(dim=1)
                action_has_neighbor[anchor_row, anchor_positions] = has_neighbor
                robust = torch.minimum(anchor_raw, neighbor_min)
                robust = torch.where(has_neighbor, robust, anchor_raw)
                ropd_raw[anchor_row, anchor_positions] = robust

                if sample_limit > 0 and len(samples) < sample_limit:
                    for local, position in enumerate(anchor_positions.tolist()):
                        state_ordinal = visited_states + local
                        if state_ordinal % sample_interval != 0:
                            continue
                        action_rank = 0
                        worst = int(worst_neighbor[local, action_rank])
                        neighbor_is_worst = bool(
                            has_neighbor[local, action_rank]
                            and robust[local, action_rank] < anchor_raw[local, action_rank]
                        )
                        record = {
                            "step": step,
                            "uid": uid,
                            "anchor_row": anchor_row,
                            "anchor_position": position,
                            "action_rank": action_rank,
                            "action_token_id": int(anchor_ids[local, action_rank]),
                            "opd_raw_reward": float(anchor_raw[local, action_rank]),
                            "ropd_raw_reward": float(robust[local, action_rank]),
                            "robust_penalty": float(anchor_raw[local, action_rank] - robust[local, action_rank]),
                            "neighbor_count": int(selected_valid[local].sum()),
                            "action_neighbor_available": bool(has_neighbor[local, action_rank]),
                            "worst_neighbor_row": -1,
                            "worst_neighbor_position": -1,
                            "teacher_cosine_distance": float("nan"),
                            "student_cosine_distance": float("nan"),
                            "nearest_teacher_cosine_distance": float("nan"),
                            "nearest_student_cosine_distance": float("nan"),
                        }
                        if bool(selected_valid[local, 0]):
                            record.update(
                                {
                                    "nearest_teacher_cosine_distance": float(selected_teacher_distance[local, 0]),
                                    "nearest_student_cosine_distance": float(selected_student_distance[local, 0]),
                                }
                            )
                        if neighbor_is_worst:
                            record.update(
                                {
                                    "worst_neighbor_row": int(selected_rows[local, worst]),
                                    "worst_neighbor_position": int(selected_positions[local, worst]),
                                    "teacher_cosine_distance": float(selected_teacher_distance[local, worst]),
                                    "student_cosine_distance": float(selected_student_distance[local, worst]),
                                }
                            )
                        samples.append(record)
                        if len(samples) >= sample_limit:
                            break
                visited_states += len(anchor_positions)

    original_scores = raw_opd * weights
    ropd_scores = ropd_raw * weights
    expanded_mask = mask.unsqueeze(-1).expand_as(raw_opd)
    token_opd = original_scores.sum(dim=-1)[mask]
    token_ropd = ropd_scores.sum(dim=-1)[mask]
    token_penalty = token_opd - token_ropd
    raw_penalty = (raw_opd - ropd_raw)[expanded_mask]
    valid_weights = weights[expanded_mask]
    weighted_coverage = float(
        (weights * action_has_neighbor * expanded_mask).sum() / (weights * expanded_mask).sum().clamp_min(1e-12)
    )
    selected_denom = max(selected_distance_count, 1)
    student_distance_mean = student_distance_sum / selected_denom
    teacher_distance_mean = teacher_distance_sum / selected_denom
    metrics: dict[str, float] = {
        "ropd/states": float(total_valid_states),
        "ropd/groups": float(len(group_rows)),
        "ropd/groups_with_multiple_rollouts": float(sum(len(rows) > 1 for rows in group_rows.values())),
        "ropd/neighbor_count_mean": float(neighbor_count[mask].float().mean()) if total_valid_states else 0.0,
        "ropd/zero_neighbor_fraction": float((neighbor_count[mask] == 0).float().mean()) if total_valid_states else 1.0,
        "ropd/action_neighbor_weighted_coverage": weighted_coverage,
        "ropd/dual_ball_candidate_fraction": dual_ball_count / max(candidate_count, 1),
        "ropd/selected_student_distance_mean": student_distance_mean,
        "ropd/selected_teacher_distance_mean": teacher_distance_mean,
        "ropd/selected_student_distance_std": math.sqrt(
            max(student_distance_sq_sum / selected_denom - student_distance_mean**2, 0.0)
        ),
        "ropd/selected_teacher_distance_std": math.sqrt(
            max(teacher_distance_sq_sum / selected_denom - teacher_distance_mean**2, 0.0)
        ),
        "ropd/changed_action_fraction": float((raw_penalty > 1e-8).float().mean()) if raw_penalty.numel() else 0.0,
        "ropd/changed_token_fraction": float((token_penalty > 1e-8).float().mean()) if token_penalty.numel() else 0.0,
        "ropd/weighted_action_mass": float(valid_weights.sum()),
    }
    for prefix, values in (
        ("opd_token_reward", token_opd),
        ("ropd_token_reward", token_ropd),
        ("robust_penalty", token_penalty),
        ("raw_action_penalty", raw_penalty),
    ):
        metrics.update({f"ropd/{prefix}_{name}": value for name, value in _quantiles(values).items()})
    if torch.any(ropd_raw[expanded_mask] > raw_opd[expanded_mask] + 1e-6):
        raise AssertionError("Hard-min ROPD reward exceeded the anchor OPD reward")
    return DenseDiscreteROPDResult(ropd_scores=ropd_scores, metrics=metrics, samples=samples)


class DenseDiscreteROPDEngine:
    """Lazy embedding cache plus dense discrete ROPD computation."""

    def __init__(
        self,
        config: dict[str, Any],
        actor_model_path: str | Path,
        teacher_model_path: str | Path,
        cache_root: str | Path,
    ) -> None:
        validate_dense_discrete_config(config)
        self.config = dict(config)
        projection_dim = int(config.get("projection_dim", 32))
        seed = int(config.get("projection_seed", 314159))
        projection_method = config.get("projection_method", "count_sketch")
        self.student_table = load_projected_embedding_table(
            actor_model_path,
            cache_root,
            projection_dim=projection_dim,
            seed=seed,
            projection_method=projection_method,
        )
        self.teacher_table = load_projected_embedding_table(
            teacher_model_path,
            cache_root,
            projection_dim=projection_dim,
            seed=seed,
            projection_method=projection_method,
        )

    def _state_features(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.config.get("representation", "token_embedding_tail_mean")
        tail_tokens = int(self.config.get("tail_tokens", 16))
        common = {
            "prompts": batch.batch["prompts"],
            "responses": batch.batch["responses"],
            "attention_mask": batch.batch["attention_mask"],
            "response_mask": batch.batch["response_mask"],
            "representation": representation,
            "tail_tokens": tail_tokens,
        }
        return (
            build_prefix_state_features(embedding_table=self.student_table, **common),
            build_prefix_state_features(embedding_table=self.teacher_table, **common),
        )

    def prepare_sampled_action_requests(self, batch: Any) -> DenseDiscreteLCBSupport:
        """Prepare LCB neighbors before the teacher and exact student forwards."""

        if self.config.get("aggregation", "hard_min") != "lcb_gate":
            raise ValueError("Sampled-action request preparation requires aggregation='lcb_gate'")
        student_features, teacher_features = self._state_features(batch)
        return prepare_dense_discrete_lcb_support(
            responses=batch.batch["responses"],
            response_mask=batch.batch["response_mask"],
            student_features=student_features,
            teacher_features=teacher_features,
            uids=batch.non_tensor_batch["uid"],
            config=self.config,
        )

    def compute(
        self,
        batch: Any,
        step: int,
        support: DenseDiscreteLCBSupport | None = None,
    ) -> DenseDiscreteROPDResult:
        if self.config.get("aggregation", "hard_min") == "lcb_gate":
            if support is None:
                raise ValueError("LCB computation requires prepared sampled-action support")
            return compute_dense_discrete_lcb(
                sampled_student_log_probs=batch.batch["old_log_probs"],
                sampled_teacher_log_probs=batch.batch["teacher_sampled_token_log_probs"],
                neighbor_student_log_probs=batch.batch["student_neighbor_request_log_probs"],
                neighbor_teacher_log_probs=batch.batch["teacher_neighbor_request_log_probs"],
                opd_raw_rewards=batch.batch["opd_raw_rewards"],
                opd_reward_weights=batch.batch["opd_reward_weights"],
                response_mask=batch.batch["response_mask"],
                responses=batch.batch["responses"],
                uids=batch.non_tensor_batch["uid"],
                support=support,
                config=self.config,
                step=step,
            )
        student_features, teacher_features = self._state_features(batch)
        return compute_dense_discrete_ropd(
            student_top_k_ids=batch.batch["student_top_k_ids"],
            student_top_k_log_probs=batch.batch["student_top_k_log_probs"],
            teacher_on_student_log_probs=batch.batch["teacher_on_student_log_probs"],
            opd_reward_weights=batch.batch["opd_reward_weights"],
            response_mask=batch.batch["response_mask"],
            student_features=student_features,
            teacher_features=teacher_features,
            uids=batch.non_tensor_batch["uid"],
            config=self.config,
            step=step,
        )


def append_ropd_diagnostics(
    run_dir: str | Path,
    step: int,
    metrics: dict[str, float],
    samples: list[dict[str, Any]],
    apply_to_training: bool,
) -> None:
    """Append durable driver-side diagnostics for later analysis."""

    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "step": step,
        "apply_to_training": apply_to_training,
        **metrics,
    }
    with (metrics_dir / "ropd_step_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, allow_nan=True) + "\n")
    if samples:
        with (metrics_dir / "ropd_reward_samples.jsonl").open("a", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=True) + "\n")
