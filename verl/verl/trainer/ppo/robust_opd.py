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

The implementation deliberately uses real student-generated prefix states.  It
does not create continuous embedding perturbations.  For an anchor action v,
only neighbor states whose student Top-K contains v provide an exact OPD reward
for v; unmatched actions are skipped and reported as coverage diagnostics.
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


def validate_dense_discrete_config(config: dict[str, Any]) -> None:
    if config.get("method", "dense_discrete") != "dense_discrete":
        raise ValueError("robust_opd.method must be 'dense_discrete'")
    if config.get("aggregation", "hard_min") != "hard_min":
        raise ValueError("Only hard_min is implemented for the primary ROPD definition")
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


def _model_fingerprint(model_path: Path, projection_dim: int, seed: int, projection_method: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(model_path.resolve()).encode())
    for name in ("config.json", "model.safetensors", "model.safetensors.index.json"):
        path = model_path / name
        if path.is_file():
            stat = path.stat()
            digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(
        f"projection_dim={projection_dim};seed={seed};projection_method={projection_method}".encode()
    )
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
        cumulative = torch.cat(
            [torch.zeros(1, projection_dim, dtype=torch.float32), vectors.cumsum(dim=0)], dim=0
        )
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


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {name: float("nan") for name in ("mean", "std", "p05", "p50", "p95")}
    values = values.float()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
    }


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
                    candidate_valid
                    & (student_rank < keep_count[:, None])
                    & (teacher_rank < keep_count[:, None])
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
                                    "nearest_teacher_cosine_distance": float(
                                        selected_teacher_distance[local, 0]
                                    ),
                                    "nearest_student_cosine_distance": float(
                                        selected_student_distance[local, 0]
                                    ),
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
        (weights * action_has_neighbor * expanded_mask).sum()
        / (weights * expanded_mask).sum().clamp_min(1e-12)
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

    def compute(self, batch: Any, step: int) -> DenseDiscreteROPDResult:
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
        student_features = build_prefix_state_features(embedding_table=self.student_table, **common)
        teacher_features = build_prefix_state_features(embedding_table=self.teacher_table, **common)
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
