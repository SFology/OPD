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

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from verl.trainer.ppo.robust_opd import (
    DenseDiscreteLCBSupport,
    _evenly_spaced_indices,
    _quantiles,
    build_prefix_state_features,
    compute_dense_discrete_lcb,
    compute_dense_discrete_ropd,
    load_projected_embedding_table,
    prepare_dense_discrete_lcb_support,
    validate_dense_discrete_config,
)
from verl.utils.sparse_log_probs import (
    gather_sparse_packed_response_log_probs,
    gather_sparse_response_log_probs,
)


def _config() -> dict:
    return {
        "method": "dense_discrete",
        "aggregation": "hard_min",
        "representation": "token_embedding_tail_mean",
        "projection_method": "count_sketch",
        "projection_dim": 2,
        "tail_tokens": 4,
        "candidate_stride": 1,
        "candidate_offsets": 1,
        "progress_window": 1.0,
        "radius_quantile": 1.0,
        "neighbor_k": 1,
        "minimum_neighbors": 1,
        "anchor_chunk_size": 2,
        "sample_records_per_step": 4,
    }


def _inputs(neighbor_ids: torch.Tensor | None = None, uids: np.ndarray | None = None) -> dict:
    ids = torch.tensor(
        [
            [[10, 11], [12, 13]],
            [[10, 11], [12, 13]],
        ]
    )
    if neighbor_ids is not None:
        ids[1] = neighbor_ids
    student_logp = torch.zeros(2, 2, 2)
    teacher_logp = torch.tensor(
        [
            [[1.0, 2.0], [1.5, 2.5]],
            [[0.5, 3.0], [0.75, 3.5]],
        ]
    )
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    return {
        "student_top_k_ids": ids,
        "student_top_k_log_probs": student_logp,
        "teacher_on_student_log_probs": teacher_logp,
        "opd_reward_weights": torch.full_like(student_logp, 0.5),
        "response_mask": torch.ones(2, 2, dtype=torch.bool),
        "student_features": features,
        "teacher_features": features,
        "uids": np.array(["prompt-a", "prompt-a"], dtype=object) if uids is None else uids,
        "config": _config(),
    }


def test_prefix_features_exclude_current_action_token() -> None:
    embedding_table = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    result = build_prefix_state_features(
        prompts=torch.tensor([[0, 1]]),
        responses=torch.tensor([[2, 3]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        response_mask=torch.ones(1, 2, dtype=torch.long),
        embedding_table=embedding_table,
        representation="token_embedding_prefix_mean",
        tail_tokens=4,
    )
    expected_t0 = F.normalize(torch.tensor([0.5, 0.5]), dim=0)
    expected_t1 = F.normalize(torch.tensor([2.0 / 3.0, 2.0 / 3.0]), dim=0)
    torch.testing.assert_close(result[0, 0], expected_t0)
    torch.testing.assert_close(result[0, 1], expected_t1)


def test_large_quantiles_use_deterministic_bounded_sample() -> None:
    values = torch.arange(100, dtype=torch.float32)
    first = _quantiles(values, max_quantile_samples=10)
    second = _quantiles(values, max_quantile_samples=10)

    assert first == second
    assert first["mean"] == pytest.approx(49.5)
    assert first["p50"] == pytest.approx(49.5)


def test_large_sample_indices_never_round_past_tensor_end() -> None:
    # Regression for CUDA float32 linspace rounding 26_535_087 up to
    # 26_535_088, which produced an out-of-bounds diagnostic sample.
    num_values = 26_535_088
    indices = _evenly_spaced_indices(num_values, 1_000_000, torch.device("cpu"))

    assert indices.numel() == 1_000_000
    assert int(indices.min()) >= 0
    assert int(indices.max()) < num_values


def test_sparse_response_log_probs_match_dense_log_softmax() -> None:
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 7
    positions = torch.tensor([[0, 2, 0], [1, 0, 0]])
    actions = torch.tensor([[3, 1, 0], [2, 0, 0]])
    valid = torch.tensor([[True, True, False], [True, True, False]])

    actual = gather_sparse_response_log_probs(logits, positions, actions, valid)
    dense = torch.log_softmax(logits, dim=-1)

    torch.testing.assert_close(actual[0, :2], dense[0, [0, 2], [3, 1]])
    torch.testing.assert_close(actual[1, :2], dense[1, [1, 0], [2, 0]])
    assert torch.equal(actual[:, 2], torch.zeros(2))


def test_sparse_packed_log_probs_preserve_response_alignment() -> None:
    full_logits = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4) / 9
    attention_mask = torch.tensor([[False, True, True, True, True], [True, True, True, True, True]])
    indices = attention_mask.reshape(-1).nonzero(as_tuple=True)[0]
    packed_logits = full_logits.reshape(-1, 4)[indices]
    positions = torch.tensor([[0, 1], [1, 0]])
    actions = torch.tensor([[1, 3], [2, 0]])
    valid = torch.ones_like(positions, dtype=torch.bool)

    actual = gather_sparse_packed_response_log_probs(
        packed_logits,
        indices,
        sequence_length=5,
        response_length=2,
        positions=positions,
        action_ids=actions,
        valid=valid,
    )
    dense = torch.log_softmax(full_logits[:, 2:4], dim=-1)

    torch.testing.assert_close(actual[0], dense[0, [0, 1], [1, 3]])
    torch.testing.assert_close(actual[1], dense[1, [1, 0], [2, 0]])


def test_count_sketch_embedding_cache_is_reusable(tmp_path) -> None:
    model_dir = tmp_path / "model"
    cache_dir = tmp_path / "cache"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    embedding = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    save_file({"model.embed_tokens.weight": embedding}, model_dir / "model.safetensors")
    first = load_projected_embedding_table(model_dir, cache_dir, projection_dim=3, seed=7)
    second = load_projected_embedding_table(model_dir, cache_dir, projection_dim=3, seed=7)
    assert first.shape == (6, 3)
    assert first.dtype == torch.float16
    torch.testing.assert_close(first, second)
    assert len(list(cache_dir.glob("*.pt"))) == 1


def test_hard_min_uses_same_action_from_neighbor() -> None:
    result = compute_dense_discrete_ropd(**_inputs())
    torch.testing.assert_close(result.ropd_scores[0, 0], torch.tensor([0.25, 1.0]))
    assert result.metrics["ropd/changed_action_fraction"] > 0
    assert result.metrics["ropd/action_neighbor_weighted_coverage"] == pytest.approx(1.0)
    assert torch.all(result.ropd_scores <= _inputs()["teacher_on_student_log_probs"] * 0.5)


def test_missing_neighbor_action_falls_back_to_anchor_reward() -> None:
    inputs = _inputs(neighbor_ids=torch.tensor([[20, 21], [22, 23]]))
    result = compute_dense_discrete_ropd(**inputs)
    expected = inputs["teacher_on_student_log_probs"] * inputs["opd_reward_weights"]
    torch.testing.assert_close(result.ropd_scores, expected)
    assert result.metrics["ropd/action_neighbor_weighted_coverage"] == 0.0


def test_states_from_different_prompts_are_never_neighbors() -> None:
    inputs = _inputs(uids=np.array(["prompt-a", "prompt-b"], dtype=object))
    result = compute_dense_discrete_ropd(**inputs)
    expected = inputs["teacher_on_student_log_probs"] * inputs["opd_reward_weights"]
    torch.testing.assert_close(result.ropd_scores, expected)
    assert result.metrics["ropd/groups_with_multiple_rollouts"] == 0.0
    assert result.metrics["ropd/zero_neighbor_fraction"] == 1.0


def test_rejects_impossible_minimum_neighbor_setting() -> None:
    config = _config()
    config["minimum_neighbors"] = 2
    with pytest.raises(ValueError, match="must not exceed"):
        validate_dense_discrete_config(config)


def _lcb_config() -> dict:
    config = _config()
    config.update(
        {
            "aggregation": "lcb_gate",
            "action_evaluation": "sampled_token_exact",
            "risk_aggregation": "max",
            "lcb_lambda": 0.5,
            "lcb_epsilon": 1e-6,
            "max_sparse_requests_per_trajectory": 32,
        }
    )
    return config


def test_lcb_support_requests_anchor_sampled_action_at_neighbor_state() -> None:
    responses = torch.tensor([[10, 12], [20, 22]])
    mask = torch.ones_like(responses, dtype=torch.bool)
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    support = prepare_dense_discrete_lcb_support(
        responses=responses,
        response_mask=mask,
        student_features=features,
        teacher_features=features,
        uids=np.array(["prompt-a", "prompt-a"], dtype=object),
        config=_lcb_config(),
    )

    assert support.neighbor_valid.all()
    for row, position, neighbor_rank in support.neighbor_valid.nonzero().tolist():
        target_row = int(support.neighbor_rows[row, position, neighbor_rank])
        target_position = int(support.neighbor_positions[row, position, neighbor_rank])
        request_slot = int(support.request_slots[row, position, neighbor_rank])
        assert support.request_positions[target_row, request_slot] == target_position
        assert support.request_action_ids[target_row, request_slot] == responses[row, position]
        assert support.request_valid[target_row, request_slot]
    assert support.metrics["ropd/action_neighbor_weighted_coverage"] == pytest.approx(1.0)


def test_lcb_sparse_support_does_not_impose_a_global_per_state_width() -> None:
    batch_size = 130
    responses = torch.arange(10, 10 + batch_size).reshape(batch_size, 1)
    mask = torch.ones_like(responses, dtype=torch.bool)
    features = torch.tensor([1.0, 0.0]).repeat(batch_size, 1, 1)
    config = _lcb_config()
    config.update(
        {
            "neighbor_k": batch_size - 1,
            "max_sparse_requests_per_trajectory": 256,
        }
    )

    support = prepare_dense_discrete_lcb_support(
        responses=responses,
        response_mask=mask,
        student_features=features,
        teacher_features=features,
        uids=np.full(batch_size, "prompt-a", dtype=object),
        config=config,
    )

    assert support.metrics["ropd/exact_request_max_per_state"] == batch_size - 1
    assert support.request_action_ids.shape == (batch_size, batch_size - 1)
    assert support.request_valid.all()


def test_lcb_shrinks_magnitude_without_changing_reward_sign() -> None:
    mask = torch.ones(2, 2, dtype=torch.bool)
    neighbor_rows = torch.tensor([[[1], [1]], [[0], [0]]])
    neighbor_positions = torch.tensor([[[0], [1]], [[0], [1]]])
    support = DenseDiscreteLCBSupport(
        request_positions=torch.tensor([[0, 1], [0, 1]]),
        request_action_ids=torch.tensor([[10, 12], [10, 12]]),
        request_valid=torch.ones(2, 2, dtype=torch.bool),
        neighbor_rows=neighbor_rows,
        neighbor_positions=neighbor_positions,
        request_slots=torch.tensor([[[0], [1]], [[0], [1]]]),
        neighbor_valid=torch.ones_like(neighbor_rows, dtype=torch.bool),
        student_distances=torch.zeros_like(neighbor_rows, dtype=torch.float32),
        teacher_distances=torch.zeros_like(neighbor_rows, dtype=torch.float32),
        metrics={"ropd/states": 4.0},
    )
    sampled_teacher = torch.tensor([[1.0, -2.0], [0.5, -1.0]])
    sampled_student = torch.zeros_like(sampled_teacher)
    raw_opd = torch.tensor(
        [
            [[1.0, -2.0], [3.0, -4.0]],
            [[2.0, -1.0], [0.5, -0.25]],
        ]
    )
    result = compute_dense_discrete_lcb(
        sampled_student_log_probs=sampled_student,
        sampled_teacher_log_probs=sampled_teacher,
        neighbor_student_log_probs=torch.zeros(2, 2),
        neighbor_teacher_log_probs=sampled_teacher,
        opd_raw_rewards=raw_opd,
        opd_reward_weights=torch.full_like(raw_opd, 0.5),
        response_mask=mask,
        responses=torch.tensor([[10, 12], [10, 12]]),
        uids=np.array(["prompt-a", "prompt-a"], dtype=object),
        support=support,
        config=_lcb_config(),
    )
    original = raw_opd * 0.5

    assert torch.all(result.ropd_scores.abs() <= original.abs() + 1e-7)
    assert torch.all(result.ropd_scores.sign() == original.sign())
    assert result.metrics["ropd/trust_mean"] < 1.0
    assert result.metrics["ropd/absolute_reward_reduction_mean"] >= 0.0


def test_lcb_no_neighbor_falls_back_to_original_opd() -> None:
    mask = torch.ones(1, 1, dtype=torch.bool)
    support = DenseDiscreteLCBSupport(
        request_positions=torch.zeros(1, 1, dtype=torch.long),
        request_action_ids=torch.zeros(1, 1, dtype=torch.long),
        request_valid=torch.zeros(1, 1, dtype=torch.bool),
        neighbor_rows=torch.zeros(1, 1, 1, dtype=torch.long),
        neighbor_positions=torch.zeros(1, 1, 1, dtype=torch.long),
        request_slots=torch.zeros(1, 1, 1, dtype=torch.long),
        neighbor_valid=torch.zeros(1, 1, 1, dtype=torch.bool),
        student_distances=torch.full((1, 1, 1), float("nan")),
        teacher_distances=torch.full((1, 1, 1), float("nan")),
        metrics={"ropd/states": 1.0},
    )
    raw_opd = torch.tensor([[[1.0, -2.0]]])
    weights = torch.tensor([[[0.25, 0.75]]])
    result = compute_dense_discrete_lcb(
        sampled_student_log_probs=torch.zeros(1, 1),
        sampled_teacher_log_probs=torch.ones(1, 1),
        neighbor_student_log_probs=torch.zeros(1, 1),
        neighbor_teacher_log_probs=torch.zeros(1, 1),
        opd_raw_rewards=raw_opd,
        opd_reward_weights=weights,
        response_mask=mask,
        responses=torch.tensor([[10]]),
        uids=np.array(["prompt-a"], dtype=object),
        support=support,
        config=_lcb_config(),
    )

    torch.testing.assert_close(result.ropd_scores, raw_opd * weights)
    assert result.metrics["ropd/trust_mean"] == pytest.approx(1.0)
