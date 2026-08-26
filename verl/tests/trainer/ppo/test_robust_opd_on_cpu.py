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
    build_prefix_state_features,
    compute_dense_discrete_ropd,
    load_projected_embedding_table,
    validate_dense_discrete_config,
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
