from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "scripts" / "trustworthy_opd"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_lcb_reliability_calibration import (  # noqa: E402
    average_precision,
    binary_auc,
    build_calibration,
    build_coverage,
    build_metric_summary,
    plot_calibration,
    plot_forest,
    plot_representation,
)
from four_group_common import atomic_write_jsonl  # noqa: E402
from lcb_reliability_common import (  # noqa: E402
    aggregate_risk,
    candidate_positions,
    lcb_trust,
    rollout_is_available,
    select_dual_ball_neighbors,
    stable_fold,
)
from run_lcb_reliability_calibration import (  # noqa: E402
    build_neighbors_and_requests,
    compute_metrics,
)


def test_candidate_positions_match_stride_and_progress_geometry() -> None:
    positions = candidate_positions(
        anchor_position=50,
        anchor_length=101,
        target_length=201,
        stride=10,
        offsets=2,
        progress_window=0.2,
    )
    assert positions == [80, 90, 100, 110, 120]


def test_rollout_block_reproduces_online_four_rollout_support() -> None:
    assert rollout_is_available(0, 3, 4)
    assert not rollout_is_available(0, 4, 4)
    assert rollout_is_available(0, 4, None)
    assert not rollout_is_available(2, 2, None)


def test_dual_ball_requires_both_model_ranks() -> None:
    anchor = np.asarray([1.0, 0.0], dtype=np.float32)
    student = np.asarray([[0.99, 0.01], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    teacher = np.asarray([[0.0, 1.0], [0.8, 0.2], [0.99, 0.01]], dtype=np.float32)
    student /= np.linalg.norm(student, axis=1, keepdims=True)
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True)
    chosen, student_distance, teacher_distance = select_dual_ball_neighbors(
        anchor_student=anchor,
        anchor_teacher=anchor,
        candidate_student=student,
        candidate_teacher=teacher,
        radius_quantile=2 / 3,
        neighbor_k=3,
        minimum_neighbors=1,
        max_student_distance=None,
        max_teacher_distance=None,
    )
    assert chosen.tolist() == [1]
    assert student_distance.shape == teacher_distance.shape == (1,)


def test_lcb_risk_and_trust_are_sign_agnostic_and_conservative() -> None:
    risk = aggregate_risk([0.1, 0.3], "softmax", 1.0)
    assert 0.2 < risk < 0.3
    positive = lcb_trust(0.5, risk, 1.0, 1e-6)
    negative = lcb_trust(-0.5, risk, 1.0, 1e-6)
    assert positive == negative
    assert 0.0 < positive < 1.0
    assert lcb_trust(0.5, float("nan"), 1.0, 1e-6) == 1.0


def test_auc_and_average_precision_handle_ties() -> None:
    labels = np.asarray([0, 0, 1, 1])
    perfect = np.asarray([0.0, 0.1, 0.9, 1.0])
    tied = np.ones(4)
    assert binary_auc(labels, perfect) == 1.0
    assert average_precision(labels, perfect) == 1.0
    assert binary_auc(labels, tied) == 0.5
    assert average_precision(labels, tied) == 0.5


def test_prompt_fold_is_deterministic_and_bounded() -> None:
    values = [stable_fold(prompt, 42, 5) for prompt in range(100)]
    assert values == [stable_fold(prompt, 42, 5) for prompt in range(100)]
    assert set(values) <= set(range(5))


def test_neighbor_request_and_metric_pipeline_on_synthetic_data(tmp_path: Path) -> None:
    config = {
        "representations": {"definitions": [{"name": "toy"}]},
        "neighborhood": {
            "radius_quantile": 1.0,
            "neighbor_k": 1,
            "minimum_neighbors": 1,
            "max_student_cosine_distance": 0.35,
            "max_teacher_cosine_distance": 0.35,
            "variants": [
                {"name": "online4", "rollout_block_size": 4},
                {"name": "dense8", "rollout_block_size": None},
            ],
        },
        "risk": {
            "aggregation": "softmax",
            "temperature": 1.0,
            "epsilon": 1e-6,
            "lambdas": [0.0, 0.1],
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    for directory in (
        "artifacts",
        "features/student",
        "features/teacher",
        "scores/student",
        "scores/teacher",
        "results",
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    anchor = {
        "state_id": "anchor",
        "prompt_index": 1,
        "rollout_index": 0,
        "base_trajectory_id": "r0",
        "position": 1,
        "student_action_token_id": 5,
        "group": "teacher_correct_student_wrong",
        "fraction_name": "q20",
        "fold": 0,
    }
    points = [
        {"point_id": "r0@1", "prompt_index": 1, "rollout_index": 0, "base_trajectory_id": "r0", "position": 1},
        {"point_id": "r1@1", "prompt_index": 1, "rollout_index": 1, "base_trajectory_id": "r1", "position": 1},
        {"point_id": "r4@1", "prompt_index": 1, "rollout_index": 4, "base_trajectory_id": "r4", "position": 1},
    ]
    edges = [
        {"state_id": "anchor", "anchor_point_id": "r0@1", "candidate_point_id": "r1@1", "candidate_rollout_index": 1},
        {"state_id": "anchor", "anchor_point_id": "r0@1", "candidate_point_id": "r4@1", "candidate_rollout_index": 4},
    ]
    atomic_write_jsonl(tmp_path / "artifacts/anchors.jsonl", [anchor])
    atomic_write_jsonl(tmp_path / "artifacts/points.jsonl", points)
    atomic_write_jsonl(tmp_path / "artifacts/candidate_edges.jsonl", edges)
    feature = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=np.float16)
    for role in ("student", "teacher"):
        with (tmp_path / f"features/{role}/shard_000.npz").open("wb") as handle:
            np.savez(handle, point_ids=np.asarray([row["point_id"] for row in points]), representation__toy=feature)
    build_neighbors_and_requests(tmp_path, config)
    requests = [
        __import__("json").loads(line) for line in (tmp_path / "artifacts/requests.jsonl").read_text().splitlines()
    ]
    # Both neighborhood variants select the same closest target, so exact
    # sampled-action requests are deduplicated across variants.
    assert len(requests) == 2
    for role in ("student", "teacher"):
        values = []
        for request in requests:
            if role == "student":
                log_prob = -2.0
            else:
                log_prob = -1.0 if request["base_trajectory_id"] == "r0" else -1.2
            values.append({"request_id": request["request_id"], "role": role, "log_prob": log_prob})
        atomic_write_jsonl(tmp_path / f"scores/{role}/shard_000.jsonl", values)
    compute_metrics(tmp_path, config)
    result = pd.read_parquet(tmp_path / "results/state_metrics.parquet")
    assert len(result) == 2
    assert set(result.neighborhood) == {"online4", "dense8"}
    assert np.allclose(result.lcb_risk, 0.2)
    assert np.allclose(result.trust_lambda_0p1, 0.98)


def test_statistical_summary_and_figures_on_clustered_synthetic_data(
    tmp_path: Path,
) -> None:
    config = {
        "experiment": {"seed": 42},
        "risk": {"lambdas": [0.0, 0.003, 0.01, 0.03, 1.0]},
        "analysis": {
            "bootstrap_samples": 20,
            "calibration_bins": 5,
            "primary_neighborhood": "online4",
            "primary_representation": "token_embedding_tail_mean",
        },
    }
    rows = []
    for prompt in range(8):
        for q_index, q in enumerate(("q20", "q40", "q60", "q80")):
            for label, group in enumerate(("both_wrong", "teacher_correct_student_wrong")):
                reliable = float(label) + 0.01 * prompt + 0.001 * q_index
                rows.append(
                    {
                        "state_id": f"p{prompt}_{q}_{group}",
                        "prompt_index": prompt,
                        "fold": prompt % 5,
                        "fraction_name": q,
                        "group": group,
                        "neighborhood": "online4",
                        "representation": "token_embedding_tail_mean",
                        "neighbor_count": 3,
                        "teacher_sensitivity": -reliable,
                        "relative_sensitivity": -reliable,
                        "lcb_risk": -reliable,
                        "risk_to_abs_reward": -reliable,
                        "trust_lambda_0p0": reliable,
                        "trust_lambda_0p003": reliable,
                        "trust_lambda_0p01": reliable,
                        "trust_lambda_0p03": reliable,
                        "trust_lambda_1p0": reliable,
                        "student_distance_mean": 0.1,
                        "teacher_distance_mean": 0.1,
                    }
                )
    frame = pd.DataFrame(rows)
    summary, folds = build_metric_summary(frame, config)
    calibration = build_calibration(frame, config)
    coverage = build_coverage(frame)
    assert not summary.empty and not folds.empty and not calibration.empty
    macro = summary[summary.stratum == "macro_q"]
    assert np.allclose(macro.auroc, 1.0)
    plot_forest(summary, config, tmp_path / "forest.svg")
    plot_representation(summary, tmp_path / "representations.svg")
    plot_calibration(calibration, tmp_path / "calibration.svg")
    assert (tmp_path / "forest.svg").is_file()
    assert (tmp_path / "representations.svg").is_file()
    assert (tmp_path / "calibration.svg").is_file()
    assert np.allclose(coverage.coverage, 1.0)
