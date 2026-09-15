from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "scripts" / "trustworthy_opd"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_semantic_neighborhood_audit import precision_summary  # noqa: E402
from prepare_semantic_neighborhood_audit import sample_selected  # noqa: E402
from semantic_neighborhood_audit_common import (  # noqa: E402
    balanced_sample,
    clustered_precision_interval,
    cohens_kappa,
    consensus_annotations,
    semantic_pair_id,
)
from serve_semantic_neighborhood_audit import safe_annotator  # noqa: E402


def test_pair_identifier_and_annotator_validation() -> None:
    assert semantic_pair_id("state", "neighbor") == semantic_pair_id("state", "neighbor")
    assert semantic_pair_id("state", "neighbor") != semantic_pair_id("state", "other")
    assert safe_annotator("rater_2") == "rater_2"
    try:
        safe_annotator("../../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe annotator was accepted")


def test_balanced_sampling_spans_distance_range() -> None:
    frame = pd.DataFrame({"joint_distance": np.arange(30, dtype=float), "value": np.arange(30)})
    sampled = balanced_sample(frame, 9, seed=7)
    assert len(sampled) == 9
    bins = sampled.joint_distance // 10
    assert set(bins) == {0.0, 1.0, 2.0}


def test_consensus_and_kappa() -> None:
    empty = consensus_annotations({})
    assert empty.empty
    assert "audit_id" in empty.columns
    annotations = {
        "a": {
            "x": {"label": "comparable"},
            "y": {"label": "not_comparable"},
            "z": {"label": "uncertain"},
        },
        "b": {
            "x": {"label": "comparable"},
            "y": {"label": "comparable"},
            "z": {"label": "uncertain"},
        },
    }
    result = consensus_annotations(annotations).set_index("audit_id")
    assert result.loc["x", "consensus_label"] == "comparable"
    assert result.loc["y", "consensus_label"] == "disputed"
    assert result.loc["z", "consensus_label"] == "uncertain"
    assert (
        cohens_kappa(
            ["comparable", "not_comparable"],
            ["comparable", "not_comparable"],
        )
        == 1.0
    )


def test_clustered_precision_and_summary() -> None:
    rows = []
    for prompt in range(12):
        for group in ("teacher_correct_student_wrong", "both_wrong"):
            rows.append(
                {
                    "audit_id": f"{prompt}-{group}",
                    "prompt_index": prompt,
                    "arm": "selected",
                    "neighborhood": "online4",
                    "representation": "toy",
                    "fraction_name": "q20",
                    "group": group,
                    "consensus_label": "comparable" if prompt < 9 else "not_comparable",
                }
            )
    frame = pd.DataFrame(rows)
    estimate, low, high = clustered_precision_interval(frame, samples=100, seed=9, confidence=0.95)
    assert estimate == 0.75
    assert low <= estimate <= high
    config = {
        "experiment": {"seed": 42},
        "sampling": {
            "groups": ["teacher_correct_student_wrong", "both_wrong"],
            "q_points": ["q20"],
        },
        "analysis": {"bootstrap_samples": 100, "confidence_level": 0.95},
    }
    summary = precision_summary(frame, config)
    pooled = summary[(summary.fraction_name == "all_q") & (summary.group == "all_primary_groups")].iloc[0]
    assert pooled.semantic_precision == 0.75
    assert pooled.decided == 24


def test_selected_sampling_is_balanced_by_group_and_cell() -> None:
    anchors = []
    neighbors = []
    for group_index, group in enumerate(("teacher_correct_student_wrong", "both_wrong")):
        for index in range(6):
            state = f"s-{group_index}-{index}"
            anchors.append(
                {
                    "state_id": state,
                    "prompt_index": index,
                    "fraction_name": "q20",
                    "group": group,
                    "base_trajectory_id": f"t-{state}",
                    "position": 10,
                    "student_action_token_id": 3,
                }
            )
            neighbors.append(
                {
                    "state_id": state,
                    "neighborhood": "online4",
                    "representation": "toy",
                    "neighbor_rank": 0,
                    "neighbor_point_id": f"n-{state}",
                    "neighbor_request_id": f"r-{state}",
                    "student_cosine_distance": index / 10,
                    "teacher_cosine_distance": index / 9,
                }
            )
    config = {
        "experiment": {"seed": 7},
        "sampling": {
            "selected_pairs_per_group_cell": 4,
            "q_points": ["q20"],
            "groups": ["teacher_correct_student_wrong", "both_wrong"],
        },
    }
    sampled = sample_selected(pd.DataFrame(neighbors), pd.DataFrame(anchors), config)
    assert len(sampled) == 8
    assert pd.Series([row["group"] for row in sampled]).value_counts().to_dict() == {
        "teacher_correct_student_wrong": 4,
        "both_wrong": 4,
    }
