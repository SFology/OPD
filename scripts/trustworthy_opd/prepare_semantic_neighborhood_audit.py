from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from common import create_run_dir, load_config, load_run, read_jsonl, update_status
from four_group_common import atomic_write_jsonl, write_json
from lcb_reliability_common import rollout_is_available
from run_lcb_reliability_calibration import load_feature_arrays, sha256_file
from semantic_neighborhood_audit_common import (
    balanced_sample,
    semantic_pair_id,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a blinded semantic-neighborhood annotation audit"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.config is None and args.run_dir is None:
        parser.error("--config is required when --run-dir is not supplied")
    return args


def validate_source(source_run: Path, required_status: str) -> None:
    status_path = source_run / "status.yaml"
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    status = yaml.safe_load(status_path.read_text(encoding="utf-8"))
    if status.get("status") != required_status:
        raise RuntimeError(
            f"Source run status is {status.get('status')!r}, expected {required_status!r}"
        )
    required = (
        "config.yaml",
        "artifacts/anchors.jsonl",
        "artifacts/candidate_edges.jsonl",
        "artifacts/neighbors.jsonl",
        "artifacts/points.jsonl",
        "artifacts/trajectories.jsonl",
        "results/state_metrics.parquet",
    )
    for relative in required:
        path = source_run / relative
        if not path.is_file():
            raise FileNotFoundError(path)


def normalized(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values


def sample_selected(
    neighbors: pd.DataFrame,
    anchors: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    sampling = config["sampling"]
    merged = neighbors.merge(
        anchors[
            [
                "state_id",
                "prompt_index",
                "fraction_name",
                "group",
                "base_trajectory_id",
                "position",
                "student_action_token_id",
            ]
        ],
        on="state_id",
        validate="many_to_one",
    )
    merged["joint_distance"] = merged[
        ["student_cosine_distance", "teacher_cosine_distance"]
    ].max(axis=1)
    rows: list[dict[str, Any]] = []
    count = int(sampling["selected_pairs_per_group_cell"])
    for neighborhood in sorted(merged.neighborhood.unique()):
        for representation in sorted(merged.representation.unique()):
            for q in sampling["q_points"]:
                for group in sampling["groups"]:
                    subset = merged[
                        (merged.neighborhood == neighborhood)
                        & (merged.representation == representation)
                        & (merged.fraction_name == q)
                        & (merged.group == group)
                    ]
                    chosen = balanced_sample(
                        subset,
                        count,
                        seed=stable_seed(
                            config["experiment"]["seed"],
                            "selected",
                            neighborhood,
                            representation,
                            q,
                            group,
                        ),
                    )
                    if len(chosen) < count:
                        raise RuntimeError(
                            "Insufficient selected neighbors for "
                            f"{neighborhood}/{representation}/{q}/{group}: "
                            f"{len(chosen)} < {count}"
                        )
                    for row in chosen.to_dict("records"):
                        rows.append(
                            {
                                **row,
                                "arm": "selected",
                                "selection_cell": (
                                    f"{neighborhood}|{representation}|{q}|{group}"
                                ),
                            }
                        )
    return rows


def control_candidates(
    *,
    source_run: Path,
    anchors: pd.DataFrame,
    points: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    source_config = load_config(source_run / "config.yaml")
    representations = [
        item["name"] for item in source_config["representations"]["definitions"]
    ]
    point_index = {row["point_id"]: index for index, row in enumerate(points)}
    student = {
        name: normalized(values)
        for name, values in load_feature_arrays(
            source_run, "student", points, representations
        ).items()
    }
    teacher = {
        name: normalized(values)
        for name, values in load_feature_arrays(
            source_run, "teacher", points, representations
        ).items()
    }
    edges = pd.DataFrame(read_jsonl(source_run / "artifacts" / "candidate_edges.jsonl"))
    edges = edges.merge(
        anchors[
            [
                "state_id",
                "prompt_index",
                "fraction_name",
                "group",
                "rollout_index",
                "base_trajectory_id",
                "position",
                "student_action_token_id",
            ]
        ],
        on="state_id",
        validate="many_to_one",
    )
    anchor_indices = np.asarray(
        [point_index[item] for item in edges.anchor_point_id], dtype=np.int64
    )
    candidate_indices = np.asarray(
        [point_index[item] for item in edges.candidate_point_id], dtype=np.int64
    )
    frames = []
    variants = source_config["neighborhood"]["variants"]
    for variant in variants:
        block = variant.get("rollout_block_size")
        block = None if block is None else int(block)
        allowed = np.asarray(
            [
                rollout_is_available(int(anchor), int(candidate), block)
                for anchor, candidate in zip(
                    edges.rollout_index, edges.candidate_rollout_index
                )
            ],
            dtype=bool,
        )
        for representation in representations:
            student_distance = 1.0 - np.einsum(
                "ij,ij->i",
                student[representation][anchor_indices],
                student[representation][candidate_indices],
            )
            teacher_distance = 1.0 - np.einsum(
                "ij,ij->i",
                teacher[representation][anchor_indices],
                teacher[representation][candidate_indices],
            )
            frame = edges.loc[allowed].copy()
            frame["neighborhood"] = variant["name"]
            frame["representation"] = representation
            frame["neighbor_point_id"] = frame.candidate_point_id
            frame["student_cosine_distance"] = student_distance[allowed]
            frame["teacher_cosine_distance"] = teacher_distance[allowed]
            frame["joint_distance"] = np.maximum(
                student_distance[allowed], teacher_distance[allowed]
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def sample_controls(
    candidates: pd.DataFrame,
    selected: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    sampling = config["sampling"]
    selected_keys = {
        (
            row["state_id"],
            row["neighborhood"],
            row["representation"],
            row["neighbor_point_id"],
        )
        for row in selected
    }
    rows: list[dict[str, Any]] = []
    count = int(sampling["controls_per_group_cell"])
    for neighborhood in sorted(candidates.neighborhood.unique()):
        for representation in sorted(candidates.representation.unique()):
            for q in sampling["q_points"]:
                for group in sampling["groups"]:
                    subset = candidates[
                        (candidates.neighborhood == neighborhood)
                        & (candidates.representation == representation)
                        & (candidates.fraction_name == q)
                        & (candidates.group == group)
                    ].copy()
                    subset = subset[
                        [
                            (
                                row.state_id,
                                neighborhood,
                                representation,
                                row.neighbor_point_id,
                            )
                            not in selected_keys
                            for row in subset.itertuples()
                        ]
                    ]
                    used: set[tuple[str, str]] = set()
                    for arm in sampling["control_arms"]:
                        available = subset[
                            [
                                (row.state_id, row.neighbor_point_id) not in used
                                for row in subset.itertuples()
                            ]
                        ]
                        if arm == "far":
                            chosen = available.nlargest(count, "joint_distance")
                        elif arm == "random":
                            if len(available) < count:
                                chosen = available
                            else:
                                rng = np.random.default_rng(
                                    stable_seed(
                                        config["experiment"]["seed"],
                                        arm,
                                        neighborhood,
                                        representation,
                                        q,
                                        group,
                                    )
                                )
                                chosen = available.iloc[
                                    rng.choice(len(available), count, replace=False)
                                ]
                        else:
                            raise ValueError(f"Unsupported control arm: {arm}")
                        if len(chosen) < count:
                            raise RuntimeError(
                                f"Insufficient {arm} controls for "
                                f"{neighborhood}/{representation}/{q}/{group}"
                            )
                        for row in chosen.to_dict("records"):
                            used.add((row["state_id"], row["neighbor_point_id"]))
                            rows.append(
                                {
                                    **row,
                                    "arm": arm,
                                    "neighbor_rank": -1,
                                    "selection_cell": (
                                        f"{neighborhood}|{representation}|{q}|{group}"
                                    ),
                                }
                            )
    return rows


def build_blinded_pairs(
    memberships: list[dict[str, Any]],
    *,
    source_run: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    source_config = load_config(source_run / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        source_config["models"]["student"],
        local_files_only=True,
        trust_remote_code=True,
    )
    trajectories = {
        row["base_trajectory_id"]: row
        for row in read_jsonl(source_run / "artifacts" / "trajectories.jsonl")
    }
    points = {
        row["point_id"]: row
        for row in read_jsonl(source_run / "artifacts" / "points.jsonl")
    }
    unique: dict[str, dict[str, Any]] = {}
    tail = int(config["sampling"]["context_tail_tokens"])
    for membership in memberships:
        audit_id = semantic_pair_id(
            membership["state_id"], membership["neighbor_point_id"]
        )
        if audit_id in unique:
            continue
        anchor = trajectories[membership["base_trajectory_id"]]
        neighbor_point = points[membership["neighbor_point_id"]]
        neighbor = trajectories[neighbor_point["base_trajectory_id"]]
        anchor_prefix = anchor["generated_token_ids"][: int(membership["position"])]
        neighbor_prefix = neighbor["generated_token_ids"][
            : int(neighbor_point["position"])
        ]
        secondary_value = (
            stable_seed(config["experiment"]["seed"], "secondary", audit_id) / 2**64
        )
        unique[audit_id] = {
            "audit_id": audit_id,
            "prompt_text": tokenizer.decode(
                anchor["prompt_token_ids"], skip_special_tokens=True
            ),
            "anchor_excerpt": tokenizer.decode(
                anchor_prefix[-tail:], skip_special_tokens=True
            ),
            "neighbor_excerpt": tokenizer.decode(
                neighbor_prefix[-tail:], skip_special_tokens=True
            ),
            "anchor_action": tokenizer.decode(
                [int(membership["student_action_token_id"])],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "secondary_required": secondary_value
            < float(config["sampling"]["secondary_fraction"]),
        }
    maximum = int(config["sampling"]["maximum_unique_pairs"])
    if len(unique) > maximum:
        raise RuntimeError(
            f"Prepared {len(unique)} unique pairs, exceeding maximum {maximum}"
        )
    rows = list(unique.values())
    rows.sort(
        key=lambda row: stable_seed(config["experiment"]["seed"], row["audit_id"])
    )
    for index, row in enumerate(rows, start=1):
        row["audit_index"] = index
    return rows


def prepare(run_dir: Path, config: dict[str, Any]) -> None:
    output = run_dir / "artifacts" / "blinded_pairs.jsonl"
    if output.exists():
        print(f"RUN_DIR={run_dir}")
        return
    source_run = Path(config["source"]["run_dir"]).resolve()
    validate_source(source_run, str(config["source"]["required_status"]))
    anchors = pd.DataFrame(read_jsonl(source_run / "artifacts" / "anchors.jsonl"))
    anchors = anchors[anchors.group.isin(config["sampling"]["groups"])].copy()
    neighbors = pd.DataFrame(read_jsonl(source_run / "artifacts" / "neighbors.jsonl"))
    points = read_jsonl(source_run / "artifacts" / "points.jsonl")
    selected = sample_selected(neighbors, anchors, config)
    candidates = control_candidates(
        source_run=source_run, anchors=anchors, points=points, config=config
    )
    controls = sample_controls(candidates, selected, config)
    memberships = selected + controls
    for row in memberships:
        row["audit_id"] = semantic_pair_id(row["state_id"], row["neighbor_point_id"])
    blinded = build_blinded_pairs(memberships, source_run=source_run, config=config)
    keep_columns = [
        "audit_id",
        "state_id",
        "prompt_index",
        "fraction_name",
        "group",
        "arm",
        "neighborhood",
        "representation",
        "neighbor_rank",
        "neighbor_point_id",
        "student_cosine_distance",
        "teacher_cosine_distance",
        "joint_distance",
        "selection_cell",
    ]
    atomic_write_jsonl(
        run_dir / "artifacts" / "sample_membership.jsonl",
        [{column: row[column] for column in keep_columns} for row in memberships],
    )
    atomic_write_jsonl(output, blinded)
    source_files = [
        "status.yaml",
        "config.yaml",
        "artifacts/anchors.jsonl",
        "artifacts/candidate_edges.jsonl",
        "artifacts/neighbors.jsonl",
        "artifacts/points.jsonl",
        "artifacts/trajectories.jsonl",
        "results/state_metrics.parquet",
    ]
    repository = Path(__file__).resolve().parents[2]
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    snapshot = {
        "source_run": str(source_run),
        "git_revision": git_revision,
        "git_dirty_at_prepare": git_dirty,
        "source_sha256": {
            relative: sha256_file(source_run / relative) for relative in source_files
        },
        "membership_count": len(memberships),
        "unique_pair_count": len(blinded),
        "selected_membership_count": len(selected),
        "control_membership_count": len(controls),
        "secondary_pair_count": sum(row["secondary_required"] for row in blinded),
    }
    write_json(run_dir / "artifacts" / "source_snapshot.json", snapshot)
    (run_dir / "annotations").mkdir(exist_ok=True)
    update_status(run_dir, "awaiting_annotations", **snapshot)
    print(f"RUN_DIR={run_dir}")
    print(f"UNIQUE_PAIRS={len(blinded)}")
    print(f"SECONDARY_PAIRS={snapshot['secondary_pair_count']}")


def main() -> int:
    args = parse_args()
    if args.run_dir is None:
        config_path = args.config.resolve()
        config = load_config(config_path)
        run_dir = create_run_dir(config, config_path)
    else:
        run_dir, config = load_run(args.run_dir)
    try:
        prepare(run_dir, config)
    except Exception as error:
        update_status(run_dir, "failed", error=repr(error))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
