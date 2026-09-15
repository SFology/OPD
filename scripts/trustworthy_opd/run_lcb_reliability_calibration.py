from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from common import (
    create_run_dir,
    load_config,
    load_run,
    read_jsonl,
    update_status,
)
from four_group_common import (
    atomic_write_jsonl,
    load_sharded,
    logical_shard,
    write_json,
)
from lcb_reliability_common import (
    aggregate_risk,
    candidate_positions,
    lcb_trust,
    point_id,
    request_id,
    rollout_is_available,
    select_dual_ball_neighbors,
    source_path,
    stable_fold,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline LCB reliability calibration"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    if args.run_dir is None and args.config is None:
        parser.error("--config is required when --run-dir is not supplied")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_revision() -> None:
    expected = os.environ.get("OPD_FROZEN_GIT_REVISION")
    if not expected:
        return
    repository = Path(__file__).resolve().parents[2]
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != expected or dirty:
        raise RuntimeError(
            "Repository changed during the frozen calibration run: "
            f"expected={expected}, current={current}, dirty={bool(dirty)}"
        )


def prepare_source(run_dir: Path, config: dict[str, Any]) -> None:
    output = run_dir / "artifacts" / "candidate_edges.jsonl"
    if output.exists():
        return
    source_run = Path(config["source"]["run_dir"]).resolve()
    frozen = source_path(source_run, config["source"]["frozen_manifest"])
    labels_path = source_path(source_run, config["source"]["labels"])
    states_path = source_run / "artifacts" / "states.jsonl"
    for required in (frozen, labels_path, states_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("freeze_four_group_rounds.py")),
            "--run-dir",
            str(source_run),
            "--completed-rounds",
            str(config["source"]["completed_rounds"]),
            "--verify-only",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    states = {row["state_id"]: row for row in read_jsonl(states_path)}
    fresh_labels = read_jsonl(labels_path)
    anchors = []
    for label in fresh_labels:
        if not label.get("eligible_pair", False):
            continue
        state = states.get(label["state_id"])
        if state is None:
            raise KeyError(f"Fresh label has no frozen state: {label['state_id']}")
        anchors.append(
            {
                **label,
                "round": int(state["round"]),
                "rollout_index": int(state["rollout_index"]),
                "base_trajectory_id": state["base_trajectory_id"],
                "fraction": float(state["fraction"]),
                "position": int(state["position"]),
                "normalized_position": float(state["normalized_position"]),
                "student_action_token_id": int(state["student_action_token_id"]),
                "fold": stable_fold(
                    int(state["prompt_index"]),
                    int(config["experiment"]["seed"]),
                    int(config["analysis"]["prompt_folds"]),
                ),
            }
        )

    trajectories = []
    for round_index in range(int(config["source"]["completed_rounds"])):
        trajectories.extend(
            load_sharded(source_run / "rounds" / f"round_{round_index:03d}" / "base")
        )
    trajectories.sort(
        key=lambda row: (int(row["prompt_index"]), int(row["rollout_index"]))
    )
    trajectory_lookup = {row["base_trajectory_id"]: row for row in trajectories}
    by_prompt: dict[int, list[dict]] = defaultdict(list)
    for row in trajectories:
        by_prompt[int(row["prompt_index"])].append(row)
    bad_prompts = [prompt for prompt, rows in by_prompt.items() if len(rows) != 8]
    if bad_prompts:
        raise RuntimeError(f"Expected eight frozen rollouts per prompt: {bad_prompts}")

    neighborhood = config["neighborhood"]
    edges: list[dict] = []
    point_rows: dict[str, dict] = {}
    for anchor in anchors:
        anchor_trajectory = trajectory_lookup[anchor["base_trajectory_id"]]
        anchor_length = len(anchor_trajectory["generated_token_ids"])
        anchor_point = point_id(anchor["base_trajectory_id"], anchor["position"])
        point_rows[anchor_point] = {
            "point_id": anchor_point,
            "prompt_index": int(anchor["prompt_index"]),
            "rollout_index": int(anchor["rollout_index"]),
            "base_trajectory_id": anchor["base_trajectory_id"],
            "position": int(anchor["position"]),
        }
        for candidate_trajectory in by_prompt[int(anchor["prompt_index"])]:
            if int(candidate_trajectory["rollout_index"]) == int(
                anchor["rollout_index"]
            ):
                continue
            positions = candidate_positions(
                anchor_position=int(anchor["position"]),
                anchor_length=anchor_length,
                target_length=len(candidate_trajectory["generated_token_ids"]),
                stride=int(neighborhood["candidate_stride"]),
                offsets=int(neighborhood["candidate_offsets"]),
                progress_window=float(neighborhood["progress_window"]),
            )
            for position in positions:
                candidate_point = point_id(
                    candidate_trajectory["base_trajectory_id"], position
                )
                point_rows[candidate_point] = {
                    "point_id": candidate_point,
                    "prompt_index": int(anchor["prompt_index"]),
                    "rollout_index": int(candidate_trajectory["rollout_index"]),
                    "base_trajectory_id": candidate_trajectory["base_trajectory_id"],
                    "position": position,
                }
                edges.append(
                    {
                        "state_id": anchor["state_id"],
                        "anchor_point_id": anchor_point,
                        "candidate_point_id": candidate_point,
                        "candidate_rollout_index": int(
                            candidate_trajectory["rollout_index"]
                        ),
                    }
                )

    atomic_write_jsonl(run_dir / "artifacts" / "trajectories.jsonl", trajectories)
    atomic_write_jsonl(run_dir / "artifacts" / "anchors.jsonl", anchors)
    atomic_write_jsonl(
        run_dir / "artifacts" / "points.jsonl",
        sorted(point_rows.values(), key=lambda row: row["point_id"]),
    )
    atomic_write_jsonl(output, edges)
    source_record = {
        "source_run": str(source_run),
        "frozen_manifest": str(frozen),
        "frozen_manifest_sha256": sha256_file(frozen),
        "fresh_labels": str(labels_path),
        "fresh_labels_sha256": sha256_file(labels_path),
        "states": str(states_path),
        "states_sha256": sha256_file(states_path),
        "trajectory_count": len(trajectories),
        "anchor_count": len(anchors),
        "point_count": len(point_rows),
        "candidate_edge_count": len(edges),
        "group_counts": dict(Counter(row["group"] for row in anchors)),
        "q_group_counts": {
            q: dict(
                Counter(row["group"] for row in anchors if row["fraction_name"] == q)
            )
            for q in ("q20", "q40", "q60", "q80")
        },
    }
    write_json(run_dir / "artifacts" / "source_snapshot.json", source_record)
    update_status(run_dir, "prepared", **source_record)


def query_idle_gpus(config: dict[str, Any]) -> list[int]:
    parallel = config["parallel"]
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = []
    for line in result.stdout.splitlines():
        index, memory, utilization = [int(item.strip()) for item in line.split(",")]
        if memory >= int(parallel["min_free_mb"]) and utilization <= int(
            parallel["max_utilization"]
        ):
            candidates.append((index, memory, utilization))
    candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
    maximum = parallel.get("max_gpus")
    if maximum is not None:
        candidates = candidates[: int(maximum)]
    return [item[0] for item in candidates]


def wait_for_idle_gpus(config: dict[str, Any]) -> list[int]:
    parallel = config["parallel"]
    deadline = time.monotonic() + int(parallel["gpu_wait_seconds"])
    while True:
        selected = query_idle_gpus(config)
        if selected:
            time.sleep(int(parallel["gpu_stability_seconds"]))
            confirmed = set(query_idle_gpus(config))
            stable = [gpu for gpu in selected if gpu in confirmed]
            if stable:
                print(f"Selected idle physical GPUs: {stable}", flush=True)
                return stable
        if time.monotonic() >= deadline:
            raise RuntimeError("No GPU satisfied the configured idle thresholds")
        print("Waiting for at least one idle GPU", flush=True)
        time.sleep(int(parallel["gpu_poll_seconds"]))


def run_workers(
    run_dir: Path,
    config: dict[str, Any],
    mode: str,
    role: str,
    required_shards: list[int],
) -> None:
    worker = Path(__file__).with_name("lcb_reliability_worker.py")
    output_root = run_dir / ("features" if mode == "features" else "scores") / role
    output_suffix = ".npz" if mode == "features" else ".jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    for retry in range(3):
        missing = [
            shard
            for shard in required_shards
            if not (output_root / f"shard_{shard:03d}{output_suffix}").exists()
        ]
        if not missing:
            return
        gpus = wait_for_idle_gpus(config)
        assignments = [
            (gpu, missing[index :: len(gpus)])
            for index, gpu in enumerate(gpus)
            if missing[index :: len(gpus)]
        ]
        processes = []
        for worker_index, (gpu, shards) in enumerate(assignments):
            log_path = (
                run_dir
                / "logs"
                / (f"{mode}_{role}_retry{retry}_worker{worker_index}_gpu{gpu}.log")
            )
            handle = log_path.open("w", encoding="utf-8")
            command = [
                sys.executable,
                "-u",
                str(worker),
                mode,
                "--run-dir",
                str(run_dir),
                "--role",
                role,
                "--shards",
                ",".join(map(str, shards)),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[{mode}:{role}] GPU {gpu}: shards={shards}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, handle, log_path))
        failures = []
        for process, handle, log_path in processes:
            return_code = process.wait()
            handle.close()
            if return_code:
                failures.append((return_code, str(log_path)))
        remaining = [
            shard
            for shard in missing
            if not (output_root / f"shard_{shard:03d}{output_suffix}").exists()
        ]
        if not remaining:
            return
        print(
            f"Retry {retry + 1}/3 for {mode}:{role}; remaining={remaining}; "
            f"failures={failures}",
            flush=True,
        )
    raise RuntimeError(f"Incomplete {mode}:{role} shards after retries")


def load_feature_arrays(
    run_dir: Path, role: str, points: list[dict], representation_names: list[str]
) -> dict[str, np.ndarray]:
    point_order = {row["point_id"]: index for index, row in enumerate(points)}
    result: dict[str, np.ndarray] = {}
    seen = np.zeros(len(points), dtype=bool)
    for path in sorted((run_dir / "features" / role).glob("shard_*.npz")):
        with np.load(path) as payload:
            ids = payload["point_ids"].tolist()
            indices = np.asarray([point_order[item] for item in ids], dtype=np.int64)
            seen[indices] = True
            for name in representation_names:
                values = payload[f"representation__{name}"].astype(np.float32)
                if name not in result:
                    result[name] = np.empty(
                        (len(points), values.shape[1]), dtype=np.float32
                    )
                result[name][indices] = values
    if not bool(seen.all()):
        raise RuntimeError(f"Missing {role} features for {int((~seen).sum())} points")
    return result


def build_neighbors_and_requests(run_dir: Path, config: dict[str, Any]) -> None:
    neighbors_path = run_dir / "artifacts" / "neighbors.jsonl"
    requests_path = run_dir / "artifacts" / "requests.jsonl"
    if neighbors_path.exists() and requests_path.exists():
        return
    anchors = read_jsonl(run_dir / "artifacts" / "anchors.jsonl")
    points = read_jsonl(run_dir / "artifacts" / "points.jsonl")
    edges = read_jsonl(run_dir / "artifacts" / "candidate_edges.jsonl")
    point_index = {row["point_id"]: index for index, row in enumerate(points)}
    point_lookup = {row["point_id"]: row for row in points}
    representation_names = [
        item["name"] for item in config["representations"]["definitions"]
    ]
    student = load_feature_arrays(run_dir, "student", points, representation_names)
    teacher = load_feature_arrays(run_dir, "teacher", points, representation_names)
    edges_by_anchor: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        edges_by_anchor[edge["state_id"]].append(edge)
    neighborhood = config["neighborhood"]
    neighbor_rows: list[dict] = []
    request_rows: dict[str, dict] = {}
    for anchor_number, anchor in enumerate(anchors):
        action = int(anchor["student_action_token_id"])
        anchor_request = request_id(
            anchor["base_trajectory_id"], anchor["position"], action
        )
        request_rows[anchor_request] = {
            "request_id": anchor_request,
            "prompt_index": int(anchor["prompt_index"]),
            "base_trajectory_id": anchor["base_trajectory_id"],
            "position": int(anchor["position"]),
            "action_token_id": action,
        }
        all_edges = edges_by_anchor[anchor["state_id"]]
        for variant in neighborhood["variants"]:
            block = variant.get("rollout_block_size")
            block = None if block is None else int(block)
            allowed = [
                edge
                for edge in all_edges
                if rollout_is_available(
                    int(anchor["rollout_index"]),
                    int(edge["candidate_rollout_index"]),
                    block,
                )
            ]
            candidate_ids = [edge["candidate_point_id"] for edge in allowed]
            candidate_indices = np.asarray(
                [point_index[item] for item in candidate_ids], dtype=np.int64
            )
            anchor_index = point_index[
                point_id(anchor["base_trajectory_id"], anchor["position"])
            ]
            for representation in representation_names:
                chosen, student_distance, teacher_distance = select_dual_ball_neighbors(
                    anchor_student=student[representation][anchor_index],
                    anchor_teacher=teacher[representation][anchor_index],
                    candidate_student=student[representation][candidate_indices],
                    candidate_teacher=teacher[representation][candidate_indices],
                    radius_quantile=float(neighborhood["radius_quantile"]),
                    neighbor_k=int(neighborhood["neighbor_k"]),
                    minimum_neighbors=int(neighborhood["minimum_neighbors"]),
                    max_student_distance=neighborhood.get(
                        "max_student_cosine_distance"
                    ),
                    max_teacher_distance=neighborhood.get(
                        "max_teacher_cosine_distance"
                    ),
                )
                for rank, candidate_local_index in enumerate(chosen.tolist()):
                    candidate_id = candidate_ids[candidate_local_index]
                    candidate = point_lookup[candidate_id]
                    neighbor_request = request_id(
                        candidate["base_trajectory_id"], candidate["position"], action
                    )
                    request_rows[neighbor_request] = {
                        "request_id": neighbor_request,
                        "prompt_index": int(candidate["prompt_index"]),
                        "base_trajectory_id": candidate["base_trajectory_id"],
                        "position": int(candidate["position"]),
                        "action_token_id": action,
                    }
                    neighbor_rows.append(
                        {
                            "state_id": anchor["state_id"],
                            "neighborhood": variant["name"],
                            "representation": representation,
                            "neighbor_rank": rank,
                            "neighbor_point_id": candidate_id,
                            "neighbor_request_id": neighbor_request,
                            "student_cosine_distance": float(student_distance[rank]),
                            "teacher_cosine_distance": float(teacher_distance[rank]),
                        }
                    )
        if (anchor_number + 1) % 100 == 0:
            print(f"selected neighbors {anchor_number + 1}/{len(anchors)}", flush=True)
    atomic_write_jsonl(neighbors_path, neighbor_rows)
    atomic_write_jsonl(
        requests_path,
        sorted(request_rows.values(), key=lambda row: row["request_id"]),
    )
    update_status(
        run_dir,
        "neighbors_selected",
        neighbor_edges=len(neighbor_rows),
        exact_requests=len(request_rows),
    )


def load_score_map(run_dir: Path, role: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for path in sorted((run_dir / "scores" / role).glob("shard_*.jsonl")):
        for row in read_jsonl(path):
            request = row["request_id"]
            if request in result:
                raise RuntimeError(f"Duplicate {role} score: {request}")
            result[request] = float(row["log_prob"])
    return result


def compute_metrics(run_dir: Path, config: dict[str, Any]) -> None:
    output = run_dir / "results" / "state_metrics.parquet"
    if output.exists():
        return
    anchors = read_jsonl(run_dir / "artifacts" / "anchors.jsonl")
    neighbors = read_jsonl(run_dir / "artifacts" / "neighbors.jsonl")
    student_scores = load_score_map(run_dir, "student")
    teacher_scores = load_score_map(run_dir, "teacher")
    requests = read_jsonl(run_dir / "artifacts" / "requests.jsonl")
    expected_requests = {row["request_id"] for row in requests}
    for role, scores in (("student", student_scores), ("teacher", teacher_scores)):
        missing = expected_requests - scores.keys()
        if missing:
            raise RuntimeError(f"Missing {len(missing)} {role} exact request scores")
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in neighbors:
        by_key[(row["state_id"], row["neighborhood"], row["representation"])].append(
            row
        )
    rows = []
    risk_config = config["risk"]
    representations = [
        item["name"] for item in config["representations"]["definitions"]
    ]
    variants = [item["name"] for item in config["neighborhood"]["variants"]]
    for anchor in anchors:
        action = int(anchor["student_action_token_id"])
        anchor_request = request_id(
            anchor["base_trajectory_id"], anchor["position"], action
        )
        anchor_student = student_scores[anchor_request]
        anchor_teacher = teacher_scores[anchor_request]
        anchor_reward = anchor_teacher - anchor_student
        for variant in variants:
            for representation in representations:
                selected = sorted(
                    by_key[(anchor["state_id"], variant, representation)],
                    key=lambda row: int(row["neighbor_rank"]),
                )
                neighbor_teacher = np.asarray(
                    [teacher_scores[row["neighbor_request_id"]] for row in selected],
                    dtype=np.float64,
                )
                neighbor_student = np.asarray(
                    [student_scores[row["neighbor_request_id"]] for row in selected],
                    dtype=np.float64,
                )
                neighbor_rewards = neighbor_teacher - neighbor_student
                deviations = np.abs(neighbor_rewards - anchor_reward)
                risk = aggregate_risk(
                    deviations,
                    str(risk_config["aggregation"]),
                    float(risk_config["temperature"]),
                )
                row: dict[str, Any] = {
                    **anchor,
                    "neighborhood": variant,
                    "representation": representation,
                    "neighbor_count": len(selected),
                    "anchor_teacher_log_prob": anchor_teacher,
                    "anchor_student_log_prob": anchor_student,
                    "anchor_opd_reward": anchor_reward,
                    "teacher_sensitivity": float(
                        np.max(anchor_teacher - neighbor_teacher)
                    )
                    if len(selected)
                    else float("nan"),
                    "student_sensitivity": float(
                        np.max(anchor_student - neighbor_student)
                    )
                    if len(selected)
                    else float("nan"),
                    "relative_sensitivity": float(
                        np.max(anchor_reward - neighbor_rewards)
                    )
                    if len(selected)
                    else float("nan"),
                    "lcb_risk": risk,
                    "risk_to_abs_reward": (
                        risk / max(abs(anchor_reward), float(risk_config["epsilon"]))
                        if np.isfinite(risk)
                        else float("nan")
                    ),
                    "student_distance_mean": float(
                        np.mean([item["student_cosine_distance"] for item in selected])
                    )
                    if selected
                    else float("nan"),
                    "teacher_distance_mean": float(
                        np.mean([item["teacher_cosine_distance"] for item in selected])
                    )
                    if selected
                    else float("nan"),
                    "neighbor_point_ids": json.dumps(
                        [item["neighbor_point_id"] for item in selected]
                    ),
                }
                for value in risk_config["lambdas"]:
                    slug = str(value).replace(".", "p")
                    trust = lcb_trust(
                        anchor_reward,
                        risk,
                        float(value),
                        float(risk_config["epsilon"]),
                    )
                    row[f"trust_lambda_{slug}"] = trust
                    row[f"trust_drop_unsupported_lambda_{slug}"] = (
                        trust if len(selected) else 0.0
                    )
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output, index=False)
    frame.drop(columns=["neighbor_point_ids"]).to_csv(
        run_dir / "results" / "state_metrics.csv", index=False
    )
    update_status(
        run_dir,
        "metrics_computed",
        metric_rows=len(frame),
        metric_states=int(frame.state_id.nunique()),
        supported_rows=int((frame.neighbor_count > 0).sum()),
    )


def required_prompt_shards(run_dir: Path, config: dict[str, Any]) -> list[int]:
    count = int(config["parallel"]["logical_shards"])
    return sorted(
        {
            logical_shard(f"prompt:{row['prompt_index']}", count)
            for row in read_jsonl(run_dir / "artifacts" / "points.jsonl")
        }
    )


def run_analysis(run_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("analyze_lcb_reliability_calibration.py")),
            "--run-dir",
            str(run_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def main() -> int:
    args = parse_args()
    if args.run_dir:
        run_dir, config = load_run(args.run_dir)
    else:
        config = load_config(args.config)
        validate_config(config)
        run_dir = create_run_dir(config, args.config)
        for directory in ("scores/student", "scores/teacher"):
            (run_dir / directory).mkdir(parents=True, exist_ok=True)
    validate_config(config)
    print(f"RUN_DIR={run_dir}", flush=True)
    try:
        prepare_source(run_dir, config)
        assert_frozen_revision()
        if args.prepare_only:
            print("PREPARE_ONLY=1", flush=True)
            return 0
        if args.analysis_only:
            compute_metrics(run_dir, config)
            run_analysis(run_dir)
            return 0
        shards = required_prompt_shards(run_dir, config)
        for role in ("student", "teacher"):
            update_status(run_dir, "extracting_features", role=role)
            run_workers(run_dir, config, "features", role, shards)
            assert_frozen_revision()
        update_status(run_dir, "selecting_neighbors")
        build_neighbors_and_requests(run_dir, config)
        assert_frozen_revision()
        for role in ("student", "teacher"):
            update_status(run_dir, "scoring_exact_actions", role=role)
            run_workers(run_dir, config, "scores", role, shards)
            assert_frozen_revision()
        compute_metrics(run_dir, config)
        run_analysis(run_dir)
        return 0
    except Exception as error:
        update_status(
            run_dir,
            "failed",
            stage="lcb_reliability_calibration",
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
