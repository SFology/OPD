from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from common import create_run_dir, load_config, load_run, read_jsonl, update_status
from four_group_common import (
    atomic_write_jsonl,
    load_sharded,
    logical_shard,
    write_json,
)
from lcb_reliability_common import (
    aggregate_risk,
    lcb_trust,
    point_id,
    request_id,
    rollout_is_available,
)
from objective_reliability_common import stable_seed, validate_config
from run_lcb_reliability_calibration import (
    assert_frozen_revision,
    load_feature_arrays,
    load_score_map,
    wait_for_idle_gpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Objective teacher-reliability study")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    if args.config is None and args.run_dir is None:
        parser.error("one of --config or --run-dir is required")
    return args


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_source(run_dir: Path, config: dict[str, Any]) -> None:
    output = run_dir / "artifacts" / "state_manifest.jsonl"
    if output.exists():
        return
    lcb_run = Path(config["source"]["lcb_run"]).resolve()
    decode_run = Path(config["source"]["decode_run"]).resolve()
    for required in (
        lcb_run / "artifacts" / "anchors.jsonl",
        lcb_run / "artifacts" / "trajectories.jsonl",
        lcb_run / "artifacts" / "candidate_edges.jsonl",
        lcb_run / "artifacts" / "neighbors.jsonl",
        lcb_run / "results" / "state_metrics.parquet",
        decode_run / "state_manifest.jsonl",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    lcb_status = yaml.safe_load((lcb_run / "status.yaml").read_text())
    if lcb_status.get("status") != "completed":
        raise RuntimeError(f"LCB source is not completed: {lcb_status.get('status')}")

    decode_states = {
        row["state_id"]: row for row in read_jsonl(decode_run / "state_manifest.jsonl")
    }
    teacher_rows = {
        row["state_id"]: row
        for row in load_sharded(decode_run / "teacher_continuations")
    }
    selected = []
    for anchor in read_jsonl(lcb_run / "artifacts" / "anchors.jsonl"):
        if bool(anchor["student_correct"]) != bool(
            config["selection"]["student_correct"]
        ):
            continue
        if not anchor.get("eligible_pair", False):
            continue
        state = decode_states.get(anchor["state_id"])
        teacher = teacher_rows.get(anchor["state_id"])
        if state is None or teacher is None:
            raise KeyError(f"Missing frozen continuation: {anchor['state_id']}")
        if config["selection"]["require_existing_teacher_eligible"] and not teacher.get(
            "eligible", False
        ):
            continue
        selected.append(
            {
                **anchor,
                "input_ids": state["input_ids"],
                "ground_truth": state["ground_truth"],
                "existing_repeat": teacher,
            }
        )
    selected.sort(key=lambda row: row["state_id"])
    if not selected:
        raise RuntimeError("No states survived the objective selection")
    atomic_write_jsonl(output, selected)
    hardlink_or_copy(
        lcb_run / "artifacts" / "trajectories.jsonl",
        run_dir / "artifacts" / "trajectories.jsonl",
    )
    snapshot = {
        "lcb_run": str(lcb_run),
        "decode_run": str(decode_run),
        "state_count": len(selected),
        "q_counts": dict(Counter(row["fraction_name"] for row in selected)),
        "student_correct": bool(config["selection"]["student_correct"]),
    }
    write_json(run_dir / "artifacts" / "source_snapshot.json", snapshot)
    update_status(run_dir, "prepared", **snapshot)


def cosine_distances(anchor: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    anchor = anchor.astype(np.float32)
    candidates = candidates.astype(np.float32)
    anchor /= max(float(np.linalg.norm(anchor)), 1e-12)
    candidates /= np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12)
    return 1.0 - candidates @ anchor


def choose_control_indices(
    available: list[int], joint_distance: np.ndarray, count: int, seed: int
) -> tuple[list[int], list[int]]:
    if count == 0:
        return [], []
    if len(available) < count:
        raise RuntimeError(f"Only {len(available)} controls for matched count {count}")
    rng = np.random.default_rng(seed)
    random_choice = rng.choice(
        np.asarray(available), size=count, replace=False
    ).tolist()
    far_choice = sorted(available, key=lambda index: (-joint_distance[index], index))[
        :count
    ]
    return random_choice, far_choice


def build_supports_and_requests(run_dir: Path, config: dict[str, Any]) -> None:
    supports_path = run_dir / "artifacts" / "supports.jsonl"
    requests_path = run_dir / "artifacts" / "requests.jsonl"
    if supports_path.exists() and requests_path.exists():
        return
    source = Path(config["source"]["lcb_run"])
    source_config = load_run(source)[1]
    states = read_jsonl(run_dir / "artifacts" / "state_manifest.jsonl")
    state_ids = {row["state_id"] for row in states}
    points = read_jsonl(source / "artifacts" / "points.jsonl")
    point_lookup = {row["point_id"]: row for row in points}
    point_index = {row["point_id"]: index for index, row in enumerate(points)}
    edges_by_state: dict[str, list[dict]] = defaultdict(list)
    for edge in read_jsonl(source / "artifacts" / "candidate_edges.jsonl"):
        if edge["state_id"] in state_ids:
            edges_by_state[edge["state_id"]].append(edge)
    selected_by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in read_jsonl(source / "artifacts" / "neighbors.jsonl"):
        if row["state_id"] in state_ids:
            selected_by_key[
                (row["state_id"], row["neighborhood"], row["representation"])
            ].append(row)

    representations = list(config["support"]["representations"])
    student = load_feature_arrays(source, "student", points, representations)
    teacher = load_feature_arrays(source, "teacher", points, representations)
    variants = {row["name"]: row for row in source_config["neighborhood"]["variants"]}
    support_rows: list[dict] = []
    requests: dict[str, dict] = {}
    for number, state in enumerate(states, start=1):
        action = int(state["student_action_token_id"])
        anchor_pid = point_id(state["base_trajectory_id"], state["position"])
        anchor_request = request_id(
            state["base_trajectory_id"], state["position"], action
        )
        requests[anchor_request] = {
            "request_id": anchor_request,
            "prompt_index": int(state["prompt_index"]),
            "base_trajectory_id": state["base_trajectory_id"],
            "position": int(state["position"]),
            "action_token_id": action,
        }
        for neighborhood in config["support"]["neighborhoods"]:
            block = variants[neighborhood].get("rollout_block_size")
            allowed_edges = [
                edge
                for edge in edges_by_state[state["state_id"]]
                if rollout_is_available(
                    int(state["rollout_index"]),
                    int(edge["candidate_rollout_index"]),
                    None if block is None else int(block),
                )
            ]
            candidate_ids = list(
                dict.fromkeys(edge["candidate_point_id"] for edge in allowed_edges)
            )
            candidate_indices = np.asarray(
                [point_index[item] for item in candidate_ids]
            )
            for representation in representations:
                anchor_index = point_index[anchor_pid]
                sd = cosine_distances(
                    student[representation][anchor_index],
                    student[representation][candidate_indices],
                )
                td = cosine_distances(
                    teacher[representation][anchor_index],
                    teacher[representation][candidate_indices],
                )
                joint = np.maximum(sd, td)
                selected_rows = sorted(
                    selected_by_key[(state["state_id"], neighborhood, representation)],
                    key=lambda row: int(row["neighbor_rank"]),
                )
                selected_ids = [row["neighbor_point_id"] for row in selected_rows]
                selected_set = set(selected_ids)
                available = [
                    index
                    for index, candidate in enumerate(candidate_ids)
                    if candidate not in selected_set
                ]
                random_indices, far_indices = choose_control_indices(
                    available,
                    joint,
                    len(selected_ids),
                    stable_seed(state["state_id"], neighborhood, representation),
                )
                arm_ids = {
                    "selected": selected_ids,
                    "random": [candidate_ids[index] for index in random_indices],
                    "far": [candidate_ids[index] for index in far_indices],
                }
                candidate_position = {
                    item: index for index, item in enumerate(candidate_ids)
                }
                for arm, ids in arm_ids.items():
                    for rank, candidate_id in enumerate(ids):
                        index = candidate_position[candidate_id]
                        candidate = point_lookup[candidate_id]
                        rid = request_id(
                            candidate["base_trajectory_id"],
                            candidate["position"],
                            action,
                        )
                        requests[rid] = {
                            "request_id": rid,
                            "prompt_index": int(candidate["prompt_index"]),
                            "base_trajectory_id": candidate["base_trajectory_id"],
                            "position": int(candidate["position"]),
                            "action_token_id": action,
                        }
                        support_rows.append(
                            {
                                "state_id": state["state_id"],
                                "neighborhood": neighborhood,
                                "representation": representation,
                                "arm": arm,
                                "neighbor_rank": rank,
                                "neighbor_point_id": candidate_id,
                                "neighbor_request_id": rid,
                                "student_cosine_distance": float(sd[index]),
                                "teacher_cosine_distance": float(td[index]),
                            }
                        )
        if number % 100 == 0:
            print(f"built matched supports {number}/{len(states)}", flush=True)
    atomic_write_jsonl(supports_path, support_rows)
    atomic_write_jsonl(
        requests_path, sorted(requests.values(), key=lambda row: row["request_id"])
    )
    update_status(
        run_dir,
        "supports_prepared",
        support_edges=len(support_rows),
        exact_requests=len(requests),
    )


def required_shards(
    run_dir: Path, config: dict[str, Any], *, by_state: bool
) -> list[int]:
    count = int(config["parallel"]["logical_shards"])
    rows = read_jsonl(run_dir / "artifacts" / "state_manifest.jsonl")
    if by_state:
        return sorted({logical_shard(row["state_id"], count) for row in rows})
    return sorted(
        {logical_shard(f"prompt:{row['prompt_index']}", count) for row in rows}
    )


def run_workers(
    run_dir: Path, config: dict[str, Any], kind: str, role: str | None = None
) -> None:
    state_mode = kind == "generate"
    shards = required_shards(run_dir, config, by_state=state_mode)
    worker = Path(__file__).with_name(
        "objective_reliability_worker.py" if state_mode else "lcb_reliability_worker.py"
    )
    for retry in range(3):
        if state_mode:
            states = read_jsonl(run_dir / "artifacts" / "state_manifest.jsonl")
            pending_states = [
                row
                for row in states
                if not (
                    run_dir / "repeats" / "done" / f"{row['state_id']}.done"
                ).exists()
            ]
            missing = sorted(
                {
                    logical_shard(
                        row["state_id"], int(config["parallel"]["logical_shards"])
                    )
                    for row in pending_states
                }
            )
        else:
            root = run_dir / "scores" / str(role)
            missing = [
                s for s in shards if not (root / f"shard_{s:03d}.jsonl").exists()
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
        for worker_index, (gpu, assigned) in enumerate(assignments):
            log = (
                run_dir
                / "logs"
                / f"{kind}_{role or 'teacher'}_retry{retry}_worker{worker_index}_gpu{gpu}.log"
            )
            handle = log.open("w", encoding="utf-8")
            command = [sys.executable, "-u", str(worker)]
            if state_mode:
                command += [
                    "--run-dir",
                    str(run_dir),
                    "--shards",
                    ",".join(map(str, assigned)),
                ]
            else:
                command += [
                    "scores",
                    "--run-dir",
                    str(run_dir),
                    "--role",
                    str(role),
                    "--shards",
                    ",".join(map(str, assigned)),
                ]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, handle, log))
            print(
                f"[{kind}:{role or 'teacher'}] GPU {gpu}: shards={assigned}", flush=True
            )
        failures = []
        for process, handle, log in processes:
            code = process.wait()
            handle.close()
            if code:
                failures.append((code, str(log)))
        if failures:
            print(f"retry={retry + 1}/3 failures={failures}", flush=True)
    raise RuntimeError(f"Incomplete {kind}:{role or 'teacher'} after three attempts")


def collect_competence(run_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    target = int(config["selection"]["target_valid_teacher_repeats"])
    rows = []
    for state in read_jsonl(run_dir / "artifacts" / "state_manifest.jsonl"):
        repeats = [{**state["existing_repeat"], "attempt_index": 0}]
        path = run_dir / "repeats" / "by_state" / f"{state['state_id']}.jsonl"
        if path.exists():
            repeats.extend(read_jsonl(path))
        valid = sorted(
            (row for row in repeats if row.get("eligible", False)),
            key=lambda row: int(row["attempt_index"]),
        )[:target]
        rows.append(
            {
                "state_id": state["state_id"],
                "successes": sum(bool(row["correct"]) for row in valid),
                "trials": len(valid),
                "target_trials": target,
                "teacher_success_rate": (
                    sum(bool(row["correct"]) for row in valid) / len(valid)
                    if valid
                    else np.nan
                ),
                "complete": len(valid) == target,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(run_dir / "results" / "teacher_competence.parquet", index=False)
    frame.to_csv(run_dir / "results" / "teacher_competence.csv", index=False)
    return frame


def compute_metrics(run_dir: Path, config: dict[str, Any]) -> None:
    output = run_dir / "results" / "objective_state_metrics.parquet"
    if output.exists():
        return
    states = read_jsonl(run_dir / "artifacts" / "state_manifest.jsonl")
    supports = read_jsonl(run_dir / "artifacts" / "supports.jsonl")
    student = load_score_map(run_dir, "student")
    teacher = load_score_map(run_dir, "teacher")
    expected = {
        row["request_id"]
        for row in read_jsonl(run_dir / "artifacts" / "requests.jsonl")
    }
    for role, scores in (("student", student), ("teacher", teacher)):
        missing = expected - scores.keys()
        if missing:
            raise RuntimeError(f"Missing {len(missing)} {role} exact scores")
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in supports:
        grouped[
            (row["state_id"], row["arm"], row["neighborhood"], row["representation"])
        ].append(row)
    rows = []
    for state in states:
        action = int(state["student_action_token_id"])
        anchor_id = request_id(state["base_trajectory_id"], state["position"], action)
        anchor_teacher = teacher[anchor_id]
        anchor_student = student[anchor_id]
        anchor_reward = anchor_teacher - anchor_student
        for arm in config["support"]["arms"]:
            for neighborhood in config["support"]["neighborhoods"]:
                for representation in config["support"]["representations"]:
                    selected = grouped[
                        (state["state_id"], arm, neighborhood, representation)
                    ]
                    nt = np.asarray(
                        [teacher[row["neighbor_request_id"]] for row in selected]
                    )
                    ns = np.asarray(
                        [student[row["neighbor_request_id"]] for row in selected]
                    )
                    nr = nt - ns
                    deviations = np.abs(nr - anchor_reward)
                    risk = aggregate_risk(
                        deviations,
                        config["risk"]["aggregation"],
                        float(config["risk"]["temperature"]),
                    )
                    row = {
                        **{
                            key: value
                            for key, value in state.items()
                            if key not in {"input_ids", "existing_repeat"}
                        },
                        "arm": arm,
                        "neighborhood": neighborhood,
                        "representation": representation,
                        "neighbor_count": len(selected),
                        "anchor_teacher_log_prob": anchor_teacher,
                        "anchor_student_log_prob": anchor_student,
                        "anchor_opd_reward": anchor_reward,
                        "teacher_sensitivity": float(np.max(anchor_teacher - nt))
                        if len(nt)
                        else np.nan,
                        "student_sensitivity": float(np.max(anchor_student - ns))
                        if len(ns)
                        else np.nan,
                        "relative_sensitivity": float(np.max(anchor_reward - nr))
                        if len(nr)
                        else np.nan,
                        "lcb_risk": risk,
                        "risk_to_abs_reward": risk
                        / max(abs(anchor_reward), float(config["risk"]["epsilon"]))
                        if np.isfinite(risk)
                        else np.nan,
                    }
                    for value in config["risk"]["lambdas"]:
                        slug = str(value).replace(".", "p")
                        row[f"trust_lambda_{slug}"] = lcb_trust(
                            anchor_reward,
                            risk,
                            float(value),
                            float(config["risk"]["epsilon"]),
                        )
                    rows.append(row)
    metrics = pd.DataFrame(rows)
    competence = collect_competence(run_dir, config)
    metrics = metrics.merge(competence, on="state_id", validate="many_to_one")
    metrics.to_parquet(output, index=False)
    metrics.to_csv(run_dir / "results" / "objective_state_metrics.csv", index=False)
    update_status(
        run_dir,
        "metrics_computed",
        complete_states=int(competence.complete.sum()),
        incomplete_states=int((~competence.complete).sum()),
        metric_rows=len(metrics),
    )


def run_analysis(run_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(
                Path(__file__).with_name("analyze_objective_reliability_calibration.py")
            ),
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
        for directory in (
            "repeats/by_state",
            "repeats/done",
            "scores/student",
            "scores/teacher",
        ):
            (run_dir / directory).mkdir(parents=True, exist_ok=True)
    validate_config(config)
    print(f"RUN_DIR={run_dir}", flush=True)
    try:
        prepare_source(run_dir, config)
        build_supports_and_requests(run_dir, config)
        assert_frozen_revision()
        if args.prepare_only:
            print("PREPARE_ONLY=1", flush=True)
            return 0
        if args.analysis_only:
            compute_metrics(run_dir, config)
            run_analysis(run_dir)
            return 0
        update_status(run_dir, "generating_teacher_repeats")
        run_workers(run_dir, config, "generate")
        assert_frozen_revision()
        for role in ("student", "teacher"):
            update_status(run_dir, "scoring_exact_actions", role=role)
            run_workers(run_dir, config, "scores", role)
            assert_frozen_revision()
        compute_metrics(run_dir, config)
        run_analysis(run_dir)
        return 0
    except Exception as error:
        update_status(
            run_dir,
            "failed",
            stage="objective_reliability_calibration",
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
