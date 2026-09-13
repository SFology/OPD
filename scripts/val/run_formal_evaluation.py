from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from formal_eval_common import (
    generation_path,
    load_yaml,
    read_jsonl,
    sha256_file,
    update_status,
    utc_now,
    validate_generation_shard,
    write_jsonl,
    write_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched OPD/ROPD held-out evaluation.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if not args.status and not args.config and not (args.run_dir / "config.yaml").exists():
        parser.error("--config is required when preparing a new run directory")
    return args


def git_metadata() -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status),
        "status": status,
    }


def validate_model_source(model: dict[str, Any]) -> dict[str, Any]:
    path = Path(model["path"]).resolve()
    if model["type"] == "huggingface":
        required = [path / "config.json", path / "tokenizer_config.json"]
        if not any((path / name).exists() for name in ("model.safetensors", "model.safetensors.index.json")):
            raise FileNotFoundError(f"No HuggingFace weights found in {path}")
    elif model["type"] == "fsdp_checkpoint":
        required = [path / "fsdp_config.json", path / "huggingface" / "config.json"]
        fsdp = load_yaml(path / "fsdp_config.json")
        world_size = int(fsdp["world_size"])
        required.extend(path / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size))
        source_run = Path(model["source_run"])
        status = load_yaml(source_run / "status.yaml")
        if status.get("status") != "completed" or int(status.get("exit_code", -1)) != 0:
            raise RuntimeError(f"Source run is not a successful completed run: {source_run}")
        latest = int((source_run / "checkpoints" / "latest_checkpointed_iteration.txt").read_text().strip())
        if f"global_step_{latest}" not in str(path):
            raise RuntimeError(f"Checkpoint {path} is not source run's latest step {latest}")
    else:
        raise ValueError(f"Unsupported model type: {model['type']}")
    for required_path in required:
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    return {
        "name": model["name"],
        "type": model["type"],
        "path": str(path),
        "files": [
            {"path": str(item), "bytes": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns}
            for item in required
        ],
    }


def dataset_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise TypeError(f"Invalid prompt messages: {value!r}")
    messages = []
    for item in value:
        if not isinstance(item, dict) or "role" not in item or "content" not in item:
            raise TypeError(f"Invalid prompt message: {item!r}")
        messages.append({"role": str(item["role"]), "content": str(item["content"])})
    return messages


def build_prompt_manifest(run_dir: Path, config: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    from transformers import AutoTokenizer

    output = run_dir / "artifacts" / "prompt_manifest.jsonl"
    initial_model = next(item for item in config["models"] if item["type"] == "huggingface")
    tokenizer = AutoTokenizer.from_pretrained(
        initial_model["path"], local_files_only=True, trust_remote_code=True
    )
    rows = []
    dataset_records = []
    for dataset_index, dataset in enumerate(config["datasets"]):
        path = Path(dataset["path"]).resolve()
        frame = pd.read_parquet(path)
        dataset_records.append(
            {"name": dataset["name"], "path": str(path), "rows": len(frame), "sha256": sha256_file(path)}
        )
        for example_id, record in frame.iterrows():
            messages = dataset_messages(record["prompt"])
            prompt_tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            if len(prompt_tokens) > int(config["generation"]["max_prompt_tokens"]):
                raise ValueError(
                    f"{dataset['name']}:{example_id} has {len(prompt_tokens)} prompt tokens; "
                    f"limit={config['generation']['max_prompt_tokens']}"
                )
            reward_model = record["reward_model"]
            ground_truth = reward_model["ground_truth"]
            rows.append(
                {
                    "prompt_id": f"{dataset['name']}:{int(example_id)}",
                    "dataset": dataset["name"],
                    "dataset_index": dataset_index,
                    "example_id": int(example_id),
                    "messages": messages,
                    "ground_truth": str(ground_truth),
                    "prompt_tokens_initial_tokenizer": len(prompt_tokens),
                }
            )
    write_jsonl(output, rows)
    return rows, dataset_records


def snapshot_sources(run_dir: Path, config_path: Path) -> list[dict[str, Any]]:
    relative_paths = [
        Path("scripts/launch_opd_ropd_evaluation_tmux.sh"),
        Path("scripts/val/formal_eval_common.py"),
        Path("scripts/val/formal_eval_worker.py"),
        Path("scripts/val/run_formal_evaluation.py"),
        Path("scripts/val/analyze_formal_evaluation.py"),
        Path("scripts/val/audit_eval_contamination.py"),
        Path("scripts/val/prepare_extended_math_eval.py"),
        Path("verl/verl/utils/reward_score/ttrl_math/__init__.py"),
        Path("verl/verl/utils/reward_score/ttrl_math/math_utils.py"),
        Path("verl/tests/trainer/ppo/test_math_answer_parsing_on_cpu.py"),
        Path("verl/tests/trainer/ppo/test_formal_evaluation_on_cpu.py"),
    ]
    config_record = {
        "path": str(config_path.resolve()),
        "snapshot": "config.yaml",
        "bytes": config_path.stat().st_size,
        "source_sha256": sha256_file(config_path),
        "resolved_snapshot_sha256": sha256_file(run_dir / "config.yaml"),
    }
    records = []
    for relative in relative_paths:
        source = REPO_ROOT / relative
        destination = run_dir / "artifacts" / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": str(relative),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    return [config_record, *records]


def prepare_run(run_dir: Path, config_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if (run_dir / "config.yaml").exists():
        frozen = load_yaml(run_dir / "config.yaml")
        requested = load_yaml(config_path.resolve())
        if requested != frozen:
            raise RuntimeError(
                f"Refusing to resume {run_dir} with a config different from its frozen config.yaml"
            )
        return frozen
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to initialize non-empty run directory: {run_dir}")
    config = load_yaml(config_path.resolve())
    for name in ("artifacts", "generations", "logs", "merged_models", "results"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.yaml", config)
    update_status(
        run_dir,
        "preparing",
        created_at_utc=utc_now(),
        source_config=str(config_path.resolve()),
        run_path=str(run_dir),
    )
    prompt_rows, datasets = build_prompt_manifest(run_dir, config)
    models = [validate_model_source(model) for model in config["models"]]
    sources = snapshot_sources(run_dir, config_path)
    contamination = None
    contamination_config = config.get("contamination_audit")
    if contamination_config:
        report_path = Path(contamination_config["report"]).resolve()
        report = load_yaml(report_path)
        maximum = int(contamination_config.get("max_exact_train_overlap_unique_prompts", 0))
        observed = int(report["exact_train_overlap_unique_prompts"])
        if observed > maximum:
            raise RuntimeError(
                f"Contamination audit found {observed} exact train overlaps; allowed={maximum}"
            )
        contamination = {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "exact_train_overlap_unique_prompts": observed,
            "exact_train_overlap_rows": int(report["exact_train_overlap_rows"]),
            "duplicate_evaluation_unique_prompts": int(
                report["duplicate_evaluation_unique_prompts"]
            ),
            "scope_note": report["scope_note"],
        }
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "run_dir": str(run_dir),
        "source_config": str(config_path.resolve()),
        "config_sha256": sha256_file(run_dir / "config.yaml"),
        "git": git_metadata(),
        "datasets": datasets,
        "models": models,
        "source_snapshot": sources,
        "contamination_audit": contamination,
        "unique_prompts": len(prompt_rows),
        "rollouts_per_prompt": int(config["generation"]["rollouts_per_prompt"]),
        "expected_generations": len(prompt_rows)
        * int(config["generation"]["rollouts_per_prompt"])
        * len(config["models"]),
        "scientific_contract": {
            "paired_request_seeds_across_models": True,
            "unparseable_outputs_count_as_incorrect": True,
            "at_token_limit_outputs_count_as_incorrect_unless_answer_is_parseable_and_correct": True,
            "primary_comparison": " - ".join(
                config["analysis"].get(
                    "primary_comparison",
                    config["analysis"]["comparisons"][-1],
                )
            ),
        },
    }
    write_yaml(run_dir / "manifest.yaml", manifest)
    update_status(
        run_dir,
        "prepared",
        unique_prompts=len(prompt_rows),
        expected_generations=manifest["expected_generations"],
    )
    return config


def merge_models(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    resolved = {}
    for model in config["models"]:
        source = Path(model["path"]).resolve()
        if model["type"] == "huggingface":
            inference_path = source
        else:
            target = run_dir / "merged_models" / model["name"]
            marker = target / "merge_complete.yaml"
            if marker.exists():
                inference_path = target
            else:
                if target.exists():
                    raise RuntimeError(
                        f"Incomplete merged target exists: {target}. Preserve or move it before retrying."
                    )
                temporary = target.with_name(f"{target.name}.partial-{os.getpid()}")
                if temporary.exists():
                    raise RuntimeError(f"Temporary merge directory already exists: {temporary}")
                log_path = run_dir / "logs" / f"merge_{model['name']}.log"
                update_status(
                    run_dir,
                    "running",
                    stage=f"merge:{model['name']}",
                    clear_fields=("model", "completed_shards", "total_shards"),
                )
                command = [
                    sys.executable,
                    "-m",
                    "verl.model_merger",
                    "merge",
                    "--backend",
                    "fsdp",
                    "--local_dir",
                    str(source),
                    "--target_dir",
                    str(temporary),
                    "--use_cpu_initialization",
                ]
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                if not (temporary / "config.json").exists() or not any(temporary.glob("*.safetensors")):
                    raise RuntimeError(f"Merge did not create complete HuggingFace weights: {temporary}")
                write_yaml(
                    temporary / "merge_complete.yaml",
                    {"completed_at_utc": utc_now(), "source_checkpoint": str(source)},
                )
                temporary.replace(target)
                inference_path = target
        resolved[model["name"]] = {
            "type": model["type"],
            "source_path": str(source),
            "source_run": model.get("source_run"),
            "inference_path": str(inference_path),
        }
    write_yaml(run_dir / "artifacts" / "model_paths.yaml", resolved)
    return resolved


def query_idle_gpus(config: dict[str, Any]) -> list[int]:
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
    parallel = config["parallel"]
    candidates = []
    for line in result.stdout.splitlines():
        index, free_memory, utilization = [int(item.strip()) for item in line.split(",")]
        if free_memory >= int(parallel["min_free_mb"]) and utilization <= int(parallel["max_utilization"]):
            candidates.append((index, free_memory, utilization))
    candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
    maximum = parallel.get("max_gpus")
    if maximum is not None:
        candidates = candidates[: int(maximum)]
    return [item[0] for item in candidates]


def wait_for_idle_gpus(run_dir: Path, config: dict[str, Any], model: str) -> list[int]:
    parallel = config["parallel"]
    deadline = time.monotonic() + int(parallel["gpu_wait_seconds"])
    while True:
        selected = query_idle_gpus(config)
        if selected:
            time.sleep(int(parallel["gpu_stability_seconds"]))
            confirmed = set(query_idle_gpus(config))
            stable = [gpu for gpu in selected if gpu in confirmed]
            if stable:
                print(f"[{model}] selected idle physical GPUs: {stable}", flush=True)
                return stable
        update_status(run_dir, "waiting_for_gpu", stage=f"generation:{model}")
        if time.monotonic() >= deadline:
            raise RuntimeError("No GPU satisfied the configured idle thresholds before the deadline")
        print(f"[{model}] waiting for an idle GPU", flush=True)
        time.sleep(int(parallel["gpu_poll_seconds"]))


def expected_prompt_counts(prompt_rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["dataset"] for row in prompt_rows))


def missing_shards(
    run_dir: Path,
    config: dict[str, Any],
    model: str,
    *,
    quarantine_invalid: bool = True,
) -> list[tuple[str, int]]:
    prompt_rows = read_jsonl(run_dir / "artifacts" / "prompt_manifest.jsonl")
    counts = expected_prompt_counts(prompt_rows)
    prompt_ids = {
        dataset: {row["prompt_id"] for row in prompt_rows if row["dataset"] == dataset}
        for dataset in counts
    }
    expected = [
        (dataset["name"], rollout)
        for dataset in config["datasets"]
        for rollout in range(int(config["generation"]["rollouts_per_prompt"]))
    ]
    missing = []
    for dataset, rollout in expected:
        path = generation_path(run_dir, model, dataset, rollout)
        if validate_generation_shard(
            path,
            model=model,
            dataset=dataset,
            rollout=rollout,
            expected_prompts=counts[dataset],
            expected_prompt_ids=prompt_ids[dataset],
        ):
            continue
        if path.exists() and quarantine_invalid:
            quarantine = path.with_suffix(path.suffix + f".invalid-{int(time.time())}")
            path.replace(quarantine)
            print(f"Quarantined invalid shard: {quarantine}", flush=True)
        missing.append((dataset, rollout))
    return missing


def run_generation(run_dir: Path, config: dict[str, Any], model: str) -> None:
    worker = Path(__file__).with_name("formal_eval_worker.py")
    retries = int(config["parallel"]["retries"])
    total_shards = len(config["datasets"]) * int(config["generation"]["rollouts_per_prompt"])
    for retry in range(retries):
        missing = missing_shards(run_dir, config, model)
        if not missing:
            return
        completed = total_shards - len(missing)
        update_status(
            run_dir,
            "running",
            stage=f"generation:{model}",
            model=model,
            completed_shards=completed,
            total_shards=total_shards,
        )
        gpus = wait_for_idle_gpus(run_dir, config, model)
        assignments = [(gpu, missing[index :: len(gpus)]) for index, gpu in enumerate(gpus)]
        processes = []
        for worker_index, (gpu, shards) in enumerate(assignments):
            if not shards:
                continue
            shard_arg = ",".join(f"{dataset}:{rollout}" for dataset, rollout in shards)
            command = [
                sys.executable,
                "-u",
                str(worker),
                "--run-dir",
                str(run_dir),
                "--model",
                model,
                "--shards",
                shard_arg,
            ]
            log_path = run_dir / "logs" / f"generate_{model}_retry{retry}_worker{worker_index}_gpu{gpu}.log"
            log = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["TOKENIZERS_PARALLELISM"] = "true"
            environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            print(f"[{model}] GPU {gpu}: {len(shards)} shards", flush=True)
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, log, log_path))
        failures = []
        for process, log, log_path in processes:
            return_code = process.wait()
            log.close()
            if return_code:
                failures.append({"return_code": return_code, "log": str(log_path)})
        remaining = missing_shards(run_dir, config, model)
        if not remaining:
            return
        print(
            f"[{model}] retry {retry + 1}/{retries}: remaining={len(remaining)}, failures={failures}",
            flush=True,
        )
    raise RuntimeError(f"Generation for {model} remains incomplete after {retries} attempts")


def record_source_run_pointers(run_dir: Path, config: dict[str, Any]) -> None:
    for model in config["models"]:
        source_run = model.get("source_run")
        if not source_run:
            continue
        pointer = Path(source_run) / "evaluation" / f"{run_dir.name}.yaml"
        write_yaml(
            pointer,
            {
                "evaluation_run": str(run_dir),
                "model": model["name"],
                "checkpoint": model["path"],
                "summary": str(run_dir / "results" / "summary.json"),
                "dashboard": str(run_dir / "results" / "figures" / "dashboard.html"),
                "completed_at_utc": utc_now(),
            },
        )


def show_status(run_dir: Path) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    status = load_yaml(run_dir / "status.yaml")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    config = load_yaml(run_dir / "config.yaml")
    for model in config["models"]:
        complete = len(config["datasets"]) * int(config["generation"]["rollouts_per_prompt"])
        missing = len(
            missing_shards(
                run_dir, config, model["name"], quarantine_invalid=False
            )
        )
        print(f"{model['name']}: shards={complete - missing}/{complete}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.status:
        show_status(run_dir)
        return
    if args.config is not None:
        config_path = args.config.resolve()
    elif (run_dir / "status.yaml").exists():
        config_path = Path(load_yaml(run_dir / "status.yaml")["source_config"])
    else:
        raise ValueError("--config is required for a new run")
    config = prepare_run(run_dir, config_path)
    if args.prepare_only:
        print(f"Prepared evaluation run: {run_dir}")
        print(f"Expected generations: {load_yaml(run_dir / 'manifest.yaml')['expected_generations']}")
        return
    try:
        current_status = load_yaml(run_dir / "status.yaml")
        start_fields = {}
        if "started_at_utc" not in current_status:
            start_fields["started_at_utc"] = utc_now()
        update_status(
            run_dir,
            "running",
            stage="merge",
            clear_fields=("model", "completed_shards", "total_shards"),
            **start_fields,
        )
        merge_models(run_dir, config)
        for model in config["models"]:
            run_generation(run_dir, config, model["name"])
        update_status(
            run_dir,
            "running",
            stage="analysis",
            clear_fields=("model", "completed_shards", "total_shards"),
        )
        analysis_log = run_dir / "logs" / "analysis.log"
        with analysis_log.open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).with_name("analyze_formal_evaluation.py")),
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        record_source_run_pointers(run_dir, config)
        update_status(
            run_dir,
            "completed",
            stage="complete",
            clear_fields=("model", "completed_shards", "total_shards", "error", "traceback"),
            finished_at_utc=utc_now(),
            dashboard=str(run_dir / "results" / "figures" / "dashboard.html"),
        )
        print(f"Formal evaluation completed: {run_dir}", flush=True)
    except Exception as error:
        update_status(
            run_dir,
            "failed",
            failed_at_utc=utc_now(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
