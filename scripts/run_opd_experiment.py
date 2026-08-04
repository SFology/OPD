#!/usr/bin/env python3
"""Managed launcher for reproducible OPD experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"cyclic config inheritance at {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_data = load_config(path.parent / parent, seen)
        data = deep_merge(parent_data, data)
    return data


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"unresolved environment variable in {value!r}")
        return os.path.expanduser(expanded)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def set_dotted(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"override must be KEY=VALUE: {expression}")
    dotted, raw_value = expression.split("=", 1)
    value = yaml.safe_load(raw_value)
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def run_capture(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return result.stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def git_info() -> dict[str, Any]:
    commit = run_capture(["git", "rev-parse", "HEAD"])
    status = run_capture(["git", "status", "--short"])
    return {
        "commit": commit,
        "short_commit": commit[:7],
        "branch": run_capture(["git", "branch", "--show-current"]),
        "dirty": bool(status),
        "status": status,
        "remotes": run_capture(["git", "remote", "-v"]),
    }


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-")


def make_run_id(config: dict[str, Any], git: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = slug(config["experiment"]["name"])
    seed = config["experiment"]["seed"]
    return f"{stamp}_{name}_seed{seed}_{git['short_commit']}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, hash_limit: int = 32 * 1024 * 1024) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {"path": str(path.resolve()), "bytes": stat.st_size}
    if stat.st_size <= hash_limit:
        record["sha256"] = sha256(path)
    return record


def model_record(path: Path) -> dict[str, Any]:
    files = []
    for item in sorted(path.iterdir()):
        if item.is_file() and (item.suffix in {".json", ".model", ".safetensors"}):
            files.append(file_record(item))
    revisions = set()
    metadata_root = path / ".cache" / "huggingface" / "download"
    if metadata_root.is_dir():
        for metadata in metadata_root.rglob("*.metadata"):
            lines = metadata.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                revisions.add(lines[0])
    return {"path": str(path.resolve()), "hub_revisions": sorted(revisions), "files": files}


def hvalue(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def override(key: str, value: Any, add: bool = False) -> str:
    return f"{'+' if add else ''}{key}={hvalue(value)}"


def build_command(config: dict[str, Any], run_dir: Path, resume: bool) -> list[str]:
    data = config["data"]
    models = config["models"]
    distill = config["distillation"]
    optim = config["optimization"]
    rollout = config["rollout"]
    reward = config["reward_model"]
    trainer = config["trainer"]
    logging = config["logging"]

    max_model_len = max(
        data["max_prompt_length"] + data["max_response_length"],
        data["max_prompt_length"] + data["max_val_response_length"],
    )
    max_tokens_per_gpu = max(data["max_prompt_length"] + data["max_response_length"], 32768)
    global_batch = optim["mini_batch_size"] * rollout["sequence_parallel_size"]
    rollout_dir = str(run_dir / "rollouts") if logging["dump_rollouts"] else None

    values: list[tuple[str, Any, bool]] = [
        ("algorithm.adv_estimator", distill["advantage_estimator"], False),
        ("algorithm.grpo_outcome_weight", distill["grpo_outcome_weight"], False),
        ("data.shuffle", data["shuffle"], False),
        ("data.seed", config["experiment"]["seed"], False),
        ("data.train_files", data["train_files"], False),
        ("data.val_files", data["val_files"], False),
        ("data.train_batch_size", global_batch, False),
        ("data.max_prompt_length", data["max_prompt_length"], False),
        ("data.max_response_length", data["max_response_length"], False),
        ("data.filter_overlong_prompts", data["filter_overlong_prompts"], False),
        ("data.truncation", data["truncation"], False),
        ("data.return_raw_chat", True, False),
        ("actor_rollout_ref.model.path", models["actor_path"], False),
        ("actor_rollout_ref.model.use_remove_padding", True, False),
        ("actor_rollout_ref.model.enable_activation_offload", optim["activation_offload"], False),
        ("actor_rollout_ref.model.enable_gradient_checkpointing", optim["gradient_checkpointing"], False),
        ("actor_rollout_ref.actor.optim.lr", optim["learning_rate"], False),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", optim["mini_batch_size"], False),
        ("actor_rollout_ref.actor.use_dynamic_bsz", True, False),
        ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", optim["micro_batch_size_per_gpu"], False),
        ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", max_tokens_per_gpu, False),
        ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", rollout["sequence_parallel_size"], False),
        ("actor_rollout_ref.actor.use_kl_loss", distill["use_kl"], False),
        ("actor_rollout_ref.actor.loss_agg_mode", optim["loss_agg_mode"], False),
        ("actor_rollout_ref.actor.fsdp_config.param_offload", optim["param_offload"], False),
        ("actor_rollout_ref.actor.fsdp_config.optimizer_offload", optim["optimizer_offload"], False),
        ("actor_rollout_ref.actor.fsdp_config.forward_prefetch", True, False),
        ("actor_rollout_ref.actor.fsdp_config.model_dtype", models["dtype"], False),
        ("actor_rollout_ref.ref.fsdp_config.param_offload", True, False),
        ("actor_rollout_ref.ref.fsdp_config.model_dtype", models["dtype"], False),
        ("actor_rollout_ref.ref.log_prob_use_dynamic_bsz", True, False),
        ("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 1, False),
        ("actor_rollout_ref.rollout.name", rollout["backend"], False),
        ("actor_rollout_ref.rollout.temperature", rollout["temperature"], False),
        ("actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", True, False),
        ("actor_rollout_ref.rollout.log_prob_top_k", distill["log_prob_top_k"], True),
        ("actor_rollout_ref.rollout.top_k_strategy", distill["top_k_strategy"], True),
        ("actor_rollout_ref.rollout.reward_weight_mode", distill["reward_weight_mode"], True),
        ("actor_rollout_ref.rollout.teacher_temperature", distill["teacher_temperature"], True),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", rollout["tensor_model_parallel_size"], False),
        ("actor_rollout_ref.rollout.gpu_memory_utilization", rollout["gpu_memory_utilization"], False),
        ("actor_rollout_ref.rollout.max_model_len", max_model_len, False),
        ("actor_rollout_ref.rollout.max_num_batched_tokens", max_tokens_per_gpu, False),
        ("actor_rollout_ref.rollout.n", rollout["n"], False),
        ("actor_rollout_ref.rollout.val_kwargs.do_sample", True, False),
        ("actor_rollout_ref.rollout.val_kwargs.max_tokens", data["max_val_response_length"], True),
        ("actor_rollout_ref.rollout.val_kwargs.n", rollout["validation_n"], False),
        ("actor_rollout_ref.rollout.val_kwargs.temperature", rollout["validation_temperature"], False),
        ("actor_rollout_ref.rollout.val_kwargs.top_p", rollout["validation_top_p"], False),
        ("actor_rollout_ref.rollout.repetition_penalty", rollout["repetition_penalty"], False),
        ("actor_rollout_ref.rollout.calculate_log_probs", True, False),
        ("reward_model.enable", True, False),
        ("reward_model.reward_kwargs.enable_format_reward", distill["enable_format_reward"], True),
        ("reward_model.model.path", models["teacher_path"], False),
        ("reward_model.model.input_tokenizer", None, False),
        ("reward_model.model.use_remove_padding", True, False),
        ("reward_model.model.fsdp_config.param_offload", False, False),
        ("reward_model.model.dtype", models["dtype"], True),
        ("reward_model.micro_batch_size_per_gpu", reward["micro_batch_size_per_gpu"], False),
        ("custom_reward_function.path", str(REPO_ROOT / "verl/verl/utils/reward_score/ttrl_math/__init__.py"), False),
        ("custom_reward_function.name", "reward_func", False),
        ("trainer.val_before_train", trainer["val_before_train"], False),
        ("trainer.log_val_generations", trainer["log_val_generations"], False),
        ("trainer.logger", logging["backends"], False),
        ("trainer.project_name", config["experiment"]["project_name"], False),
        ("trainer.experiment_name", run_dir.name, False),
        ("trainer.validation_data_dir", str(run_dir / "validation"), False),
        ("trainer.rollout_data_dir", rollout_dir, False),
        ("trainer.n_gpus_per_node", trainer["n_gpus_per_node"], False),
        ("trainer.nnodes", trainer["nnodes"], False),
        ("trainer.save_freq", trainer["save_freq"], False),
        ("trainer.test_freq", trainer["test_freq"], False),
        ("trainer.total_epochs", trainer["total_epochs"], False),
        ("trainer.total_training_steps", trainer["total_training_steps"], False),
        ("trainer.default_local_dir", str(run_dir / "checkpoints"), False),
        ("trainer.max_actor_ckpt_to_keep", trainer["max_actor_ckpt_to_keep"], False),
        ("trainer.resume_mode", "auto" if resume else trainer["resume_mode"], False),
        ("trainer.is_plot", trainer["is_plot"], False),
        ("hydra.run.dir", str(run_dir / "hydra"), False),
        ("hydra.job.chdir", False, False),
    ]
    if distill["use_kl"]:
        values.extend(
            [
                ("actor_rollout_ref.actor.kl_loss_coef", distill["kl_loss_coef"], False),
                ("actor_rollout_ref.actor.kl_loss_type", distill["kl_loss_type"], False),
            ]
        )
    if optim["lr_scheduler"] == "cosine":
        values.extend(
            [
                ("actor_rollout_ref.actor.optim.warmup_style", "cosine", False),
                ("actor_rollout_ref.actor.optim.lr_warmup_steps_ratio", 0.03, False),
            ]
        )
    return [sys.executable, "-m", "verl.trainer.main_ppo"] + [override(*item) for item in values]


def managed_environment(config: dict[str, Any], run_dir: Path) -> dict[str, str]:
    runtime = config["runtime"]
    return {
        "CUDA_LAUNCH_BLOCKING": "1" if runtime["cuda_launch_blocking"] else "0",
        "HYDRA_FULL_ERROR": "1",
        "NCCL_DEBUG": str(runtime["nccl_debug"]),
        "NCCL_TIMEOUT": str(runtime["nccl_timeout"]),
        "OUTLINES_CACHE_DIR": str(run_dir / "cache" / "outlines"),
        "PYTHONUNBUFFERED": "1",
        "SWANLAB_LOG_DIR": str(run_dir / "swanlab"),
        "SWANLAB_MODE": str(config["logging"]["swanlab_mode"]),
        "TOKENIZERS_PARALLELISM": "true",
        "TORCH_DISTRIBUTED_DEBUG": str(runtime["torch_distributed_debug"]),
        "TORCH_NCCL_BLOCKING_WAIT": "1",
    }


def render_command_script(command: list[str], config: dict[str, Any], run_dir: Path) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO_ROOT))}"]
    for key, value in managed_environment(config, run_dir).items():
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.extend(["unset RAY_ADDRESS", f"exec {shlex.join(command)}"])
    return "\n".join(lines) + "\n"


def validate(config: dict[str, Any]) -> None:
    required_sections = [
        "experiment", "storage", "data", "models", "distillation", "optimization",
        "rollout", "reward_model", "trainer", "logging", "runtime",
    ]
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"missing config sections: {missing}")
    for name in [*config["data"]["train_files"], *config["data"]["val_files"]]:
        if not Path(name).is_file():
            raise FileNotFoundError(f"dataset not found: {name}")
    for key in ["actor_path", "teacher_path"]:
        path = Path(config["models"][key])
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"model not found or incomplete: {path}")
    if config["trainer"]["n_gpus_per_node"] < 1:
        raise ValueError("trainer.n_gpus_per_node must be positive")


def write_yaml(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(temporary, path)


def prepare_run(
    config: dict[str, Any], source_config: Path, run_dir: Path, git: dict[str, Any], command: list[str], resume: bool
) -> None:
    for name in ["checkpoints", "environment", "evaluation", "hydra", "logs", "rollouts", "swanlab", "validation"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.yaml", config)
    config_digest = hashlib.sha256((run_dir / "config.yaml").read_bytes()).hexdigest()
    datasets = [file_record(Path(path)) for path in [*config["data"]["train_files"], *config["data"]["val_files"]]]
    models = {name: model_record(Path(path)) for name, path in config["models"].items() if name.endswith("_path")}
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at_utc": utc_now(),
        "source_config": str(source_config.resolve()),
        "config_sha256": config_digest,
        "description": config["experiment"]["description"],
        "tags": config["experiment"]["tags"],
        "git": git,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "datasets": datasets,
        "models": models,
        "managed_environment": managed_environment(config, run_dir),
        "resume": resume,
    }
    write_yaml(run_dir / "manifest.yaml", manifest)
    (run_dir / "command.sh").write_text(render_command_script(command, config, run_dir), encoding="utf-8")
    (run_dir / "command.sh").chmod(0o755)
    (run_dir / "environment" / "launcher.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "environment" / "source-config.yaml").write_text(
        source_config.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "environment" / "pip-freeze.txt").write_text(run_capture([sys.executable, "-m", "pip", "freeze"]) + "\n")
    (run_dir / "environment" / "conda-list.txt").write_text(run_capture(["conda", "list"]) + "\n")
    (run_dir / "environment" / "hardware.txt").write_text(run_capture(["nvidia-smi"]) + "\n")
    (run_dir / "environment" / "git-diff.patch").write_text(run_capture(["git", "diff", "--binary"]) + "\n")


def update_status(run_dir: Path, status: str, **extra: Any) -> None:
    path = run_dir / "status.yaml"
    current = yaml.safe_load(path.read_text()) if path.is_file() else {}
    current.update({"run_id": run_dir.name, "status": status, "updated_at_utc": utc_now(), **extra})
    write_yaml(path, current)


def launch(command: list[str], run_dir: Path, config: dict[str, Any]) -> int:
    env = os.environ.copy()
    env.update(managed_environment(config, run_dir))
    env.pop("RAY_ADDRESS", None)
    (run_dir / "cache" / "outlines").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "train.log"
    update_status(run_dir, "running", started_at_utc=utc_now(), pid=os.getpid())
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        header = f"Run: {run_dir.name}\nCommand: {shlex.join(command)}\n"
        print(header, end="")
        log.write(header)
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def forward_signal(signum: int, _frame: Any) -> None:
            if process.poll() is None:
                process.send_signal(signum)

        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/experiments/opd_default.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true", help="validate and print without creating files")
    parser.add_argument("--prepare-only", action="store_true", help="materialize a run without launching training")
    parser.add_argument("--resume-run", help="existing run ID or absolute run directory")
    args = parser.parse_args()

    source_config = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = expand_env(load_config(source_config))
    for expression in args.set:
        set_dotted(config, expression)
    config = expand_env(config)
    validate(config)
    git = git_info()
    experiments_root = Path(config["storage"]["experiments_root"])
    resume = bool(args.resume_run)
    if resume:
        candidate = Path(args.resume_run)
        run_dir = candidate if candidate.is_absolute() else experiments_root / candidate
        if not (run_dir / "config.yaml").is_file():
            raise FileNotFoundError(f"managed run not found: {run_dir}")
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        for expression in args.set:
            set_dotted(config, expression)
    else:
        run_id = args.run_id or make_run_id(config, git)
        run_dir = experiments_root / run_id
        if run_dir.exists() and not args.dry_run:
            raise FileExistsError(f"run already exists: {run_dir}")

    command = build_command(config, run_dir, resume)
    print(f"RUN_ID={run_dir.name}")
    print(f"RUN_DIR={run_dir}")
    print(f"COMMAND={shlex.join(command)}")
    if args.dry_run:
        return 0

    if not resume:
        run_dir.mkdir(parents=True)
        prepare_run(config, source_config, run_dir, git, command, resume)
        update_status(run_dir, "prepared")
    else:
        resume_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        resume_command = run_dir / f"resume-command-{resume_stamp}.sh"
        resume_command.write_text(render_command_script(command, config, run_dir), encoding="utf-8")
        resume_command.chmod(0o755)
        update_status(run_dir, "prepared", resume_requested_at_utc=utc_now())
    if args.prepare_only:
        return 0

    started = time.monotonic()
    try:
        exit_code = launch(command, run_dir, config)
    except BaseException as exc:
        update_status(run_dir, "failed", error=repr(exc), duration_seconds=round(time.monotonic() - started, 3))
        raise
    status = "completed" if exit_code == 0 else "failed"
    update_status(
        run_dir,
        status,
        exit_code=exit_code,
        finished_at_utc=utc_now(),
        duration_seconds=round(time.monotonic() - started, 3),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
