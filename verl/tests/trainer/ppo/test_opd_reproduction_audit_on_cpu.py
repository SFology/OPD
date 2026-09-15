from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_script(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_script("run_opd_experiment_for_test", "scripts/run_opd_experiment.py")
audit = load_script("audit_opd_reproduction_for_test", "scripts/audit_opd_reproduction.py")
lambda_analysis = load_script(
    "analyze_lcb_lambda_diagnostic_for_test",
    "scripts/analyze_lcb_lambda_diagnostic.py",
)


def test_low_precision_actor_requires_explicit_non_scientific_opt_in(capsys):
    with pytest.raises(ValueError, match="updates can quantize to zero"):
        launcher.validate_actor_training_precision({"models": {"dtype": "bfloat16"}})

    launcher.validate_actor_training_precision(
        {"models": {"dtype": "bfloat16", "allow_low_precision_actor_training": True}}
    )
    assert "do not use this run as an OPD effectiveness result" in capsys.readouterr().err


@pytest.mark.parametrize("dtype", ["fp32", "float32", "torch.float32"])
def test_fp32_actor_precision_is_accepted(dtype):
    launcher.validate_actor_training_precision({"models": {"dtype": dtype}})


def test_paper_exact_and_completion_configs_resolve_to_fp32():
    paper = launcher.load_config(REPO_ROOT / "configs/experiments/opd_paper_exact_8gpu.yaml")
    completion = launcher.load_config(REPO_ROOT / "configs/experiments/opd_fp32_offload.yaml")

    assert paper["models"]["dtype"] == "fp32"
    assert paper["trainer"]["n_gpus_per_node"] == 8
    assert paper["optimization"]["param_offload"] is False
    assert completion["models"]["dtype"] == "fp32"
    assert completion["optimization"]["param_offload"] is True


def test_metric_parser_keeps_latest_record_per_step(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "\x1b[36m(worker)\x1b[0m step:1 - val-topk/overlap_ratio:0.72 - actor/grad_norm:2.0\n"
        "step:2 - val-topk/overlap_ratio:0.73 - actor/grad_norm:1.8\n"
        "step:2 - val-topk/overlap_ratio:0.74 - actor/grad_norm:1.7\n",
        encoding="utf-8",
    )

    rows = audit.parse_training_metrics(log)
    summary = audit.summarize_metrics(rows, window=1)

    assert [int(row["step"]) for row in rows] == [1, 2]
    assert rows[-1]["val-topk/overlap_ratio"] == pytest.approx(0.74)
    assert summary["metrics"]["val-topk/overlap_ratio"]["late_minus_early"] == pytest.approx(0.02)


def test_contract_flags_low_precision_and_robust_instrumentation():
    config = launcher.load_config(REPO_ROOT / "configs/experiments/opd_default.yaml")
    config["models"]["dtype"] = "bfloat16"
    config["distillation"]["robust_opd"] = {"enabled": True, "apply_to_training": False}

    deviations = audit.paper_contract(config)
    paths = {item["path"] for item in deviations}

    assert "models.dtype" in paths
    assert "distillation.robust_opd.enabled" in paths


def test_topk_opd_reward_sign_reduces_reverse_kl():
    """The upstream top-k reward/loss sign must move student toward teacher."""

    student_logits = torch.nn.Parameter(torch.tensor([[[1.4, 0.0]]], dtype=torch.float64))
    teacher_log_probs = torch.log_softmax(torch.tensor([[[-1.0, 1.0]]], dtype=torch.float64), dim=-1)
    optimizer = torch.optim.SGD([student_logits], lr=0.2)
    config = OmegaConf.create(
        {"clip_ratio": 0.2, "clip_ratio_low": None, "clip_ratio_high": None, "clip_ratio_c": 3.0}
    )

    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    student_weights = student_log_probs.detach().exp()
    advantages = student_weights * (teacher_log_probs - student_log_probs.detach())
    mask = torch.ones((1, 1), dtype=torch.float64)
    before = torch.sum(student_weights * (student_log_probs.detach() - teacher_log_probs)).item()

    loss, _ = compute_policy_loss_vanilla(
        old_log_prob=student_log_probs.detach(),
        log_prob=student_log_probs,
        advantages=advantages,
        response_mask=mask,
        config=config,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    updated_log_probs = torch.log_softmax(student_logits.detach(), dim=-1)
    after = torch.sum(updated_log_probs.exp() * (updated_log_probs - teacher_log_probs)).item()
    assert after < before


def test_representative_parameter_change_summary_is_weighted_by_numel():
    changes = {
        "a": {
            "numel": 3,
            "changed_elements": 3,
            "absolute_delta_mean": 2.0,
            "delta_l2": 3.0,
            "source_l2": 4.0,
        },
        "b": {
            "numel": 1,
            "changed_elements": 0,
            "absolute_delta_mean": 4.0,
            "delta_l2": 4.0,
            "source_l2": 3.0,
        },
        "bad": {"error": "shape mismatch"},
    }

    result = audit.summarize_parameter_changes(changes)

    assert result["parameter_count"] == 2
    assert result["numel"] == 4
    assert result["changed_fraction"] == pytest.approx(0.75)
    assert result["absolute_delta_mean"] == pytest.approx(2.5)
    assert result["delta_l2"] == pytest.approx(5.0)
    assert result["source_l2"] == pytest.approx(5.0)
    assert result["relative_delta_l2"] == pytest.approx(1.0)
    assert result["delta_rms"] == pytest.approx(2.5)


def test_lambda_diagnostic_summary_aggregates_steps_and_normalizes_rms():
    fields = {
        "trust_mean": [0.5, 0.7],
        "zero_trust_fraction": [0.4, 0.2],
        "supported_trust_mean": [0.4, 0.6],
        "supported_zero_trust_fraction": [0.5, 0.3],
        "selected_token_rms": [1.0, 1.4],
        "effective_abs_reward_mass_fraction": [0.6, 0.8],
        "unsupported_selected_abs_mass_fraction": [0.2, 0.4],
    }
    rows = []
    for index in range(2):
        row = {"ropd/training_opd_token_rms": 2.0}
        row.update({f"ropd/cf_lambda_0p1/{key}": values[index] for key, values in fields.items()})
        rows.append(row)

    opd_rms, summaries = lambda_analysis.summarize_lambda_grid(rows, [0.1])

    assert opd_rms == pytest.approx(2.0)
    assert summaries[0]["trust_mean"] == pytest.approx(0.6)
    assert summaries[0]["selected_token_rms"] == pytest.approx(1.2)
    assert summaries[0]["selected_token_rms_relative_to_opd"] == pytest.approx(0.6)
