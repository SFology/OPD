# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_module("formal_eval_common", REPO_ROOT / "scripts" / "val" / "formal_eval_common.py")
analysis = load_module(
    "analyze_formal_evaluation", REPO_ROOT / "scripts" / "val" / "analyze_formal_evaluation.py"
)
audit = load_module(
    "audit_eval_contamination", REPO_ROOT / "scripts" / "val" / "audit_eval_contamination.py"
)


def test_pass_at_k_uses_unbiased_estimator():
    assert analysis.pass_at_k(correct=0, total=16, k=4) == 0.0
    assert analysis.pass_at_k(correct=16, total=16, k=4) == 1.0
    assert analysis.pass_at_k(correct=1, total=16, k=1) == pytest.approx(1 / 16)
    assert analysis.pass_at_k(correct=1, total=16, k=16) == 1.0


def test_request_seed_is_model_independent_and_prompt_specific():
    config = {"generation": {"seed_base": 420000}}
    first = common.expected_seed(config, dataset_index=1, example_id=7, rollout=3)
    assert first == common.expected_seed(config, dataset_index=1, example_id=7, rollout=3)
    assert first != common.expected_seed(config, dataset_index=1, example_id=8, rollout=3)
    assert first != common.expected_seed(config, dataset_index=1, example_id=7, rollout=4)


def test_contamination_canonicalization_ignores_known_suffix_and_whitespace():
    plain = "Solve   x + 1 = 2"
    templated = [
        {
            "role": "user",
            "content": "solve x + 1 = 2 Please reason step by step, and put your final answer within \\boxed{}.",
        }
    ]
    assert audit.canonical_prompt(plain) == audit.canonical_prompt(templated)


def test_generation_shard_validation_rejects_partial_output(tmp_path):
    path = tmp_path / "rollout_000.jsonl"
    row = {
        "model": "opd",
        "dataset": "AIME24",
        "prompt_id": "AIME24:0",
        "rollout": 0,
    }
    common.write_jsonl(path, [row])
    assert common.validate_generation_shard(
        path,
        model="opd",
        dataset="AIME24",
        rollout=0,
        expected_prompts=1,
        expected_prompt_ids={"AIME24:0"},
    )
    assert not common.validate_generation_shard(
        path,
        model="opd",
        dataset="AIME24",
        rollout=0,
        expected_prompts=2,
        expected_prompt_ids={"AIME24:0", "AIME24:1"},
    )


def test_update_status_can_clear_transient_progress_fields(tmp_path):
    common.write_yaml(
        tmp_path / "status.yaml",
        {
            "status": "running",
            "stage": "generation:ropd",
            "model": "ropd",
            "completed_shards": 47,
            "total_shards": 48,
        },
    )
    common.update_status(
        tmp_path,
        "completed",
        stage="complete",
        clear_fields=("model", "completed_shards", "total_shards"),
    )
    status = common.load_yaml(tmp_path / "status.yaml")
    assert status["status"] == "completed"
    assert status["stage"] == "complete"
    assert "model" not in status
    assert "completed_shards" not in status
    assert "total_shards" not in status


def test_analysis_builds_complete_dashboard(tmp_path, monkeypatch):
    model_names = ["initial_student", "opd_step279", "ropd_step279"]
    dataset_names = ["AIME24", "AIME25", "AMC23"]
    config = {
        "models": [{"name": name} for name in model_names],
        "datasets": [{"name": name} for name in dataset_names],
        "generation": {"rollouts_per_prompt": 4},
        "analysis": {
            "pass_k": [1, 4],
            "comparisons": [
                ["opd_step279", "initial_student"],
                ["ropd_step279", "initial_student"],
                ["ropd_step279", "opd_step279"],
            ],
            "bootstrap_samples": 50,
            "confidence_level": 0.95,
            "bootstrap_seed": 7,
        },
    }
    common.write_yaml(tmp_path / "config.yaml", config)
    for model_index, model in enumerate(model_names):
        for dataset in dataset_names:
            for rollout in range(4):
                rows = []
                for example_id in range(2):
                    correct = rollout < 2 + model_index
                    answer = "25" if correct else "24"
                    rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "prompt_id": f"{dataset}:{example_id}",
                            "example_id": example_id,
                            "rollout": rollout,
                            "seed": rollout,
                            "ground_truth": "025",
                            "response": f"Answer: {answer}",
                            "prompt_tokens": 10,
                            "response_tokens": 3,
                            "finish_reason": "stop",
                            "stop_reason": None,
                            "at_token_limit": False,
                        }
                    )
                common.write_jsonl(
                    common.generation_path(tmp_path, model, dataset, rollout), rows
                )
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_formal_evaluation.py", "--run-dir", str(tmp_path)],
    )
    analysis.main()
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert summary["unique_prompts"] == 6
    assert summary["generations"] == 72
    paired_rows = list(
        csv.DictReader((tmp_path / "results" / "paired_comparisons.csv").open())
    )
    overall_ropd_opd = next(
        row
        for row in paired_rows
        if row["dataset"] == "ALL"
        and row["treatment"] == "ropd_step279"
        and row["control"] == "opd_step279"
    )
    assert float(overall_ropd_opd["pass_at_4_delta"]) == pytest.approx(0.0)
    assert int(overall_ropd_opd["pass_at_1_treatment_better_prompts"]) == 6
    assert (tmp_path / "results" / "figures" / "dashboard.html").exists()
    assert (tmp_path / "results" / "figures" / "accuracy_comparison.svg").exists()
    assert (tmp_path / "results" / "figures" / "paired_pass_at_k_deltas.svg").exists()
    dashboard = (tmp_path / "results" / "figures" / "dashboard.html").read_text()
    assert "每题 4 次采样" in dashboard
    assert "pass@8" not in dashboard
