from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "trustworthy_opd"
sys.path.insert(0, str(SCRIPT_DIR))

from objective_reliability_common import cross_validated_predictions
from run_objective_reliability_calibration import choose_control_indices


def test_matched_controls_are_disjoint_and_deterministic() -> None:
    available = [1, 3, 4, 6]
    distances = np.asarray([0.0, 0.2, 0.0, 0.8, 0.5, 0.0, 0.9])
    random_a, far_a = choose_control_indices(available, distances, 2, 123)
    random_b, far_b = choose_control_indices(available, distances, 2, 123)
    assert random_a == random_b
    assert far_a == far_b == [6, 3]
    assert len(random_a) == len(set(random_a)) == 2
    assert set(random_a) <= set(available)


def test_cross_validated_predictive_metric_improves_brier() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for prompt in range(80):
        q = ("q20", "q40", "q60", "q80")[prompt % 4]
        metric = rng.normal()
        probability = 1 / (1 + np.exp(-(1.8 * metric - 0.25 * (prompt % 4))))
        trials = 8
        rows.append(
            {
                "state_id": f"s{prompt}",
                "prompt_index": prompt,
                "fraction_name": q,
                "successes": rng.binomial(trials, probability),
                "trials": trials,
                "teacher_success_rate": probability,
                "metric": metric,
                "fold": prompt % 5,
            }
        )
    prediction = cross_validated_predictions(
        pd.DataFrame(rows), "metric", folds=5, l2=1e-4
    )
    assert prediction.metric_brier_sum.sum() < prediction.baseline_brier_sum.sum()
