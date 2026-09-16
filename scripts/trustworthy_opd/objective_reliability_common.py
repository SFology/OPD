from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

Q_ORDER = ("q20", "q40", "q60", "q80")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def binomial_log_loss_sum(
    successes: np.ndarray, trials: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    return -(
        successes * np.log(probability) + (trials - successes) * np.log1p(-probability)
    )


def binomial_brier_sum(
    successes: np.ndarray, trials: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    probability = np.asarray(probability, dtype=float)
    return successes * (1 - probability) ** 2 + (trials - successes) * probability**2


def q_design(q_values: pd.Series | np.ndarray) -> np.ndarray:
    q_values = np.asarray(q_values)
    return np.column_stack([q_values == q for q in Q_ORDER]).astype(float)


def metric_design(
    q_values: pd.Series | np.ndarray,
    metric: np.ndarray,
    *,
    q_means: dict[str, float],
    q_scales: dict[str, float],
) -> np.ndarray:
    q_values = np.asarray(q_values)
    metric = np.asarray(metric, dtype=float)
    base = q_design(q_values)
    standardized = np.asarray(
        [
            (value - q_means[str(q)]) / q_scales[str(q)]
            for q, value in zip(q_values, metric)
        ],
        dtype=float,
    )
    return np.column_stack([base, base * standardized[:, None]])


def q_standardization(
    q_values: pd.Series | np.ndarray, metric: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    q_values = np.asarray(q_values)
    metric = np.asarray(metric, dtype=float)
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    global_mean = float(np.mean(metric))
    global_scale = max(float(np.std(metric)), 1e-6)
    for q in Q_ORDER:
        values = metric[q_values == q]
        means[q] = float(np.mean(values)) if len(values) else global_mean
        scales[q] = max(float(np.std(values)), 1e-6) if len(values) else global_scale
    return means, scales


def fit_binomial_logistic(
    design: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    *,
    l2: float,
    penalty_start: int,
) -> np.ndarray:
    design = np.asarray(design, dtype=float)
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ beta
        probability = expit(logits)
        value = float(np.sum(np.logaddexp(0.0, logits) * trials - successes * logits))
        gradient = design.T @ (trials * probability - successes)
        if l2:
            value += (
                0.5 * l2 * float(np.dot(beta[penalty_start:], beta[penalty_start:]))
            )
            gradient[penalty_start:] += l2 * beta[penalty_start:]
        return value, gradient

    initial = np.zeros(design.shape[1], dtype=float)
    result = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Binomial logistic fit failed: {result.message}")
    return result.x


def cross_validated_predictions(
    frame: pd.DataFrame,
    metric: str,
    *,
    folds: int,
    l2: float,
) -> pd.DataFrame:
    required = frame.dropna(subset=[metric]).copy()
    predictions = []
    for fold in range(folds):
        train = required[required.fold != fold]
        test = required[required.fold == fold]
        if train.empty or test.empty:
            continue
        base_train = q_design(train.fraction_name)
        base_beta = fit_binomial_logistic(
            base_train,
            train.successes.to_numpy(float),
            train.trials.to_numpy(float),
            l2=0.0,
            penalty_start=base_train.shape[1],
        )
        means, scales = q_standardization(
            train.fraction_name, train[metric].to_numpy(float)
        )
        metric_train = metric_design(
            train.fraction_name,
            train[metric].to_numpy(float),
            q_means=means,
            q_scales=scales,
        )
        metric_beta = fit_binomial_logistic(
            metric_train,
            train.successes.to_numpy(float),
            train.trials.to_numpy(float),
            l2=l2,
            penalty_start=len(Q_ORDER),
        )
        base_probability = expit(q_design(test.fraction_name) @ base_beta)
        metric_probability = expit(
            metric_design(
                test.fraction_name,
                test[metric].to_numpy(float),
                q_means=means,
                q_scales=scales,
            )
            @ metric_beta
        )
        output = test[
            [
                "state_id",
                "prompt_index",
                "fraction_name",
                "successes",
                "trials",
                "teacher_success_rate",
            ]
        ].copy()
        output["fold"] = fold
        output["metric_value"] = test[metric].to_numpy(float)
        output["baseline_probability"] = base_probability
        output["metric_probability"] = metric_probability
        output["baseline_log_loss_sum"] = binomial_log_loss_sum(
            output.successes, output.trials, base_probability
        )
        output["metric_log_loss_sum"] = binomial_log_loss_sum(
            output.successes, output.trials, metric_probability
        )
        output["baseline_brier_sum"] = binomial_brier_sum(
            output.successes, output.trials, base_probability
        )
        output["metric_brier_sum"] = binomial_brier_sum(
            output.successes, output.trials, metric_probability
        )
        predictions.append(output)
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def prompt_bootstrap_improvement(
    frame: pd.DataFrame,
    *,
    baseline_column: str,
    model_column: str,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float, float]:
    if frame.empty:
        return float("nan"), float("nan"), float("nan")
    grouped = frame.groupby("prompt_index", as_index=False).agg(
        baseline=(baseline_column, "sum"),
        model=(model_column, "sum"),
        trials=("trials", "sum"),
    )
    estimate = float(
        (grouped.baseline.sum() - grouped.model.sum()) / grouped.trials.sum()
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(samples, len(grouped)))
    baseline = grouped.baseline.to_numpy(float)[indices].sum(axis=1)
    model = grouped.model.to_numpy(float)[indices].sum(axis=1)
    trials = grouped.trials.to_numpy(float)[indices].sum(axis=1)
    values = (baseline - model) / trials
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return estimate, float(low), float(high)


def validate_config(config: dict[str, Any]) -> None:
    selection = config["selection"]
    if int(selection["target_valid_teacher_repeats"]) < 2:
        raise ValueError("target_valid_teacher_repeats must be at least two")
    if int(selection["maximum_new_attempts_per_state"]) < (
        int(selection["target_valid_teacher_repeats"]) - 1
    ):
        raise ValueError("maximum_new_attempts_per_state is too small")
    arms = list(config["support"]["arms"])
    if arms != ["selected", "random", "far"]:
        raise ValueError("support.arms must be [selected, random, far]")
    if tuple(config["analysis"]["q_points"]) != Q_ORDER:
        raise ValueError(f"analysis.q_points must be {list(Q_ORDER)}")
