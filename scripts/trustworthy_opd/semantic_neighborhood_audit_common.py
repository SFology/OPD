from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VALID_LABELS = {"comparable", "not_comparable", "uncertain"}
PRIMARY_GROUPS = ("teacher_correct_student_wrong", "both_wrong")
Q_ORDER = ("q20", "q40", "q60", "q80")


def semantic_pair_id(state_id: str, neighbor_point_id: str) -> str:
    payload = f"{state_id}|{neighbor_point_id}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def latest_annotations(path: Path) -> dict[str, dict[str, Any]]:
    from common import read_jsonl

    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("label") not in VALID_LABELS:
            continue
        latest[str(row["audit_id"])] = row
    return latest


def consensus_annotations(
    annotations: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    audit_ids = sorted({audit_id for rows in annotations.values() for audit_id in rows})
    result = []
    for audit_id in audit_ids:
        labels = [
            rows[audit_id]["label"] for rows in annotations.values() if audit_id in rows
        ]
        decided = [label for label in labels if label != "uncertain"]
        counts = Counter(decided)
        if not decided:
            consensus = "uncertain"
        elif (
            len(counts) > 1 and counts.most_common()[0][1] == counts.most_common()[1][1]
        ):
            consensus = "disputed"
        else:
            consensus = counts.most_common(1)[0][0]
        result.append(
            {
                "audit_id": audit_id,
                "consensus_label": consensus,
                "annotation_count": len(labels),
                "decided_annotation_count": len(decided),
            }
        )
    return pd.DataFrame(
        result,
        columns=[
            "audit_id",
            "consensus_label",
            "annotation_count",
            "decided_annotation_count",
        ],
    )


def cohens_kappa(first: list[str], second: list[str]) -> float:
    pairs = [
        (left, right)
        for left, right in zip(first, second)
        if left in VALID_LABELS and right in VALID_LABELS
    ]
    if not pairs:
        return float("nan")
    labels = sorted(VALID_LABELS)
    observed = sum(left == right for left, right in pairs) / len(pairs)
    first_counts = Counter(left for left, _ in pairs)
    second_counts = Counter(right for _, right in pairs)
    expected = (
        sum(first_counts[label] * second_counts[label] for label in labels)
        / len(pairs) ** 2
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else float("nan")
    return float((observed - expected) / (1.0 - expected))


def clustered_precision_interval(
    frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float, float]:
    decided = frame[frame.consensus_label.isin(["comparable", "not_comparable"])]
    if decided.empty:
        return float("nan"), float("nan"), float("nan")
    estimate = float((decided.consensus_label == "comparable").mean())
    prompts = np.asarray(sorted(decided.prompt_index.unique()), dtype=np.int64)
    prompt_counts = decided.groupby("prompt_index").consensus_label.agg(
        comparable=lambda values: int((values == "comparable").sum()),
        total="size",
    )
    comparable = np.asarray(
        [prompt_counts.loc[prompt, "comparable"] for prompt in prompts], dtype=float
    )
    total = np.asarray(
        [prompt_counts.loc[prompt, "total"] for prompt in prompts], dtype=float
    )
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(prompts), np.full(len(prompts), 1.0 / len(prompts)), size=samples
    )
    denominator = weights @ total
    values = np.divide(
        weights @ comparable,
        denominator,
        out=np.full(samples, np.nan),
        where=denominator > 0,
    )
    alpha = (1.0 - confidence) / 2.0
    finite = values[np.isfinite(values)]
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return estimate, float(low), float(high)


def balanced_sample(
    frame: pd.DataFrame,
    count: int,
    *,
    seed: int,
    distance_column: str = "joint_distance",
) -> pd.DataFrame:
    """Deterministically cover the distance range without leaking order to raters."""

    if count <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    if len(frame) <= count:
        return frame.copy()
    ranked = frame.sort_values(distance_column, kind="stable").copy()
    ranked["_distance_bin"] = np.minimum(
        np.floor(np.arange(len(ranked)) * 3 / len(ranked)).astype(int), 2
    )
    rng = np.random.default_rng(seed)
    pieces = []
    base, remainder = divmod(count, 3)
    for distance_bin in range(3):
        subset = ranked[ranked._distance_bin == distance_bin]
        take = min(len(subset), base + (distance_bin < remainder))
        if take:
            pieces.append(subset.iloc[rng.choice(len(subset), take, replace=False)])
    selected = pd.concat(pieces) if pieces else ranked.iloc[:0]
    if len(selected) < count:
        remaining = ranked[~ranked.index.isin(selected.index)]
        take = min(count - len(selected), len(remaining))
        selected = pd.concat(
            [selected, remaining.iloc[rng.choice(len(remaining), take, replace=False)]]
        )
    return selected.drop(columns="_distance_bin", errors="ignore")
