from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, load_run, read_jsonl, update_status


FRACTIONS = ["q20", "q40", "q60", "q80"]
FRACTION_POSITION = {"q20": 20.0, "q40": 40.0, "q60": 60.0, "q80": 80.0}
GROUPS = ["teacher_correct_student_wrong", "both_wrong"]
GROUP_LABELS = {
    "teacher_correct_student_wrong": "Teacher corrected (T+/S−)",
    "both_wrong": "Teacher failed (T−/S−)",
    "both_correct": "Both correct (T+/S+)",
    "teacher_wrong_student_correct": "Teacher harmed (T−/S+)",
}
GROUP_COLORS = {
    "teacher_correct_student_wrong": "#2563EB",
    "both_wrong": "#F59E0B",
    "both_correct": "#059669",
    "teacher_wrong_student_correct": "#DC2626",
}
BRANCH_LABELS = {
    "teacher": "Teacher continuation",
    "student": "Original student continuation",
}
BRANCH_COLORS = {"teacher": "#7C3AED", "student": "#059669"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze original student prefixes and paired continuation PPL."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def load_score_rows(directory: Path) -> list[dict]:
    return [
        row
        for path in sorted(directory.glob("shard_*.jsonl"))
        for row in read_jsonl(path)
    ]


def load_frames(
    run_dir: Path, branch_dir: Path, roles: list[str], primary_window: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.DataFrame(read_jsonl(branch_dir / "manifest.jsonl"))
    metadata = manifest.set_index("state_id")
    expected = set(metadata.index)
    post_rows, prefix_rows = [], []
    local_key = f"local_log_ppl_w{primary_window}"

    for role in roles:
        teacher_items = load_score_rows(
            run_dir / "ppl_recovery_curve" / "scores" / role
        )
        student_items = load_score_rows(
            branch_dir / "scores" / "student_branch" / role
        )
        teacher_by_id = {row["state_id"]: row for row in teacher_items}
        student_by_id = {row["state_id"]: row for row in student_items}
        for branch, items in (("teacher", teacher_by_id), ("student", student_by_id)):
            missing = expected - set(items)
            if missing:
                raise RuntimeError(
                    f"Missing {len(missing)} {branch}-branch scores for role={role}"
                )
            for state_id in sorted(expected):
                item = items[state_id]
                meta = metadata.loc[state_id]
                for point in item["curve"]:
                    post_rows.append(
                        {
                            "state_id": state_id,
                            "role": role,
                            "branch": branch,
                            "prompt_index": int(meta.prompt_index),
                            "base_trajectory_id": meta.base_trajectory_id,
                            "fraction_name": meta.fraction_name,
                            "normalized_position": float(meta.normalized_position),
                            "group": meta.group,
                            **point,
                        }
                    )
        for state_id in sorted(expected):
            item = student_by_id[state_id]
            meta = metadata.loc[state_id]
            for point in item["prefix_curve"]:
                prefix_rows.append(
                    {
                        "state_id": state_id,
                        "role": role,
                        "prompt_index": int(meta.prompt_index),
                        "base_trajectory_id": meta.base_trajectory_id,
                        "fraction_name": meta.fraction_name,
                        "normalized_position": float(meta.normalized_position),
                        "group": meta.group,
                        **point,
                    }
                )

    post = pd.DataFrame(post_rows)
    prefix = pd.DataFrame(prefix_rows)
    if post.empty or prefix.empty:
        raise RuntimeError("Branch scores are empty")
    for frame in (post, prefix):
        frame["local_ppl"] = np.exp(frame[local_key].clip(upper=50))
    intervention_baseline = (
        prefix[prefix.progress == 1.0]
        .set_index(["state_id", "role"])[local_key]
        .rename("intervention_local_log_ppl")
    )
    prefix = prefix.join(intervention_baseline, on=["state_id", "role"])
    post = post.join(intervention_baseline, on=["state_id", "role"])
    prefix["local_log_ppl_delta_from_intervention"] = (
        prefix[local_key] - prefix.intervention_local_log_ppl
    )
    post["local_log_ppl_delta_from_intervention"] = (
        post[local_key] - post.intervention_local_log_ppl
    )
    post["continuation_cumulative_ppl"] = np.exp(
        post.continuation_cumulative_log_ppl.clip(upper=50)
    )
    post["trajectory_cumulative_ppl"] = np.exp(
        post.trajectory_cumulative_log_ppl.clip(upper=50)
    )
    prefix["prefix_cumulative_ppl"] = np.exp(
        prefix.prefix_cumulative_log_ppl.clip(upper=50)
    )
    cumulative_baseline = (
        prefix[prefix.progress == 1.0]
        .set_index(["state_id", "role"])["prefix_cumulative_log_ppl"]
        .rename("intervention_cumulative_log_ppl")
    )
    prefix = prefix.join(cumulative_baseline, on=["state_id", "role"])
    post = post.join(cumulative_baseline, on=["state_id", "role"])
    prefix["cumulative_log_ppl_delta_from_intervention"] = (
        prefix.prefix_cumulative_log_ppl
        - prefix.intervention_cumulative_log_ppl
    )
    post["trajectory_cumulative_log_ppl_delta_from_intervention"] = (
        post.trajectory_cumulative_log_ppl
        - post.intervention_cumulative_log_ppl
    )
    prefix["original_trajectory_position_percent"] = (
        100.0 * prefix.normalized_position * prefix.progress
    )
    return post, prefix


def prompt_cluster_curve(
    subset: pd.DataFrame,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for progress, point in subset.groupby("progress"):
        values = (
            point.groupby("prompt_index")[metric]
            .median()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(float)
        )
        if not len(values):
            continue
        boot = np.median(
            rng.choice(values, size=(samples, len(values)), replace=True), axis=1
        )
        rows.append(
            {
                "progress": float(progress),
                "estimate": float(np.median(values)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "prompt_clusters": len(values),
                "states": int(point.state_id.nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("progress")


def setup_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def finish_four_panel(fig, axes, title: str, subtitle: str, output: Path) -> None:
    from matplotlib import pyplot as plt

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.set_size_inches(15.4, 11.6, forward=True)
    wrapped_title = textwrap.fill(
        title, width=82, break_long_words=False, break_on_hyphens=False
    )
    wrapped_subtitle = textwrap.fill(
        subtitle, width=118, break_long_words=False, break_on_hyphens=False
    )
    fig.suptitle(
        wrapped_title,
        fontsize=17,
        fontweight="bold",
        y=0.99,
        va="top",
        linespacing=1.22,
    )
    fig.text(
        0.5,
        0.905,
        wrapped_subtitle,
        ha="center",
        va="top",
        fontsize=10,
        color="#4B5563",
        linespacing=1.25,
    )
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.855),
            ncol=min(3, len(labels)),
            frameon=False,
            fontsize=10.2,
            columnspacing=2.2,
            handlelength=2.8,
        )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.79,
        bottom=0.105,
        hspace=0.52,
        wspace=0.27,
    )
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(axis) -> None:
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelbottom=True, pad=6)
    axis.tick_params(axis="y", pad=4)
    axis.title.set_y(1.025)
    axis.xaxis.labelpad = 10
    axis.yaxis.labelpad = 8


def make_intervention_figure(
    prefix: pd.DataFrame,
    post: pd.DataFrame,
    role: str,
    metric: str,
    ylabel: str,
    output: Path,
    samples: int,
    seed: int,
) -> list[dict]:
    """Join student-prefix and teacher-continuation curves at intervention x=0."""
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), sharex=True)
    rng = np.random.default_rng(seed)
    aggregate = []
    for axis, fraction in zip(axes.flat, FRACTIONS):
        prefix_panel = prefix[
            (prefix.role == role) & (prefix.fraction_name == fraction)
        ]
        post_panel = post[
            (post.role == role)
            & (post.fraction_name == fraction)
            & (post.branch == "teacher")
        ]
        axis.axvspan(-100, 0, color="#EFF6FF", alpha=0.8, zorder=0)
        axis.axvspan(0, 100, color="#F5F3FF", alpha=0.8, zorder=0)
        for group in GROUPS:
            before = prompt_cluster_curve(
                prefix_panel[prefix_panel.group == group], metric, samples, rng
            )
            after = prompt_cluster_curve(
                post_panel[post_panel.group == group], metric, samples, rng
            )
            if not before.empty:
                x_before = -100 * (1 - before.progress.to_numpy())
                axis.plot(
                    x_before,
                    before.estimate,
                    marker="o",
                    linewidth=2.2,
                    markersize=4.2,
                    linestyle="--",
                    color=GROUP_COLORS[group],
                    label=GROUP_LABELS[group],
                )
                axis.fill_between(
                    x_before,
                    before.ci_low,
                    before.ci_high,
                    color=GROUP_COLORS[group],
                    alpha=0.14,
                )
                for row in before.to_dict(orient="records"):
                    aggregate.append(
                        {
                            "figure": "intervention",
                            "phase": "student_prefix",
                            "role": role,
                            "metric": metric,
                            "fraction_name": fraction,
                            "series": group,
                            **row,
                        }
                    )
            if not after.empty:
                x_after = 100 * after.progress.to_numpy()
                axis.plot(
                    x_after,
                    after.estimate,
                    marker="o",
                    linewidth=2.4,
                    markersize=4.2,
                    color=GROUP_COLORS[group],
                )
                axis.fill_between(
                    x_after,
                    after.ci_low,
                    after.ci_high,
                    color=GROUP_COLORS[group],
                    alpha=0.14,
                )
                for row in after.to_dict(orient="records"):
                    aggregate.append(
                        {
                            "figure": "intervention",
                            "phase": "teacher_continuation",
                            "role": role,
                            "metric": metric,
                            "fraction_name": fraction,
                            "series": group,
                            **row,
                        }
                    )
        axis.axvline(0, color="#DC2626", linewidth=2.6, linestyle="--", zorder=6)
        if metric == "local_log_ppl_delta_from_intervention":
            axis.axhline(0, color="#6B7280", linewidth=1.1, linestyle=":")
        axis.text(
            0.23,
            0.96,
            "Student prefix",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="#1D4ED8",
            fontweight="bold",
        )
        axis.text(
            0.77,
            0.96,
            "Teacher continuation",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="#6D28D9",
            fontweight="bold",
        )
        counts = prefix_panel.groupby("group").state_id.nunique().to_dict()
        axis.set_title(
            f"{fraction} intervention | corrected n={counts.get(GROUPS[0], 0)}, "
            f"failed n={counts.get(GROUPS[1], 0)}",
            fontsize=11.1,
            fontweight="bold",
        )
        axis.set_xlim(-102, 102)
        axis.set_xticks(
            [-100, -50, 0, 50, 100],
            ["prefix\nstart", "−50", "intervention\n0", "+50", "continuation\nend"],
        )
        axis.set_xlabel("Normalized progress relative to teacher intervention (%)")
        axis.set_ylabel(ylabel)
        style_axis(axis)
    finish_four_panel(
        fig,
        axes,
        f"PPL before and after teacher intervention | {role} scorer",
        "Dashed: original student prefix; solid: teacher continuation; red line: intervention; bands: prompt-cluster bootstrap 95% CIs",
        output,
    )
    return aggregate


def make_prefix_figure(
    prefix: pd.DataFrame,
    role: str,
    metric: str,
    ylabel: str,
    output: Path,
    samples: int,
    seed: int,
) -> list[dict]:
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), sharey=False)
    rng = np.random.default_rng(seed)
    aggregate = []
    for axis, fraction in zip(axes.flat, FRACTIONS):
        panel = prefix[(prefix.role == role) & (prefix.fraction_name == fraction)]
        intervention = FRACTION_POSITION[fraction]
        for group in GROUPS:
            curve = prompt_cluster_curve(
                panel[panel.group == group], metric, samples, rng
            )
            if curve.empty:
                continue
            x = intervention * curve.progress.to_numpy()
            axis.plot(
                x,
                curve.estimate,
                marker="o",
                linewidth=2.2,
                markersize=4.2,
                color=GROUP_COLORS[group],
                label=GROUP_LABELS[group],
            )
            axis.fill_between(
                x, curve.ci_low, curve.ci_high, color=GROUP_COLORS[group], alpha=0.16
            )
            for row in curve.to_dict(orient="records"):
                aggregate.append(
                    {"figure": "prefix", "role": role, "metric": metric,
                     "fraction_name": fraction, "series": group, **row}
                )
        axis.axvline(intervention, color="#DC2626", linewidth=2.3, linestyle="--")
        axis.text(
            intervention,
            0.97,
            f" intervention {fraction}",
            transform=axis.get_xaxis_transform(),
            va="top",
            ha="left",
            color="#B91C1C",
            fontsize=9.5,
            fontweight="bold",
        )
        counts = panel.groupby("group").state_id.nunique().to_dict()
        axis.set_title(
            f"{fraction} | corrected n={counts.get(GROUPS[0], 0)}, "
            f"failed n={counts.get(GROUPS[1], 0)}",
            fontsize=11.2,
            fontweight="bold",
        )
        axis.set_xlim(0, 100)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xlabel("Position in original student response (%)")
        axis.set_ylabel(ylabel)
        style_axis(axis)
    finish_four_panel(
        fig,
        axes,
        f"Student-prefix {ylabel} | scored by {role} model",
        "Only tokens generated before teacher intervention are included; bands are prompt-cluster bootstrap 95% CIs",
        output,
    )
    return aggregate


def make_standard_trajectory_figure(
    prefix: pd.DataFrame,
    post: pd.DataFrame,
    role: str,
    group: str,
    prefix_metric: str,
    post_metric: str,
    ylabel: str,
    output: Path,
    samples: int,
    seed: int,
) -> list[dict]:
    """Plot standard all-response-token PPL continuously across intervention."""
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), sharex=True)
    rng = np.random.default_rng(seed)
    aggregate = []
    for axis, fraction in zip(axes.flat, FRACTIONS):
        before_points = prefix[
            (prefix.role == role)
            & (prefix.group == group)
            & (prefix.fraction_name == fraction)
        ]
        after_points = post[
            (post.role == role)
            & (post.group == group)
            & (post.fraction_name == fraction)
        ]
        axis.axvspan(-100, 0, color="#EFF6FF", alpha=0.8, zorder=0)
        axis.axvspan(0, 100, color="#F5F3FF", alpha=0.8, zorder=0)

        before = prompt_cluster_curve(
            before_points, prefix_metric, samples, rng
        )
        if not before.empty:
            x_before = -100 * (1 - before.progress.to_numpy())
            axis.plot(
                x_before,
                before.estimate,
                marker="o",
                linewidth=2.3,
                markersize=4.2,
                linestyle="--",
                color="#2563EB",
                label="Original student prefix",
            )
            axis.fill_between(
                x_before,
                before.ci_low,
                before.ci_high,
                color="#2563EB",
                alpha=0.14,
            )
            for row in before.to_dict(orient="records"):
                aggregate.append(
                    {
                        "figure": "standard_trajectory",
                        "phase": "student_prefix",
                        "role": role,
                        "metric": prefix_metric,
                        "fraction_name": fraction,
                        "group": group,
                        "series": "shared_prefix",
                        **row,
                    }
                )

        for branch in ("teacher", "student"):
            after = prompt_cluster_curve(
                after_points[after_points.branch == branch],
                post_metric,
                samples,
                rng,
            )
            if after.empty:
                continue
            x_after = 100 * after.progress.to_numpy()
            axis.plot(
                x_after,
                after.estimate,
                marker="o",
                linewidth=2.4,
                markersize=4.2,
                color=BRANCH_COLORS[branch],
                label=BRANCH_LABELS[branch],
            )
            axis.fill_between(
                x_after,
                after.ci_low,
                after.ci_high,
                color=BRANCH_COLORS[branch],
                alpha=0.14,
            )
            for row in after.to_dict(orient="records"):
                aggregate.append(
                    {
                        "figure": "standard_trajectory",
                        "phase": f"{branch}_continuation",
                        "role": role,
                        "metric": post_metric,
                        "fraction_name": fraction,
                        "group": group,
                        "series": branch,
                        **row,
                    }
                )

        axis.axvline(0, color="#DC2626", linewidth=2.6, linestyle="--", zorder=6)
        if "delta_from_intervention" in post_metric:
            axis.axhline(0, color="#6B7280", linewidth=1.1, linestyle=":")
        axis.text(
            0.23,
            0.96,
            "Student prefix",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="#1D4ED8",
            fontweight="bold",
        )
        axis.text(
            0.77,
            0.96,
            "Paired continuations",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="#6D28D9",
            fontweight="bold",
        )
        axis.set_title(
            f"{fraction} intervention | n={before_points.state_id.nunique()}",
            fontsize=11.2,
            fontweight="bold",
        )
        axis.set_xlim(-102, 102)
        axis.set_xticks(
            [-100, -50, 0, 50, 100],
            ["prefix\nstart", "−50", "intervention\n0", "+50", "continuation\nend"],
        )
        axis.set_xlabel("Normalized progress relative to intervention (%)")
        axis.set_ylabel(ylabel)
        style_axis(axis)
    finish_four_panel(
        fig,
        axes,
        f"Standard cumulative trajectory PPL | {GROUP_LABELS[group]} | {role} scorer",
        "Every point averages all response tokens so far; prompt tokens are conditioning context and are not included in the loss",
        output,
    )
    return aggregate


def make_branch_figure(
    post: pd.DataFrame,
    role: str,
    group: str,
    metric: str,
    ylabel: str,
    output: Path,
    samples: int,
    seed: int,
) -> list[dict]:
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), sharex=True)
    rng = np.random.default_rng(seed)
    aggregate = []
    for axis, fraction in zip(axes.flat, FRACTIONS):
        panel = post[
            (post.role == role)
            & (post.group == group)
            & (post.fraction_name == fraction)
        ]
        for branch in ("teacher", "student"):
            curve = prompt_cluster_curve(
                panel[panel.branch == branch], metric, samples, rng
            )
            if curve.empty:
                continue
            x = 100 * curve.progress.to_numpy()
            axis.plot(
                x,
                curve.estimate,
                marker="o",
                linewidth=2.2,
                markersize=4.2,
                color=BRANCH_COLORS[branch],
                label=BRANCH_LABELS[branch],
            )
            axis.fill_between(
                x, curve.ci_low, curve.ci_high, color=BRANCH_COLORS[branch], alpha=0.16
            )
            for row in curve.to_dict(orient="records"):
                aggregate.append(
                    {"figure": "branches", "role": role, "metric": metric,
                     "fraction_name": fraction, "group": group,
                     "series": branch, **row}
                )
        axis.axvline(0, color="#DC2626", linewidth=2.3, linestyle="--")
        axis.text(
            0.025,
            0.97,
            f"Divergence / intervention: {fraction}",
            transform=axis.transAxes,
            va="top",
            color="#B91C1C",
            fontsize=9.5,
            fontweight="bold",
        )
        axis.set_title(f"{fraction} | n={panel.state_id.nunique()}", fontweight="bold")
        axis.set_xlim(-4, 102)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xlabel("Continuation progress after the shared prefix (%)")
        axis.set_ylabel(ylabel)
        style_axis(axis)
    finish_four_panel(
        fig,
        axes,
        f"Paired continuation {ylabel} | {GROUP_LABELS[group]} | {role} scorer",
        "Both branches start from the exact same state; their x-axes are normalized separately by branch length",
        output,
    )
    return aggregate


def build_gap_frame(post: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = [
        "state_id", "role", "prompt_index", "base_trajectory_id",
        "fraction_name", "group", "progress",
    ]
    wide = post.pivot_table(index=keys, columns="branch", values=metric).reset_index()
    wide = wide.dropna(subset=["teacher", "student"])
    wide["teacher_minus_student"] = wide.teacher - wide.student
    return wide


def make_gap_figure(
    gap: pd.DataFrame,
    role: str,
    ylabel: str,
    output: Path,
    samples: int,
    seed: int,
) -> list[dict]:
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), sharex=True)
    rng = np.random.default_rng(seed)
    aggregate = []
    for axis, fraction in zip(axes.flat, FRACTIONS):
        panel = gap[(gap.role == role) & (gap.fraction_name == fraction)]
        for group in GROUPS:
            curve = prompt_cluster_curve(
                panel[panel.group == group], "teacher_minus_student", samples, rng
            )
            if curve.empty:
                continue
            x = 100 * curve.progress.to_numpy()
            axis.plot(
                x,
                curve.estimate,
                marker="o",
                linewidth=2.2,
                markersize=4.2,
                color=GROUP_COLORS[group],
                label=GROUP_LABELS[group],
            )
            axis.fill_between(
                x, curve.ci_low, curve.ci_high, color=GROUP_COLORS[group], alpha=0.16
            )
            for row in curve.to_dict(orient="records"):
                aggregate.append(
                    {"figure": "gap", "role": role, "metric": ylabel,
                     "fraction_name": fraction, "series": group, **row}
                )
        axis.axhline(0, color="#6B7280", linewidth=1.2, linestyle=":")
        axis.axvline(0, color="#DC2626", linewidth=2.3, linestyle="--")
        axis.text(
            0.025, 0.97, f"Divergence: {fraction}", transform=axis.transAxes,
            va="top", color="#B91C1C", fontsize=9.5, fontweight="bold"
        )
        axis.set_title(f"{fraction} | n={panel.state_id.nunique()}", fontweight="bold")
        axis.set_xlim(-4, 102)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xlabel("Continuation progress after the shared prefix (%)")
        axis.set_ylabel(ylabel)
        style_axis(axis)
    finish_four_panel(
        fig,
        axes,
        f"Teacher-branch minus student-branch gap | {role} scorer",
        "Negative values mean the teacher continuation has lower PPL than the original student continuation",
        output,
    )
    return aggregate


def bootstrap_summary(
    post: pd.DataFrame,
    prefix: pd.DataFrame,
    primary_window: int,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 70000)
    local = f"local_log_ppl_w{primary_window}"
    gap = build_gap_frame(post, local)
    rows = []

    def add_summary(kind: str, values: pd.DataFrame, value_key: str) -> None:
        for (role, fraction, group), subset in values.groupby(
            ["role", "fraction_name", "group"]
        ):
            clustered = subset.groupby("prompt_index")[value_key].median().dropna()
            array = clustered.to_numpy(float)
            if not len(array):
                continue
            boot = np.median(
                rng.choice(array, size=(samples, len(array)), replace=True), axis=1
            )
            rows.append(
                {
                    "summary": kind,
                    "role": role,
                    "fraction_name": fraction,
                    "group": group,
                    "states": int(subset.state_id.nunique()),
                    "prompt_clusters": len(array),
                    "median": float(np.median(array)),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                }
            )

    endpoint_gap = gap[gap.progress == 1.0]
    add_summary(
        "endpoint_teacher_minus_student_local_log_ppl",
        endpoint_gap,
        "teacher_minus_student",
    )
    auc = (
        gap[gap.progress > 0]
        .groupby(
            ["state_id", "role", "prompt_index", "fraction_name", "group"],
            as_index=False,
        )
        .teacher_minus_student.mean()
    )
    add_summary(
        "mean_curve_teacher_minus_student_local_log_ppl",
        auc,
        "teacher_minus_student",
    )
    prefix_finite = prefix.dropna(subset=[local]).sort_values("progress")
    prefix_change = (
        prefix_finite.groupby(
            ["state_id", "role", "prompt_index", "fraction_name", "group"]
        )[local]
        .agg(lambda values: values.iloc[-1] - values.iloc[0])
        .rename("prefix_endpoint_minus_first_log_ppl")
        .reset_index()
    )
    add_summary(
        "student_prefix_endpoint_minus_first_local_log_ppl",
        prefix_change,
        "prefix_endpoint_minus_first_log_ppl",
    )
    return pd.DataFrame(rows)


def standard_trajectory_summary(
    post: pd.DataFrame, samples: int, seed: int
) -> pd.DataFrame:
    metric = "trajectory_cumulative_log_ppl_delta_from_intervention"
    endpoint = post[post.progress == 1.0].copy()
    rng = np.random.default_rng(seed + 91000)
    rows = []

    def summarize(
        subset: pd.DataFrame,
        values: pd.Series,
        *,
        contrast: str,
        role: str,
        fraction: str,
        group: str,
        branch: str,
    ) -> None:
        array = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if not len(array):
            return
        boot = np.median(
            rng.choice(array, size=(samples, len(array)), replace=True), axis=1
        )
        rows.append(
            {
                "contrast": contrast,
                "role": role,
                "fraction_name": fraction,
                "group": group,
                "branch": branch,
                "states": int(subset.state_id.nunique()),
                "prompt_clusters": len(array),
                "median_log_ppl_change": float(np.median(array)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
            }
        )

    for (role, fraction, group, branch), subset in endpoint.groupby(
        ["role", "fraction_name", "group", "branch"]
    ):
        clustered = subset.groupby("prompt_index")[metric].median()
        summarize(
            subset,
            clustered,
            contrast="endpoint_minus_intervention",
            role=role,
            fraction=fraction,
            group=group,
            branch=branch,
        )

    paired_branch = endpoint.pivot_table(
        index=[
            "state_id",
            "role",
            "prompt_index",
            "fraction_name",
            "group",
        ],
        columns="branch",
        values=metric,
    ).dropna(subset=["teacher", "student"])
    paired_branch["teacher_minus_student"] = (
        paired_branch.teacher - paired_branch.student
    )
    paired_branch = paired_branch.reset_index()
    for (role, fraction, group), subset in paired_branch.groupby(
        ["role", "fraction_name", "group"]
    ):
        clustered = subset.groupby("prompt_index").teacher_minus_student.median()
        summarize(
            subset,
            clustered,
            contrast="teacher_minus_student_endpoint_change",
            role=role,
            fraction=fraction,
            group=group,
            branch="teacher_minus_student",
        )

    teacher_endpoint = endpoint[endpoint.branch == "teacher"]
    for (role, fraction), subset in teacher_endpoint.groupby(
        ["role", "fraction_name"]
    ):
        by_prompt = subset.pivot_table(
            index="prompt_index", columns="group", values=metric, aggfunc="median"
        )
        if not set(GROUPS).issubset(by_prompt.columns):
            continue
        paired = by_prompt.dropna(subset=GROUPS)
        difference = paired[GROUPS[0]] - paired[GROUPS[1]]
        summarize(
            subset[subset.prompt_index.isin(paired.index)],
            difference,
            contrast="corrected_minus_failed_teacher_endpoint_change",
            role=role,
            fraction=fraction,
            group="corrected_minus_failed",
            branch="teacher",
        )
    return pd.DataFrame(rows)


def explanation(kind: str, metric: str | None = None) -> str:
    if kind == "standard_trajectory":
        if metric == "prefix_cumulative_ppl":
            meaning = "标准条件序列 PPL：每个点累计回答开始以来的全部 token；prompt 仅作为条件，不计入 loss。介入后紫线加入教师 token，绿线加入原学生后缀 token。"
            reading = "三条曲线在红线处对应完全相同的学生状态。介入后紫线低于绿线，表示在该评分模型下，教师分支的整条累计轨迹更可预测。"
        else:
            meaning = "仍使用全部回答 token 的累计 log-PPL，但逐状态减去介入点数值，从而将共同起点校准为 0。"
            reading = "负值表示完整轨迹累计 PPL 相对介入点下降。紫线减绿线反映教师分支相对原学生分支的改变，且不会发生滑动窗口 token 替换。"
    elif kind == "intervention":
        if metric == "local_ppl":
            meaning = "把同一状态的介入前学生前缀（虚线）和介入后教师续写（实线）连接起来；每点是最近 128 个 token 的局部 PPL。"
            reading = "红线处是真实教师介入点。介入后实线若明显低于红线前的水平，说明 PPL 在教师接手后下降；还需比较蓝线成功组与橙线失败组的下降幅度。"
        else:
            meaning = "对每个状态都将介入点的 local log-PPL 减为 0，再连接介入前后的相对变化。"
            reading = "介入后负值表示 PPL 低于介入时，正值表示升高。若成功组比失败组更负且置信区间支持该差异，才说明下降与成功纠偏相关。"
    elif kind == "prefix":
        if metric == "local_ppl":
            meaning = "沿原始学生回答，从回答起点到教师介入点，统计最近 128 个 token 的局部 PPL。"
            reading = "下降表示学生前缀在该评分模型下局部更可预测；蓝橙分离若早于介入出现，说明后续能否纠偏可能由前缀状态预先决定。"
        else:
            meaning = "沿原始学生回答，累计统计从回答起点到当前位置的 PPL。"
            reading = "它展示总体前缀质量，但对早期 token 有较强记忆，局部变化会被平滑。红线是该面板的教师介入分位点。"
    elif kind == "branches":
        if metric == "local_ppl":
            meaning = "从完全相同的学生状态出发，紫线为教师续写、绿线为原始学生续写；每点统计最近 128 token 的 PPL。"
            reading = "同一评分模型下紫线低于绿线，表示教师分支在该阶段更符合评分模型分布；成功组与失败组应分别解读。"
        else:
            meaning = "从共同前缀之后开始，分别累计教师分支与原始学生分支的 PPL。"
            reading = "该图比较整段续写的总体可预测性。两个分支长度不同，因此相同百分比仅表示各自完成比例，不是相同 token 时刻。"
    else:
        meaning = "逐状态配对计算“教师续写 log-PPL − 学生续写 log-PPL”，再按 prompt 聚类汇总。"
        reading = "0 表示两分支相同；负值表示教师续写 PPL 更低。只有该差距在成功组比失败组更负，才支持它与纠偏可靠性有关，而不只是自生成偏好。"
    return (
        f'<div class="analysis"><h3>图表解释</h3><dl>'
        f'<dt>指标含义</dt><dd>{html.escape(meaning)}</dd>'
        f'<dt>如何读图</dt><dd>{html.escape(reading)}</dd>'
        f'<dt>统计单位</dt><dd>先在 prompt 内取中位数，再对 prompt 做 cluster bootstrap；阴影为 95% CI。</dd>'
        f'</dl></div>'
    )


def standard_card_finding(card: dict, summary: pd.DataFrame) -> str:
    role = card["role"]
    group = card["group"]
    endpoint = summary[
        (summary.contrast == "endpoint_minus_intervention")
        & (summary.role == role)
        & (summary.group == group)
    ]
    gap = summary[
        (summary.contrast == "teacher_minus_student_endpoint_change")
        & (summary.role == role)
        & (summary.group == group)
    ]

    def values(frame: pd.DataFrame, branch: str | None = None) -> str:
        if branch is not None:
            frame = frame[frame.branch == branch]
        by_q = frame.set_index("fraction_name").reindex(FRACTIONS)
        return "、".join(
            f"{fraction}={row.median_log_ppl_change:+.3f}"
            for fraction, row in by_q.iterrows()
            if pd.notna(row.median_log_ppl_change)
        )

    teacher_values = values(endpoint, "teacher")
    student_values = values(endpoint, "student")
    gap_values = values(gap)
    significant = int(((gap.ci_low > 0) | (gap.ci_high < 0)).sum())
    return (
        '<div class="finding"><h3>当前数据</h3>'
        f'<p>终点相对介入点的累计 log-PPL：教师分支 {html.escape(teacher_values)}；'
        f'原学生分支 {html.escape(student_values)}。</p>'
        f'<p>教师减学生的逐状态配对差：{html.escape(gap_values)}。'
        f'其中 {significant}/4 个分位点的 prompt-cluster bootstrap 95% CI 不含 0。'
        '负值表示教师分支累计 PPL 更低。</p></div>'
    )


def make_dashboard(
    branch_dir: Path, cards: list[dict], trajectory_summary: pd.DataFrame
) -> Path:
    sections = []
    labels = {
        "standard_trajectory": "一、核心视图：标准全前缀 PPL 的介入前后曲线",
        "prefix": "二、教师介入前：原始学生前缀",
        "branches": "三、分歧后：教师续写与原始学生续写",
        "gap": "四、逐状态配对：两条续写曲线的差距",
    }
    for kind in ("standard_trajectory", "prefix", "branches", "gap"):
        body = []
        for card in [item for item in cards if item["kind"] == kind]:
            finding = (
                standard_card_finding(card, trajectory_summary)
                if kind == "standard_trajectory"
                else ""
            )
            body.append(
                f'<article><h2>{html.escape(card["title"])}</h2>'
                f'<a href="{html.escape(card["file"])}" target="_blank">'
                f'<img src="{html.escape(card["file"])}" alt="{html.escape(card["title"])}"></a>'
                f'{explanation(kind, card.get("metric"))}{finding}</article>'
            )
        sections.append(f'<section><h1>{labels[kind]}</h1>{"".join(body)}</section>')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>学生前缀与双续写 PPL 对照</title>
<style>body{{margin:0;background:#f3f4f6;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#111827}}main{{max-width:1580px;margin:auto;padding:26px}}header{{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:18px 22px;line-height:1.65}}section>h1{{margin:34px 0 14px}}article{{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:12px;margin:0 0 22px;box-shadow:0 2px 8px #0000000d}}article h2{{font-size:19px;margin:8px 12px}}img{{display:block;width:100%;height:auto}}.analysis,.finding{{margin:8px 16px 16px;padding:15px 19px;border-radius:10px;line-height:1.65}}.analysis{{background:#f8fafc;border:1px solid #dbeafe}}.finding{{background:#f0fdf4;border:1px solid #bbf7d0}}.analysis h3,.finding h3{{margin:0 0 8px}}.finding p{{margin:6px 0}}dl{{display:grid;grid-template-columns:90px 1fr;gap:7px 14px;margin:0}}dt{{font-weight:700;color:#1e3a8a}}dd{{margin:0;color:#374151}}code{{background:#eef2ff;padding:2px 5px;border-radius:4px}}a{{color:#2563eb}}@media(max-width:800px){{main{{padding:12px}}dl{{grid-template-columns:1fr}}}}</style></head>
<body><main><header><h1>学生前缀与教师/学生双续写 PPL 对照</h1>
<p>本看板同时报告教师模型 PPL 与学生模型 PPL。所有续写分支共享同一个学生状态：教师分支复用既有 teacher continuation，学生分支复用冻结的原始 student rollout 后缀，没有重新采样。</p>
<p><strong>核心判据：</strong>第一部分使用标准条件序列 PPL：始终累计回答开始以来的全部 token，红线为教师介入点，之后分别延伸为教师续写和原学生续写。横轴两侧按各自长度归一化，因此左右相同百分比不代表相同 token 数，但纵轴统计量在介入前后完全一致。</p></header>
{''.join(sections)}
<p>原始数据：<a href="../prefix_curve_points.csv">prefix_curve_points.csv</a> · <a href="../branch_curve_points.csv">branch_curve_points.csv</a> · <a href="../paired_gap_points.csv">paired_gap_points.csv</a> · <a href="../trajectory_cumulative_summary.csv">trajectory_cumulative_summary.csv</a> · <a href="../statistical_summary.csv">statistical_summary.csv</a></p>
</main></body></html>'''
    output = branch_dir / "results" / "figures" / "ppl_branch_dashboard.html"
    output.write_text(page, encoding="utf-8")
    return output


def main() -> int:
    args = parse_args()
    run_dir, _ = load_run(args.run_dir)
    branch_dir = run_dir / "ppl_branch_comparison"
    config = load_config(branch_dir / "config.yaml")
    roles = list(config["scoring"]["roles"])
    primary_window = int(config["scoring"]["primary_local_window"])
    samples = int(config["analysis"]["bootstrap_samples"])
    seed = int(config["experiment"]["seed"])
    post, prefix = load_frames(run_dir, branch_dir, roles, primary_window)
    results = branch_dir / "results"
    figures = results / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    post.to_csv(results / "branch_curve_points.csv", index=False)
    post.to_parquet(results / "branch_curve_points.parquet", index=False)
    prefix.to_csv(results / "prefix_curve_points.csv", index=False)
    prefix.to_parquet(results / "prefix_curve_points.parquet", index=False)

    local_key = f"local_log_ppl_w{primary_window}"
    gap_local = build_gap_frame(post, local_key)
    gap_cumulative = build_gap_frame(post, "continuation_cumulative_log_ppl")
    gap_local["metric"] = "local_log_ppl"
    gap_cumulative["metric"] = "continuation_cumulative_log_ppl"
    paired_gap = pd.concat([gap_local, gap_cumulative], ignore_index=True)
    paired_gap.to_csv(results / "paired_gap_points.csv", index=False)
    paired_gap.to_parquet(results / "paired_gap_points.parquet", index=False)

    cards, aggregate = [], []
    prefix_metrics = {
        "local_ppl": "Local PPL (recent 128 tokens)",
        "prefix_cumulative_ppl": "Cumulative student-prefix PPL",
    }
    branch_metrics = {
        "local_ppl": "Local PPL (recent 128 tokens)",
        "continuation_cumulative_ppl": "Continuation cumulative PPL",
    }
    for role_index, role in enumerate(roles):
        standard_metrics = (
            (
                "prefix_cumulative_ppl",
                "trajectory_cumulative_ppl",
                "Standard cumulative trajectory PPL",
            ),
            (
                "cumulative_log_ppl_delta_from_intervention",
                "trajectory_cumulative_log_ppl_delta_from_intervention",
                "Cumulative log-PPL change from intervention",
            ),
        )
        for group_index, group in enumerate(GROUPS):
            for metric_index, (prefix_metric, post_metric, ylabel) in enumerate(
                standard_metrics
            ):
                filename = (
                    f"standard_trajectory__{role}__{group}__{metric_index}.svg"
                )
                aggregate.extend(
                    make_standard_trajectory_figure(
                        prefix,
                        post,
                        role,
                        group,
                        prefix_metric,
                        post_metric,
                        ylabel,
                        figures / filename,
                        samples,
                        seed
                        + 30000
                        + role_index * 1000
                        + group_index * 200
                        + metric_index * 100,
                    )
                )
                cards.append(
                    {
                        "kind": "standard_trajectory",
                        "metric": prefix_metric,
                        "role": role,
                        "group": group,
                        "file": filename,
                        "title": (
                            f"{GROUP_LABELS[group]}：{ylabel}"
                            f"（{role} 模型打分）"
                        ),
                    }
                )
        for metric_index, (metric, ylabel) in enumerate(prefix_metrics.items()):
            filename = f"prefix__{role}__{metric}.svg"
            aggregate.extend(
                make_prefix_figure(
                    prefix, role, metric, ylabel, figures / filename, samples,
                    seed + role_index * 1000 + metric_index * 100,
                )
            )
            cards.append(
                {"kind": "prefix", "metric": metric, "file": filename,
                 "title": f"学生前缀 {ylabel}（{role} 模型打分）"}
            )
        for group_index, group in enumerate(GROUPS):
            for metric_index, (metric, ylabel) in enumerate(branch_metrics.items()):
                filename = f"branches__{role}__{group}__{metric}.svg"
                aggregate.extend(
                    make_branch_figure(
                        post, role, group, metric, ylabel, figures / filename,
                        samples, seed + 10000 + role_index * 1000 + group_index * 200 + metric_index * 100,
                    )
                )
                cards.append(
                    {"kind": "branches", "metric": metric, "file": filename,
                     "title": f"{GROUP_LABELS[group]}：教师/学生续写 {ylabel}（{role} 模型打分）"}
                )
        for metric_index, (gap, ylabel) in enumerate(
            (
                (gap_local, "Teacher − student local log-PPL"),
                (gap_cumulative, "Teacher − student cumulative log-PPL"),
            )
        ):
            filename = f"gap__{role}__{metric_index}.svg"
            aggregate.extend(
                make_gap_figure(
                    gap, role, ylabel, figures / filename, samples,
                    seed + 20000 + role_index * 1000 + metric_index * 100,
                )
            )
            cards.append(
                {"kind": "gap", "file": filename,
                 "title": f"教师续写 − 学生续写 PPL 差距（{role} 模型打分；{ylabel}）"}
            )
    pd.DataFrame(aggregate).to_csv(results / "curve_aggregate.csv", index=False)
    summary_frame = bootstrap_summary(
        post, prefix, primary_window, samples, seed
    )
    summary_frame.to_csv(results / "statistical_summary.csv", index=False)
    trajectory_summary = standard_trajectory_summary(post, samples, seed)
    trajectory_summary.to_csv(
        results / "trajectory_cumulative_summary.csv", index=False
    )
    dashboard = make_dashboard(branch_dir, cards, trajectory_summary)
    summary = {
        "states": int(post.state_id.nunique()),
        "roles": roles,
        "branches": ["teacher", "student"],
        "primary_local_window": primary_window,
        "dashboard": str(dashboard),
        "statistical_summary": summary_frame.to_dict(orient="records"),
        "trajectory_cumulative_summary": trajectory_summary.to_dict(
            orient="records"
        ),
    }
    (results / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    update_status(
        branch_dir,
        "completed",
        states=summary["states"],
        roles=roles,
        dashboard=str(dashboard),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
