#!/usr/bin/env python3
"""Create uniform mean-plus-SEM figures from matrix sweep manifests."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


FIGURES = (
    "fig_hidden",
    "fig_phase_observability",
    "fig_q_reset",
    "fig_response_reward",
)
EXPERIMENT_FOR_FIGURE = {
    "fig_hidden": "hidden_queries",
    "fig_phase_observability": "phase_observability",
    "fig_q_reset": "q_reset",
    "fig_response_reward": "response_reward",
}
PAPER_BASENAME = {
    "fig_hidden": "fig_hidden",
    "fig_phase_observability": "fig_memory_pg",
    "fig_q_reset": "fig_reset",
    "fig_response_reward": "fig_bots_leaderreward",
}
COLORS = {
    "observed": "#4C72B0",
    "hidden": "#DD8452",
    "visible": "#4C72B0",
    "reset": "#4C72B0",
    "ongoing": "#DD8452",
    "excluded": "#4C72B0",
    "included": "#DD8452",
}
LINESTYLES = {
    "observed": "-",
    "hidden": "--",
    "visible": "-",
    "phase_hidden": "--",
    "reset": "-",
    "ongoing": "--",
    "excluded": "-",
    "included": "--",
}
DEEP_PALETTE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
)
LINE_WIDTH = 1.8
STANDARD_ERROR_ALPHA = 0.18
MAX_X_TICKS = 5
LABELS = {
    "observed": "Observed queries",
    "hidden": "Hidden queries",
    "visible": "Phase indicator available",
    "phase_hidden": "Phase indicator unavailable",
    "reset": "Q-table reset",
    "ongoing": "Q-table carried",
    "excluded": "Response reward excluded",
    "included": "Response reward included",
}
NATIVE_RLLIB_ES_IMPLEMENTATION = "ray_rllib_es_2_0_1"
ES_IMPLEMENTATION_NOT_APPLICABLE = "not_applicable"
PLOT_UNIT_ENVIRONMENT_TRANSITIONS = "environment_transitions"
PLOT_UNIT_ES_ITERATIONS = "es_iterations"
ALGORITHM_TITLES = {
    "PG": "Policy gradient",
    "A2C": "A2C",
    "PPO": "PPO",
    "ES": "Evolution strategies",
}
PAPER_REQUIRED_CELLS = {
    "fig_hidden": {
        ("hidden_queries", "modified_pd", "PG", "observed"),
        ("hidden_queries", "modified_pd", "PG", "hidden"),
        ("hidden_queries", "modified_pd", "ES", "observed"),
        ("hidden_queries", "modified_pd", "ES", "hidden"),
    },
}
PAPER_REQUIRED_PROFILES = {
    "fig_hidden": "paper_joint_v1",
}
PAPER_CELL_COLUMNS = ("experiment", "matrix", "algorithm", "condition")


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config):
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_step_sizes(config, config_path):
    """Return the algorithm-specific step sizes recorded for plotting.

    Corrected ES runs intentionally leave ``learning_rate`` null and expose
    their effective Adam update size as ``es_stepsize``.  Other leader
    algorithms use ``learning_rate``; the always-present ES CLI default is
    irrelevant for those runs and must not define their cohort.
    """
    algorithm = config.get("algorithm")
    if algorithm == "ES":
        if config.get("learning_rate") is not None:
            raise ValueError(
                "corrected ES config must record learning_rate as null: {}".format(
                    config_path
                )
            )
        name = "es_protocol.stepsize"
        value = (config.get("es_protocol") or {}).get("stepsize")
        learning_rate = None
    else:
        name = "learning_rate"
        value = config.get(name)
        learning_rate = value
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "{} config has no numeric effective {}: {}".format(
                algorithm, name, config_path
            )
        )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "{} config has invalid effective {}: {}".format(
                algorithm, name, config_path
            )
        )
    if algorithm == "ES":
        return None, value
    return float(learning_rate), None


def collect_runs(root):
    runs = []
    history = []
    configs = []
    runs_root = root / "runs" if (root / "runs").is_dir() else root
    for manifest_path in runs_root.glob("**/run_manifest.json"):
        manifest = read_json(manifest_path)
        config_path = manifest_path.with_name("config.json")
        progress_path = manifest_path.with_name("progress.jsonl")
        if not config_path.exists():
            raise ValueError("manifest has no config: {}".format(manifest_path))
        config = read_json(config_path)
        if config.get("stage") != "leader":
            continue
        if not progress_path.exists():
            raise ValueError("leader manifest has no progress: {}".format(
                manifest_path
            ))
        if manifest.get("config_sha256") != config_sha256(config):
            raise ValueError("config hash mismatch in {}".format(config_path))
        if manifest.get("status") == "completed":
            progress_artifact = (manifest.get("artifacts") or {}).get("progress")
            if not progress_artifact or not progress_artifact.get("sha256"):
                raise ValueError(
                    "completed manifest has no progress checksum: {}".format(
                        manifest_path
                    )
                )
            if progress_artifact["sha256"] != sha256(progress_path):
                raise ValueError("progress hash mismatch in {}".format(progress_path))
        algorithm = config["algorithm"]
        es_implementation = ES_IMPLEMENTATION_NOT_APPLICABLE
        if algorithm == "ES":
            es_implementation = (config.get("es_protocol") or {}).get(
                "implementation"
            )
            # Historical custom ES artifacts remain immutable provenance, but
            # they are not an active paper treatment.  Non-ES runs in the same
            # result root remain eligible for aggregation.
            if es_implementation != NATIVE_RLLIB_ES_IMPLEMENTATION:
                continue
        learning_rate, es_stepsize = effective_step_sizes(config, config_path)
        run_id = manifest.get("config_sha256", manifest_path.parent.name)
        base = {
            "run_id": run_id,
            "run_dir": str(manifest_path.parent),
            "status": manifest.get("status"),
            "profile_id": config.get("profile_id"),
            "experiment": config["experiment"],
            "condition": config["condition"],
            "matrix": config["matrix"],
            "algorithm": algorithm,
            "es_implementation": es_implementation,
            "seed": int(config["seed"]),
            "learning_rate": learning_rate,
            "es_stepsize": es_stepsize,
        }
        runs.append(base)
        configs.append({
            **base,
            "config_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        })
        for row_number, row in enumerate(read_jsonl(progress_path), start=1):
            if algorithm == "ES":
                evaluation_keys = {
                    "iteration", "measured_timesteps_total",
                    "native_episode_reward_mean",
                    "native_leader_reward_per_stage",
                }
                present = evaluation_keys.intersection(row)
                if not present:
                    continue
                if present != evaluation_keys:
                    raise ValueError(
                        "partial native RLlib ES row {} in {}".format(
                            row_number, progress_path
                        )
                    )
                native_return = float(row["native_episode_reward_mean"])
                reward = float(row["native_leader_reward_per_stage"])
                expected_reward = native_return / float(config["episode_length"])
                if not np.isclose(reward, expected_reward, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        "native RLlib return/per-stage mismatch at row {} in {}"
                        .format(row_number, progress_path)
                    )
                plot_step = int(row["iteration"])
                plot_step_unit = PLOT_UNIT_ES_ITERATIONS
                evaluation_metric = "native_episode_reward_mean_per_stage"
                evaluation_target_step = plot_step
                measured_timesteps = int(row["measured_timesteps_total"])
                if measured_timesteps < 1:
                    raise ValueError(
                        "native RLlib ES row has invalid measured timesteps at "
                        "row {} in {}".format(
                            row_number, progress_path
                        )
                    )
                evaluation_window_size = 1
                evaluation_window_capacity = 1
            else:
                evaluation_keys = {
                    "evaluation_target_step", "evaluation_mean"
                }
                present = evaluation_keys.intersection(row)
                if not present:
                    continue
                if present != evaluation_keys:
                    raise ValueError(
                        "partial evaluation row {} in {}".format(
                            row_number, progress_path
                        )
                    )
                reward = float(row["evaluation_mean"])
                evaluation_target_step = int(row["evaluation_target_step"])
                plot_step = evaluation_target_step
                plot_step_unit = PLOT_UNIT_ENVIRONMENT_TRANSITIONS
                evaluation_metric = "evaluation_mean"
                evaluation_window_size = 1
                evaluation_window_capacity = 1
            within_run_sem = float(row.get("evaluation_sem", 0.0))
            if not np.isfinite(reward) or not np.isfinite(within_run_sem):
                raise ValueError(
                    "non-finite evaluation row {} in {}".format(
                        row_number, progress_path
                    )
                )
            if within_run_sem < 0.0:
                raise ValueError(
                    "negative evaluation SEM row {} in {}".format(
                        row_number, progress_path
                    )
                )
            history.append({
                **base,
                "evaluation_target_step": evaluation_target_step,
                "global_step": int(row["global_step"]),
                "measured_timesteps_total": int(row.get(
                    "measured_timesteps_total", row["global_step"]
                )),
                "plot_step": plot_step,
                "plot_step_unit": plot_step_unit,
                "evaluation_metric": evaluation_metric,
                "evaluation_window_size": evaluation_window_size,
                "evaluation_window_capacity": evaluation_window_capacity,
                "leader_reward": reward,
                "within_run_eval_sem": within_run_sem,
                "evaluation_episodes": int(row.get("evaluation_episodes", 0)),
            })
    if not runs:
        raise ValueError("no completed matrix leader runs found under {}".format(root))
    return pd.DataFrame(runs), pd.DataFrame(history), pd.DataFrame(configs)


def collect_input_roots(roots):
    """Collect compatible immutable run roots without copying artifacts."""
    collected = [collect_runs(root) for root in roots]
    merged = []
    for index in range(3):
        frames = [items[index] for items in collected]
        columns = list(dict.fromkeys(
            column for frame in frames for column in frame.columns
        ))
        # Omitting per-root all-NA columns before concatenation follows the
        # pandas forward-compatible dtype rule; reindexing restores the full
        # auditable schema afterward.
        nonempty_columns = [
            frame.dropna(axis=1, how="all") for frame in frames
        ]
        merged.append(
            pd.concat(nonempty_columns, ignore_index=True).reindex(columns=columns)
        )
    return tuple(merged)


def validate_runs(runs, expected_seeds, allow_incomplete):
    completed = runs[runs["status"] == "completed"]
    if completed["profile_id"].isna().any():
        raise ValueError("completed run is missing an immutable profile_id")
    profile_groups = completed.groupby(
        ["experiment", "matrix", "algorithm"]
    )["profile_id"].nunique()
    mixed = profile_groups[profile_groups > 1]
    if len(mixed):
        raise ValueError(
            "mixed experiment profiles are forbidden: {}".format(
                mixed.to_dict()
            )
        )
    for group_key, group in completed.groupby(
        ["experiment", "matrix", "algorithm"], dropna=False
    ):
        step_sizes = group[["learning_rate", "es_stepsize"]].drop_duplicates()
        if len(step_sizes) > 1:
            raise ValueError(
                "mixed effective step sizes are forbidden for {}: {}".format(
                    group_key, step_sizes.to_dict("records")
                )
            )
    if len(completed) != len(runs) and not allow_incomplete:
        raise ValueError("incomplete or failed runs are present")
    if "es_implementation" in completed:
        es_completed = completed[completed["algorithm"] == "ES"].copy()
        if not es_completed.empty:
            es_completed["_implementation"] = es_completed[
                "es_implementation"
            ].fillna(
                "<missing>"
            )
            implementation_groups = es_completed.groupby(
                ["experiment", "matrix"], dropna=False
            )["_implementation"].nunique()
            mixed = implementation_groups[implementation_groups > 1]
            if len(mixed):
                raise ValueError(
                    "mixed ES implementations are forbidden: {}".format(
                        mixed.to_dict()
                    )
                )
    key = ["experiment", "matrix", "algorithm", "condition", "seed"]
    duplicates = completed.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(
            "duplicate run cells: {}".format(
                completed.loc[duplicates, key].to_dict("records")
            )
        )
    for group_key, group in completed.groupby(key[:-1]):
        found = set(int(value) for value in group["seed"])
        if found != expected_seeds and not allow_incomplete:
            raise ValueError(
                "seed mismatch for {}: expected {}, found {}".format(
                    group_key, sorted(expected_seeds), sorted(found)
                )
            )


def validate_required_seed_count(expected_seeds, required_seed_count,
                                 allow_incomplete):
    """Make a paper-grade seed-count requirement explicit and non-bypassable."""
    if required_seed_count is None:
        return
    if required_seed_count < 2:
        raise ValueError("required seed count must be at least two")
    if allow_incomplete:
        raise ValueError(
            "--require-seeds-per-cell cannot be combined with --allow-incomplete"
        )
    if len(expected_seeds) != required_seed_count:
        raise ValueError(
            "expected seed set has {} distinct seeds, but {} are required per "
            "cell".format(len(expected_seeds), required_seed_count)
        )


def validate_required_figure_cells(frame, figure, required_seed_count,
                                   artifact_name):
    """Require the complete paper panel/condition schema when requested."""
    if required_seed_count is None:
        return
    names = FIGURES if figure == "all" else (figure,)
    for name in names:
        expected = PAPER_REQUIRED_CELLS.get(name)
        if expected is None:
            continue
        experiments = {cell[0] for cell in expected}
        scoped = frame[frame["experiment"].isin(experiments)]
        found = {
            tuple(row[column] for column in PAPER_CELL_COLUMNS)
            for _, row in scoped.iterrows()
        }
        if found != expected:
            raise ValueError(
                "{} cells for {} differ from the required paper schema; "
                "missing={}, extra={}".format(
                    artifact_name, name, sorted(expected - found),
                    sorted(found - expected),
                )
            )
        required_profile = PAPER_REQUIRED_PROFILES.get(name)
        if required_profile is not None:
            profiles = set(scoped["profile_id"].dropna().unique())
            if profiles != {required_profile}:
                raise ValueError(
                    "{} profiles for {} must be exactly {}; found {}".format(
                        artifact_name, name, required_profile,
                        sorted(profiles),
                    )
                )


def validate_plan_cells(runs, plan, allow_incomplete):
    if allow_incomplete:
        return
    keys = ("experiment", "matrix", "algorithm", "condition", "seed")
    expected = {
        tuple(record["match"][key] for key in keys)
        for record in plan["records"] if record["stage"] == "leader"
    }
    completed = runs[runs["status"] == "completed"]
    found = {
        tuple(row[key] for key in keys)
        for _, row in completed.iterrows()
    }
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(
            "completed cells differ from plan; missing={}, extra={}".format(
                missing, extra
            )
        )


def validate_history(history, expected_seeds, allow_incomplete):
    keys = [
        "experiment", "matrix", "algorithm", "condition",
        "learning_rate", "es_stepsize", "evaluation_target_step",
    ]
    for column in (
            "es_implementation", "plot_step_unit", "plot_step",
            "evaluation_metric", "evaluation_window_capacity",
            "evaluation_window_size",
    ):
        if column in history:
            keys.append(column)
    if "plot_step" in history:
        canonical_keys = [
            column for column in keys
            if column != "evaluation_target_step"
        ]
        canonical_duplicates = history.duplicated(
            [*canonical_keys, "seed"], keep=False
        )
        if canonical_duplicates.any():
            raise ValueError(
                "duplicate seed rows on a canonical plot coordinate"
            )
    duplicates = history.duplicated([*keys, "seed"], keep=False)
    if duplicates.any():
        raise ValueError("duplicate seed rows on an evaluation grid point")
    if allow_incomplete:
        return
    for group_key, group in history.groupby(keys, dropna=False):
        found = set(int(value) for value in group["seed"])
        if found != expected_seeds:
            raise ValueError(
                "history seed mismatch for {}: expected {}, found {}".format(
                    group_key, sorted(expected_seeds), sorted(found)
                )
            )


def sample_sem(values):
    """Return sample SEM across independent run-level values.

    A sample standard error is not defined for one independent run.  Returning
    NaN in that case keeps diagnostic, incomplete figures from implying zero
    uncertainty.  Final paper figures require the complete seed cohort.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("sample SEM requires at least one run-level value")
    if not np.isfinite(values).all():
        raise ValueError("sample SEM received a non-finite run-level value")
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def summarize(history):
    keys = [
        "experiment", "matrix", "algorithm", "condition",
        "learning_rate", "es_stepsize", "evaluation_target_step",
    ]
    for column in (
            "es_implementation", "plot_step_unit", "plot_step",
            "evaluation_metric", "evaluation_window_capacity",
            "evaluation_window_size",
    ):
        if column in history:
            keys.append(column)
    rows = []
    for values, group in history.groupby(keys, dropna=False):
        if group["seed"].duplicated().any():
            raise ValueError(
                "sample SEM requires one independent value per training seed"
            )
        rewards = group["leader_reward"].to_numpy(dtype=float)
        n = len(rewards)
        std = float(np.std(rewards, ddof=1)) if n > 1 else float("nan")
        sem = sample_sem(rewards)
        row = dict(zip(keys, values))
        row.update({
            "mean": float(np.mean(rewards)),
            "std": std,
            "sem": sem,
            "n": n,
            "n_independent_seeds": int(group["seed"].nunique()),
        })
        row["lower"] = row["mean"] - sem
        row["upper"] = row["mean"] + sem
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def configure_style():
    plt.rcParams.update({
        "axes.axisbelow": True,
        "axes.edgecolor": "white",
        "axes.facecolor": "#EAEAF2",
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.grid.which": "major",
        "axes.labelcolor": "#262626",
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "axes.prop_cycle": matplotlib.cycler(color=DEEP_PALETTE),
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "figure.dpi": 120,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "figure.titleweight": "normal",
        "font.size": 10,
        "font.weight": "normal",
        "grid.alpha": 1.0,
        "grid.color": "white",
        "grid.linewidth": 0.8,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.title_fontsize": 9.5,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.04,
        "text.color": "#262626",
        "xtick.direction": "out",
        "xtick.color": "#262626",
        "xtick.labelsize": 9.5,
        "ytick.direction": "out",
        "ytick.color": "#262626",
        "ytick.labelsize": 9.5,
    })


def format_training_step(value, _position=None):
    """Format training-step ticks compactly (for example, 100k or 1M)."""
    if not np.isfinite(value):
        return ""
    sign = "−" if value < 0 else ""
    magnitude = abs(float(value))
    if magnitude >= 1_000_000:
        scaled, suffix = magnitude / 1_000_000, "M"
    elif magnitude >= 1_000:
        scaled, suffix = magnitude / 1_000, "k"
    else:
        scaled, suffix = magnitude, ""
    if np.isclose(scaled, round(scaled)):
        number = str(int(round(scaled)))
    else:
        number = "{:.1f}".format(scaled).rstrip("0").rstrip(".")
    return "{}{}{}".format(sign, number, suffix)


def draw_curve(ax, data, condition, label=None):
    step_column = "plot_step" if "plot_step" in data else "evaluation_target_step"
    line = data[data["condition"] == condition].sort_values(step_column)
    if line.empty:
        return
    x = line[step_column].to_numpy()
    mean = line["mean"].to_numpy()
    sem = line["sem"].to_numpy()
    color = COLORS[condition]
    ax.plot(
        x, mean, color=color, linestyle=LINESTYLES[condition],
        linewidth=LINE_WIDTH,
        label=label or LABELS[condition],
    )
    ax.fill_between(
        x, mean - sem, mean + sem, color=color,
        where=np.isfinite(sem), alpha=STANDARD_ERROR_ALPHA, linewidth=0,
    )


def finish_axis(ax, ylabel=True, xlabel="Training steps",
                ylabel_text="Leader reward"):
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel_text)
    ax.xaxis.set_major_locator(MaxNLocator(
        nbins=MAX_X_TICKS, steps=(1, 2, 2.5, 5, 10), min_n_ticks=3,
    ))
    ax.xaxis.set_major_formatter(FuncFormatter(format_training_step))
    ax.margins(x=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_uniform_legend(figure, axes, right):
    """Add one frameless, outside-right legend with stable label ordering."""
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = (axes,)
    handles_by_label = {}
    for ax in np.asarray(axes, dtype=object).flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    labels = list(handles_by_label)
    figure.legend(
        [handles_by_label[label] for label in labels], labels,
        loc="center right", bbox_to_anchor=(0.995, 0.5),
        borderaxespad=0.0,
    )
    figure.subplots_adjust(right=right)


def plot_hidden(summary):
    preferred_order = ("PG", "A2C", "PPO", "ES")
    present = set(summary["algorithm"].unique())
    algorithms = tuple(
        algorithm for algorithm in preferred_order if algorithm in present
    )
    if not algorithms:
        raise ValueError("hidden-query summary contains no supported algorithm")
    legend_width = 1.85
    figure_width = 3.0 * len(algorithms) + legend_width
    figure, axes = plt.subplots(
        1, len(algorithms), figsize=(figure_width, 3.2), sharey=True,
    )
    axes = np.atleast_1d(axes)
    for index, algorithm in enumerate(algorithms):
        panel = summary[summary["algorithm"] == algorithm]
        draw_curve(axes[index], panel, "observed")
        draw_curve(axes[index], panel, "hidden")
        axes[index].set_title(ALGORITHM_TITLES.get(algorithm, algorithm))
        if "plot_step_unit" in panel:
            units = set(panel["plot_step_unit"].dropna().unique())
            if len(units) != 1:
                raise ValueError(
                    "hidden-query panel {} has mixed x-axis units: {}".format(
                        algorithm, sorted(units)
                    )
                )
            unit = next(iter(units))
            xlabels = {
                PLOT_UNIT_ENVIRONMENT_TRANSITIONS: "Environment steps",
                PLOT_UNIT_ES_ITERATIONS: "ES iterations",
            }
            if unit not in xlabels:
                raise ValueError("unknown hidden-query x-axis unit: {}".format(unit))
            xlabel = xlabels[unit]
        else:
            xlabel = "Training steps"
        finish_axis(
            axes[index], ylabel=index == 0, xlabel=xlabel,
            ylabel_text="Leader reward per stage",
        )
    add_uniform_legend(
        figure, axes, right=1.0 - legend_width / figure_width,
    )
    figure.subplots_adjust(wspace=0.22)
    return figure


def plot_phase(summary):
    figure, ax = plt.subplots(figsize=(7.4, 3.6))
    draw_curve(ax, summary, "visible", LABELS["visible"])
    draw_curve(ax, summary, "hidden", LABELS["phase_hidden"])
    finish_axis(ax)
    add_uniform_legend(figure, ax, right=0.67)
    return figure


def plot_q_reset(summary):
    figure, ax = plt.subplots(figsize=(7.4, 3.6))
    draw_curve(ax, summary, "reset")
    draw_curve(ax, summary, "ongoing")
    finish_axis(ax)
    add_uniform_legend(figure, ax, right=0.70)
    return figure


def plot_response_reward(summary):
    panels = (
        ("coordination_zero_miscoordination", "No coordination penalty"),
        ("coordination_penalized_miscoordination", "Coordination penalty (−5)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), sharey=True)
    for index, (matrix, title) in enumerate(panels):
        panel = summary[summary["matrix"] == matrix]
        draw_curve(axes[index], panel, "excluded")
        draw_curve(axes[index], panel, "included")
        axes[index].set_title(title)
        finish_axis(axes[index], ylabel=index == 0)
    add_uniform_legend(figure, axes, right=0.74)
    figure.subplots_adjust(wspace=0.20)
    return figure


PLOTTERS = {
    "fig_hidden": plot_hidden,
    "fig_phase_observability": plot_phase,
    "fig_q_reset": plot_q_reset,
    "fig_response_reward": plot_response_reward,
}


def write_manifest(path, files, records):
    payload = {
        "schema_version": 1,
        "uncertainty": "sample SEM across independent training-seed means",
        "uncertainty_specification": {
            "sampling_unit": "independent leader-training seed",
            "formula": "sample_std(ddof=1) / sqrt(n_independent_seeds)",
            "minimum_independent_seeds": 2,
            "single_seed_sem": "undefined (NaN)",
            "within_run_evaluation_sem_pooled": False,
        },
        "records": records,
        "files": {
            Path(os.path.relpath(file, path.parent)).as_posix(): sha256(file)
            for file in files if file.exists()
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def render_figure(name, output, paper_logs_root, runs, history, configs):
    experiment = EXPERIMENT_FOR_FIGURE[name]
    run_sort = ["matrix", "algorithm", "condition", "seed", "run_id"]
    history_sort = [
        "matrix", "algorithm", "condition", "seed",
        "evaluation_target_step", "run_id",
    ]
    if "plot_step" in history:
        history_sort.insert(-2, "plot_step")
    figure_runs = runs[runs["experiment"] == experiment].copy().sort_values(
        run_sort
    )
    figure_history = history[
        history["experiment"] == experiment
    ].copy().sort_values(history_sort)
    figure_configs = configs[
        configs["experiment"] == experiment
    ].copy().sort_values(run_sort)
    completed_ids = set(figure_runs.loc[
        figure_runs["status"] == "completed", "run_id"
    ])
    figure_history = figure_history[figure_history["run_id"].isin(completed_ids)]
    summary = summarize(figure_history)

    figures_dir = output / "figures"
    basename = PAPER_BASENAME[name]
    logs_dir = paper_logs_root / basename
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / "{}_summary.csv".format(basename)
    pdf_path = figures_dir / "{}.pdf".format(basename)
    png_path = figures_dir / "{}.png".format(basename)
    history_path = logs_dir / "history.csv"
    runs_path = logs_dir / "runs.csv"
    configs_path = logs_dir / "configs.csv"
    summary.to_csv(summary_path, index=False, na_rep="NaN")
    figure_history.to_csv(history_path, index=False)
    figure_runs.to_csv(runs_path, index=False)
    figure_configs.to_csv(configs_path, index=False)

    figure = PLOTTERS[name](summary)
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=400, facecolor="white")
    plt.close(figure)
    files = [summary_path, pdf_path, png_path, history_path, runs_path, configs_path]
    write_manifest(
        logs_dir / "manifest.json",
        files,
        {
            "figure": name,
            "paper_basename": basename,
            "experiment": experiment,
            "runs": int(len(figure_runs)),
            "completed_runs": int(len(completed_ids)),
            "history_rows": int(len(figure_history)),
            "evaluation_metrics": sorted(
                str(value) for value in figure_history.get(
                    "evaluation_metric", pd.Series(dtype=str)
                ).dropna().unique()
            ),
            "x_axis_units": sorted(
                str(value) for value in figure_history.get(
                    "plot_step_unit", pd.Series(dtype=str)
                ).dropna().unique()
            ),
        },
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--additional-input",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional immutable result root to aggregate; may be repeated. "
            "All roots must form one non-overlapping compatible cohort."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--paper-logs-root", type=Path,
        help="Separate figure-grouped paper-log destination.",
    )
    parser.add_argument("--figure", choices=("all", *FIGURES), default="all")
    parser.add_argument("--seeds")
    parser.add_argument(
        "--require-seeds-per-cell", type=int,
        help=(
            "Require this many distinct seeds in the exact --seeds set for "
            "every plotted cell; incompatible with --allow-incomplete."
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def parse_seed_set(value):
    result = set()
    for token in value.split(","):
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    return result


def main():
    args = build_parser().parse_args()
    output = args.output or args.input
    paper_logs_root = args.paper_logs_root or output / "paper_logs"
    input_roots = [args.input, *args.additional_input]
    runs, history, configs = collect_input_roots(input_roots)
    plan_path = args.input / "plan.json"
    if len(input_roots) > 1 and any(
        (root / "plan.json").exists() for root in input_roots
    ):
        raise ValueError(
            "multi-root aggregation does not support component sweep plans"
        )
    plan = (
        read_json(plan_path)
        if len(input_roots) == 1 and plan_path.exists()
        else None
    )
    if plan is not None and not args.allow_incomplete:
        expected = int(plan["leader_runs"])
        if len(runs) != expected:
            raise ValueError(
                "leader-run count mismatch: expected {}, found {}".format(
                    expected, len(runs)
                )
            )
    if args.seeds is not None:
        expected_seeds = parse_seed_set(args.seeds)
    elif plan is not None:
        expected_seeds = set(int(seed) for seed in plan["seeds"])
    else:
        expected_seeds = parse_seed_set("1-10")
    validate_required_seed_count(
        expected_seeds, args.require_seeds_per_cell, args.allow_incomplete
    )
    if plan is not None:
        validate_plan_cells(runs, plan, args.allow_incomplete)
    validate_runs(runs, expected_seeds, args.allow_incomplete)
    completed_runs = runs[runs["status"] == "completed"]
    completed_ids = set(completed_runs["run_id"])
    validate_required_figure_cells(
        completed_runs, args.figure, args.require_seeds_per_cell,
        "completed run",
    )
    completed_history = history[history["run_id"].isin(completed_ids)]
    validate_history(
        completed_history,
        expected_seeds,
        args.allow_incomplete,
    )
    validate_required_figure_cells(
        completed_history, args.figure, args.require_seeds_per_cell,
        "evaluation-history",
    )
    configure_style()
    selected = FIGURES if args.figure == "all" else (args.figure,)
    for name in selected:
        render_figure(
            name, output, paper_logs_root, runs, history, configs
        )
        print("wrote {}".format(name), flush=True)


if __name__ == "__main__":
    main()
