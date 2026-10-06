"""Exact MSPM benchmarks, cohort validation and appendix-panel drawing."""
from __future__ import annotations

import csv
from fractions import Fraction
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import METHOD_COLORS, configure_mspm_style

COHORTS = {
    "uniform_std05": dict(seeds=set(range(96, 121)),
                          episodes=set(range(0, 10001, 200))),
}
EPISODES = np.arange(0, 10001, 200)
SEEDS = range(96, 121)
TYPES = range(3, 7)


def spm_optimum(types: int) -> float:
    if types == 2:
        return -0.06
    return float(-min(
        Fraction(math.comb(types - k + 1, 3) + math.comb(k + 1, 3),
                 types**2 * (types - 1))
        for k in range(1, types + 1)
    ))


def ex_post_optimum(types: int) -> float:
    if types == 2:
        return 0.44
    numerator = sum(k * (2 * k + 1) for k in range(types))
    return float(Fraction(numerator, types**2 * (types - 1)))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_data(curve_path: Path, seed_path: Path, *, cohort: dict) -> tuple[np.ndarray, np.ndarray]:
    expected_seeds = cohort["seeds"]
    expected_episodes = cohort["episodes"]
    curves = read_rows(curve_path)
    seeds = read_rows(seed_path)
    seeds = [row for row in seeds if int(row.get("num_types", 2)) == 2]
    curves = [row for row in curves if int(row.get("num_types", 2)) == 2]
    if len(seeds) != 25 or {int(row["seed"]) for row in seeds} != expected_seeds:
        raise ValueError("Expected exactly one completed result for each selected seed")
    if any(row["status"] != "completed"
           or int(row["curve_points"]) != len(expected_episodes)
           for row in seeds):
        raise ValueError("The seed summary contains an incomplete validation run")
    if len(curves) != 25 * len(expected_episodes):
        raise ValueError("Missing current-policy evaluations")

    seen = set()
    values: dict[int, dict[int, float]] = {seed: {} for seed in expected_seeds}
    for row in curves:
        seed = int(row["seed"])
        episode = int(row["episodes"])
        key = (seed, episode)
        if seed not in expected_seeds or episode not in expected_episodes or key in seen:
            raise ValueError(f"Unexpected or duplicate curve cell: {key}")
        seen.add(key)
        reward = float(row["raw_reward"])
        gap = float(row["bcce_gap"])
        if not np.isfinite(reward) or not np.isfinite(gap):
            raise ValueError(f"Nonfinite evaluation: {key}")
        if reward > 1e-8 or row["bcce_certified"] != "1" or gap > 0.05 + 1e-12:
            raise ValueError(f"Invalid welfare bound or BCCE certificate: {key}")
        values[seed][episode] = reward
    if any(set(seed_values) != expected_episodes for seed_values in values.values()):
        raise ValueError("A seed has missing evaluation episodes")
    episodes = np.array(sorted(expected_episodes), dtype=float)
    matrix = np.array([[values[seed][int(ep)] for ep in episodes]
                       for seed in sorted(expected_seeds)], dtype=float)
    return episodes, matrix


def load_multitype_runs(root: Path) -> dict[int, np.ndarray]:
    histories = {}
    with root.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    for types in TYPES:
        for seed in SEEDS:
            group = [r for r in records if int(r["num_types"]) == types and int(r["seed"]) == seed]
            if [int(r["episodes"]) for r in group] != EPISODES.tolist():
                raise ValueError(f"Incomplete evaluation history: {types}, {seed}")
            if any(r["bcce_certified"] != "1" or float(r["bcce_gap"]) > .05+1e-12 for r in group):
                raise ValueError("Invalid BCCE certificate")
            rewards = np.asarray([float(r["raw_reward"]) for r in group])
            if not np.isfinite(rewards).all() or rewards.max() > 1e-8:
                raise ValueError("Invalid welfare history")
            histories[types, seed] = rewards
    if len(records) != 5100:
        raise ValueError("Expected 5,100 current-policy evaluations")
    if set(histories) != {(types, seed) for types in TYPES for seed in SEEDS}:
        raise ValueError("Expected all 25 runs for each of types 3 through 6")
    return {types: np.stack([histories[types, seed] for seed in SEEDS])
            for types in TYPES}


def make_multitype_figure(values: dict[int, np.ndarray]) -> plt.Figure:
    configure_mspm_style()
    plt.rcParams.update({"axes.labelsize": 10, "axes.titlesize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    x = EPISODES / 1000
    for label, types, ax in zip("abcd", TYPES, axes.flat):
        matrix = values[types]
        mean = matrix.mean(axis=0)
        se = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
        ax.plot(x, matrix.T, color=METHOD_COLORS["MSPM"], alpha=.16,
                lw=.65, zorder=1)
        ax.fill_between(x, mean-se, mean+se, color=METHOD_COLORS["MSPM"],
                        alpha=.18, linewidth=0, zorder=2)
        ax.plot(x, mean, color=METHOD_COLORS["MSPM"], lw=2,
                label="MSPM mean", zorder=5)
        ax.axhline(spm_optimum(types), color=METHOD_COLORS["SPM"],
                   ls=":", lw=1.6, label="Exact optimal SPM", zorder=3)
        ax.axhline(0, color=".25", ls="--", lw=1.2,
                   label="Ex post optimum", zorder=4)
        ax.set(title=f"({label}) {types} types per buyer", xlim=(0, 10),
               ylim=(-.12, .005), xticks=np.arange(0, 11, 2),
               yticks=[-.12, -.08, -.04, 0])
    fig.supylabel(r"Welfare gap $W-W^*$", x=.012, fontsize=11)
    fig.supxlabel("Training episodes (thousands)", y=.105, fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .015),
               ncol=3, columnspacing=1.6, handlelength=2.5)
    fig.subplots_adjust(left=.095, right=.99, top=.94, bottom=.20,
                        hspace=.29, wspace=.12)
    return fig
