#!/usr/bin/env python3
"""Create the paper-style PI training-reward comparison for SPM and MSPM.

The source logs can contain millions of reward rows, so this script streams
them into fixed-width training-step bins rather than loading raw rows into
memory.  The bold curves are centered moving averages of the binned rewards;
the lightly drawn curves show the unsmoothed bin means.  No uncertainty band
is shown because the current clean comparison contains one seed per method.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REWARD_RE = re.compile(
    r"^\[reward\]\s+steps=(?P<step>\d+)\s+.*?"
    r"reward_phase_avg=(?P<reward>[-+0-9.eE]+)(?:\s|$)"
)

PAPER_COLORS = {
    "MSPM": "#4C72B0",
    "SPM": "#DD8452",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spm-log", required=True, type=Path)
    parser.add_argument("--mspm-log", required=True, type=Path)
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Output path without an extension; writes PDF, PNG, CSV, and JSON.",
    )
    parser.add_argument("--bin-width", type=int, default=10_000)
    parser.add_argument(
        "--smooth-steps",
        type=int,
        default=100_000,
        help="Width of the centered moving average in training steps.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Optional common x-axis cutoff. Defaults to the shorter run.",
    )
    parser.add_argument("--spm-run-id", default="6ijiy88e")
    parser.add_argument("--mspm-run-id", default="5xhgdpgt")
    return parser.parse_args()


def stream_bin_rewards(path, bin_width):
    reward_sums = defaultdict(float)
    reward_counts = defaultdict(int)
    max_step = 0
    matched_rows = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("[reward]"):
                continue
            match = REWARD_RE.match(line)
            if match is None:
                continue
            step = int(match.group("step"))
            reward = float(match.group("reward"))
            bin_index = step // bin_width
            reward_sums[bin_index] += reward
            reward_counts[bin_index] += 1
            max_step = max(max_step, step)
            matched_rows += 1

    if not matched_rows:
        raise ValueError(f"No [reward] rows found in {path}")

    return reward_sums, reward_counts, max_step, matched_rows


def binned_series(reward_sums, reward_counts, bin_width, max_step):
    max_bin = max_step // bin_width
    steps = []
    rewards = []
    counts = []
    for bin_index in range(max_bin + 1):
        count = reward_counts.get(bin_index, 0)
        if count == 0:
            continue
        steps.append((bin_index + 0.5) * bin_width)
        rewards.append(reward_sums[bin_index] / count)
        counts.append(count)
    return (
        np.asarray(steps, dtype=float),
        np.asarray(rewards, dtype=float),
        np.asarray(counts, dtype=int),
    )


def centered_moving_average(values, window):
    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values), kernel, mode="same")
    return numerator / denominator


def configure_paper_style():
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        plt.style.use("seaborn-darkgrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 9.5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_data_csv(path, series_by_method):
    rows_by_step = defaultdict(dict)
    for method, series in series_by_method.items():
        for step, reward, smoothed, count in zip(
            series["steps"],
            series["rewards"],
            series["smoothed"],
            series["counts"],
        ):
            rows_by_step[int(step)][method] = (reward, smoothed, int(count))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "training_step",
                "mspm_bin_reward",
                "mspm_smoothed_reward",
                "mspm_observations",
                "spm_bin_reward",
                "spm_smoothed_reward",
                "spm_observations",
            ]
        )
        for step in sorted(rows_by_step):
            mspm = rows_by_step[step].get("MSPM", ("", "", ""))
            spm = rows_by_step[step].get("SPM", ("", "", ""))
            writer.writerow([step, *mspm, *spm])


def main():
    args = parse_args()
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    if args.smooth_steps <= 0:
        raise ValueError("--smooth-steps must be positive")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    for method, path in (("MSPM", args.mspm_log), ("SPM", args.spm_log)):
        sums, counts, max_step, matched_rows = stream_bin_rewards(
            path, args.bin_width
        )
        raw[method] = {
            "sums": sums,
            "counts": counts,
            "max_step": max_step,
            "matched_rows": matched_rows,
            "path": path,
        }

    common_max_step = min(raw["MSPM"]["max_step"], raw["SPM"]["max_step"])
    if args.max_step is not None:
        common_max_step = min(common_max_step, args.max_step)
    common_max_step = (common_max_step // args.bin_width) * args.bin_width
    if common_max_step < args.bin_width:
        raise ValueError("The common run horizon is shorter than one bin.")

    smooth_bins = max(1, round(args.smooth_steps / args.bin_width))
    series_by_method = {}
    for method in ("MSPM", "SPM"):
        steps, rewards, counts = binned_series(
            raw[method]["sums"],
            raw[method]["counts"],
            args.bin_width,
            common_max_step,
        )
        keep = steps <= common_max_step
        steps = steps[keep]
        rewards = rewards[keep]
        counts = counts[keep]
        series_by_method[method] = {
            "steps": steps,
            "rewards": rewards,
            "counts": counts,
            "smoothed": centered_moving_average(rewards, smooth_bins),
        }

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    fig.subplots_adjust(left=0.14, right=0.75, bottom=0.19, top=0.97)

    for method in ("MSPM", "SPM"):
        series = series_by_method[method]
        color = PAPER_COLORS[method]
        ax.plot(
            series["steps"],
            series["rewards"],
            color=color,
            linewidth=0.9,
            alpha=0.22,
            zorder=1,
        )
        ax.plot(
            series["steps"],
            series["smoothed"],
            color=color,
            linewidth=2.25,
            label=method,
            zorder=3,
        )

    all_rewards = np.concatenate(
        [series_by_method[method]["smoothed"] for method in ("MSPM", "SPM")]
    )
    lower = min(-0.10, np.floor((np.nanmin(all_rewards) - 0.005) / 0.02) * 0.02)
    ax.set_xlim(0, common_max_step)
    ax.set_ylim(lower, 0.005)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Leader Reward")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle=(0, (2, 2)), zorder=2)
    ax.legend(
        title="Mechanism",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=2.2,
    )

    pdf_path = args.output_prefix.with_suffix(".pdf")
    png_path = args.output_prefix.with_suffix(".png")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")

    metadata = {
        "Title": "SPM versus MSPM training reward in the two-type PI setting",
        "Subject": "Seed-1 clean-run comparison; 100,000-step centered smoothing",
        "Keywords": "Stackelberg POMDP, SPM, MSPM, PI, training reward",
    }
    fig.savefig(pdf_path, metadata=metadata, facecolor="white")
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)

    write_data_csv(csv_path, series_by_method)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure": {
            "pdf": str(pdf_path),
            "png": str(png_path),
            "data_csv": str(csv_path),
        },
        "sources": {
            "SPM": {
                "run_id": args.spm_run_id,
                "log": str(args.spm_log),
                "observed_max_step": raw["SPM"]["max_step"],
                "reward_rows": raw["SPM"]["matched_rows"],
            },
            "MSPM": {
                "run_id": args.mspm_run_id,
                "log": str(args.mspm_log),
                "observed_max_step": raw["MSPM"]["max_step"],
                "reward_rows": raw["MSPM"]["matched_rows"],
            },
        },
        "processing": {
            "common_max_step": common_max_step,
            "bin_width_steps": args.bin_width,
            "centered_smoothing_steps": smooth_bins * args.bin_width,
            "uncertainty_band": None,
            "seeds_per_method": 1,
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
