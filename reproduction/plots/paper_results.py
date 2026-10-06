#!/usr/bin/env python3
"""Recompute the manuscript's numerical claims from the complete rerun cohort."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.stats import ttest_ind

from _support.platform import payoffs
from _support.platform import load_search

LOGS = Path(__file__).resolve().parents[1]
DATA = LOGS / "data"
from _support import data
INPUTS = data.INPUTS




def mean_se(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = data.platform_results()
    scores = []
    for seed in range(801, 826):
        result = results["PPO", seed]
        costs = [float(np.mean([x["frozen_cs"] for x in result["responses"]
                               if x["cost"] == c])) for c in (1., 1.5)]
        assert np.isclose(np.mean(costs), result["cs"], atol=1e-12)
        scores.append([*costs, result["cs"]])
    a = np.asarray(scores)
    means = a.mean(axis=0)
    ses = a.std(axis=0, ddof=1) / 5
    medians = np.median(a, axis=0)
    fixed = results["Fixed cap", ""]
    clipped = results["Clipped rule", ""]
    search = load_search()
    macros = {"PlatformN": 25, "PlatformSellerN": 25,
              "PlatformMean": f"{means[2]:.4f}", "PlatformSE": f"{ses[2]:.4f}",
              "PlatformLow": f"{means[0]:.4f}", "PlatformLowSE": f"{ses[0]:.4f}",
              "PlatformHigh": f"{means[1]:.4f}", "PlatformHighSE": f"{ses[1]:.4f}",
              "PlatformFixed": f"{fixed['cs']:.4f}", "PlatformClipped": f"{clipped['cs']:.4f}",
              "PlatformUniform": f"{search['cs']:.4f}",
              "PlatformMedian": f"{medians[2]:.4f}",
              "PlatformLowMedian": f"{medians[0]:.4f}",
              "PlatformHighMedian": f"{medians[1]:.4f}",
              "PlatformGain": f"{means[2]-fixed['cs']:.4f}",
              "PlatformGainPercent": f"{100*(means[2]/fixed['cs']-1):.1f}",
              "PlatformAbove": int(sum(a[:, 2] > fixed["cs"] + 1e-12)),
              "PlatformMatchingClipped": int(sum(abs(a[:, 2]-clipped["cs"]) <= 1e-12)),
              "PlatformSeedScores": ", ".join(f"{v:.4f}" for v in a[:, 2])}
    for name, result in (("Fixed", fixed), ("Clipped", clipped), ("Uniform", search)):
        for label, cost in (("Low", 1.), ("High", 1.5)):
            value = (result["per_cost"][str(cost)] if "per_cost" in result else
                     np.mean([r["frozen_cs"] for r in result["responses"] if r["cost"] == cost]))
            macros[f"Platform{name}{label}"] = f"{value:.4f}"
    numbers = "% Generated from the complete on-demand rerun cohort; do not edit.\n"
    numbers += "".join("\\newcommand{\\" + k + "}{" + str(v) + "}\n" for k, v in macros.items())
    (out / "hidden_cost_numbers.tex").write_text(numbers)

    chosen_seed, prices, admissions, _ = data.platform_policy()
    chosen = results["PPO", chosen_seed]
    i, j = np.indices((15, 15))
    n = admissions.sum(axis=2)
    rule = np.where(n == 2, np.maximum(i, j), np.where(n == 1, np.minimum(i, j), np.minimum(i, j)-1))
    clip_rule = np.minimum(7, np.maximum(4, np.minimum(i, j)))
    clip_admissions = np.stack((i <= clip_rule, j <= clip_rule), axis=-1)
    mechanism = {"different_ordered_pairs": int(np.any(admissions != clip_admissions, axis=-1).sum()),
                 "equal_price_admissions_match": bool(np.array_equal(admissions[i == j], clip_admissions[i == j])),
                 "tied_top_policies": int(np.isclose(a[:, 2], chosen["cs"], atol=1e-12, rtol=0).sum()),
                 "selected_score": chosen["cs"], "price_pair_counts": {}}
    for cost in (1., 1.5):
        for name, result in (("learned", chosen), ("clipped", clipped)):
            counts = {}
            for row in result["responses"]:
                if row["cost"] != cost:
                    continue
                pair = str(row["cycle_actions"])
                counts[pair] = counts.get(pair, 0) + 1
            mechanism["price_pair_counts"][f"{name}:{cost}"] = counts
    mechanism["profits"] = {name: {
        "at_4_4": float(payoffs(prices, mapping, 4, 4, 1.)[0][0]),
        "cut_to_3": float(payoffs(prices, mapping, 3, 4, 1.)[0][0]),
        "at_1_1": float(payoffs(prices, mapping, 1, 1, 1.)[0][0]),
        "raise_to_4": float(payoffs(prices, mapping, 4, 1, 1.)[0][0]),
    } for name, mapping in (("learned", rule), ("clipped", clip_rule))}
    facts = {"platform": {"macros": macros, "below_fixed": int(sum(a[:, 2] <= fixed['cs'])),
                          "below_fixed_range": [float(a[a[:, 2] <= fixed['cs'], 2].min()), float(a[a[:, 2] <= fixed['cs'], 2].max())],
                          "above_fixed_range": [float(a[a[:, 2] > fixed['cs'], 2].min()), float(a[:, 2].max())],
                          "above_clipped": int(sum(a[:, 2] > clipped['cs'])), "mechanism": mechanism,
                          "random_search": search}}

    metrics = ("mean_accepted_price", "mean_seller_payoff", "mean_buyer_payoff")
    groups = {role: [] for role in ("buyer", "seller")}
    curves = data.atari_curves()
    for role in groups:
        for seed in range(1, 26):
            records = sorted([r for r in curves if r["leader_role"] == role and r["seed"] == seed],
                             key=lambda r: r["training_total_timesteps"])
            groups[role].append([{"summary": r} for r in records])
    atari = {"means": {}, "welch_holm": {}}
    raw = []
    for metric in metrics:
        result = ttest_ind([r[-1]["summary"][metric] for r in groups["seller"]],
                           [r[-1]["summary"][metric] for r in groups["buyer"]], equal_var=False)
        raw.append(float(result.pvalue))
    order = np.argsort(raw)
    adjusted = np.maximum.accumulate([(len(raw)-rank) * raw[index] for rank, index in enumerate(order)])
    for index, value in zip(order, adjusted):
        atari["welch_holm"][metrics[index]] = {"p_raw": raw[index], "p_adjusted": min(1., float(value)), "n_per_role": 25}
    for role in groups:
        atari["means"][role] = {str(point): {m: dict(zip(("mean", "se"), mean_se([r[point]["summary"][m] for r in groups[role]])))
                                                 for m in metrics} for point in (0, 1, 5)}
    jobs = data.csv_rows("table_atari_compute")
    assert len(jobs) == 50 and all(r["allocated_cores"] == 5 and r["memory_gb"] == 24 for r in jobs)
    atari["compute"] = {"runs": 50, "wall_hours": sum(r["wall_seconds"] for r in jobs) / 3600,
                         "allocated_core_hours": sum(r["allocated_core_seconds"] for r in jobs) / 3600,
                         "total_cpu_hours": sum(r["total_cpu_seconds"] for r in jobs) / 3600}
    facts["atari"] = atari
    (out / "paper_results.json").write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")
    (out / "claim_inputs.json").write_text(json.dumps({str(p.relative_to(LOGS)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                      for p in sorted(INPUTS)}, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
