#!/usr/bin/env python3
"""Reproduce the retrospective comparisons specified in statistical_protocol.json."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, ttest_ind

from _support.platform import load_search
from _support.mspm import spm_optimum

LOGS = Path(__file__).resolve().parents[1]
DATA = LOGS / "data"
PROTOCOL = Path(__file__).parent / "_support/statistical_protocol.json"
from _support import data
INPUTS = data.INPUTS


def read_text(path):
    INPUTS.add(path.resolve())
    return path.read_text()


def read(path):
    return json.loads(read_text(path))




def mean_se(values):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) > 1 and np.isfinite(values).all()
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    assert np.isfinite(p_values).all() and ((0 <= p_values) & (p_values <= 1)).all()
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(order))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        (len(order) - np.arange(len(order))) * p_values[order]))
    return adjusted.tolist()


def sign_comparison(identifier, family, label, differences, tolerance):
    differences = np.asarray(differences, dtype=float)
    positive = int(np.sum(differences > tolerance))
    negative = int(np.sum(differences < -tolerance))
    nonzero = positive + negative
    effect, se = mean_se(differences)
    return dict(id=identifier, family=family, comparison=label, method="S",
                effect=effect, se=se, n=len(differences), positive=positive,
                negative=negative, ties=len(differences)-nonzero,
                p_raw=float(binomtest(positive, nonzero).pvalue) if nonzero else 1.,
                differences=differences.tolist())


def mspm(tolerance):
    groups = {types: {} for types in range(2, 7)}
    curves = data.mspm_curves()
    for run in data.csv_rows("table_mspm_welfare"):
        assert run["status"] == "completed" and run["curve_points"] == 51
        curve = [r for r in curves if r["num_types"] == run["num_types"] and r["seed"] == run["seed"]]
        assert [r["episodes"] for r in curve] == list(range(0, 10001, 200))
        assert all(r["bcce_certified"] == 1 and r["bcce_gap"] <= .05+tolerance for r in curve)
        best = max([r["raw_reward"] for r in curve] + [run["final_separate_greedy"]])
        assert best == run["best_greedy"]
        group = groups[run["num_types"]]
        assert run["seed"] not in group
        group[run["seed"]] = best - spm_optimum(run["num_types"])
    rows = []
    for types, values in groups.items():
        assert set(values) == set(range(96, 121))
        row = sign_comparison(f"mspm_{types}", "MSPM", f"MSPM minus SPM, {types} types",
                              [values[s] for s in sorted(values)], tolerance)
        row["seeds"] = sorted(values)
        rows.append(row)
    return rows


def seller_means(responses, seeds):
    cells = {(row["cost"], row["seed"]): row["frozen_cs"] for row in responses}
    assert len(cells) == len(responses) == 2 * len(seeds)
    assert set(cells) == {(cost, seed) for cost in (1., 1.5) for seed in seeds}
    return np.array([np.mean([cells[cost, seed] for cost in (1., 1.5)]) for seed in seeds])


def crossed_mean(matrix, row_counts, column_counts):
    """Bootstrap means without constructing repeated copies of each crossed matrix."""
    n, m = matrix.shape
    return np.einsum("bi,ij,bj->b", row_counts, matrix, column_counts,
                     optimize=False) / (n * m)


def exact_bootstrap_variance(matrix):
    """Conditional variance of independent multinomial row/column resampling."""
    n, m = matrix.shape
    row = matrix.mean(axis=1) - matrix.mean()
    column = matrix.mean(axis=0) - matrix.mean()
    interaction = matrix - matrix.mean() - row[:, None] - column[None, :]
    return float(row @ row / n**2 + column @ column / m**2
                 + np.sum(interaction**2) / (n**2 * m**2))


def bootstrap_platform(matrix, fixed, clipped, random_policy, settings):
    n, m = matrix.shape
    k = len(random_policy)
    rng = np.random.default_rng(settings["seed"])
    draws = settings["draws"]
    samples = np.empty((draws, 3))
    for start in range(0, draws, settings["batch_size"]):
        stop = min(start + settings["batch_size"], draws)
        size = stop - start
        rows = rng.multinomial(n, np.full(n, 1/n), size=size)
        columns = rng.multinomial(m, np.full(m, 1/m), size=size)
        other = rng.multinomial(k, np.full(k, 1/k), size=size)
        ppo = crossed_mean(matrix, rows, columns)
        samples[start:stop, 0] = ppo - columns @ fixed / m
        samples[start:stop, 1] = ppo - columns @ clipped / m
        samples[start:stop, 2] = ppo - other @ random_policy / k
    effects = matrix.mean() - np.array([fixed.mean(), clipped.mean(), random_policy.mean()])
    variances = [exact_bootstrap_variance(matrix - fixed),
                 exact_bootstrap_variance(matrix - clipped),
                 exact_bootstrap_variance(matrix) + float(np.var(random_policy) / k)]
    results = []
    for index, effect in enumerate(effects):
        bootstrap = samples[:, index]
        q_low, q_high = np.quantile(bootstrap, [.025, .975])
        exceedances = int(np.sum(np.abs(bootstrap - effect) >= abs(effect)))
        results.append(dict(effect=float(effect), se=float(bootstrap.std(ddof=1)),
                            p_raw=(1 + exceedances) / (draws + 1),
                            exceedances=exceedances,
                            basic_ci_95=[float(2*effect-q_high), float(2*effect-q_low)],
                            exact_bootstrap_se=float(np.sqrt(variances[index]))))
    return results


def platform(settings):
    confirmations = data.platform_results()
    seeds = sorted({r["seed"] for r in confirmations["PPO", 801]["responses"]})
    assert len(seeds) == 25
    training_seeds = list(range(801, 826))
    matrix = np.asarray([seller_means(confirmations["PPO", seed]["responses"], seeds) for seed in training_seeds])
    baselines = [seller_means(confirmations[name, ""]["responses"], seeds) for name in ("Fixed cap", "Clipped rule")]
    random_responses = confirmations["Random search", ""]["responses"]
    random_seeds = sorted({r["seed"] for r in random_responses})
    assert len(random_seeds) == 25 and set(seeds).isdisjoint(random_seeds)
    random_policy = seller_means(random_responses, random_seeds)
    assert abs(random_policy.mean() - load_search()["cs"]) < 1e-12
    results = bootstrap_platform(matrix, *baselines, random_policy, settings)
    for result, name, label in zip(results, ("fixed", "clipped", "selected_random"),
                                   ("fixed cap", "clipped rule", "selected random policy")):
        result.update(id="platform_"+name, family="Platform", comparison="PPO minus "+label,
                      method="B", n=25)
        assert abs(result["se"] / result["exact_bootstrap_se"] - 1) < .006
    units = dict(training_seeds=training_seeds, seller_seeds=seeds,
                 random_seller_seeds=random_seeds, ppo=matrix.tolist(),
                 fixed=baselines[0].tolist(), clipped=baselines[1].tolist(),
                 selected_random=random_policy.tolist())
    return results, units


def atari():
    groups = {role: [] for role in ("seller", "buyer")}
    curves = data.atari_curves()
    for role in groups:
        for seed in range(1, 26):
            rows = [r for r in curves if r["leader_role"] == role and r["seed"] == seed]
            groups[role].append(max(rows, key=lambda r: r["training_total_timesteps"]))
    results = []
    for identifier, metric, label in (
            ("price", "mean_accepted_price", "Accepted price"),
            ("seller_payoff", "mean_seller_payoff", "Seller payoff"),
            ("buyer_payoff", "mean_buyer_payoff", "Buyer payoff")):
        seller, buyer = [np.array([r[metric] for r in groups[role]]) for role in ("seller", "buyer")]
        test = ttest_ind(seller, buyer, equal_var=False)
        results.append(dict(id="atari_"+identifier, family="Atari", comparison=label,
                            method="W", effect=float(seller.mean()-buyer.mean()),
                            se=float(np.sqrt(seller.var(ddof=1)/25 + buyer.var(ddof=1)/25)),
                            n=25, p_raw=float(test.pvalue), statistic=float(test.statistic),
                            degrees_of_freedom=float(test.df), seeds=list(range(1, 26)),
                            seller_led=seller.tolist(), buyer_led=buyer.tolist()))
    return results


def ablations(tolerance):
    records = data.csv_rows("response_credit_ablation")
    terminal = {(r["method"], int(r["seed"])): float(r["exact_value"])
                for r in records if int(r["update"]) == 1000}
    assert set(terminal) == {(m, s) for m in ("full", "gameplay_only") for s in range(1, 26)}
    results = [sign_comparison("response_credit", "Ablations", "Full minus gameplay-only credit",
                              [terminal["full", s]-terminal["gameplay_only", s]
                               for s in range(1, 26)], tolerance)]
    results[0]["seeds"] = list(range(1, 26))
    for experiment in ("commitment_consistency", "reward_timing"):
        groups = {}
        for config, _, evaluation in data.matrix_runs(experiment, 150 if experiment == "commitment_consistency" else 100):
            key = (config["condition"], config["matrix"], config["learning_rate"], config["seed"])
            assert key not in groups
            groups[key] = (evaluation["per_stage_summary"]["mean"], config["response_checkpoint_sha256"])
        if experiment == "commitment_consistency":
            specifications = [("commitment_008", "Phase visible minus hidden, lr=0.008", "prisoners_dilemma", .008),
                              ("commitment_015", "Phase visible minus hidden, lr=0.015", "prisoners_dilemma", .015)]
            left, right = "visible", "hidden"
        else:
            specifications = [("reward_timing_zero", "Reward exclusion minus inclusion, c=0", "coordination_zero_miscoordination", .1),
                              ("reward_timing_penalized", "Reward exclusion minus inclusion, c=5", "coordination_penalized_miscoordination", .1)]
            left, right = "excluded", "included"
        for identifier, label, matrix, rate in specifications:
            differences = []
            for seed in range(25):
                a, checkpoint_a = groups[left, matrix, rate, seed]
                b, checkpoint_b = groups[right, matrix, rate, seed]
                assert checkpoint_a == checkpoint_b
                differences.append(a-b)
            row = sign_comparison(identifier, "Ablations", label, differences, tolerance)
            row["seeds"] = list(range(25))
            results.append(row)
    return results


def tex_p(value):
    if value >= .001:
        return f"{value:.4f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def table_tex(rows):
    lines = ["% Generated by statistical_comparisons.py; do not edit.",
             r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Comparison & Mean difference (SE) & Test & Adjusted $p$ \\", r"\midrule"]
    family = None
    for row in rows:
        if row["family"] != family:
            if family is not None:
                lines.append(r"\addlinespace")
            family = row["family"]
            label = family + (": seller-led minus buyer-led" if family == "Atari" else "")
            lines.append(r"\multicolumn{4}{@{}l}{\textit{"+label+r"}} \\")
        lines.append(f"{row['comparison']} & ${row['effect']:.4f}\\;({row['se']:.4f})$ & "
                     f"{row['method']} & ${tex_p(row['p_adjusted'])}$ " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    platform_rows, platform_units = platform(protocol["bootstrap"])
    rows = mspm(protocol["zero_tolerance"]) + platform_rows + atari() + ablations(protocol["zero_tolerance"])
    for family, specification in protocol["families"].items():
        group = [row for row in rows if row["family"] == family]
        assert [row["id"] for row in group] == specification["comparisons"]
        for row, adjusted in zip(group, holm([row["p_raw"] for row in group])):
            row["p_adjusted"] = adjusted
    output = args.output_dir.resolve()
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "statistical_comparisons.json").write_text(json.dumps(
        dict(protocol=protocol, comparisons=rows, platform_units=platform_units), indent=2, allow_nan=False) + "\n")
    fields = ["id", "family", "comparison", "method", "effect", "se", "n", "p_raw", "p_adjusted"]
    with (output / "tables/statistical_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output / "statistical_comparisons.tex").write_text(table_tex(rows))
    (output / "statistical_inputs.json").write_text(json.dumps(
        {str(p.relative_to(LOGS)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(INPUTS)}, indent=2) + "\n")
    for row in rows:
        print(f"{row['id']}: difference {row['effect']:.6f} (SE {row['se']:.6f}), "
              f"Holm p={row['p_adjusted']:.6g}")


if __name__ == "__main__":
    main()
