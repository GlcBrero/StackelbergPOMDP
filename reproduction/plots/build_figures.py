"""Generate the paper figures and tables directly from the named CSVs."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent
DATA = LOGS / "data"
OUT = Path(sys.argv[1]).resolve()
FIGS = OUT / "figures"
TABLES = OUT / "tables"
for folder in (OUT, FIGS, TABLES, OUT / "work"):
    folder.mkdir(parents=True, exist_ok=True)
from _support import data
INPUTS = data.INPUTS

from _support import style
from _support import platform as platform_plot
from _support import atari
from _support import mspm as mspm_plot
from _support.mspm import ex_post_optimum, spm_optimum
from _support.platform import load_search
from _support import response_credit as scalar


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = np.asarray(values, dtype=float)
    assert len(values) > 1 and np.isfinite(values).all()
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def save(fig, name):
    fig.savefig(FIGS / (name + ".pdf"), bbox_inches="tight", pad_inches=0.06,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def band(ax, x, values, *, label=None, color=None, linestyle="-"):
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / math.sqrt(len(values))
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, lw=1.8)
    ax.fill_between(x, mean-se, mean+se, color=color, alpha=.18, linewidth=0)


summary = {"snapshot": {"status": "complete", "target_training_seeds_per_condition": 25}}
curves = data.mspm_curves()
seeds = data.csv_rows("table_mspm_welfare")
assert len(seeds) == 125 and {(r["num_types"], r["seed"]) for r in seeds} == {
    (t, s) for t in range(2, 7) for s in range(96, 121)}
assert len(curves) == 125 * 51
assert all(r["bcce_certified"] == 1 and r["bcce_gap"] <= .05+1e-12 for r in curves)
write_csv(OUT / "work/mspm_curves.csv", curves)
write_csv(OUT / "work/mspm_seeds.csv", seeds)
x, values = mspm_plot.load_data(OUT / "work/mspm_curves.csv", OUT / "work/mspm_seeds.csv",
                               cohort=mspm_plot.COHORTS["uniform_std05"])
style.configure_mspm_style()
plt.rcParams.update({"axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 8.5})
fig, ax = plt.subplots(figsize=(3.9, 2.65))
ax.axhline(.380, color=style.METHOD_COLORS["SPM"], ls=":", lw=1.5, label="Exact optimal SPM")
ax.axhline(.440, color=".25", ls="--", lw=1.2, label="Ex post optimum")
band(ax, x / 1000, values + .44, color=style.METHOD_COLORS["MSPM"], label="MSPM mean")
ax.set(xlabel="Training episodes (thousands)", ylabel="Expected welfare", xlim=(0, 10), ylim=(.35, .45))
h, labels = ax.get_legend_handles_labels()
ax.legend([h[i] for i in [2, 0, 1]], [labels[i] for i in [2, 0, 1]], loc="lower right",
          frameon=True, facecolor="white", edgecolor="none", framealpha=.95)
fig.tight_layout()
save(fig, "pi_spm_vs_mspm_exact_eval_allseeds")
table = []
for types in range(2, 7):
    group = [row for row in seeds if row["num_types"] == types]
    best = [row["best_greedy"] for row in group]
    final = [row["final_separate_greedy"] for row in group]
    bm, bs = stats(best)
    fm, fs = stats(final)
    optimum = ex_post_optimum(types)
    table.append(dict(types=types, seeds=25, ex_post_welfare=optimum,
                      optimal_spm_welfare=optimum+spm_optimum(types),
                      best_welfare_mean=optimum+bm, best_welfare_se=bs,
                      final_welfare_mean=optimum+fm, final_welfare_se=fs,
                      best_exactly_efficient=sum(abs(x) <= 1e-8 for x in best),
                      final_exactly_efficient=sum(abs(x) <= 1e-8 for x in final)))
write_csv(TABLES / "mspm_welfare.csv", table)
summary["mspm"] = table


# Collusion: final policy and current-policy validation, never archive-best.
learning, benchmarks = data.platform_learning()
results = data.platform_results()
chosen_seed, prices, admissions, outcomes = data.platform_policy()
confirmations = [results["PPO", seed] for seed in range(801, 826)]
histories = [[row["consumer_surplus"] for row in learning[seed]] for seed in range(801, 826)]
validation = [(learning[seed][-1]["consumer_surplus"], seed) for seed in range(801, 826)]
protocol = {"confirmation_seeds": sorted({r["seed"] for r in confirmations[0]["responses"]})}
assert chosen_seed == min(validation, key=lambda pair: (-pair[0], pair[1]))[1]


def cost_scores(result):
    cells = result["responses"]
    assert {(x["cost"], x["seed"]) for x in cells} == {
        (c, s) for c in [1., 1.5] for s in protocol["confirmation_seeds"]}
    scores = [float(np.mean([x["frozen_cs"] for x in cells if x["cost"] == c])) for c in [1., 1.5]]
    assert np.isclose(np.mean(scores), result["cs"], atol=1e-12)
    return scores + [float(np.mean(scores))]


fixed = results["Fixed cap", ""]
clipped = results["Clipped rule", ""]
search = load_search()
validation_benchmarks = {"fixed": benchmarks["Fixed cap"], "clipped": benchmarks["Clipped rule"]}
no_intervention = benchmarks["No intervention"]
search_validation = benchmarks["Random search"]
chosen = results["PPO", chosen_seed]
assert admissions.shape == (15, 15, 2)
assert np.array_equal(admissions, admissions.transpose(1, 0, 2)[:, :, ::-1])
i, j = np.indices((15, 15))
counts = admissions.sum(axis=2)
assert np.all((counts != 1) | ((i < j) & admissions[:, :, 0]) | ((j < i) & admissions[:, :, 1]))
caps = np.where(counts == 2, np.maximum(i, j), np.where(counts == 1, np.minimum(i, j), np.minimum(i, j)-1))
assert np.array_equal(np.stack([i <= caps, j <= caps], axis=-1), admissions)
for cost in [1., 1.5]:
    pairs = {tuple(tuple(pair) for pair in x["cycle_actions"]) for x in chosen["responses"] if x["cost"] == cost}
    assert len(pairs) == 1 and len(next(iter(pairs))) == 1
    assert outcomes[str(cost)]["price_indices"] == list(next(iter(pairs))[0])
style.configure_paper_style()
fig = plt.figure(figsize=(3.25, 3.9))
ax = fig.add_axes([.21, .29, .756, .63])
platform_plot.draw_policy(ax, prices, caps, outcomes)
platform_plot.save_panel(fig, FIGS / "hidden_cost_policy")
fig = plt.figure(figsize=(4.7, 3.9))
ax = fig.add_axes([.155, .29, .78, .63])
platform_plot.draw_learning(ax, np.arange(400, 20001, 400), np.asarray(histories),
    {"validation_fixed_cap_cs": validation_benchmarks["fixed"], "validation_clipped_cs": validation_benchmarks["clipped"]},
    {"no_intervention": {"mean": no_intervention}}, {"validation_cs": search_validation})
platform_plot.save_panel(fig, FIGS / "hidden_cost_learning")
ppo_scores = np.asarray([cost_scores(x) for x in confirmations])
table = []
for name, scores in [("Fixed threshold", cost_scores(fixed)), ("Clipped minimum", cost_scores(clipped)),
                     ("Random search (one run)", [search["per_cost"]["1.0"], search["per_cost"]["1.5"], search["cs"]])]:
    table.append(dict(method=name, leader_policies=1, seller_seeds_per_cost=25,
                      low_cost=scores[0], low_cost_se="", high_cost=scores[1], high_cost_se="",
                      equal_cost_average=scores[2], equal_cost_se=""))
means, ses = ppo_scores.mean(0), ppo_scores.std(0, ddof=1)/5
medians = np.median(ppo_scores, axis=0)
table.append(dict(method="PPO mean", leader_policies=25, seller_seeds_per_cost=25,
                  low_cost=means[0], low_cost_se=ses[0], high_cost=means[1], high_cost_se=ses[1],
                  equal_cost_average=means[2], equal_cost_se=ses[2]))
table.append(dict(method="PPO median", leader_policies=25, seller_seeds_per_cost=25,
                  low_cost=medians[0], low_cost_se="", high_cost=medians[1], high_cost_se="",
                  equal_cost_average=medians[2], equal_cost_se=""))
write_csv(TABLES / "collusion_consumer_surplus.csv", table)
summary["platform"] = dict(table=table, illustration_seed=chosen_seed, marked_outcomes=outcomes,
    above_fixed=int(sum(ppo_scores[:, 2] > fixed["cs"]+1e-12)),
    above_clipped=int(sum(ppo_scores[:, 2] > clipped["cs"]+1e-12)))


# Frozen Atari responses: recompute uncertainty from the 20 individual episodes.
response_rows = []
observations = data.csv_rows("atari_e1_responses")
assert len(observations) == 440
for role in ["buyer", "seller"]:
    contexts = list(dict.fromkeys(r["opponent_value"] for r in observations if r["controlled_role"] == role))
    assert len(contexts) == 11
    for context in contexts:
        episodes = [r for r in observations if r["controlled_role"] == role and r["opponent_value"] == context]
        assert len(episodes) == 20 and {r["evaluation_seed"] for r in episodes} == set(range(3700001, 3700021))
        row = dict(controlled_role=role, opponent_value=context, n=20)
        for key in ("transfers", "controlled_shots", "quoted_price", "controlled_payoff"):
            row[key+"_mean"], row[key+"_sem"] = stats([r[key] for r in episodes])
        response_rows.append(row)
response_data = pd.DataFrame(response_rows)
save(atari.make_response_figure(response_data), "atari_e1_responses")
write_csv(TABLES / "atari_frozen_responses.csv", response_rows)


# Each row is one training seed's prespecified endpoint evaluation.
expected_seeds = {role: list(range(1, 26)) for role in ("buyer", "seller")}
policy_rows, fresh_rows = [], []
included = {"buyer": [], "seller": []}
for row in data.atari_curves():
    role, seed = row["leader_role"], row["seed"]
    if seed not in included[role]:
        included[role].append(seed)
    for metric in atari.EXPECTED_METRICS:
        policy_rows.append(dict(leader_role=role, seed=seed,
            training_total_timesteps=row["training_total_timesteps"], metric=metric, value=row[metric]))
for row in data.atari_fresh():
    for metric in atari.EXPECTED_METRICS:
        fresh_rows.append(dict(leader_role=row["leader_role"], seed=row["seed"], metric=metric, value=row[metric]))
leader_data = pd.DataFrame(policy_rows)
leader_summary = []
for (role, step, metric), group in leader_data.groupby(["leader_role", "training_total_timesteps", "metric"]):
    mean, se = stats(group["value"])
    assert len(group) == len(included[role])
    leader_summary.append(dict(metric=metric, leader_role=role, training_total_timesteps=int(step),
                               mean_across_policy_means=mean, sem_across_policy_means=se, n_policies=len(group)))
counts = {role: len(seeds) for role, seeds in included.items()}
assert all(sorted(included[role]) == sorted(expected_seeds[role]) for role in included)
for role in counts:
    atari.ROLE_LABELS[role] = role.title()+f" leader (n={counts[role]})"
fig = atari.make_leader_figure(pd.DataFrame(leader_summary))
save(fig, "atari_meta_stackpomdp")
write_csv(TABLES / "atari_leader_curves.csv", leader_summary)
fresh_summary = []
for (role, metric), group in pd.DataFrame(fresh_rows).groupby(["leader_role", "metric"]):
    assert sorted(group["seed"].tolist()) == sorted(expected_seeds[role])
    mean, se = stats(group["value"])
    fresh_summary.append(dict(metric=metric, leader_role=role, n_policies=counts[role],
                              episodes_per_policy=100, mean=mean, se=se))
write_csv(TABLES / "atari_fresh_final.csv", fresh_summary)
summary["atari_leaders"] = dict(included_seeds={k: sorted(v) for k, v in included.items()},
                                 final_rows=[x for x in leader_summary if x["training_total_timesteps"] == 2000040],
                                 fresh_final=fresh_summary, counts=counts, target_per_role=25)


# Scalar response credit: validate recorded updates, do not rerun simulation.
config = scalar.CONFIG
scalar_records = data.csv_rows("response_credit_ablation")
values = scalar.validate(scalar_records, config)
style.configure_paper_style()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.65))
p = np.linspace(0, 1, 301)
for index, method in enumerate(scalar.METHODS):
    color = style.DEEP_PALETTE[index]
    ls = "-" if index == 0 else "--"
    axes[0].plot(p, [scalar.exact_moments(float(x), method)[1] for x in p],
                 color=color, ls=ls, label=scalar.LABELS[index])
    band(axes[1], np.arange(1001)*.128, values[:, index, :], label=scalar.LABELS[index], color=color, linestyle=ls)
axes[0].axhline(0, color=".35", lw=.8)
axes[0].set(xlabel=r"Leader probability $p$", ylabel=r"Expected gradient in $z$",
            title="(a) Exact expected gradients", xlim=(0, 1), ylim=(-.28, .28))
axes[1].set(xlabel="Training transitions (thousands)", ylabel=r"Exact leader value $J(p)$",
            title="(b) Learning from sampled episodes", xlim=(0, 128), ylim=(0, 1))
fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, -.01), ncol=2)
fig.tight_layout(rect=(0, .1, 1, 1))
save(fig, "response_credit_ablation")
summary["response_credit"] = scalar.summarize(values, config)["primary"]
ablation_rows = []
for method in scalar.METHODS:
    mean, se = stats(values[:, scalar.METHODS.index(method), -1])
    ablation_rows.append(dict(experiment="Response credit", condition=method, learning_rate=.1,
                              seeds=25, mean=mean, se=se, metric="exact leader value"))
mean, se = stats(values[:, 0, -1] - values[:, 1, -1])
ablation_rows.append(dict(experiment="Response credit", condition="paired_full_minus_gameplay",
                          learning_rate=.1, seeds=25, mean=mean, se=se, metric="paired value difference"))


matrix_runs = data.matrix_runs


memory = matrix_runs("commitment_consistency", 150)
style.configure_paper_style()
plt.rcParams.update({"axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10})
fig, ax = plt.subplots(figsize=(5.6, 3.45))
styles = {0.004: "-", .008: (0, (4, 2)), .015: (0, (1, 1.4)), .03: (0, (5, 1.8, 1.2, 1.8))}
for condition in ["visible", "hidden"]:
    for rate in ([.004, .008, .015, .03] if condition == "visible" else [.008, .015]):
        group = [x for x in memory if x[0]["condition"] == condition and x[0]["learning_rate"] == rate]
        assert len(group) == 25 and {x[0]["seed"] for x in group} == set(range(25))
        steps = [r.get("policy_updates", 0) for r in group[0][1]]
        assert len(steps) == 101
        assert all([r.get("policy_updates", 0) for r in run[1]] == steps for run in group)
        band(ax, steps, [[r["evaluation_mean"] for r in run[1]] for run in group],
             color=style.DEEP_PALETTE[0 if condition == "visible" else 1], linestyle=styles[rate],
             label=f"Phase {condition}, lr={rate:g}")
        final = [x[2]["per_stage_summary"]["mean"] for x in group]
        mean, se = stats(final)
        ablation_rows.append(dict(experiment="Commitment consistency", condition=condition,
                                  learning_rate=rate, seeds=25, mean=mean, se=se, metric="centered per-stage reward"))
ax.set(xlabel="Policy-gradient updates", ylabel="Leader reward per stage", xlim=(0, steps[-1]))
fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
           bbox_to_anchor=(.5, .005), ncol=2, fontsize=9.5, columnspacing=1.0,
           handlelength=2.4, labelspacing=0.35)
fig.tight_layout(rect=(0, .20, 1, 1))
save(fig, "fig_memory_pg")

timing = matrix_runs("reward_timing", 100)
timing_successes = {}
style.configure_paper_style()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, matrix, title in zip(axes, ["coordination_zero_miscoordination", "coordination_penalized_miscoordination"],
                             ["No coordination penalty", "Coordination penalty (-5)"]):
    for condition, color in zip(["excluded", "included"], style.DEEP_PALETTE):
        group = [x for x in timing if x[0]["matrix"] == matrix and x[0]["condition"] == condition]
        assert len(group) == 25 and {x[0]["seed"] for x in group} == set(range(25))
        steps = [r["global_step"] for r in group[0][1]]
        assert all([r["global_step"] for r in run[1]] == steps for run in group)
        band(ax, np.array(steps)/1000, [[r["evaluation_mean"] for r in run[1]] for run in group],
             color=color, label="Response rewards "+condition)
        mean, se = stats([x[2]["per_stage_summary"]["mean"] for x in group])
        timing_successes[str((matrix, condition))] = sum(
            abs(x[2]["per_stage_summary"]["mean"] - 2.0) < 1e-8 for x in group)
        ablation_rows.append(dict(experiment="Reward timing: "+title, condition=condition,
                                  learning_rate=.1, seeds=25, mean=mean, se=se, metric="final-game reward"))
    ax.set(title=title, xlabel="Training transitions (thousands)", xlim=(0, 22))
axes[0].set_ylabel("Leader reward in final game")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2)
fig.tight_layout(rect=(0, .12, 1, 1))
save(fig, "fig_bots_leaderreward")
write_csv(TABLES / "ablation_endpoints.csv", ablation_rows)
summary["ablations"] = ablation_rows
summary["reward_timing_successes"] = timing_successes
# The appendix uses all 100 runs for three to six buyer types.
multitype = mspm_plot.load_multitype_runs(DATA / "mspm_multitype_training.csv")
save(mspm_plot.make_multitype_figure(multitype), "mspm_multitype_training")

(OUT / "work/summary.json").write_text(json.dumps(summary, indent=2)+"\n")
(OUT / "figure_inputs.json").write_text(json.dumps({os.path.relpath(p, LOGS): hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in sorted(INPUTS)}, indent=2) + "\n")
print(json.dumps(summary, indent=2))
