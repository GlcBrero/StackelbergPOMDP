# Normal-form appendix experiments

All three appendix diagnostics have maintained training and plotting support.

Their PG/SimpleQ recipes are explicit algorithm-specific exceptions in
[PARAMETERS.md](../PARAMETERS.md). Optional PPO/A2C/DQN alternatives use SB3
optimizer defaults; their outputs are separate from the paper's PG/SimpleQ
cohorts.

| Figure | Comparison | Leader / follower | Full cohort |
|---|---|---|---|
| `fig_memory_pg` | Phase indicator visible versus unavailable in iterated Prisoner's Dilemma | Linear PG / pretrained REINFORCE | 60 leaders + 10 shared follower models |
| `fig_reset` | Fresh versus carried Q-table in Battle of the Sexes | Linear PG / tabular Q | 18 leaders: 3 learning rates × 2 conditions × 3 seeds |
| `fig_bots_leaderreward` | Include versus exclude adaptation rewards, with coordination penalty 0 or −5 | Linear SimpleQ / tabular Q | 40 leaders: 4 conditions × 10 seeds |

The phase figure uses visible-phase learning rates 0.004, 0.008, 0.015, and
0.03, with complete historical controls at 0.008 and 0.015. Reset uses 0.008,
0.015, and 0.03; SimpleQ uses 0.1. The full plan contains **128 runs**.

The maintained implementations use the current paper's five-state repeated
game, correct terminal Q updates, and fixed actions within each outer episode.
They need scientific reruns before replacing archived figures. Remaining
differences from historical training are explicit in
[LEGACY_PROVENANCE.md](LEGACY_PROVENANCE.md). The published curves still
regenerate from retained data using the journal artifact's
`reproducibility/scripts/plot_theory_diagnostics.py`.

## Run one treatment

Commitment consistency first needs a frozen follower. This command prints its
run directory and saves `model.zip`, a response contract, and an exhaustive
response-quality assessment:

```bash
python -m stackelberg_pomdp.experiments.matrix_ablations meta-follower \
  --algorithm REINFORCE --matrix prisoners_dilemma --memory-mode joint --seed 1
```

Use its checkpoint for either leader condition:

```bash
python -m stackelberg_pomdp.experiments.matrix_ablations leader \
  --figure fig_memory_pg --condition visible \
  --response-checkpoint /path/to/model.zip --seed 1
```

Change `visible` to `hidden` for the control. Both execute and store all
response queries; `hidden` here refers only to the **phase indicator**.
Both use PG and share the same follower checkpoint within a seed. A mismatched
contract or response above the regret threshold is rejected.
`--allow-uncertified-response` is available for tiny integration checks; the
full replication plan does not bypass the response-quality gate.

The other two diagnostics need no pretrained checkpoint:

```bash
python -m stackelberg_pomdp.experiments.matrix_ablations leader \
  --figure fig_reset --condition reset --seed 1
python -m stackelberg_pomdp.experiments.matrix_ablations leader \
  --figure fig_bots_leaderreward --condition included --seed 1
```

The paired conditions are `ongoing` and `excluded`. Add
`--matrix coordination_penalized_miscoordination` for the −5 penalty panel.
Default leader budgets are 200,000 transitions for commitment consistency,
55,000 for reset, and 22,000 for reward timing. Complete-episode collection can
exceed a requested limit; logs record actual steps and optimizer updates.
Runs save resolved settings, source hashes, `model.zip`, held-out evaluation,
and `progress.jsonl`. Existing artifacts cannot be overwritten.

PG shares the follower's REINFORCE loss, without a baseline or entropy bonus.
SimpleQ uses uniform replay, a target network, Huber loss, and Gaussian parameter
noise fixed for an outer episode. Both tabular followers use parameter noise.
A2C/PPO remain explicit `--algorithm` alternatives, labeled separately in plots.

Evaluation always reports reward **after** adaptation. Each carried-state
measurement copies the training follower's current Q-table, preserving its
history without modifying the training table. PG/SimpleQ curves are measured
after completed optimizer updates.
The final `evaluation.json` retains `response_q_values`; pass that value to
`evaluate_leader_policy(..., response_q_values=...)` when reevaluating a carried-Q
leader loaded without its training environment.

## Plan and plot the cohort

```bash
python replication/matrix_ablations/sweep.py plan --sweep-id appendix-v1
python replication/matrix_ablations/sweep.py status --sweep-id appendix-v1
```

Planning executes no training. Use `plan --figure fig_reset` for one figure;
`--seeds` overrides the paper counts for a smaller check. The cluster scripts
use the full plan: 10 follower tasks, 58 independent leaders, and 60 phase
leaders. Run the E1 gate before dependent leaders. Each array task executes one
plan record; retries receive separate immutable directories.

The public `replication/run.py` targets also expand the named curves:
`fig_memory_phase_ablation`, `fig_continuous_follower_learning_ablation`, and
`fig_reward_during_learning_ablation`. Use `--dry-run` to inspect commands;
the first requires `--response-checkpoint` as above.

```bash
python replication/matrix_ablations/plot.py \
  --input replication/matrix_ablations/results/appendix-v1 --figure all
```

The plotter reads each curve's required seeds from the plan, keeps learning
rates separate, checks hashes and complete treatment coverage, and writes
PDFs, PNGs, and summary CSVs. Bands are sample standard errors across independent
training seeds. `trend.py` pairs treatment differences within a learning rate
and counts retries as attempts, not additional seeds.

## Implementation map

`stackelberg_pomdp/experiments/matrix_ablations.py` is the CLI. Under
`stackelberg_pomdp/matrix_ablations/`:

- `presets.py`: treatments, learning rates, and default budgets.
- `meta_training.py`: follower training and certification.
- `leader_training.py`: treatment construction and leader training.
- `reinforce.py`, `simple_q.py`: the small learner specializations.
- `evaluation.py`: held-out reward-play evaluation.
- `artifacts.py`: manifests, hashes, and run files.
- `profiles.py`: explicit state and reward conventions.

Game and response environments live in `stackelberg_pomdp/envs/matrix.py`.
Matrix Design is separate: [../normal_form/README.md](../normal_form/README.md).
