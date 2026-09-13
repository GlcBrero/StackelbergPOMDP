# Training parameters: defaults and justified exceptions

New runs use the selected algorithm's Stable-Baselines3 defaults unless the
economic objective, the commitment protocol, or an explicitly retained paper
experiment requires an exception. `stackelberg_pomdp/training_defaults.py`
reads constructor defaults from the installed library. Use the pinned
`environment.yml` for the publication environment.

| Ordinary optimizer setting | PPO | A2C |
|---|---:|---:|
| Learning rate | 0.0003 | 0.0007 |
| Minibatch size | 64 | One rollout; no PPO-style minibatches |
| Optimization epochs | 10 | One update per rollout |
| GAE lambda | 0.95 | 1 |
| PPO clipping | 0.2 | Not applicable |
| Value coefficient / gradient cap | 0.5 / 0.5 | 0.5 / 0.5 |

These agree with the pinned [SB3 PPO](https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/ppo/ppo.py)
and [A2C](https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/a2c/a2c.py)
constructors. Unexposed optimizer settings are left to SB3. The shared
64-unit actor/critic networks match SB3's MLP widths; the actor/critic
observation split is a defining experimental treatment.

## Exceptions used by the maintained experiments

| Scope | Exception | Reason |
|---|---|---|
| Economic leader and response learners | `gamma=1` | Optimize the undiscounted episode objective in the paper. |
| Generic PPO | Round SB3's 2,048-step rollout budget up to whole maximum-length episode budgets | Keep commitment episodes together without the old response-ratio multiplier. Minibatches remain 64 independently of episode length. |
| Generic A2C | One maximum-length episode per rollout | Include the delayed reward phase in each update, especially the 50,030-step Bertrand episode. |
| Cached generic leaders | `ent_coef=0.01` | Preserve exploration over deterministic episode commitments. This is a retained empirical exception, not a mathematical requirement or a claim that every domain needs it. |
| MSPM | One maximum-length episode per PPO rollout; entropy 0 | Retain the certified-response benchmark's update cadence. Its other optimizer settings are SB3 defaults. |
| Standard SPM paper target | Entropy 0.01 | Retain the referenced learning comparator's exploration setting. The ordinary SPM CLI uses SB3's entropy default of 0. |
| Atari E0/E1/E2 | Named `ATARI_PAPER_PPO` settings and stage-specific transfer rates | Reproduce the released Atari training/checkpoint protocol; see below. |
| Normal-form appendix PG/SimpleQ | Linear policies and the recorded RLlib-derived optimizer/exploration recipes | These implement the named appendix experiments, including learning rate as an explicit plotted treatment. They are different algorithms from PPO/A2C. |
| MW and tabular Q followers | Domain response rates, reset rules, exploration and certification thresholds | These define the follower-response experiment; there is no universal SB3 default for them. They are explicit in target arguments or appendix protocol records. |

For variable-length episodes, the generic rollout size is a multiple of the
maximum episode budget and guarantees a minimum number of completed episodes;
it does not promise that every rollout ends on a terminal state. Additional
certification queries are excluded from the learner buffer by the existing
response protocol. Fixed-length Simple Allocation/Matrix Design rollouts
retain the corresponding complete stored episode blocks.

`--ppo_episodes_per_batch` explicitly overrides the automatic episode count.
The compatibility option `historical_ratio_scaled` retains the old formula
only when explicitly requested; it does not recover missing historical run
settings. It no longer silently selects an episode-sized minibatch. A
September replay requires its explicit multiplier, minibatch, epoch count,
learning rate, and frozen source.

## Atari is an explicit paper-protocol exception

`stackelberg_pomdp/atari/training.py::ATARI_PAPER_PPO` supplies the shared four
epochs, entropy 0.01, clipping 0.1, gameplay rate 0.00025, and transfer-stage
rate 0.0001. The first three settings and the initial gameplay rate resemble
the [RL Zoo Atari recipe](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml).
The repository uses constant rates and clipping rather than Zoo's linear
schedules, and different rollout geometry. Do not call this the unmodified
SB3 or Zoo default.

Atari uses `gamma=gae_lambda=1` because payments are immediate while gameplay
benefits arrive later. One complete episode per environment and a full-rollout
minibatch support the phase-balanced objective: gameplay and economic rows
are averaged separately, so 200 gameplay decisions do not overwhelm five
trade decisions. Its implementation explicitly rejects smaller minibatches
in balanced mode.

The fine-tuning stages retain their checkpoint-tested module rates, including
the seller-v5 live rate 0.0005, its separate context/critic groups, and the
pretrained gameplay scale. Those choices are retained for artifact
reproduction; this audit does not prove that each is individually necessary
for convergence. Changing them requires a separately evaluated Atari recipe,
not relabeling the existing checkpoint cohort.

## Appendix and follower algorithms

`matrix_ablations/presets.py` records the PG/SimpleQ protocols and the exact
learning-rate curves. Their zero entropy, linear policies, REINFORCE return
calculation, SimpleQ replay/target updates, and parameter noise are deliberate
algorithm choices. See [appendix provenance](matrix_ablations/LEGACY_PROVENANCE.md).

Optional PPO/A2C/DQN appendix trainers now use their own SB3 learning rates,
default MLP widths, and optimizer/replay defaults. Their episodic objective
still uses gamma 1. PPO/A2C rollout sizes round up to complete episode budgets.
Short DQN smoke tests must explicitly reduce `--dqn-learning-starts`; the
publication default is no longer silently changed to zero.

Bertrand calibration has an explicit `(alpha, beta)` grid in the paper.
Intervention sellers use `(0.25, 0.0001)` and discount 0.95. These are treatment
parameters, not arbitrary leader optimizer overrides. MW uses the maintained
rate 0.1; the archived Simple Allocation weights identify 0.01. That historical
response-rate discrepancy remains to be reconciled independently of restoring
PPO defaults.

Simple Allocation and Matrix Design now use the same `--mw_response_cycles 33`
in every condition, including the one- and two-message capacity controls.
This is a shared experimental budget, not an optimizer default. MW always
rounds a raw follower-game budget down to whole joint-action sweeps. MSPM's
existing fixed prefixes and certification extensions are unchanged.

The planned matched comparison varies only MW epsilon (0.01 versus 0.1)
within each configuration and seed, holding these 33 updates and the current
optimizer recipe fixed. No qualitative equivalence is established yet, and
retained curves must not be relabeled as results under the new cycle budget.

## Preventing future silent overrides

Mechanism replication targets with a nondefault optimizer argument must have
a matching `parameter_exceptions` entry with the exact value and its reason.
`replication/run.py` checks this when expanding commands, including variants.
Explicit user CLI overrides remain available for diagnostics. A reason is
scientific documentation, not proof of necessity; record comparative evidence
before adopting a newly tuned paper recipe.

Generic training also writes `optimizer_parameters.json` with the actual SB3
version, rollout/minibatch sizes, epochs, learning rate, discount, GAE, clipping,
entropy, and loss coefficients. This records resolved model values rather
than relying on names such as "historical" or on parser defaults.

## Effect on existing results

The September 10 defaults cleanup changes future Simple Allocation and Matrix
Design training: learning rate 0.0007 to 0.0003, four epochs to ten, minibatches
to 64, and ratio-scaled rollouts to the automatic episode rule. It also removes
unjustified defaults from the optional appendix PPO/A2C/DQN trainers. The
published appendix PG/SimpleQ and Atari recipes retain their behavior.

Completed runs retain their original configurations and source hashes. They
must not be described as having used these new defaults. The earlier claim
that the 131,072/8,192 PPO geometry reproduced the original paper cohort was
incorrect: archived policy histories contradict that update cadence. New
full-cohort training and a matching methods/figure update are required before
claiming results for the new recipe. This cleanup submits no training jobs.
