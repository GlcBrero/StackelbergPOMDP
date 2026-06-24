# StackelbergPOMDP Cleanup Roadmap

Goal: make this repository the main, easy-to-run replication code for the paper
experiments it owns, with no unused source or script clutter.

## Current Diagnosis

- The core package is small, but `custom_envs.py` is too large and mixes base
  games, follower wrappers, StackPOMDP phase logic, Bertrand economics, reward
  wrappers, and logging.
- The active experiment interface is fragmented across `main_args.py`,
  `calvano_replication.py`, `calvano_deviation.py`, `eval_intervention.py`,
  `experiments/*.py`, and many overlapping top-level `run_*.sh` scripts.
- Several script/config parameters had drifted from the CLI. The CLI now
  exposes `price_min`, `price_max`, `leader_k`, `sort_obs`, `ent_coef`, and PPO
  batch controls; `grid_lower_bound` and `grid_upper_bound` are accepted as
  legacy aliases.
- `requirements.txt` still reflects the broader historical stack, including
  Ray/TensorFlow/Atari dependencies that are not required for this SB3/tabular
  package.
- Generated artifacts (`logs/`, `__pycache__/`, model checkpoints) should stay
  out of version control.

## Keep

- `main_args.py`, `run_setups.py`, `env_setups.py`, `rl_trainer_setup.py`,
  `baselines_utils.py`, and `callbacks.py`.
- `games.py`.
- `leader_policies.py`, `policy_enumeration.py`, and the verifier utilities
  while MSPM `StopOnThreshold` and policy enumeration remain supported.
- `calvano_replication.py`, `calvano_deviation.py`, and `eval_intervention.py`
  as paper-replication tools, but they should move under a clearer
  `scripts/` or `replication/` namespace.
- `calvano.py` as the shared home for Calvano constants and metrics used by
  those tools.

## Restructure

Split `stackelberg_pomdp/gym_envs/envs/custom_envs.py` into:

- `envs/base.py`: `BaseEnv`.
- `envs/mechanism_design.py`: normal-form, matrix-design, simple-allocation,
  and MSPM base environments.
- `envs/bertrand.py`: `BertrandCompetitionEnv`.
- `calvano.py`: constants and diagnostic metrics for Calvano-style
  Q-learning seller calibration.
- `followers/mw.py`: `MWFollowersWrapper`.
- `followers/q_learning.py`: `QLearningFollowersWrapper`.
- `followers/round_robin.py`: `RoundRobinFollowersWrapper`.
- `wrappers/stack_pomdp.py`: `StackPOMDPWrapper`.
- `wrappers/leader_observation.py`: `ReactiveLeaderWrapper`.
- `wrappers/rewards.py`: `StationaryCycleRewardWrapper`,
  `OpennessEvaluationWrapper`, `StopOnThresholdWrapper`.
- `wrappers/logging.py`: `LoggingWrapper`.

Do this with compatibility re-exports first so existing scripts keep working.

## Replace Script Sprawl

Replace top-level `run_*.sh` scripts with:

- `replication/experiments.yaml`: one manifest of named paper targets.
- `replication/run.py`: local runner for one target/seed.
- `replication/slurm.py`: emits or submits SLURM arrays from the same manifest.
- `replication/README.md`: maps paper figures/tables to target names.

The first step is in place as `replication/bertrand_targets.json` with a
stdlib-only local runner in `replication/run.py`.

Until that exists, treat the current `run_*.sh` files as historical working
scripts, not the public replication interface.

## Dependency Cleanup

Create a minimal SB3 environment file for this package. Candidate direct
runtime dependencies:

- `stable-baselines3`
- `gym`
- `numpy`
- `torch`
- `pandas` or `matplotlib` only if plotting/result aggregation remains in this
  repo.
- `wandb` as an optional extra.

Move Ray, TensorFlow, Atari, OpenCV, Redis, and dashboard dependencies to the
`StackeRLberg`/Atari companion environment unless a local import proves they
are still required.

## Tests / Smoke Checks

Add fast tests for:

- Environment construction for every `experiment_type`.
- One complete StackPOMDP episode for MW, Q-learning, and RoundRobin followers.
- Action caching: same `base_environment` observation reuses the same action
  within an episode and resets across episodes.
- Critic observation shape for `critic_obs=full`.
- Bertrand buy-box rules: `no_intervene`, `pdp`, `dpdp`, `learn_threshold`,
  `learn_binary_threshold`, `block_equal`.

Keep full 50M-step replication runs out of tests; they belong in the
replication manifest.
