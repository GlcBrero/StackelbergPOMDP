# StackelbergPOMDP Cleanup Roadmap

Goal: make this repository the main, easy-to-run replication code for the paper
experiments it owns, with no unused source or script clutter.

## Current Diagnosis

- The core package is small, but `custom_envs.py` is too large and mixes base
  games, follower wrappers, StackPOMDP phase logic, Bertrand economics, reward
  wrappers, and logging.
- The active experiment interface is `replication/run.py` and
  `replication/targets.json`; historical top-level `run_*.sh` scripts have been
  removed from the public tree and remain recoverable from git history.
- Several script/config parameters had drifted from the CLI. The CLI now
  exposes `price_min`, `price_max`, `leader_k`, `sort_obs`, `ent_coef`, and PPO
  batch controls; `grid_lower_bound` and `grid_upper_bound` are accepted as
  legacy aliases.
- `environment.yml` is now the clean local environment for this SB3/tabular
  package. Ray/TensorFlow/Atari dependencies belong in companion-code
  environments unless a local import proves they are required here.
- Generated artifacts (`logs/`, `__pycache__/`, model checkpoints) should stay
  out of version control.

## Keep

- `main_args.py`, `run_setups.py`, `env_setups.py`, `rl_trainer_setup.py`,
  `baselines_utils.py`, and `callbacks.py`.
- `games.py`.
- `leader_policies.py` and verifier utilities while MSPM `StopOnThreshold`
  remains supported.
- `tools/policy_enumeration.py` as a developer utility, not as a paper
  replication target.
- `replication/bertrand/calibrate_price_learners.py`,
  `replication/bertrand/punishment_diagnostic.py`, and
  `replication/bertrand/evaluate_intervention.py` as paper-replication tools.
- `replication/bertrand/price_collusion.py` as the shared home for
  price-collusion constants and metrics used by those tools.

## Restructure

The first split is in place:

- `gym_envs/envs/base_envs.py`: `BaseEnv`, mechanism-design base envs, and
  `BertrandCompetitionEnv`.
- `gym_envs/envs/atari_envs.py`: Atari bullet-pricing envs and Atari buyer-env
  construction.
- `gym_envs/envs/wrappers.py`: follower wrappers, `StackPOMDPWrapper`,
  leader-observation, reward, openness, logging, and stopping wrappers.
- `gym_envs/envs/custom_envs.py`: compatibility re-exports for legacy scripts.

If the files keep growing, split one level further into:

- `envs/mechanism_design.py`
- `envs/price_collusion.py`
- `wrappers/followers.py`
- `wrappers/stack_pomdp.py`
- `wrappers/rewards.py`
- `wrappers/logging.py`

## Replace Script Sprawl

Historical top-level `run_*.sh` scripts have been removed. The replacement
interface is:

- `replication/targets.json`: one manifest of named paper targets.
- `replication/run.py`: local runner for one target/seed.
- `replication/slurm.py`: emits or submits SLURM arrays from the same manifest.
- `replication/README.md`: maps paper figures/tables to target names.

The first manifest is in place as `replication/targets.json` with a stdlib-only
local runner in `replication/run.py`. Future SLURM support should emit jobs from
that same manifest instead of reintroducing independent shell-script sprawl.

## Dependency Cleanup

Maintain a minimal SB3 environment file for this package. Direct runtime
dependencies are:

- `stable-baselines3`
- `gym`
- `numpy`
- `torch`

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
