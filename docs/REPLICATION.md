# Replication Guide

This repository is the canonical Stable-Baselines3 implementation for the
paper experiments that use tabular follower oracles and centralized critics.

## Scope

Covered here:

- Simple allocation mechanisms.
- Message sequential price mechanisms (MSPMs).
- Maintain/randomized-strategy normal-form experiment.
- Matrix design experiment.
- Bertrand price-collusion and platform-intervention experiments.

Companion code still needed unless ported:

- Hidden-query, phase-memory, reward-during-learning, and continuous-follower
  ablations that use neural/alternating follower training.
- Atari Space Invaders bilateral trade experiments.

Those currently belong to `Code/StackeRLberg`.

## Main Entrypoint

For named paper targets, prefer the replication runner:

```bash
python replication/run.py --list
python replication/run.py platform_intervention_state --seed 1 --dry-run
```

For direct ad hoc runs, use:

```bash
python -m stackelberg_pomdp.main_args --experiment_type <experiment>
```

The legacy form also works from inside `stackelberg_pomdp/`:

```bash
python main_args.py --experiment_type <experiment>
```

## Paper Experiment Families

### Simple Allocation

Figure: `SA-Ablation_new`.

Canonical command shape:

```bash
python -m stackelberg_pomdp.main_args \
  --experiment_type simple_allocation:3 \
  --tot_num_reward_episodes 30 \
  --followers_algorithm MW \
  --critic_obs full \
  --algorithm PPO
```

### MSPM

Table: MSPM welfare loss and optimal-found rates.

Canonical command shape:

```bash
python -m stackelberg_pomdp.main_args \
  --learning_method RL:StopOnThreshold \
  --experiment_type mspm:MSGSpace:5:2 \
  --tot_num_eq_episodes 1000 \
  --tot_num_reward_episodes 100 \
  --followers_algorithm MW \
  --critic_obs full \
  --algorithm PPO
```

### Normal Form / Maintain

Figure: `simpleMatrixGame2_new`.

Canonical command shape:

```bash
python -m stackelberg_pomdp.main_args \
  --experiment_type normal_form:game_2:True \
  --followers_algorithm MW \
  --critic_obs full \
  --algorithm PPO
```

### Matrix Design

Figure: `MatrixDesign-Ablation_new`.

Canonical command shape:

```bash
python -m stackelberg_pomdp.main_args \
  --experiment_type matrix_design \
  --followers_algorithm MW \
  --critic_obs full \
  --algorithm PPO
```

### Bertrand Collusion Calibration

Figure: `deviation_m4`; calibration for the platform-intervention experiments.

Canonical command shape:

```bash
python -m stackelberg_pomdp.calvano_replication \
  --m 4 \
  --alpha 0.25 \
  --beta 1e-4 \
  --n_sessions 5 \
  --output_dir results/calvano_m4
```

### Bertrand Platform Intervention

Figure: `collusion_learning_25seeds`.

Canonical command shape:

```bash
python -m stackelberg_pomdp.main_args \
  --experiment_type bertrand \
  --platform_intervention learn_threshold \
  --platform_observation_space price_profile \
  --price_grid_length 4 \
  --price_min 1.05 \
  --price_max 1.7 \
  --tot_num_eq_episodes 50000 \
  --tot_num_reward_episodes 30 \
  --algorithm A2C \
  --max_steps 50000000 \
  --critic_obs full \
  --fix_episode_actions true \
  --followers_algorithm Qlearning \
  --seed 1 \
  --learning_method RL:Standard \
  --response_phase_prob 1.0 \
  --follower_alpha 0.25 \
  --follower_beta 1e-4
```

Use `--platform_observation_space no_observation` for the no-state learned
policy.

## Consolidation Target

The current collection of top-level `run_*.sh` scripts should be replaced by a
small set of named replication targets:

- `collusion_calibration_m4`
- `collusion_deviation_m4`
- `platform_intervention_state`
- `platform_intervention_no_state`
- `simple_allocation_ablation`
- `mspm_table`
- `normal_form_randomized`
- `matrix_design_ablation`

Each target should define seeds, hyperparameters, output directory, expected
figure/table, and the exact command used locally or on SLURM.

The Bertrand/Calvano targets have started moving into
`replication/bertrand_targets.json`.
