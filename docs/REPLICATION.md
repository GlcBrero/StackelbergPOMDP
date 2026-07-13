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
python replication/run.py fig_collusion_learning_state --seed 1 --dry-run
```

For direct ad hoc runs, use:

```bash
python -m stackelberg_pomdp.experiments.simple_allocation --help
python -m stackelberg_pomdp.experiments.price_collusion --help
```

The legacy all-argument CLI remains available for debugging old commands:

```bash
python -m stackelberg_pomdp.main_args --experiment_type <experiment>
```

## Paper Experiment Families

### Simple Allocation

Figure: `SA-Ablation_new`.

Canonical command shape:

```bash
python -m stackelberg_pomdp.experiments.simple_allocation \
  --num_messages 3 \
  --seed 1
```

### MSPM

Table: MSPM welfare loss and optimal-found rates.

Canonical command shape:

```bash
python -m stackelberg_pomdp.experiments.mspm \
  --setting MSGSpace \
  --num_types 5 \
  --num_messages 2 \
  --seed 1
```

### Normal Form / Maintain

Figure: `simpleMatrixGame2_new`.

Canonical command shape:

```bash
python -m stackelberg_pomdp.experiments.normal_form \
  --game_name game_2 \
  --randomized true \
  --seed 1
```

### Matrix Design

Figure: `MatrixDesign-Ablation_new`.

Canonical command shape:

```bash
python -m stackelberg_pomdp.experiments.matrix_design --seed 1
```

### Bertrand Collusion Calibration

Figure: `deviation_m4`; calibration for the platform-intervention experiments.

Canonical command shape:

```bash
python -m replication.bertrand.calibrate_price_learners \
  --m 4 \
  --alpha 0.25 \
  --beta 1e-4 \
  --n_sessions 5 \
  --output_dir results/price_collusion_m4
```

### Bertrand Platform Intervention

Figure: `collusion_learning_25seeds`.

Canonical command shape:

```bash
python -m stackelberg_pomdp.experiments.price_collusion \
  --platform_observation_space price_profile \
  --price_grid_length 4 \
  --price_min 1.05 \
  --price_max 1.7 \
  --seed 1
```

Use `--platform_observation_space no_observation` for the no-state learned
policy.

## Current Target Coverage

`replication/targets.json` contains runnable targets for the experiments owned
by this codebase and explicit TODO entries for companion-code experiments such
as Atari and the theorem-violation ablations.
