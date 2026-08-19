# Atari bilateral-trade replication

This directory is the release pipeline for the Space Invaders experiment in
the paper.  It uses Stable-Baselines3 PPO throughout.  Historical RLlib, A3C,
stand-alone threshold-head, and intermediate seller-recovery launchers are not
part of this implementation.

## Scientific contract

Every stage uses the same composite observation and action interface:

- `image`: a four-frame `84 x 84` stack;
- `actor_state`: 14 scalars containing normalized ammunition, projectile
  activity, normalized time, trade mode, five-way event identity, and the
  opponent's five economic commitments;
- `action_mask`: feasibility of the six Atari actions;
- `critic:*`: padded, stage-private training information that is never passed
  to either actor head;
- action: `[Atari action, normalized economic action]`.

The actor contains a Nature CNN, a shared 64-unit state encoder, a six-logit
game head, and a Beta economic head.  The paper's meta-seller uses the
`seller_shared_context_beta_v5` economic parameterization.  Each curriculum
stage creates a fresh critic.  Rewards are undiscounted (`gamma =
gae_lambda = 1`) and actor credit is phase-specific: gameplay rows train the
game branch, follower trade or leader query rows train the economic branch,
and automatic/cached trade executions carry no second policy decision.
SB3 itself is unmodified; the repository subclasses PPO/policy hooks for the
two-headed distribution, credit mask, and learning-rate parameter groups.

There are five exogenous, paused trade events in 200 gameplay decisions.  A
trade gives the seller one bullet; acceptance is `price <= threshold`; and an
accepted trade transfers exactly one bullet and charges the price immediately.
The seller receives `0.1 * clipped_game_reward + payments`; the buyer receives
`clipped_game_reward - payments`.  A true ALE game-over resets that player's
emulator while preserving the outer bilateral episode and its accounting.

The implementation separates domain environments, neural policies, and Gym
adapters.  The Atari leader then composes with the same game-agnostic
`StackPOMDPWrapper` used by the normal-form and market experiments:

```text
stackelberg_pomdp/envs/atari/                 ALE, gameplay, curriculum, trade
stackelberg_pomdp/policies/atari/             actor--critic and frozen loading
stackelberg_pomdp/wrappers/atari/             preprocessing/response adapters
stackelberg_pomdp/atari/protocol.py           stable spaces and field layout
stackelberg_pomdp/atari/sampling.py           trade schedules and commitments
stackelberg_pomdp/atari/training.py           Atari PPO and training utilities
stackelberg_pomdp/wrappers/core.py            shared StackPOMDP phase wrapper
replication/atari/train_*.py                  experiment entrypoints
replication/atari/evaluate_*.py               deterministic selectors/audits
```

`stackelberg_pomdp/atari/stackpomdp_policy.py` is intentionally only a thin
compatibility import: released SB3 checkpoints serialize that historical
module path.  Maintained code imports `policies.atari.composite` directly.

## Runtime inputs

Run commands from the repository root after exporting
`PYTHONNOUSERSITE=1` and `PYTHONPATH=.`.  The ROM is not distributed; provide
it with `--rom-path` or
`STACKPOMDP_SPACE_INVADERS_ROM`.  Large checkpoints and run outputs are also
external artifacts.  The trainers fail closed when a required checkpoint is
missing, has the wrong policy class, or violates its saved provenance.

Six representative checkpoints, their sizes and SHA-256 digests, and a
verified downloader are published with the journal reproducibility release:
<https://github.com/GlcBrero/Stackelberg-Journal-Version/releases/tag/jair-2026-v1.0.0>.
The downloader installs them under
`reproducibility/artifacts/atari/checkpoints/` in that artifact checkout.
The full multi-seed paper figures are reproduced from the retained normalized
tables; readers do not need every training checkpoint merely to redraw them.

The reference software environment uses Python 3.9, Stable-Baselines3 1.8,
PyTorch, Gym 0.21, OpenCV, and `multi-agent-ale-py`.  W&B defaults to project
`StackPOMDP` and group `atari_clean_curriculum`; pass `--no-wandb` for an
offline smoke test.

## Canonical curriculum

The names below are illustrative local artifact paths. The release manifest
records SHA-256 hashes for the selected checkpoints; the ROM digest is
documented separately because the ROM is not redistributed.

### 1. Five-bullet gameplay bootstrap

```bash
python -m replication.atari.train_atari_curriculum_sb3 \
  --stage e0a --seed 1 --timesteps 50000000 --num-envs 4 \
  --rom-path /path/to/space_invaders.bin \
  --checkpoint replication/atari/checkpoints/clean/gameplay_e0a.zip
```

The large timestep value is a safety cap.  The trainer screens retained
checkpoints and stops only after the reliable-five gate passes on independent
evaluation seeds.

### 2. Randomly timed free-bullet adaptation

```bash
python -m replication.atari.train_atari_curriculum_sb3 \
  --stage e0b --seed 1 --timesteps 2000000 --num-envs 4 \
  --rom-path /path/to/space_invaders.bin \
  --init-checkpoint replication/atari/checkpoints/clean/gameplay_e0a_target.zip \
  --checkpoint replication/atari/checkpoints/clean/gameplay_e0b.zip
```

Actor modules transfer from the first stage; the critic is new.  The dedicated
`evaluate_atari_e0b_sb3.py` evaluator checks delayed free transfers, bullet
accounting, and deterministic gameplay on matched schedules.

### 3. Buyer and seller meta-responses

Buyer:

```bash
python -m replication.atari.train_atari_meta_response_sb3 \
  --role buyer --seed 1 --timesteps 2000800 --num-envs 4 \
  --actor-loss-mode balanced --e1-sampler-mode temporal-marginal-v1 \
  --e0b-checkpoint replication/atari/checkpoints/clean/gameplay_e0b_selected.zip \
  --rom-path /path/to/space_invaders.bin \
  --checkpoint replication/atari/checkpoints/clean/meta_buyer.zip
```

Seller:

```bash
python -m replication.atari.train_atari_meta_response_sb3 \
  --role seller --seed 1 --timesteps 2000800 --num-envs 4 \
  --learning-rate 0.0005 --actor-loss-mode balanced \
  --economic-architecture seller_shared_context_beta_v5 \
  --e0b-checkpoint replication/atari/checkpoints/clean/gameplay_e0b_selected.zip \
  --rom-path /path/to/space_invaders.bin \
  --checkpoint replication/atari/checkpoints/clean/meta_seller.zip
```

Use `evaluate_atari_meta_response_sb3.py` to screen all retained checkpoints
on common seeds and confirm only the screen winner on fresh seeds.  It writes
episode/event rows, fixed-context summaries, behavioral gates, and immutable
checkpoint provenance.

### 4. StackPOMDP leaders

For either role, the leader receives five event-only queries before a fresh
reward game.  Its opposite-role response is frozen.  Example seller leader:

```bash
python -m replication.atari.train_atari_stackpomdp_leader_sb3 \
  --leader-role seller --seed 1 --timesteps 2000040 \
  --num-envs 4 --n-steps 210 --batch-size 840 \
  --actor-loss-mode balanced \
  --response-checkpoint replication/atari/checkpoints/clean/meta_buyer_selected.zip \
  --leader-e1-checkpoint replication/atari/checkpoints/clean/meta_seller_selected.zip \
  --rom-path /path/to/space_invaders.bin \
  --checkpoint replication/atari/checkpoints/clean/leader_seller.zip
```

Swap the two response checkpoints and use `--leader-role buyer` for a buyer
leader.  `evaluate_atari_stackpomdp_leader_sb3.py` performs deterministic
screening, fresh-seed confirmation, endpoint counterfactuals, and the complete
210-transition protocol audit.

## Ten-seed Unity cohort

`automation/unity_atari_e2_multiseed.sbatch` is the single portable cluster
launcher.  It has no personal filesystem paths.  Set its input, output,
Python, and immutable-hash environment variables, then submit:

```bash
sbatch --array=0-19%8 \
  --export=ALL,STACKPOMDP_CODE_ROOT=$PWD,STACKPOMDP_RUN_ROOT=/scratch/$USER/atari-e2 \
  replication/atari/automation/unity_atari_e2_multiseed.sbatch
```

The 20 tasks are ten buyer-leader and ten seller-leader seeds.  Each screens
six checkpoints on 20 common seeds, confirms one winner on 100 fresh seeds,
archives the full transition report, and creates a compact content-addressed
report.  Compact reports use portable basenames, coherently rehash the
path-normalized environment configuration, and have basename-only SHA-256
sidecars.  Aggregate independent policy means (not pooled episodes) with:

```bash
python -m replication.atari.automation.aggregate_atari_e2_multiseed \
  --input-dir /scratch/$USER/atari-e2/outputs/results/canonical \
  --output atari_e2_aggregate.json
```

## Retained additional experiment: frozen gameplay actor

The paper's main runs fine-tune both actor branches.  The retained ablation
freezes the transferred CNN, state encoder, and game head while training the
leader economic head and a fresh critic.  The checkpoint stores hashes of all
three frozen modules and validates them before and after learning and on load.

Use the same launcher with a ten-task buyer-only array:

```bash
sbatch --array=0-9%8 \
  --export=ALL,STACKPOMDP_E2_VARIANT=frozen-buyer,STACKPOMDP_CODE_ROOT=$PWD,STACKPOMDP_RUN_ROOT=/scratch/$USER/atari-e2-frozen \
  replication/atari/automation/unity_atari_e2_multiseed.sbatch

python -m replication.atari.automation.aggregate_atari_e2_multiseed \
  --roles buyer \
  --input-dir /scratch/$USER/atari-e2-frozen/outputs/results/frozen-buyer \
  --output atari_e2_frozen_buyer_aggregate.json
```

For a local run, add `--freeze-gameplay-actor` to the leader trainer.  This is
an additional diagnostic, not the paper's primary architecture.

## Tests

The Atari suite is ROM-free unless the optional integration smoke is enabled:

```bash
pytest -q tests/test_atari_*.py
```

It covers the stable observation/action schema, trade and bullet accounting,
terminal handling, phase-specific policy credit, actor transfer, checkpoint
round trips, frozen-gameplay hashes, exact query-action cache reuse,
deterministic selection, and policy-level multi-seed uncertainty.
