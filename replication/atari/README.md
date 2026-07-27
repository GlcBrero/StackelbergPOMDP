# Clean Atari Stackelberg curriculum

This directory contains the single active Stable-Baselines3 implementation of
the continuous-price Space Invaders experiment. The clean rerun uses one
composite policy and one observation/action schema from E0a through E2.

No result from the older RLlib, frozen-gameplay, or event-compressed pilots is
a result for this architecture. Their source was removed from the importable
tree after being preserved under:

```text
../../Research Artifacts/legacy_atari_prototype_20260727/
```

Existing historical checkpoints remain on disk, but the clean trainers reject
them because they do not contain `StackPOMDPAtariPolicy`.

## Stable interface

Every stage emits the full action

```text
[Atari game action, normalized economic action]
```

and observes the same padded dictionary:

- `image`: four stacked 84 x 84 frames;
- `actor_state`: the 14-scalar vector
  `(ammo/5, projectile_active, gameplay_time/H, trade_mode,
  event_one_hot[5], opponent_commitment[5])`;
- `action_mask`: six deterministic Atari-action entries;
- `critic:state`: 32 padded stage-private training features;
- `critic:action_credit`: `[game, economic]` PPO log-probability gates.

The critic-prefixed fields never enter either actor head or the
observation-action cache. The actor is:

```text
frames -> Nature CNN 512 -------------------+
                                             +-> 576 -> 6 Atari logits
actor_state -> shared FC 64 ReLU -----------+
                     |
                     +-> 64 tanh -> 64 tanh -> Beta alpha,beta
```

The economic head never consumes pixels. E1 responses use the full 14-scalar
state. An E2 leader explicitly zeros every economic input except the five-way
event identity. Deterministic evaluation uses masked Atari argmax and the Beta
mean.

Each stage creates a fresh private critic. The CNN, shared state encoder, and
game head transfer between stages; transferred visual/game modules can use a
lower learning rate. The economic head starts near WTP 0.95 for an E1 buyer,
uniform for an E1 seller, and is reinitialized uniformly for an E2 leader.

Policy credit is branch-specific while rewards share one undiscounted return:

| Transition | Game PPO term | Economic PPO term |
|---|---:|---:|
| gameplay | 1 | 0 |
| E1 follower trade | 0 | 1 |
| E2 leader query | 0 | 1 |
| E2 cached reward-trade replay | 0 | 0 |
| E0b automatic free transfer | 0 | 0 |

Thus a payment or delayed game reward can credit an earlier economic query
because `gamma = gae_lambda = 1`, while cached execution is never counted as a
second policy decision.

## Environment protocol

- E0a: one Atari agent, five bullets at reset, no trades, 200 gameplay steps.
- E0b: one Atari agent, zero bullets at reset, five randomly timed paused free
  transfers, 200 gameplay plus five transfer steps.
- E1: buyer or seller response, random opponent commitment from
  `Uniform(0,1)^5`, five exogenous paused trade events, and 200 gameplay steps.
  Both actor branches of the controlled policy remain trainable.
- E2: five zero-reward event-only leader queries, a fresh bilateral reward
  game with 200 gameplay steps, and five actor-identical cached trade replays.

Each bilateral event grants exactly one bullet to the seller. Acceptance is
`price <= threshold`; an accepted trade transfers one bullet and applies the
payment immediately. Seller payoff is `0.1 * clipped_game_reward + payments`;
buyer payoff is `clipped_game_reward - payments`. Trades never advance ALE.

E2 retains the exact ordered query trace `Q`, including complete actor-visible
observations and complete two-coordinate actions. The five economic actions
form the declared response statistic `omega`; they are not described as a
lossless encoding of an arbitrary full trace. A per-episode action map caches
complete actions by exact actor-visible observations and resets only the rows
whose outer episodes end.

E2 reuses the same game-agnostic `StackPOMDPWrapper` as the other
Stackelberg experiments. Its lower layers are:

```text
StackPOMDPWrapper
  -> AtariMetaFollowerWrapper
       -> BilateralAtariRewardEnv
```

`BilateralAtariRewardEnv` owns only Atari, trade, and payoff dynamics.
`AtariMetaFollowerWrapper` implements a frozen neural PI response: it records
the five exact leader queries, finalizes their context, and then evaluates the
opposite-role E1 policy deterministically during the reward game. E1 is where
that meta-policy is learned; the response does not update online during E2.

## Setup

From `Code/StackelbergPOMDP`:

```bash
conda activate stackelbergPOMDP
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
```

The vendored ROM is discovered automatically. It can be overridden with
`STACKPOMDP_SPACE_INVADERS_ROM` or `--rom-path`.

All runs default to W&B project `StackPOMDP`, group
`atari_clean_curriculum`, and stage-specific job types such as `atari_e0a` and
`atari_e1_buyer`. Use `--no-wandb` only for tests.

## E0a: five-bullet bootstrap

```bash
python -u -m replication.atari.train_atari_curriculum_sb3 \
  --stage e0a \
  --seed 1 \
  --timesteps 50000000 \
  --num-envs 4 \
  --checkpoint replication/atari/checkpoints/clean/space_invaders_e0a_ppo_seed1.zip \
  --wandb-name atari_clean_e0a_seed1_targetstop_50mcap_local
```

The timestep argument is an absolute safety cap, not a required training
length. E0a writes a post-update checkpoint and evaluates it every 400,000
steps. It stops early only after two consecutive deterministic 20-episode
screens pass and the same checkpoint passes a fresh 100-episode confirmation.
The reliable-five gate requires mean clipped game reward at least 4.8, reward
at least 5 in at least 90% of episodes, all five bullets fired in at least 95%
of episodes, mean shots at least 4.95, and mean final ammo at most 0.05. It also
requires complete 200-step episodes, zero payments, equality of game and total
return, and exact bullet accounting. The reported 20-episode evaluation uses a
third, untouched seed block.

Every evaluated `..._stepN.zip` is retained. `..._best.zip` tracks the best
screening checkpoint, while `..._target.zip` is created only after independent
confirmation. Validation rows, seed ranges, thresholds, and selections are
recorded in `*.validation.jsonl`, per-checkpoint `*.validation.json` /
`*.confirmation_N.json`, and `*.target_selection.json`. This makes it safe to
use a generous 50M ceiling without training past an already confirmed target.

## E0b: delayed free bullets

A new stage transfers actor modules from E0a but creates a fresh critic. The
`--resume` flag is reserved for a true same-stage resume, so stage transfer is
spelled explicitly as `--init-checkpoint`.

```bash
python -u -m replication.atari.train_atari_curriculum_sb3 \
  --stage e0b \
  --init-checkpoint replication/atari/checkpoints/clean/space_invaders_e0a_ppo_seed1.zip \
  --seed 1 \
  --timesteps 2000000 \
  --num-envs 4 \
  --checkpoint replication/atari/checkpoints/clean/space_invaders_e0b_ppo_seed1.zip \
  --wandb-name atari_clean_e0b_seed1_2m_local
```

E0b should accept all five automatic transfers, fire close to all five bullets,
and recover the E0a game score despite random arrival times.

## E1: role-specific meta-responses

Buyer response to random seller price sequences:

```bash
python -u -m replication.atari.train_atari_meta_response_sb3 \
  --role buyer \
  --e0b-checkpoint replication/atari/checkpoints/clean/space_invaders_e0b_ppo_seed1.zip \
  --seed 1 \
  --timesteps 2000000 \
  --checkpoint replication/atari/checkpoints/clean/meta_buyer_e1_ppo_seed1.zip \
  --wandb-name atari_clean_e1_buyer_seed1_2m_local
```

Seller response to random buyer-threshold sequences uses the same command with
`--role seller`. E1 evaluation runs 100 random commitment sequences plus 20
episodes at each constant opponent value from 0.0 through 1.0. The JSON keeps
all random episode trade records and acceptance-by-event/time diagnostics; the
CSV contains the fixed-context table.

## E2: Stackelberg leaders

Seller leader against a frozen E1 meta-buyer:

```bash
python -u -m replication.atari.train_atari_stackpomdp_leader_sb3 \
  --leader-role seller \
  --response-checkpoint replication/atari/checkpoints/clean/meta_buyer_e1_ppo_seed1.zip \
  --leader-e1-checkpoint replication/atari/checkpoints/clean/meta_seller_e1_ppo_seed1.zip \
  --seed 1 \
  --timesteps 2000000 \
  --checkpoint replication/atari/checkpoints/clean/leader_seller_e2_ppo_seed1.zip \
  --wandb-name atari_clean_e2_seller_seed1_2m_local
```

For a buyer leader, swap the roles of the two E1 checkpoints. E2 rollouts are
exactly `5 + 200 + 5 = 210` transitions and always enable the full-action
cache callback. The generic wrapper is configured for five response
transitions and 205 reward transitions, with all five zero-reward queries kept
in PPO's rollout.

## Resume and deterministic evaluation

`--resume CHECKPOINT` restores the complete same-stage policy, critic,
optimizer groups, and timestep clock. It is not an actor-only warm start.
Evaluation without additional training uses:

```bash
python -u -m replication.atari.train_atari_curriculum_sb3 \
  --stage e0a --resume CHECKPOINT --eval-only --no-wandb
```

The E1 and E2 trainers support the same `--resume ... --eval-only` pattern and
still require the environment-side E0b or frozen-response checkpoint named by
their normal arguments.

Each checkpoint produces:

- `CHECKPOINT.training.jsonl`: completed-episode training metrics;
- `CHECKPOINT.evaluation.json`: deterministic evaluation rows and summaries;
- E1 only, `CHECKPOINT.fixed_contexts.csv`: paired fixed-context table;
- step checkpoints at completed outer-episode boundaries.

For E0a, `--timesteps` counts the complete run clock after resume. A resumed
run can retain the same W&B URL with `--wandb-id RUN_ID --wandb-resume must`.

W&B logs episode payoff and length, role-correct game reward, shots, ammo,
reward per bullet, payments, purchases, all five event prices/thresholds/
acceptance times, total timesteps, learning rate, seed, algorithm, and
checkpoint path.

## Validation

```bash
PYTHONNOUSERSITE=1 \
python -c 'import sys; sys.modules["readline"] = None; import pytest; raise SystemExit(pytest.main(["-q", "tests/test_atari_clean_e0_trainer.py", "tests/test_atari_clean_protocol.py", "tests/test_atari_clean_envs.py", "tests/test_atari_clean_meta_response_trainer.py", "tests/test_atari_clean_leader_trainer.py"]))'
```

The clean suite checks the stable 14D interface, branch-gradient isolation,
event-only leader invariance, exact full-action cache reuse, zero actor credit
on cached replays, per-vector-row cache reset, one FIRE press per max-and-skip
decision, atomic trade accounting, complete E1/E2 horizons, fresh critics,
actor transfer, and optimizer checkpoint reloadability.

## Current clean-run status

As of 2026-07-27, clean E0a seed 1 is training locally with automatic target
selection and a 50M-step safety cap:

- W&B: <https://wandb.ai/glcbrero/StackPOMDP/runs/dyaly9m7>
- run name: `atari_clean_e0a_seed1_targetstop_50mcap_local`
- group/job type: `atari_clean_curriculum` / `atari_e0a`
- tmux session: `atari_clean_e0a_seed1_target50m`
- checkpoint: `replication/atari/checkpoints/clean/space_invaders_e0a_ppo_seed1.zip`
- local log: `Research Artifacts/experiment_logs/StackelbergPOMDP/atari/clean_20260727/atari_clean_e0a_seed1_targetstop_50mcap_local.log`

The original fixed-length process reached a durable 200,000-step checkpoint
without error. The target-driven process resumes that complete model,
optimizer, critic, and clock under the same W&B run ID. This remains an active
run, not a declared result; only the auditable selection files above determine
pass or failure.
