# Atari Replication

This directory contains the staged Space Invaders replication. The current
implementation is the native Stable-Baselines3 pipeline described below. The
older RLlib experiments are retained under **Legacy RLlib reference** so their
checkpoints and completed results remain reproducible.

## Native SB3 shared pipeline (current)

One policy architecture and one environment interface are used in every stage:

- a Nature CNN and ammo/market features feed a categorical Atari-action head;
- a bounded Beta head outputs scalar willingness to pay;
- offered price is excluded structurally from both actor heads;
- one price-informed critic receives the shared features plus price;
- the environment buys exactly when `price <= threshold`;
- FIRE actions are masked while an ALE projectile is active;
- E0, free trade, E1, and joint fine-tuning differ only in environment
  configuration and which actor parameters receive gradients.

The observation keys are identical throughout: `image`, `ammo_fraction`,
`projectile_active`, `action_mask`, `offer_active`,
`opportunities_remaining`, and `critic:price`. The `critic:` prefix documents
and enforces that price is not an actor input.

Set the ROM path and ignore incompatible user-site packages before invoking
SB3 in the project conda environment:

```bash
export PYTHONNOUSERSITE=1
export STACKPOMDP_SPACE_INVADERS_ROM="$PWD/../StackeRLberg/stackerlberg/envs/roms/space_invaders.bin"
export PYTHONPATH=.
```

### E0: five initial bullets

```bash
python -m replication.atari.train_price_aware_atari_sb3 \
  --stage gameplay \
  --seed 1 \
  --timesteps 10000000 \
  --num-envs 4 \
  --checkpoint replication/atari/checkpoints/sb3/space_invaders_e0_ppo_seed1_10m.zip \
  --wandb-project StackPOMDP \
  --wandb-group atari_price_aware_sb3 \
  --wandb-job-type atari_sb3_e0_pretraining
```

The active local run is
[`sb3_e0_five_bullet_ppo_seed1_10m_local`](https://wandb.ai/glcbrero/StackPOMDP/runs/cx4srrh4).
It uses five initial bullets, no seller, no offers, no replenishment, clipped
game rewards, deterministic 20-episode evaluation every 100k steps, and saves
both latest and deterministic-best checkpoints. The E0 pass gate is median
reward 5, mean reward at least 4.8, and all five bullets fired in at least 95%
of evaluation episodes.

Standalone evaluation:

```bash
python -m replication.atari.evaluate_price_aware_atari_sb3 \
  --checkpoint replication/atari/checkpoints/sb3/space_invaders_e0_ppo_seed1_10m_best.zip \
  --stage gameplay \
  --episodes 20
```

### Free-trade bridge

The optional short bridge starts with zero ammo and presents five zero-price
offers. The threshold is still frozen; zero-price offers are therefore always
accepted. Gameplay remains trainable so it can adapt from five bullets present
at reset to bullets arriving through the final trade interface. The current E0
best checkpoint already passes this bridge without training: all 20 paired
episodes bought five, fired five, scored five, and paid zero. The primary
pipeline therefore skips bridge optimization and initializes E1 directly from
E0; the command below remains available if a future E0 checkpoint fails this
gate.

```bash
python -m replication.atari.train_price_aware_atari_sb3 \
  --stage free_trade \
  --resume replication/atari/checkpoints/sb3/space_invaders_e0_ppo_seed1_10m_best.zip \
  --seed 1 \
  --timesteps 500000 \
  --checkpoint replication/atari/checkpoints/sb3/space_invaders_free_trade_ppo_seed1_500k.zip
```

### E1: frozen gameplay, trainable WTP head

E1 starts with zero ammo, draws five offers from `Uniform(0, 1)`, freezes the
CNN/gameplay head, selects Atari actions by deterministic argmax, and trains
only the threshold head plus the single price-informed critic. PPO uses
undiscounted returns (`gamma=1`, `gae_lambda=1`).

```bash
python -m replication.atari.train_price_aware_atari_sb3 \
  --stage priced \
  --resume replication/atari/checkpoints/sb3/space_invaders_e0_ppo_seed1_10m_best.zip \
  --seed 1 \
  --timesteps 1000000 \
  --learning-rate 5e-5 \
  --entropy-coeff 0 \
  --checkpoint replication/atari/checkpoints/sb3/space_invaders_e1_ppo_seed1_1m.zip
```

The training callback selects E1 checkpoints on economic performance rather
than raw game score. It evaluates paired seeds on fixed prices `0.0, 0.1, ...,
1.0`, separately evaluates random prices, writes JSON and fixed-price CSV
artifacts, and logs the table's scalar cells to W&B. A standalone full
evaluation uses 20 episodes per fixed price and 100 random-price episodes:

```bash
python -m replication.atari.evaluate_price_aware_atari_sb3 \
  --checkpoint replication/atari/checkpoints/sb3/space_invaders_e1_ppo_seed1_1m_best.zip \
  --stage priced \
  --episodes-per-price 20 \
  --random-episodes 100
```

Joint economic/gameplay fine-tuning remains available with `--stage joint`,
but it is run only after frozen-gameplay E1 passes. It trains both actor heads
with the same observation/action interface and retains the price-informed
critic.

## Legacy RLlib reference

The previously protected RLlib E0 gameplay checkpoint is selected by
`checkpoints/space_invaders_5bullets_a3c_best.json`.

## E0: five-bullet gameplay

E0 trains Atari gameplay with five initial bullets and no replenishment. The
selected policy is ammo-aware and masks FIRE actions while an ALE projectile is
active. It is frozen and evaluated by deterministic argmax in all subsequent
economic stages.

```bash
PYTHONPATH=.:../StackeRLberg \
  python -m replication.atari.evaluate_five_bullet_checkpoint \
  --checkpoint replication/atari/checkpoints/unity/space_invaders_5bullets_a3c_ammo_mask_seed2_2m_step1629560.pkl \
  --episodes 20
```

The protected 20-episode result has mean clipped reward `4.45`, mean shots
`5.0`, and mean final ammo `0.0`.

## E1: frozen gameplay and trainable buyer threshold

E1 starts the buyer with zero bullets. A non-Atari process supplies five scalar
prices from `Uniform(0, 1)`. At each opportunity, the threshold head observes

```text
[offered price, trade flag, remaining opportunities / 5, buyer ammo / 5]
```

and outputs one scalar willingness to pay. A purchase occurs exactly when
`price <= threshold`, transfers one bullet, and immediately subtracts the
price from buyer reward. The protected E0 gameplay policy is held frozen and
selects deterministic argmax game actions. PPO uses `gamma=1` and `lambda=1`.

The preferred WTP specification keeps the full four-scalar observation for the
critic but excludes offered price from the actor. This is deliberate: a
willingness-to-pay threshold should value the next bullet from economic state,
while the external `price <= threshold` rule supplies the price dependence.
It prevents the actor from replacing a WTP with a price-contingent accept/reject
classifier. Use `--hide-price-from-actor` for this specification.

The E1 wrapper is event-driven: PPO receives exactly the five economically
meaningful threshold decisions. After the final offer, frozen Atari gameplay
runs internally to the end of the episodic-life episode and its undiscounted
reward is returned on the final transition.

### Smoke test

```bash
PYTHONPATH=.:../StackeRLberg \
  python -m replication.atari.train_buyer_threshold_head \
  --seed 991 \
  --iterations 1 \
  --num-workers 0 \
  --rollout-fragment-length 5 \
  --train-batch-size 10 \
  --sgd-minibatch-size 5 \
  --num-sgd-iter 1 \
  --checkpoint /private/tmp/e1_buyer_threshold_smoke.pkl \
  --checkpoint-every 1 \
  --fixed-eval-prices 0.2 \
  --eval-episodes-per-price 1 \
  --random-eval-episodes 1 \
  --no-wandb \
  --ray-local-mode
```

Local smoke result on 2026-07-14: **passed**. At fixed price `0.2`, the scripted
accept path exposed five opportunities, bought five bullets, paid `1.0`, fired
five shots, ended with zero ammo, earned clipped game reward `5.0`, and earned
net reward `4.0`. The reject path bought/fired zero bullets and paid zero. A
one-iteration PPO checkpoint reloaded through the standalone evaluator and
produced JSON and CSV outputs. This smoke result is a mechanics validation, not
a trained E1 scientific result.

### Full training

```bash
PYTHONPATH=.:../StackeRLberg \
  python -m replication.atari.train_buyer_threshold_head \
  --seed 1 \
  --timesteps 50000 \
  --checkpoint replication/atari/checkpoints/e1/buyer_threshold_ppo_uniform_seed1.pkl
```

Default outputs are:

- threshold checkpoint: `replication/atari/checkpoints/e1/buyer_threshold_ppo_uniform_seed1.pkl`
- JSONL training log: `replication/atari/checkpoints/e1/buyer_threshold_ppo_uniform_seed1.training.jsonl`
- evaluation JSON: `replication/atari/checkpoints/e1/buyer_threshold_ppo_uniform_seed1.evaluation.json`
- fixed-price CSV: `replication/atari/checkpoints/e1/buyer_threshold_ppo_uniform_seed1.fixed_prices.csv`

W&B defaults:

- project: `StackPOMDP`
- job type: `atari_e1_buyer_threshold`
- run name: `e1_buyer_threshold_ppo_uniform_seed1`
- seed-1 50k baseline run:
  `https://wandb.ai/glcbrero/StackPOMDP/runs/vmi9v7w0`

The seed-1 50k run completed normally. Its final checkpoint bought/fired all
five bullets through fixed price `0.50` and rejected all offers at `0.60+`.
Over 100 random-price episodes it earned mean net reward `1.581`, bought/fired
`2.87` bullets, and had positive net reward in 91 episodes. An archived
checkpoint at 45,027 decisions was better: it also bought all five at `0.60`
and achieved random-price mean net reward `1.638`. It is retained as the
price-aware baseline; the selected WTP checkpoint below supersedes it.

The direct price-aware warm-start refinement was stopped after 42,527
additional decisions because its deterministic boundary stayed near `0.60`.
Its W&B run was deleted during cleanup because it is a superseded diagnostic,
not an E1 reference; the local checkpoint and JSONL log remain available.

The WTP variant is trained from the seed-1 45,027-step actor. Its critic is
fresh because the actor and critic have different inputs. The parent exploration
standard deviation is also reset with `--initial-log-std -3` and
`--reset-log-std-on-warmstart` so PPO samples coherent WTP decisions. The loader
explicitly synchronizes the warm-started weights to remote rollout workers
before their first batch.

Selected WTP training specification:

```bash
PYTHONPATH=.:../StackeRLberg \
  python -m replication.atari.train_buyer_threshold_head \
  --seed 5 \
  --timesteps 2500 \
  --initial-checkpoint replication/atari/checkpoints/e1/buyer_threshold_ppo_uniform_seed1_step45027.pkl \
  --hide-price-from-actor \
  --initial-log-std -3 \
  --reset-log-std-on-warmstart \
  --learning-rate 5e-5 \
  --entropy-coeff 0 \
  --checkpoint replication/atari/checkpoints/e1/buyer_threshold_ppo_wtp_lownoise_seed5_reproduction.pkl
```

- W&B run: `https://wandb.ai/glcbrero/StackPOMDP/runs/m3jli3rk`
- W&B job type: `atari_e1_buyer_threshold_wtp`
- selected-candidate checkpoint:
  `checkpoints/e1/buyer_threshold_ppo_wtp_lownoise_seed5_50k_step2500.pkl`
- full deterministic evaluation: buys/fires all five at every fixed price
  through `0.80`, with net reward `0.45` at `0.80`; rejects `0.90` and `1.00`
- 100 Uniform-price episodes: mean net reward `1.968`, mean purchases/shots
  `4.33`, zero final ammo, 100% aggregate purchased-bullet firing, positive net
  reward in 100/100 episodes
- pass condition: **passed**
- the historical run targeted 50k steps and was early-stopped at 22,522 because
  the deterministic WTP boundary declined monotonically; its 20,015-step
  archive rejected price `0.80`, so further optimization was scientifically
  counterproductive
- canonical selector: `checkpoints/e1/buyer_threshold_best.json`
- the selected table/checkpoint/log are attached to the W&B run as the
  `e1-buyer-threshold-selected-seed5` model artifact, and the `selected/*`
  summary fields identify the passing archive

The W&B time series contains net episode reward, clipped gameplay reward,
payments, purchases, trade opportunities, acceptance rate, shots, final ammo,
reward per purchased bullet, threshold, offered price, decision steps, and
estimated underlying Atari steps. Learning rate, seed, algorithm, and artifact
paths are run configuration/summary fields.

### Standalone evaluation

```bash
PYTHONPATH=.:../StackeRLberg \
  python -m replication.atari.evaluate_buyer_threshold_head \
  --checkpoint replication/atari/checkpoints/e1/buyer_threshold_best.json \
  --episodes-per-price 20 \
  --random-episodes 100
```

The evaluator uses paired game seeds at fixed prices `0.0, 0.1, ..., 1.0`, a
separate random-price seed suite, deterministic threshold means, and frozen
gameplay argmax. The pass check uses the paired zero-price evaluation to
estimate five-bullet gameplay value, then requires mean purchases at least
`4.5`, at least 90% of purchased bullets fired, and positive mean net reward at
every evaluated price where five bullets remain profitable. With the current
E0 value of `4.45`, this includes fixed prices through `0.80`. It additionally
requires at least 90% aggregate purchased-bullet firing and positive mean net
reward under Uniform prices.
