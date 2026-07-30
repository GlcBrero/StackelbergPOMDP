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
live state. An E2 leader instead receives five ex-ante canonical queries; only
the five-way event identity varies across queries, and its event-only economic
head zeros every non-event input internally. Each cached reward-trade replay
uses the identical canonical query observation. Deterministic evaluation uses
masked Atari argmax and the Beta mean.

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

### PPO actor-loss modes

E1 and E2 expose two explicit `--actor-loss-mode` choices:

- `standard` is the default, backward-compatible SB3
  objective. Actor and entropy terms are averaged over rollout transitions, so
  inactive rows remain in the denominator even though their gated log
  probability is zero. A legacy checkpoint without stored actor-loss metadata
  is interpreted as this mode.
- `balanced` separately averages the PPO surrogate,
  entropy, clip fraction, and approximate KL over active gameplay rows and
  active economic rows. The two actor losses and two entropy losses are then
  summed. Cached reward-trade rows have no actor contribution. The critic still
  fits returns on every transition, and SB3 advantage normalization remains
  global over the full rollout. Target-KL early stopping uses the larger of the
  gameplay-head and economic-head KL values.

Phase-balanced training requires a single full-rollout minibatch:
`batch_size = n_steps * num_envs`. This guarantees that both sparse economic
rows and dense gameplay rows are present when their separate active-row means
are computed. The trainers reject smaller minibatches in this mode.
Both modes preserve the existing parameter-group learning-rate scaling.
Phase-balanced runs additionally log separate gameplay/economic policy loss,
entropy, clip fraction, approximate KL, and active-row counts, plus the number
of actor-inactive rows. These optimizer rows are written durably to both W&B
and the local `.training.jsonl`, including the exact optimization timestep.
Optional `--target-kl K` activates early stopping on the larger of the two
head-specific approximate KL values. It defaults to `None`; the first balanced
run should leave it unset so that phase averaging is the only objective change.

New E1/E2 checkpoints store both the actor-loss mode and the economic-head
initialization contract. Same-stage resume, including `--eval-only`, validates
these fields rather than silently changing the training objective or initial
Beta distribution; the E2 provenance evaluator and selector also distinguish
them. An E1 buyer defaults to Beta mean `0.95` and concentration `10`, exposed
as `--buyer-init-mean` and `--buyer-init-concentration`. Legacy buyer
checkpoints without initialization metadata use those historical defaults.
Default balanced checkpoint and W&B names contain a `balanced` slug; nondefault
buyer initialization and nondefault target-KL values also enter the default
checkpoint stem, preventing variants from silently sharing artifacts.

## Environment protocol

Within one max-and-skip decision, a requested FIRE action is forwarded until
ALE confirms that a projectile was created. FIRE is then removed from the
remaining raw frames. This tolerates frame-level non-registration while
guaranteeing at most one consumed bullet per policy decision.

- E0a: one Atari agent, five bullets at reset, no trades, 200 gameplay steps.
- E0b: one Atari agent, zero bullets at reset, five randomly timed paused free
  transfers, 200 gameplay plus five transfer steps.
- E1: buyer or seller response, random opponent commitment from
  `Uniform(0,1)^5`, five exogenous paused trade events, and 200 gameplay steps.
  Both actor branches of the controlled policy remain trainable.
- E2: five zero-reward event-only leader queries, a fresh bilateral reward
  game with 200 gameplay steps, and five actor-identical cached trade replays.

E0b, E1, and E2 share the same schedule: five distinct gameplay indices are
sampled uniformly without replacement from the full `{0,...,H-1}` horizon and
then sorted. Thus the fifth event may occur at `H-1`; no gameplay tail is
reserved. `--fixed-event-steps` remains available for controlled diagnostics.

Each bilateral event grants exactly one bullet to the seller. Acceptance is
`price <= threshold`; an accepted trade transfers one bullet and applies the
payment immediately. Seller payoff is `0.1 * clipped_game_reward + payments`;
buyer payoff is `clipped_game_reward - payments`. Trades never advance ALE.

An ordinary episodic-life boundary rebuilds the frame stack after one NOOP and
preserves the outer ammunition ledger. A true ALE game-over or emulator time
limit also resets only that player's emulator immediately and rebuilds its
frame stack and projectile state. The reset preserves outer ammo, cumulative
shots and reward, inventory, payments, the other emulator, the global gameplay
clock, and all five exogenous trade events. Gameplay therefore remains alive on
every one of the fixed `H` gameplay transitions; there is no inactive actor
state or absorbing gameplay action.

The clean 14-scalar actor and 32-scalar critic restore the schema used by the
existing clean E0 checkpoints, so those checkpoints remain structurally
compatible for resume, actor transfer, and frozen control. Historical
checkpoints built with other policy classes remain rejected as described above.

E2 retains the exact ordered query trace `Q`, including complete actor-visible
observations and complete two-coordinate actions. The five economic actions
form the declared response statistic `omega`; they are not described as a
lossless encoding of an arbitrary full trace. A per-episode action map caches
complete actions by exact actor-visible observations and resets only the rows
whose outer episodes end.

E1 and E2 reuse one bilateral base game. E1 merely supplies a sampled fixed
opponent commitment and frozen opponent gameplay controller:

```text
AtariFixedCommitmentResponseWrapper
  -> BilateralAtariRewardEnv
```

E2 reuses the same game-agnostic `StackPOMDPWrapper` as the non-Atari
Stackelberg experiments:

```text
StackPOMDPWrapper
  -> AtariMetaFollowerWrapper
       -> BilateralAtariRewardEnv
```

`BilateralAtariRewardEnv` is a standard multi-agent `BaseEnv`: it receives the
complete leader/follower action map, returns the leader's scalar reward, and
places both players' rewards in `info["utilities"]`. It owns only Atari, trade,
payoff, and accounting dynamics. The two response wrappers fill in follower
actions and expose a single controlled policy; phase management remains in the
generic wrapper.
`AtariMetaFollowerWrapper` implements a frozen neural PI response: it records
the five exact leader queries, finalizes their context, and then evaluates the
opposite-role E1 policy deterministically during the reward game. E1 is where
that meta-policy is learned; the response does not update online during E2.
The follower's in-game response sees the live 14-scalar state, while the
leader's ex-ante commitment remains event-indexed.

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
  --init-checkpoint replication/atari/checkpoints/clean/space_invaders_e0a_ppo_seed1_target.zip \
  --seed 1 \
  --timesteps 2000000 \
  --num-envs 4 \
  --checkpoint replication/atari/checkpoints/clean/space_invaders_e0b_ppo_seed1.zip \
  --wandb-name atari_clean_e0b_seed1_2m_local
```

E0b should accept all five automatic transfers and use bullets that arrive with
enough gameplay remaining. Full-horizon random schedules deliberately include
late transfers whose projectiles cannot always reach an alien before the
200-step horizon, so the unconditional random-schedule score is reported by
fifth-event timing rather than compared mechanically with E0a's 5/5 gate. A
fixed usable schedule provides the uncensored control: there E0b should again
fire all five bullets and recover the E0a score.

Use the dedicated evaluator to compare E0b with its E0a source on identical
seeds and schedules. It writes collision-safe artifacts under
`replication/atari/results/e0b_evaluations/` rather than overwriting either
checkpoint's training evaluation.

```bash
python -u -m replication.atari.evaluate_atari_e0b_sb3 \
  --checkpoint replication/atari/checkpoints/clean/space_invaders_e0a_ppo_seed1_firefix_retrain_target.zip \
  --checkpoint replication/atari/checkpoints/clean/space_invaders_e0b_ppo_seed1_firefix_retrain.zip \
  --episodes 100 \
  --seed-start 3100001 \
  --fixed-event-steps 0,10,20,30,40 \
  --fixed-event-steps 0,30,60,90,120 \
  --fixed-event-steps 10,50,100,150,190 \
  --run-name e0a_vs_e0b_final_seed3100001_n100
```

Every scenario is required to pass the exact `200 + 5` transition, transfer,
payment, return, and bullet-accounting audit. The report also records
checkpoint SHA-256 hashes, paired episode deltas, and outcomes by fifth-event
time.

## E1: role-specific meta-responses

Buyer response to random seller price sequences:

```bash
python -u -m replication.atari.train_atari_meta_response_sb3 \
  --role buyer \
  --e0b-checkpoint replication/atari/checkpoints/clean/space_invaders_e0b_ppo_seed1_firefix_retrain_selected.zip \
  --actor-loss-mode balanced \
  --buyer-init-mean 0.95 \
  --buyer-init-concentration 10 \
  --seed 1 \
  --timesteps 2000800 \
  --num-envs 4 \
  --checkpoint-every 400000 \
  --checkpoint replication/atari/checkpoints/clean/meta_buyer_e1_ppo_balanced_seed1_firefix_retrain.zip \
  --wandb-name atari_clean_e1_buyer_balanced_seed1_firefix_retrain_2m_local
```

Seller response to random buyer-threshold sequences uses the same command with
`--role seller` and the same phase-balanced actor-loss mode; its economic head
retains the uniform initialization rather than using the buyer-only
initialization flags. E1 evaluation runs 100 random commitment sequences plus
20 episodes at each constant opponent value from 0.0 through 1.0. Every fixed
value uses the same Atari/no-op and event-schedule seeds, making the grid a
paired comparison. The JSON keeps random and fixed-context episode-level trade
records plus acceptance-by-event/time diagnostics; the CSV contains the
fixed-context summary table.

The five opponent-commitment coordinates are identically zero in E0a and E0b,
so their incoming state-encoder weights have never received an informative
gradient. On the E0b-to-E1 actor transfer, only those five columns are therefore
initialized to zero. This makes the initial gameplay action invariant to a new
price/threshold sequence, while the columns remain trainable and can acquire
economic effects during E1 fine-tuning. All visual, game-head, and other state
weights transfer exactly, and the checkpoint provenance records the five reset
indices. A no-update ALE transfer check on the usable event schedule
`20,50,80,110,140` bought, fired, and scored all five bullets at constant
opponent values 0, 0.25, and 0.5, with exact payment and bullet accounting; the
diagnostic is retained under `replication/atari/results/e1_evaluations/`.

Select an E1 checkpoint only with the deterministic selector (repeat
`--checkpoint` for every retained step checkpoint and include the final
post-update checkpoint):

```bash
python -u -m replication.atari.evaluate_atari_meta_response_sb3 \
  --role buyer \
  --e0b-checkpoint replication/atari/checkpoints/clean/space_invaders_e0b_ppo_seed1_firefix_retrain_selected.zip \
  --checkpoint replication/atari/checkpoints/clean/meta_buyer_e1_ppo_balanced_seed1_step400160.zip \
  --checkpoint replication/atari/checkpoints/clean/meta_buyer_e1_ppo_balanced_seed1_step800320.zip \
  --selected-checkpoint replication/atari/checkpoints/clean/meta_buyer_e1_ppo_balanced_seed1_selected.zip
```

The selector binds every candidate to the exact supplied E0b bytes and rejects
the wrong policy class, role, input mode, or 205-transition/accounting
protocol. Its common training configuration explicitly includes the stored
actor-loss mode, economic-head initialization, and target KL, applying the
historical role-specific defaults only when legacy checkpoint metadata is
absent. It evaluates run-private immutable copies of the E0b, ROM, and
candidate bytes, requires one checkpoint family and PPO seed/configuration,
and enforces the canonical preprocessing, price grid, and usable event
schedule. All candidates receive the same 20 random commitments, Atari seeds,
and event schedules. Mechanically valid candidates are ranked by controlled
random payoff, then median, minimum, standard deviation, training step, and
checkpoint hash. In rank order, each candidate is tested on 100 disjoint random
episodes and a paired constant-context grid on the usable schedule
`20,50,80,110,140`. The selected alias is created only after the role-specific
behavioral gate passes. Buyer gates require close to five purchases and shots
through price 0.5, positive net payoff at every lower grid price, and lower
high-price demand. Buyer selection also includes a paired timing confirmation:
the first four free offers occur at `20,50,80,110`, the fifth occurs at either
140 or 195, and its price is 0.5, 0.75, or 0.9. Every one of the six conditions
uses the same 20 seeds. At price 0.75, forced fifth-buy and fifth-reject controls
calibrate that buying is better at step 140 and rejecting is better at step
195 by at least 0.15 mean payoff. The learned buyer must accept at least 75% of
the early offers and at most 25% of the late offers, reduce acceptance by at
least 50 percentage points, and remain within 0.15 mean payoff of the better
forced control in each schedule. The override changes only the fifth economic
coordinate; deterministic Atari actions are preserved, and every forced and
unforced episode receives the full accounting audit. This outcome-based test
replaces a raw early-minus-late threshold comparison because the threshold is
conditioned on the observed price and is therefore not uniquely interpretable
apart from its buy/reject consequence. Seller selection does not run or require
this buyer timing panel; seller gates require retained-bullet play at threshold
zero and near-threshold sales throughout the upper half of the grid. Full
episode/event rows, ranking, fixed-grid CSVs, paired-timing condition/episode/
event CSVs, exact hashes, explicit seeds, and all failed confirmation attempts
are written collision-safely under `replication/atari/results/e1_selections/`.

### Seller-v5 formal selection and E2 release

The shared-context exposure-v2 seller has a separate, fail-closed release
step. Its diagnostics gate authorizes formal training but is not itself an E1
checkpoint release. After the formal run completes, select from the five
retained checkpoints at `400160`, `800320`, `1200480`, `1600640`, and
`2000800` transitions plus the final base checkpoint with:

```bash
zsh replication/atari/automation/run_e1_seller_shared_context_v5_exposure_v2_selector.sh
```

The command first writes an immutable six-checkpoint family manifest, screens
all six byte-distinct candidates on the same 20 held-out seeds and contexts,
and confirms only the screen winner on 100 fresh seeds. It also reruns the
fixed-price grid, joint-context ablation, all eleven forced-price controls,
and the unchanged formal seller-v5 behavioral and conditioning gates. There is
no fallback to a lower-ranked candidate. A failed confirmation leaves neither
a selected alias nor an E2-visible gate. A passing confirmation publishes:

- `replication/atari/results/e1_selections/e1_seller_conditioning_recovery_v5_shared_context_exposure_v2_formal_family.json`;
- `replication/atari/results/e1_selections/e1_seller_conditioning_recovery_v5_shared_context_exposure_v2_all6_selector_v1.json`;
- `replication/atari/results/e1_selections/e1_seller_conditioning_recovery_v5_shared_context_exposure_v2_all6_selector_v1.gate.json`;
- `replication/atari/checkpoints/clean/meta_seller_e1_ppo_balanced_conditioning_recovery_v5_shared_context_exposure_v2_seed1_selected.zip`.

Together, the family, report, and release gate bind the exact candidate, E0b,
ROM, primary-buyer release, training-code, and selector-code provenance. A
rerun revalidates an existing family and passing gate and never overwrites an
artifact. For exposure-v2, the selector first revalidates the diagnostics gate
inside the exact `180c84f51005d2f36ae86f1cd7319c0e71249065` formal-training
runtime. That gate records the preregistered validator-only bridge from the
`c4a7dcd` diagnostic evidence and the shortened W&B job type
`atari_e1_seller_v5_exposure_v2_formal`.

Once that gate exists, the versioned v5 E2 pipeline can train and select both
leader roles sequentially:

```bash
zsh replication/atari/automation/run_atari_clean_e2_shared_context_v5_sequential.sh
```

The v5 launcher uses its own checkpoint namespace, cohort manifest, lock, and
Weights & Biases job types
`atari_e2_shared_context_v5_exposure_v2_buyer_leader` and
`atari_e2_shared_context_v5_exposure_v2_seller_leader`. It refuses to fall
back to a v3 seller after the v5 diagnostics gate becomes authoritative. The
shared cohort also proves that the buyer checkpoint used in E2 is exactly the
primary buyer release against which the selected seller was trained. For
manual recovery, the same profile is exposed through the individual scripts:

```bash
zsh replication/atari/automation/run_atari_clean_e2_buyer_shared_context_v5_2m.sh
zsh replication/atari/automation/run_e2_buyer_shared_context_v5_final_selector.sh
zsh replication/atari/automation/run_atari_clean_e2_seller_shared_context_v5_2m.sh
zsh replication/atari/automation/run_e2_seller_shared_context_v5_final_selector.sh
```

The unversioned E2 scripts retain the historical v3 profile by default; use
the explicit v5 wrappers above for the shared-context experiment. These are
execution commands, not claims that a run or scientific gate has completed.

## E2: Stackelberg leaders

Seller leader against a frozen E1 meta-buyer:

```bash
python -u -m replication.atari.train_atari_stackpomdp_leader_sb3 \
  --leader-role seller \
  --response-checkpoint replication/atari/checkpoints/clean/meta_buyer_e1_ppo_balanced_seed1_firefix_retrain_selected.zip \
  --leader-e1-checkpoint replication/atari/checkpoints/clean/meta_seller_e1_ppo_balanced_seed1_firefix_retrain_selected.zip \
  --actor-loss-mode balanced \
  --seed 1 \
  --timesteps 2000040 \
  --num-envs 4 \
  --n-steps 210 \
  --batch-size 840 \
  --checkpoint-every 400000 \
  --checkpoint replication/atari/checkpoints/clean/leader_seller_e2_ppo_balanced_seed1_firefix_retrain.zip \
  --wandb-name atari_clean_e2_seller_balanced_seed1_firefix_retrain_2m_local
```

For a buyer leader, swap the roles of the two E1 checkpoints. E2 rollouts are
exactly `5 + 200 + 5 = 210` transitions and always enable the full-action
cache callback. The generic wrapper is configured for five response
transitions and 205 reward transitions, with all five zero-reward queries kept
in PPO's rollout.

Every new E2 checkpoint embeds a canonical provenance manifest that binds the
exact frozen E1-response bytes, same-role E1 initialization bytes, ROM bytes,
policy architecture, reward-game protocol, and PPO configuration. Resume and
evaluation reject a missing or incompatible manifest; selection also rehashes
the candidate, frozen response, and ROM around evaluation so reported bytes
cannot drift away from the in-memory rollout. Select among retained
step checkpoints only with the deterministic selector:

```bash
python -u -m replication.atari.evaluate_atari_stackpomdp_leader_sb3 \
  --leader-role seller \
  --response-checkpoint replication/atari/checkpoints/clean/meta_buyer_e1_ppo_balanced_seed1_firefix_retrain_selected.zip \
  --checkpoint replication/atari/checkpoints/clean/leader_seller_e2_ppo_balanced_seed1_firefix_retrain_step400680.zip \
  --checkpoint replication/atari/checkpoints/clean/leader_seller_e2_ppo_balanced_seed1_firefix_retrain_step800520.zip \
  --checkpoint replication/atari/checkpoints/clean/leader_seller_e2_ppo_balanced_seed1_firefix_retrain_step1200360.zip \
  --checkpoint replication/atari/checkpoints/clean/leader_seller_e2_ppo_balanced_seed1_firefix_retrain_step1600200.zip \
  --checkpoint replication/atari/checkpoints/clean/leader_seller_e2_ppo_balanced_seed1_firefix_retrain_step2000040.zip \
  --checkpoint replication/atari/checkpoints/clean/leader_seller_e2_ppo_balanced_seed1_firefix_retrain.zip \
  --selected-checkpoint replication/atari/checkpoints/clean/leader_seller_e2_ppo_balanced_seed1_firefix_retrain_selected.zip
```

All candidates must share one provenance fingerprint and are screened on the
same 20 seeds and event schedules. Each factual rollout is paired with two
fresh counterfactuals that force the complete leader commitment to `0^5` and
`1^5`. The intervention changes only the economic coordinate at the five
queries and their five actor-identical cache replays. It does not directly
override gameplay actions; changed trades may causally change later gameplay
states and actions. A fresh frozen-follower response is recomputed from the
altered full sequence. The endpoint comparison is this five-bullet experiment's
predeclared **interior-surplus hypothesis**, not a universal equilibrium
condition. A candidate is eligible only if all three protocol audits pass, its
mean payoff is at least `0.25`, the two players fire at least four bullets in
total, and it beats each endpoint by mean payoff `0.25`, nonnegative median
payoff, and a strict paired win rate of at least `60%`. Purchases, payments,
accepted price, and buyer/seller bullet-utilization measures remain descriptive.
The audit also binds every retained economic-decision row to its complete
transition row and verifies that an accepted trade pays `+p` immediately to a
seller leader or `-p` immediately to a buyer leader, with zero on rejection.

Only the screen-preselected top candidate is confirmed on 100 disjoint matched
seeds. If it fails, selection fails without trying another checkpoint on the
confirmation set. The selected alias is created only after confirmation passes,
and is rolled back if artifact writing then fails while its bytes still match
the newly created copy. The selector writes factual and
counterfactual condition summaries, paired seed rows, episode, transition,
decision, and event CSVs and exact intervention/provenance hashes under
`replication/atari/results/e2_selections/`.
The complete set is staged privately and published without overwrite under an
exclusive per-run lock, with the JSON report linked last as its completion
marker. It never overwrites an existing selected alias or report.

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

- `CHECKPOINT.training.jsonl`: completed-episode and phase-optimizer metrics;
- `CHECKPOINT.evaluation.json`: deterministic evaluation rows and summaries;
- E1 only, `CHECKPOINT.fixed_contexts.csv`: paired fixed-context table;
- step checkpoints at completed outer-episode boundaries.

For E0a, `--timesteps` counts the complete run clock after resume. A resumed
run can retain the same W&B URL with `--wandb-id RUN_ID --wandb-resume must`.

W&B logs episode payoff and length, role-correct game reward, shots, ammo,
reward per bullet, payments, purchases, all five event prices/thresholds/
acceptance times, and early/middle/late trade summaries. The timing summaries
always log each bin's event count and, for nonempty bins, its acceptance rate,
mean threshold, and mean price. The bins use normalized gameplay time
`[0,1/3)`, `[1/3,2/3)`, and `[2/3,1]`. W&B also logs true-game-over reset
counts/rates, time-limit reset counts, whether a true game-over occurred before
the fifth event, total timesteps, learning rate, seed, algorithm, and
checkpoint path. Evaluation JSON rows retain the exact policy-step indices at
which each local reset occurred.

## Validation

```bash
PYTHONNOUSERSITE=1 \
python -c 'import sys; sys.modules["readline"] = None; import pytest; raise SystemExit(pytest.main(["-q", "tests/test_atari_clean_gameplay_terminal.py", "tests/test_atari_clean_e0_trainer.py", "tests/test_atari_clean_protocol.py", "tests/test_atari_clean_envs.py", "tests/test_atari_phase_balanced_ppo.py", "tests/test_atari_clean_meta_response_trainer.py", "tests/test_atari_clean_e1_evaluator.py", "tests/test_atari_clean_leader_trainer.py", "tests/test_atari_clean_e0b_evaluator.py", "tests/test_atari_clean_e2_evaluator.py"]))'
```

The clean suite checks the stable 14D interface, branch-gradient isolation,
event-only leader invariance, exact full-action cache reuse, zero actor credit
on cached replays, per-vector-row cache reset, at most one registered shot per
max-and-skip decision, atomic trade accounting, complete E1/E2 horizons, fresh
critics, independent local true-game-over resets with preserved outer
accounting, actor transfer, and optimizer checkpoint reloadability.
It also checks phase-balanced active-row means, cached-row actor isolation,
credit-gate validation, full-rollout batching, per-head KL diagnostics, and a
real composite-policy optimization step.

## Current clean-run status

E0a is complete. The selected 3.2M-step checkpoint passed two consecutive
deterministic 20-episode screens and an independent 100-episode confirmation.
All 100 confirmation episodes achieved clipped reward 5, fired exactly five
bullets, ended with zero ammo, completed 200 gameplay transitions, and had
exact bullet and payment accounting.

- selected checkpoint: `replication/atari/checkpoints/clean/space_invaders_e0a_ppo_seed1_target.zip`
- selection manifest: `replication/atari/checkpoints/clean/space_invaders_e0a_ppo_seed1.target_selection.json`
- confirmation result: `replication/atari/checkpoints/clean/space_invaders_e0a_ppo_seed1_best.confirmation_1.json`
- W&B validation: <https://wandb.ai/glcbrero/StackPOMDP/runs/l6r0f0d1>
- source training run: <https://wandb.ai/glcbrero/StackPOMDP/runs/dyaly9m7>

The 50M cap was only a safety ceiling. Its extension was stopped after the
FIRE-forwarding correction made the already-saved 3.2M checkpoint pass the
full target gate; the durable 6.4M checkpoint was retained for provenance.
