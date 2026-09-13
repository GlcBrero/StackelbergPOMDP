# Reruns needed for a matching submission

**September 10 update:** the approved 210 main runs and appendix leaders have
completed; their outcomes are recorded in the workspace experiment tracker.
The user excluded the critic flag as a rerun trigger. A subsequent optimizer
audit found broader recipe drift, and the user requested defaults unless
needed. Current targets now use the [defaults policy](PARAMETERS.md), which
differs from the frozen completed cohort. No new jobs have been submitted for
that recipe. The original rationale below records the earlier recommendations;
it is not a statement that the completed runs used today's target arguments.

This list is based on the current `targets.json`, the paper's empirical
figures, and the corrected implementation. It is a plan for new evidence:
unit tests and regeneration of archived plots do not show that corrected
training preserves the published conclusions. No full scientific rerun was
started during the code cleanup or appendix-support work.

## Training to rerun

| Cohort | Scope | Reason |
|---|---|---|
| Simple Allocation | Six distinct configurations × 25 seeds; the three-message MAPPO curve is shared between figures | The MAPPO variants use the corrected critic phase flag. Basic-POMDP variants are affected by the earlier excluded-query episode-boundary correction. The older constructor fix also changes private-type RNG seeding. |
| Matrix Design | Recommend the complete four-variant comparison × 25 seeds | Both MAPPO variants use the corrected flag; both Basic-POMDP variants use the corrected episode-boundary handling. Standard PPO is a control without either change; it could be retained if old/current parity and artifact provenance are established. |
| MSPM | Two-type cohort: 25 seeds; scaling: 25 seeds for each of 3–6 types | All five targets use `critic_obs=full`, so training sees the corrected first reward-phase flag. Fixed-seed MW and exact payoff evaluation do not establish invariance of the learned critic or policy. |
| Standard SPM baseline | 10 seeds, paired with the two-type MSPM comparison | No response phase or critic flag, but its `BaseSPM` constructor was among those that previously discarded the supplied seed. New training follows the declared private-type sampling seed. |
| Learned Bertrand platform | 25 platform-training seeds and subsequent policy evaluation | `fig_collusion_learning_state` uses `critic_obs=full`. The first reward-phase critic observation changes. |
| Three normal-form diagnostics | 10 REINFORCE follower models, 60 commitment-consistency leaders, 18 reset leaders, 40 reward-timing leaders | The maintained PG/SimpleQ recipes now support all three, with explicit differences from historical training. Assess the resulting curves before replacing the archived data. |

These are full-cohort recommendations, not evidence that every changed input
alters a conclusion. A narrower rerun is defensible only after demonstrating
that the specific omitted configuration is unaffected. Avoid duplicating
shared Simple Allocation runs when rebuilding its main and appendix panels.

## No retraining trigger identified from these fixes

- **Atari E0/E1/E2:** the generic critic phase flag is disabled in E2, and the
  canonical before/after evaluation traces match. Retain the checkpoints and
  recheck the final artifact; no Atari retraining requirement has been
  identified from these fixes. This does not prove parity for every possible
  training trajectory or replace scientific review of the existing cohort.
- **Fixed Bertrand calibration and deviation experiments:** they do not train
  the StackPOMDP critic. Reuse their archived data/checkpoints after verifying
  provenance; rerun learned-platform evaluation against its new checkpoints.
- **Analytical SPM optima:** recompute the table and statistics alongside new
  MSPM data; there is no analytical training run.

## What each correction changes

- Seed forwarding (`6dd369b`): five domain constructors changed positional
  `BaseEnv` calls to explicit `game=`, `logger=`, and `seed=` arguments.
  `BaseSPM` is included; its sampled buyer types depend on the game RNG.
- Hidden-query boundary handling (`6507945`): excluded response prefixes no
  longer lose the start of an outer episode in the training buffer.
- Critic phase flag (current working tree): the observation preceding the
  first reward-phase action now reports `critic:is_reward_step=1`.
- Appendix training: [LEGACY_PROVENANCE.md](matrix_ablations/LEGACY_PROVENANCE.md)
  records state encoding, reward scale, action caching, evaluation, and learner
  differences. The new algorithms are maintained implementations rather than
  exact historical random-stream replays.

Freeze the code used for the reruns, retain resolved configurations and source
hashes, compare the complete seed cohorts, and then publish matching code,
manuscript, data, and checkpoint versions. Cluster launch files are prepared;
this document records no submitted jobs.
