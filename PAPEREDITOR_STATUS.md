# Papereditor Status

- Timestamp: 2026-07-27 (America/New_York).
- Phase: clean Atari E0a--E2 implementation validated; long E0a rerun active locally.
- Current design: one Stable-Baselines3 composite policy and one 14-scalar actor state from E0a through E2, branch-specific PPO action credit, an event-only E2 leader economic path, exact full-action query replay, and a fresh private critic at every stage.
- Verification: all 46 repository tests pass; clean modules compile; real-ROM E0a, E0b, buyer/seller E1, and seller E2 mechanics smokes passed with complete horizons and exact bullet/payment accounting.
- Preservation: the complete pre-clean Atari tree, including uncommitted trainers, tests, logs, and tracked result metadata, is archived under `Research Artifacts/legacy_atari_prototype_20260727/` outside the importable source tree.
- Active run: E0a seed 1, 10M steps, four local vector environments; W&B `dyaly9m7` in project `StackPOMDP`, group `atari_clean_curriculum`, job type `atari_e0a`. Startup reached 2,400 steps without error.
- Evidence boundary: old RLlib, frozen-gameplay, stochastic-timing, and interrupted leader pilots are historical diagnostics only. The active E0a run is not yet a declared result.
- Next action: monitor E0a and retain it only if deterministic evaluation passes, then continue E0b, both E1 roles, and both E2 leader roles in order.

## Prior certified-MW status

- Timestamp: 2026-07-15 (America/New_York).
- Phase: local validation of the clean certified-MW MSPM implementation.
- Current design: store a fixed response prefix, then run complete MW updates outside the PPO buffer until one of the MW-owned candidates is an exact `0.05`-BCCE. There is no failure-reward penalty.
- Ownership: `CertifiedMWResponse` contains all MW/BCCE/history/candidate logic; `MWFollowersWrapper` is a thin Gym adapter; `StackPOMDPWrapper` remains solution-concept and reward agnostic.
- MW update: enumerate the four joint message profiles and update from expected utilities. Only private-type sampling remains random, and MSPM restarts it at fixed seed 0 by default.
- Reward evaluation: exact expectation over type profiles and the selected response's joint message distribution. Correlated empirical mixtures retain their shared snapshot index.
- Rollout safety: response-100 PI uses a conservative 232 stored transitions. Adaptive tail transitions count toward global training time but are excluded from PPO storage. Gamma remains 1.
- W&B contract: only deterministic evaluation `reward` at `global_step`.
- Verification: 15 passing regression tests, clean whitespace diff, efficient-PI and historical learned-3-type MW parameter sweeps, and one successful real 232-step PPO rollout.
- Parameter result: keep `mw_epsilon=0.1`. Values 0.02--0.1 were equivalent at the intended prefixes; 0.2/0.5 could select a lower-reward mixed certified response first.
- Existing Unity cohort warning: array `61831884` was launched with the superseded fixed-horizon `-1` semantics. It remains historical diagnostic evidence and is not a validation of the new code.
- Existing worktree: unrelated Atari work and prior artifacts remain untouched.
- Current blocker: none locally. New paper simulations require a scoped commit/sync and explicit launch approval.
- Next action: review the certified-response diff, commit it, then prepare replacement PI/MSGSpace runs only after approval.

## Matrix qualitative revalidation

- Timestamp: 2026-07-28 (America/New_York).
- Phase: the hidden-query PG cohort is complete and native RLlib 2.0.1 ES replacement array `62490130` is running 25 paired observed/hidden seeds from immutable source `b37703c`. The earlier custom mirrored-ES and independent-Q branches remain diagnostic provenance only.
- Independent Q: Unity array `62361190` completed all 60/60 cells. Across ten seeds, means $\pm$ sample SEM were: q-reset reset `1.460 +/- 0.000` versus ongoing `1.300 +/- 0.153`, paired reset-minus-ongoing `+0.160 +/- 0.153`; response-reward with zero miscoordination penalty, excluded `2.000 +/- 0.000` versus included `1.800 +/- 0.133`, paired excluded-minus-included `+0.200 +/- 0.133`; with the penalty, excluded `1.900 +/- 0.100` versus included `1.300 +/- 0.153`, paired `+0.600 +/- 0.221`.
- Interpretation: the clean q-reset separation is weak. The response-reward direction is clearest with the miscoordination penalty. These diagnostics are not paper evidence.
- E1 certification: original array `62361189` completed all ten responses, but every response failed the gate with maximum regret `2`--`3`; gate job `62361192` therefore blocked release, and meta-dependent array `62361194` plus plot job `62361196` were cancelled without starting.
- E1 diagnostics: PPO/A2C array `62362084` completed 8/8 treatments but produced 0 strict passes among 50 retained checkpoints. DQN array `62362864` completed 8/8 tasks cleanly but also produced 0 strict passes. Its best checkpoint was 150k steps, learning rate `0.001`, 32x32 network: mean regret `0.09375`, maximum regret `1`, and 29/32 optimal commitments. No E2/meta-dependent experiment is released.
- Q-reset diagnostic and forensics: policy-gradient array `62362532` completed 8/8 treatments, but all reset and ongoing cells ended at reward `2.0`, so it showed no separation. Historical q-reset runs predate terminal-Q bugfix `f6189ac`; their terminal bootstrap likely manufactured sticky carryover. That behavior will not be reproduced or promoted.
- Hidden-query response: the exposure-matched clean REINFORCE response seed 2 passed the strict exact gate on all 32 deterministic commitments; checkpoint SHA-256 is `e76ea2a593de80a93056911dfc7fcf1755544b15a674318aa9a1fa957ca7eb80`. Reusing this one immutable certified response isolates uncertainty to independent leader-training seeds.
- Hidden-query protocol: vanilla bias-free linear PG runs 200,000 executed transitions at LR `.008`. The replacement ES treatment uses native `ray.rllib.algorithms.es.ES` from Ray 2.0.1 in the dedicated `stackelberg-pomdp-ray-es` environment: one worker, the default 2x256 tanh policy, `MeanStdFilter`, sigma `.02`, Adam stepsize `.01`, L2 `.005`, 1,000 episodes per batch, and 300 optimizer iterations. Both use Modified PD, joint five-state memory, and the same certified response; the historical Matthias checkout is provenance-only.
- Superseded diagnostic outcome: Unity arrays `62368442` and `62368796` completed all 40/40 custom PG/ES cells. Their ten-seed PG result remains available, but their hand-written mirrored-ES result is not the maintained ES implementation and is not paper evidence. Its raw manifests, hashes, and tracker entries remain immutable provenance.
- Native ES release gate: the dedicated 25-task array must complete both observed and hidden conditions for seeds 1--25 (50 artifacts), pass manifest/checkpoint/response-lookup audits, and show the qualitative visibility-invariant ES trend before the figure is promoted. Curves use sample SEM across independent seeds.
- Evidence boundary: no native ES claim or replacement figure is released before that gate. The weak Q diagnostics, custom ES cohorts, ARS diagnostic, and historically inconsistent hidden-query artifacts remain unpromoted provenance.
- Current blocker: native RLlib ES array `62490130` must finish all 50 runs and pass the release audit. Its first scientific rows are healthy: all 25 observed runs have progress and the across-seed mean improves by iteration 25. Other normal-form diagnostics require their own clean protocols and evidence decisions.
- Next action: monitor the observed-to-hidden crossover, retrieve and audit all artifacts, and update the paper figure/log bundle only if the visibility-invariant ES gate passes.
