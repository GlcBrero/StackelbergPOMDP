# Autonomous Run Report

## 2026-07-15 — Certified adaptive MW response refactor

- Replaced the fixed-horizon BCCE penalty with adaptive certified stopping.
  StackPOMDP stores the configured response prefix, then continues complete MW
  updates outside the PPO buffer until a response has exact BCCE gap at most
  `0.05`. A configurable extra-update cap raises a clear runtime error; the
  implementation never substitutes an artificial `-1` reward.
- Kept `StackPOMDPWrapper` solution-concept agnostic. It only asks the follower
  adapter whether its response is ready. `CertifiedMWResponse` owns MW state,
  certification, candidate priority, the empirical trajectory, the selected
  response, and diagnostics. `MWFollowersWrapper` is limited to Gym lifecycle,
  payoff-context, seed, action, and critic-state adaptation.
- Certification checks, in order, the current mixed MW strategy, its
  deterministic argmax projection, and the correlated uniform mixture of all
  MW snapshots. The empirical candidate preserves the shared snapshot index
  rather than independently averaging follower marginals.
- Removed sampled reference-message profiles from MW. Each update enumerates
  all four joint messages in the two-buyer/two-message experiments and updates
  from exact expected utilities under the opponents' current weights. The
  remaining private-type sequence is restarted with `mw_fixed_seed=0` by
  default for MSPM.
- Randomized certified responses are evaluated exactly over type profiles and
  joint message profiles. Type probability is applied in `BaseMessageSPM`; a
  generic wrapper above StackPOMDP applies the response probability. The joint
  exact weights sum to one.
- The conservative PI PPO horizon with response 100 is now 232 stored
  transitions: `2 * (100 + 4 type profiles * 4 joint messages)`. Any adaptive
  certification tail is executed and counted in `global_step` but excluded
  from the rollout buffer. Gamma remains the economically motivated `1.0`.
- W&B's paper-facing contract is again only `reward` indexed by `global_step`;
  response diagnostics remain in local output.
- Fifteen PI regression tests pass. They cover architecture ownership,
  adaptive tail exclusion, exact joint-message MW updates, fixed-seed
  reproducibility, randomized exact reward weights, correlated empirical
  averaging, the fail-loud cap, counterfactual state isolation, policy caching,
  exact evaluation, and rollout geometry. `git diff --check` passes.
- A real one-rollout PPO smoke (seed 9917, response 100) stored all 232
  transitions, completed a full StackPOMDP episode, and reported exact reward
  `-0.06` without an exception.
- MW parameter smokes used the efficient PI mechanism and three fixed type
  seeds. At response prefixes 100 and 300, `epsilon` values 0.02, 0.05, and
  0.1 all certified without an extra update and returned reward 0. A historical
  3-type learned mechanism showed the same equivalence at prefixes 100/300.
  Larger values 0.2/0.5 sometimes certified the mixed response first and
  lowered leader reward, so the compatible default `epsilon=0.1` was retained.
- No Unity job was submitted or modified. Existing array `61831884` uses the
  older fixed-horizon `-1` implementation and must not be presented as evidence
  for this new certified-response code.
- TODO/GB comments: none modified or removed.

## 2026-07-14 — MSGSpace 3--6 type BCCE-constrained scaling cohort

- Implemented a hard MSPM feasibility rule controlled by
  `response_bcce_failure_reward`. At the fixed response boundary, the BCCE
  checker evaluates the single deterministic projection of the final MW
  weights. A gap above `0.05` suppresses all exact-profile rewards and emits
  one terminal `-1`, so the complete episode return is exactly `-1`.
- Extended exact evaluation logging with `raw_reward`,
  `allocative_efficiency`, `bcce_gap`, `bcce_certified`, `best_reward`, and
  `best_allocative_efficiency`. MSPM best metrics update only on certified
  evaluations; SPM uses the same current/best reward and efficiency fields
  without a BCCE condition.
- Added nine regression tests in total. New assertions cover failed and
  certified returns, deterministic one-hot last-iterate MW responses, raw
  reward preservation, and best-certified metric updates. All tests pass.
- Completed an actual 3-type constrained training/evaluation smoke. It
  exercised both the training penalty and deterministic BCCE-aware evaluation
  with no exception.
- Added 24-task manifest
  `replication/cluster/msgspace_spm_vs_mspm_bcce005_3to6types_3seeds_1m.sbatch`.
  For each type count 3--6 it runs MSPM seeds 1--3 and SPM seeds 1--3.
- Response games scale quadratically as `300, 532, 832, 1200`, preserving
  roughly 8.3 complete MW updates per two-buyer type profile. Exact reward
  phases contain `9, 16, 25, 36` profiles.
- Local verification: Bash syntax, all 24 task mappings, response alignment,
  reward-profile lengths, conservative rollout horizons, all nine tests, and
  `git diff --check` pass.
- Planned group:
  `msgspace_spm_vs_mspm_bcce005_3to6types_3seeds_1m_20260714`.
- MSGSpace-specific W&B job types are
  `msgspace_mspm_argmax_bcce005_eval` and `msgspace_spm_argmax_eval`, keeping
  this type-scaling cohort separate from the earlier PI paper cohort.
- Required pre-submit queue audit found 4 prior PI paper jobs running and 6
  unrelated Atari jobs pending. Synced only the scoped implementation, tests,
  and manifest. Manifest syntax and all nine tests pass on Unity.
- Submitted the author-approved 24-task array `61831884`; tasks 0--23 all
  entered the queue. Remote logs are under
  `simulation_logs/cluster_msgspace_bcce005_3to6types_3seeds_1m_20260714/`.
- Tasks 0--17 started cleanly and initialized 18 W&B runs: nine MSPM and nine
  SPM, covering all 3-, 4-, and 5-type cells. The six 6-type tasks remain
  pending for scheduler priority. No actual runtime failure appears.
- All 18 initialized runs were relabeled with MSGSpace-specific job types;
  W&B now reports nine runs under each new label and leaves the generic
  `spm_argmax_eval` label exclusively on the ten earlier PI runs.
- A live 4-type MSPM evaluation verified the hard constraint at step 120k:
  raw reward `-0.0833333` and BCCE gap `0.0966941` produced constrained reward
  `-1` and certification `0`, while the prior certified best stayed at
  `-0.0208333` and best efficiency at `0.979167`.
- W&B IDs for types 3/4/5 are, respectively, MSPM
  (`4y0brd12`, `vt4c8gsg`, `wtfufmll`), (`zvadyaj2`, `40wji8v9`,
  `q3ns2e61`), (`y2v1kzl3`, `9new6ztq`, `32pxc7by`); and SPM
  (`rglq3ro2`, `9d0rgvie`, `j0mmyjya`), (`imppe8ws`, `0xhc5qau`,
  `inmis2vx`), (`davyvq1l`, `48sni02d`, `9iqjgspq`).
- TODO/GB comments: none modified or removed.

## 2026-07-14 — Final PI 10+10 exact-evaluation paper cohort

- At the author's direction, canceled entropy-0.01 tasks 3--5 of array
  `61826647` after 32 minutes and deleted their W&B runs `s8ucsccs`,
  `6p0k6sba`, and `w0fzq45a`. Entropy-0 tasks 0--2 were preserved.
- Created W&B group
  `pi_spm_vs_mspm_exact_eval_10seeds_1m_paper_20260714`. Audited and moved
  compatible MSPM seeds 2--4 (`mdtt0ico`, `r73jkd3z`, `gyovzm3k`) and SPM
  seed 1 (`ssqn276r`) into it.
- Relabeled the group to contain exactly two job types:
  `mspm_argmax_eval` and `spm_argmax_eval`. A W&B API audit confirms ten
  unique seeds (1--10) under each type and zero runs in the obsolete entropy
  group. A project-wide audit likewise finds zero runs carrying either old
  job type `mspm_ent001` or `mspm_ent0`.
- Added
  `replication/cluster/pi_spm_vs_mspm_exact_eval_10seeds_1m_paper.sbatch`.
  Tasks 0--6 run MSPM seeds `1, 5, 6, 7, 8, 9, 10`; tasks 7--15 run SPM
  seeds `2, 3, 4, 5, 6, 7, 8, 9, 10`. Combined with the reused runs, these
  are ten unique seeds per method.
- All runs use a 1M-step horizon and deterministic exact argmax evaluation
  every 10k steps. MSPM evaluates a response-100 StackPOMDP episode followed
  by four weighted exact profiles; SPM enumerates the four profiles directly.
- Bash syntax and task mapping pass locally. All seven PI regression tests
  pass locally and on Unity. Only the evaluation code, test, and manifest were
  synced; unrelated worktree artifacts were not transferred.
- Required queue check before submission: 3 retained PI tasks running and 6
  unrelated Atari tasks pending. Submitted the author-approved 16-task array
  `61830595`; all tasks entered the queue and all 16 subsequently started.
- MSPM W&B IDs by seed 1--10: `06a35u28`, `mdtt0ico`, `r73jkd3z`,
  `gyovzm3k`, `vxlssid5`, `btherc9n`, `ux979uja`, `6i7culmf`, `prd7hqe6`,
  `k40l98wk`. SPM IDs by seed 1--10: `ssqn276r`, `bskzgr6y`, `svtplarz`,
  `e2sumbbb`, `4d6kgvc6`, `v5pyuvhq`, `24g8rmec`, `593405t6`, `qzwj8935`,
  `bzig3t3r`.
- All 20 runs produced exact-evaluation metrics. At the startup audit, MSPM
  seeds 1, 2, 4, 5, and 10 were at reward `0`, with the other MSPM seeds at
  `-0.06`; all SPM seeds were at `-0.06`. All 16 output files exist, and a
  targeted stderr scan found no traceback, exception, runtime error, kill, or
  out-of-memory event.
- Remote logs:
  `simulation_logs/cluster_pi_paper_exact_eval_10x10_1m_20260714/slurm_61830595_<task>.{out,err}`.
- TODO/GB comments: none modified or removed.

## 2026-07-14 — PI MSPM entropy and seed diagnostic

- Launched six approved 1M-step MSPM jobs with response 100 and exact argmax
  evaluation every 10k steps. Tasks 0--2 use `ent_coef=0`, seeds 2--4;
  tasks 3--5 use `ent_coef=0.01`, seeds 1--3.
- Canceled initial array `61826028` immediately after detecting that entropy
  was absent from experiment names, which allowed same-seed tasks in the two
  arms to collide on log/checkpoint paths. Added entropy to SPM and MSPM
  experiment names and added a regression test for path separation. Deleted
  the six resulting short W&B stubs; an API audit then confirmed that the
  sweep group contains exactly the six clean replacement runs.
- All seven PI tests pass locally and on Unity. Submitted clean replacement
  array `61826647`; all six tasks are running with distinct paths and have
  emitted exact-evaluation records without startup or runtime errors.
- W&B group: `pi_mspm_entropy_seed_sweep_1m_20260714`. `ent_coef=0` run IDs
  are `mdtt0ico`, `r73jkd3z`, and `gyovzm3k`; `ent_coef=0.01` run IDs are
  `s8ucsccs`, `6p0k6sba`, and `w0fzq45a`.
- W&B API verification confirms all six are `running`, carry the correct job
  type (`mspm_ent0` or `mspm_ent001`), and log evaluation `reward` against
  `global_step`. Initial 10k rewards are respectively `-0.08`, `-0.06`,
  `-0.08` and `-0.16`, `-0.06`, `-0.06`; these are startup checks, not
  convergence comparisons.
- Subsequent author decision: the entropy-0.01 arm was not promising. Tasks
  3--5 were canceled and their three W&B runs deleted; only entropy-0 seeds
  2--4 are retained for the final paper cohort.
- Manifest:
  `replication/cluster/pi_mspm_entropy_seed_sweep_1m.sbatch`.
- TODO/GB comments: none modified or removed.

## 2026-07-14 — One-million-step evaluation-only PI validation

- After author approval, synced only the evaluation-logging code, regression
  test, and `replication/cluster/pi_spm_vs_mspm_eval_only_1m.sbatch` to Unity.
  Unrelated local artifacts were not transferred.
- Verified the manifest and all six PI tests in Unity's
  `stackelberg-pomdp` environment before submission.
- Submitted two-task SLURM array `61821006`: task 0 is MSPM seed 1 and task 1
  is SPM seed 1. Both train for 1M steps and evaluate the argmax policy every
  10k steps.
- Both tasks started on `uri-cpu004`. W&B runs are MSPM `wfhubc2x` and SPM
  `ssqn276r`, under group `pi_eval_only_1m_20260714` with distinct job types.
- A read-only W&B API audit confirms that reward rows use `global_step` and
  come from evaluation. MSPM logged `-0.08` through its first 30k steps; SPM
  reached its exact `-0.06` optimum by 20k and held it through at least 60k.
- Remote logs:
  `simulation_logs/cluster_pi_eval_only_1m_20260714/slurm_61821006_{0,1}.{out,err}`.
- Status: both jobs running; no startup traceback or runtime error.
- Monitoring at approximately 17 minutes: SPM was near 890k and had held the
  exact `-0.06` optimum since 20k. MSPM was near 510k; it changed from `-0.08`
  to `-0.06` at 50k and then stayed there for 47 consecutive evaluations,
  never reaching reward `0`. Both jobs remained healthy. This points to a
  genuine MSPM local optimum under `ent_coef=0`, not another logging failure;
  mechanism/message inspection awaits the final checkpoint.
- TODO/GB comments: none modified or removed.

## 2026-07-14 — Evaluation-only PI reward logging

- Changed the W&B contract so `reward` is written only by deterministic
  evaluation callbacks and indexed by `global_step`. Sampled training rewards
  and response diagnostics remain visible in local logs but are not uploaded.
- SPM evaluates the argmax policy exactly over all four PI type profiles.
  MSPM evaluates a copied argmax leader policy in a separate StackPOMDP
  episode and sums its four weighted exact reward profiles after the response
  phase. Background evaluations now retain their launch step rather than
  reading a later training step when the thread finishes.
- Removed TensorBoard auto-patching and the unused SB3 W&B callback import so
  custom evaluation logging is the sole metric path.
- Online SPM smoke `rjdqvx4g` produced nine W&B points, all `-0.08`, while its
  local sampled training diagnostic was `-0.4` at step 1,496.
- Online MSPM smoke `tkrq2z2k` produced four W&B points at steps 104, 208, 312,
  and 416, all `-0.08`, while local exact training episodes were `-0.06`.
- Logs:
  `simulation_logs/pi_evaluation_logging_smoke_20260714/{spm,mspm}.log`.
- Verification command:
  `PYTHONNOUSERSITE=1 MPLCONFIGDIR=/private/tmp/mplconfig /Users/gbrero/miniconda3/envs/stackelberg-pomdp/bin/python -m unittest tests.test_mspm_pi -v`.
  All six tests pass, including direct assertions that both evaluators use
  deterministic predictions and send exact reward `-0.08` at the captured
  step.
- Scientific consequence: the previous figure compared exact MSPM rewards to
  sparse realized SPM training outcomes. It must be replaced by fresh runs
  using this common evaluation protocol before inclusion in the paper.
- TODO/GB comments: none modified or removed.

## 2026-07-14 — PI SPM versus MSPM best-three moving-average variant

- Kept the common 0--5M horizon and ranked each method's ten seeds by mean
  unsmoothed reward over the final 10% (4.5M--5M).
- Retained MSPM seeds `1, 2, 6` and SPM seeds `4, 3, 2`, in descending rank
  order. Their mean ranking-window rewards are `-0.00456` and `-0.04768`.
- Plotted the full selected-seed trajectories using a centered 250,000-step
  moving average, with plus/minus one standard error across the three smoothed
  seed series. The best-five figure remains unchanged and available.
- Added the exact optimal-SPM reward `-0.06` as a dashed orange benchmark.
  Exhaustive enumeration of threshold-equivalent price regimes confirms this
  value: visit A1 first at a price below `0.4`, then A0 after no sale below
  `0.2`; only the both-high profile is inefficient, contributing
  `0.2 * 0.5 * (1.0 - 0.4) = 0.06` expected welfare loss.
- Outputs: `output/pdf/spm_vs_mspm_pi_reward_best3of10_5m.{pdf,png,csv,json}`
  and `output/pdf/spm_vs_mspm_pi_reward_best3of10_5m_rankings.csv`; identical
  PDF and PNG copies are on the Desktop.
- Regenerated on 2026-07-14 at 10:54 America/New_York from the same source logs
  and locked configuration; the refreshed Desktop copies match the repository
  artifacts by SHA-256.
- Verification: selection/window assertions, complete-bin checks, script
  compilation, PDF metadata and embedded-font inspection, Poppler rendering,
  visual inspection, `git diff --check`, and repository/Desktop SHA-256
  equality checks all pass.
- Reporting constraint: a paper caption must disclose best three of ten,
  ranking on the final 10%, and the 250k-step centered moving average. The
  curves use realized training rewards and post-selection, so they may cross
  the exact `-0.06` SPM benchmark without implying that a policy beats it.
- TODO/GB comments: none modified or removed.

## 2026-07-14 — PI SPM versus MSPM best-five-of-ten figure

- Task: compare PI MSPM and SPM through the first 5M training steps, retaining
  the best five of ten seeds per method in line with the earlier best-seed
  presentation convention.
- Sources: all 20 local copies of Unity array `61798498`; tasks 0--9 are MSPM
  seeds 1--10 and tasks 10--19 are SPM seeds 1--10. Every run populates all 500
  fixed 10,000-step bins through 5M. MSPM seed 4 failed only after 6.15M and is
  therefore complete for this comparison; it ranks ninth and is not selected.
- Ranking: descending mean unsmoothed binned reward over the final 5% of the
  common displayed horizon (4.75M--5M), computed separately within each
  method. Selected MSPM seeds are `2, 1, 6, 10, 9`; selected SPM seeds are
  `9, 3, 2, 4, 10`.
- Processing: 100,000-step centered moving average per seed; plotted line is
  the selected-seed mean and the translucent band is plus/minus one standard
  error. The visual style follows the project's paper notebooks and accepted
  seed-1 comparison.
- Result: selected MSPM runs approach reward zero within roughly 1M steps. In
  the unsmoothed 250k-step ranking window, their mean is `-0.00576`, compared
  with `-0.04503` for selected SPM runs.
- Reproduction: `tools/plot_pi_spm_vs_mspm_top_seeds.py`. Outputs are
  `output/pdf/spm_vs_mspm_pi_reward_best5of10_5m.{pdf,png,csv,json}` plus
  `output/pdf/spm_vs_mspm_pi_reward_best5of10_5m_rankings.csv`.
- Delivery: identical PDF and 300-dpi PNG copies are on the Desktop.
- Verification: compile/help smoke tests, complete-bin checks, PDF metadata and
  embedded-font inspection, Poppler rendering plus visual inspection, and
  repository/Desktop SHA-256 equality checks all pass.
- Reporting constraint: any paper caption using this figure must explicitly
  state that it shows the best five of ten seeds per method, ranked on mean
  performance over the final 5% of the displayed training horizon.
- TODO/GB comments: none modified or removed.

## 2026-07-13 — PI SPM versus MSPM ten-seed Unity launch

- Working MSPM implementation committed as `d8a709a`; reproducible Unity array
  manifest committed as `d6dcbc1`.
- Submitted SLURM array `61798498` with 20 tasks: tasks 0--9 are MSPM seeds
  1--10 and tasks 10--19 are SPM seeds 1--10, each for 10M training steps.
- W&B group: `pi_spm_vs_mspm_10seeds_20260713`. MSPM run IDs are
  `3vqh4jad`, `9ic017js`, `wtg9nzex`, `o8kt1bgy`, `d5hnhwmo`, `ayzjxfxx`,
  `a6rtlw1p`, `esz18gc4`, `nxee73a9`, and `61rzh15c`. SPM run IDs are
  `2kzm425e`, `nctxnwlv`, `zj9xq68m`, `ntvmo7u9`, `tdr60qek`, `0famcbr5`,
  `7aqpjy6x`, `a4grx2bi`, `flczyzr5`, and `l05qz1cd`.
- Startup verification: all 20 tasks entered training, all 20 initialized W&B,
  and no task log contained a traceback or import/startup error.
- Unity logs:
  `simulation_logs/cluster_pi_spm_vs_mspm_10seeds/slurm_61798498_<task>.{out,err}`.
- Next: monitor completion, retrieve logs/results, aggregate ten seeds per
  method, and regenerate the paper figure with cross-seed uncertainty.

## 2026-07-13 — PI SPM versus MSPM paper figure

- Task: create a paper-style training-step versus leader-reward comparison and place it on the Desktop.
- Sources: clean seed-1 SPM run `6ijiy88e` from `simulation_logs/pi_training_plot/spm_pi_seed1_10m_reward_only.log`; clean seed-1 MSPM run `5xhgdpgt` from `simulation_logs/pi_training_plot/mspm_pi_2messages_seed1_10m_response100_defaultppo_gamma1_earlytermination.log`.
- Processing: stream reward rows into 10,000-step bins; plot a faint binned trace and a centered 100,000-step moving average; crop both series to a fixed shared 1.4M-step horizon. No uncertainty band is shown because there is one seed per method.
- Script: `tools/plot_pi_spm_vs_mspm.py`.
- Results: `output/pdf/spm_vs_mspm_pi_reward_seed1.pdf`, `.png`, `.csv`, and `.json`; matching PDF/PNG copied to `Stackelberg-Journal-Version/JAIR/Figures/` and `/Users/gbrero/Desktop/`.
- Verification: script help smoke test under the project environment; `git diff --check`; PDF metadata and embedded-font inspection with `pdfinfo` and `pdffonts`; Poppler render at 220 dpi followed by visual inspection; SHA-256 equality checks for repository, paper, and Desktop copies.
- Limitations: this is an interim single-seed figure. MSPM was still running when the 1.4M-step snapshot was fixed. Regenerate at the completed 10M horizon and add independent seeds before using the chart as final empirical evidence.
- TODO/GB comments: none modified or removed.

## 2026-07-27 -- Clean Atari curriculum implementation

- Replaced the accumulated RLlib/frozen-head/event-compressed Atari paths with
  one active Stable-Baselines3 E0a--E2 curriculum. The actor interface is fixed
  throughout: four Atari frames, a 14-scalar state, a six-action mask, and the
  complete two-coordinate gameplay/economic action.
- Implemented E0a five-bullet bootstrap, E0b five randomly timed automatic
  free transfers, full-trajectory E1 buyer and seller meta-responses, and E2
  episodes containing five event-only leader queries, 200 gameplay decisions,
  and five exact cached trade replays.
- PPO action-credit masks select only the Atari log probability on gameplay,
  only the follower economic log probability at E1 trades, and only the leader
  economic log probability at E2 queries. Automatic E0b transfers and cached
  E2 trades contribute rewards to the common undiscounted return without a
  second actor decision.
- The exact per-episode action cache keys only actor-visible fields, stores the
  complete action, isolates vector-environment rows, and resets a row only when
  its outer episode terminates. Every stage creates a fresh critic; stage
  transfer copies only the declared actor modules. Resume restores the full
  same-stage policy, critic, optimizer groups, and timestep clock.
- Archived the complete pre-clean validation tree under
  `Research Artifacts/legacy_atari_prototype_20260727/` before removing obsolete
  importable modules and scripts. Historical checkpoints remain on disk but
  are rejected by clean checkpoint metadata validation.
- Verification: 46 repository tests pass, all clean files compile, and real-ROM
  E0a, E0b, buyer/seller E1, and seller E2 mechanics smokes completed with full
  horizons, five E2 cache hits, and exact bullet/payment accounting. These are
  implementation smokes, not scientific performance results.
- Long-run status: launched clean E0a seed 1 for 10M local steps from commit
  `bbc3ce2`, using four environments. W&B run `dyaly9m7` is online in project
  `StackPOMDP`, group `atari_clean_curriculum`, job type `atari_e0a`. The tmux
  session is `atari_clean_e0a_seed1_10m`; its local log and W&B stream reached
  2,400 steps with complete 200-step episodes and no startup error. No result
  is declared until the final deterministic 20-episode evaluation completes.

## 2026-07-28: matrix-game qualitative revalidation and diagnostics

- User authorization: explicit request to launch the new experiments on the cluster and monitor qualitative trends.
- Preflight: Unity user queue contained zero running or pending jobs; W&B authentication and the `stackelberg-pomdp` environment were verified. No Atari cluster process or job was touched.
- Source isolation: synced the verified 5.4 MB worktree snapshot based on commit `8727cbcd93c471b1dbbec75a9b02d47d15f411d0` to `~/StackelbergPOMDP-matrix-qualitative-20260728`, rather than modifying the heavily dirty canonical remote checkout. Key remote source hashes matched locally.
- Verification: direct matrix regression harness, one-record array dispatch smoke, E1 gate tests, trend-summary tests, shell syntax, CLI imports, and whitespace checks passed before submission.
- Sweep: `qualitative-v1`, seeds 1--10, 10 E1 plus 140 leader runs, 200,000 training steps per logical run, plan SHA-256 `11ae184bf29e4e2349508b647b08501e4187a0a65678dfb8b0ae6571540ec2db`.
- Initial SLURM DAG `62361130`--`62361134` failed before creating manifests because strict nounset preceded Unity profile activation. Remaining tasks were cancelled; strict mode was moved after conda activation and the exact environment order was verified remotely.
- Independent-Q outcome: array `62361190` completed 60/60 cells. Across ten seeds, means $\pm$ sample SEM were: q-reset reset `1.460 +/- 0.000` versus ongoing `1.300 +/- 0.153`, paired reset-minus-ongoing `+0.160 +/- 0.153`; response-reward with zero miscoordination penalty, excluded `2.000 +/- 0.000` versus included `1.800 +/- 0.133`, paired excluded-minus-included `+0.200 +/- 0.133`; with the penalty, excluded `1.900 +/- 0.100` versus included `1.300 +/- 0.153`, paired `+0.600 +/- 0.221`. The reset contrast is weak, while the response-reward direction is strongest under the penalty. These results are retained as diagnostics, not paper evidence.
- E1 gate outcome: array `62361189` completed 10/10 responses, but all failed certification with maximum regret `2`--`3`. Gate `62361192` blocked the branch as designed; meta-dependent array `62361194` and plot job `62361196` were cancelled without starting.
- On-policy E1 diagnostic: PPO/A2C array `62362084` completed all 8 treatments, with 0 strict passes among 50 retained checkpoints.
- Replay-based E1 diagnostic: DQN array `62362864` completed all 8 tasks with clean exits, but again produced 0 strict passes. The best maximum-regret checkpoint was 150k steps, learning rate `0.001`, 32x32 network, with mean regret `0.09375`, maximum regret `1`, and 29/32 optimal commitments. This does not release any E2/meta-dependent experiment.
- Q-reset policy-gradient diagnostic: array `62362532` completed all 8 treatments, but every reset and ongoing cell ended at leader reward `2.0`, yielding no separation.
- Historical q-reset forensics: the archived runs predate terminal-Q bugfix `f6189ac`; the old terminal bootstrap likely manufactured the sticky carryover in the historical curve. Reproducing that bug would not be a valid qualitative replication, so the historical behavior will not be promoted. A clean REINFORCE implementation is in progress locally; no cluster job has been submitted for it.
- Verified local evidence: `replication/matrix_ablations/results/qualitative-v1/` (423 files), `replication/matrix_ablations/results/e1_diagnostic/` (98 files), `replication/matrix_ablations/results/e1_dqn_diagnostic/` (72 files), and `replication/matrix_ablations/results/qreset_pgdiag_v1/` (48 files). Detailed job outcomes are in `docs/experiment_tracker.md`.
- Evidence boundary: no LaTeX empirical claim, paper figure, or table was changed. The strict E1 gate remains the blocker, and no E2/meta-dependent run is authorized by these diagnostics.

## 2026-07-28: corrected hidden-query PG/ES qualitative-v2 cohort

- Historical forensics established that the retained appendix figure mixed
  Modified PD for PG with ordinary PD for ES, used three-state opponent memory,
  plotted Seaborn standard-deviation bands, and did not actually use its
  declared linear ES model. It is retained only as provenance.
- Implemented isolated maintained PG and ES treatments under one coherent
  current-paper specification. PG is vanilla bias-free linear REINFORCE with
  no critic, baseline, entropy bonus, clipping, normalization, or action cache.
  ES uses 500 independent mirrored directions, deterministic candidates,
  sigma `.02`, average centered ranks, Adam stepsize `.01`, L2 `.005`, and
  separate hashed best/final parameter artifacts.
- Bound both algorithms to the same Modified-PD, joint-five-state environment
  and the one exposure-matched response checkpoint certified exactly on all 32
  deterministic commitments (SHA-256
  `e76ea2a593de80a93056911dfc7fcf1755544b15a674318aa9a1fa957ca7eb80`).
- Calibration array `62368442` ran 12 tasks for seeds 1--3. Every task
  completed with exit code zero; all manifests and artifact hashes validated.
  PG observed ended at `-.2,-.2,0`, PG hidden at `-2,-2,-2`, and ES at `0`
  in both conditions for all three seeds.
- After the audited calibration passed, extension array `62368796` ran exactly
  28 non-overlapping tasks for seeds 4--10. All 28 completed with exit code
  zero and all downloaded manifests/artifact hashes validated.
- Across the full ten-seed cohort, final mean `+/-` sample SEM was PG observed
  `-0.160 +/- 0.027`, PG hidden `-2.000 +/- 0.000`, and ES
  `-0.060 +/- 0.031` in both conditions. Matched ES visibility conditions are
  identical within every seed, as expected for policy-space search.
- The plotter now treats corrected ES `learning_rate: null` correctly by
  grouping on effective `es_stepsize`, rejects mixed optimizer cohorts, and
  computes sample SEM only across independent leader seeds. The hidden curve
  is dashed so exact ES overlap remains visible. Eight focused plot checks and
  an end-to-end render of all 40 runs passed.
- Raw immutable runs and SLURM logs are under
  `replication/matrix_ablations/results/hidden_queries_pg_es_calibration_s1to3/`
  and `hidden_queries_pg_es_extension_s4to10/`. The checksummed figure,
  summary, cohort manifest, and figure-grouped paper logs are under
  `hidden_queries_pg_es_qualitative_v2/`.
