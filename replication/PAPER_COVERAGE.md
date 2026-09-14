# Current-paper implementation and replication coverage

This map follows the current `JAIR/StackelbergPOMDP.tex`, including its appendix.
Figure basenames are used because figure numbers change as the manuscript is
edited. Training commands are defined in [targets.json](targets.json), with
Atari's multi-stage commands in [atari/README.md](atari/README.md).

The journal repository contains the figure-generation code and retained data.
Its large data tree is on `release/jair-2026-reproducibility`; the current
manuscript branch carries updated plotting scripts and Atari initialization
summaries. Those changes must be combined for the next release. The published
`jair-2026-v1.0.0` tag alone does not reproduce the current manuscript.

## Every empirical result

Plotter paths below are relative to the journal repository's
`reproducibility/scripts/`; data paths are relative to its `reproducibility/data/`.

| Paper output | Training or evaluation route here | Retained input and plotter |
| --- | --- | --- |
| `SA-MessageComparison_1M` | Three `fig_simple_allocation_stackpomdp_mappo` targets for 1, 2, and 3 messages; 25 seeds each | `simple_allocation/`; `plot_allocation_and_matrix.py` |
| `SA-Ablation_new` | StackPOMDP/Basic POMDP × MAPPO/PPO targets; 25 seeds each | `simple_allocation/`; `plot_allocation_and_matrix.py` |
| `pi_spm_vs_mspm_exact_eval_allseeds` | `fig_pi_mspm_2types_2messages` and `fig_pi_spm_2types`; all 25 MSPM and 10 SPM seeds | `mspm/{mspm,spm}/seed_*.csv`; `plot_mspm_two_type.py` |
| MSPM scaling table | Four `table_mspm_*types_2messages` targets; all 25 seeds per type count. SPM optima are analytic. | `mspm/scaling_seed_best.csv`; `summarize_mspm_scaling.py` recomputes means, sample SEs, exact optima, and Holm-adjusted tests |
| `deviation_m4` | Seller calibration and punishment diagnostic; 25 seller pairs | `collusion/deviation/`; `plot_collusion_deviation.py` |
| `collusion_learning_25seeds` | `fig_collusion_learning_state` and three fixed-policy benchmarks; 25 seeds each. The calibration prerequisite covers all 15 hyperparameter cells. | `collusion/{pda,pdp,dpdp,no_intervention}/`; `plot_collusion_learning.py` |
| `atari_e1_responses` | E0a → E0b → E1, response evaluation and selection; 2 roles × 11 contexts × 20 episodes | `atari/atari_e1_responses.csv` and its summary; `plot_atari_e1_responses.py` |
| `atari_meta_stackpomdp` | E2 training and evaluation for 10 seeds per role; initialization reconstructed by the journal's `evaluate_atari_e2_initialization.py` | 100 trained-checkpoint policy means plus 20 initialization policy means and their summaries; `plot_atari_meta_stackpomdp.py` |
| `fig_memory_pg` | `fig_memory_phase_ablation`: 6 PG leader curves, sharing 10 pretrained REINFORCE responses. [Training instructions](matrix_ablations/README.md). | Published: `diagnostics/fig_memory_pg/`; `plot_theory_diagnostics.py`; 10 runs per curve. New runs: `replication/matrix_ablations/plot.py`. |
| `fig_bots_leaderreward` | `fig_reward_during_learning_ablation`: SimpleQ/tabular-Q, 2 reward treatments × 2 matrices × 10 seeds. | Published: `diagnostics/fig_bots_leaderreward/`; `plot_theory_diagnostics.py`. New runs use the maintained plotter. |
| `MatrixDesign-Ablation_new` | Four variants of `fig_matrix_design_ablation`; 25 seeds each | `matrix_design/`; `plot_allocation_and_matrix.py` |

These are 11 distinct data-driven figure files and one numerical results table.
The reused Simple Allocation panel needs no second experiment. The Atari
architecture diagram is supplied directly as
`JAIR/Figures/atari_stackpomdp_architecture.tex`; other conceptual diagrams and
specification tables are defined in the manuscript source.

## What was removed

The public training surface no longer includes RoundRobin, the removed
hidden-query normal-form PG/ES study and its Ray environment/checkpoint, the
frozen-gameplay E2 ablation, or the generic Simple Allocation response-reward
option. The appendix's separate tabular-Q response-reward diagnostic remains.
The current manifest uses every MSPM/SPM seed, with no top-seed filtering.

Tests, the tiny episode example, old checkpoint import paths, and provenance
checks remain necessary supporting code. E1 meta-seller gameplay freezing and
its conditioning probes remain part of the paper's actual Atari pipeline.

## Interpretation and release status

The audit regenerated all ten data-driven figures and the scaling table from
retained inputs. All ten figures match the committed PDFs pixel for pixel,
and the inventory matches the current LaTeX source. The code
passes 325 tests, including the remaining appendix training, evaluation,
plotting, and provenance checks. Its manifest expands to 47 parser-validated
commands and covers all ten appendix treatment curves. Six published Atari
archives are bundled with checksums and their CC BY 4.0 attribution.

This establishes coverage and checks implementation behavior. It does not
establish that corrected training reproduces every historical conclusion.
The seed forwarding, hidden-query episode-boundary, and critic phase-flag
changes are documented in [README.md](README.md). The two historical
normal-form diagnostics now have maintained PG/SimpleQ training as well as
archived plot reproduction. Their documented differences from historical
training still require new evidence. See [RERUN_PLAN.md](RERUN_PLAN.md) for
the affected cohorts across the paper.

Before submission, settle the rerun evidence and publish matching code,
manuscript, and artifact revisions. The working-tree changes and the assembled
artifact reviewed here are not a new published release.
