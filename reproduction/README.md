# Figure data and plotting scripts

`data/` contains one CSV for each empirical figure, five small CSVs for
supporting tables and statistical comparisons, and seven CSVs with retained
observations, training histories and recorded settings: about 22 MB in total.
`plots/` turns those files into the paper's figures and tables.
These are compact exports of the recorded results, retaining every training seed
and the available episode-level observations. Full training logs, transition traces and intermediate models
are kept separately in the author archive.

## Generate the figures and tables

From this folder, with Python 3.9:

```sh
python -m pip install -r plots/requirements.txt
python plots/reproduce.py --check --output-dir build
```

Outputs go to `build/figures/` and `build/tables/`. The command also generates the
platform numbers, statistical-comparison table and supporting numerical results.
It checks the input fingerprints and numerical outputs against the paper's
reference. It needs no training code, checkpoints, Atari ROM, Unity account or
manuscript. From the full code package root, `make reproduce` does the same work
in `build/paper/`.

```sh
# Optional pixel comparison, using the recorded Poppler version.
python plots/reproduce.py --check --compare-rendered --output-dir build
# Check the CSVs and scripts without generating figures.
python validation/check.py
```

## CSVs for the figures

Each file has the same stem as the figure it generates. There is no averaging
across training seeds in the exported learning curves; the scripts calculate
means and standard errors. Atari response uncertainty uses individual evaluation
episodes. Baseline lines and the policy grid have different units, noted below.

| CSV in `data/` | Rows | What one row represents |
| --- | ---: | --- |
| `pi_spm_vs_mspm_exact_eval_allseeds.csv` | 1,275 | Two-type MSPM: one seed at one evaluation point. `raw_reward` is the welfare gap relative to the ex post optimum; BCCE gaps are included. |
| `mspm_multitype_training.csv` | 5,100 | The same quantities for three to six types. |
| `hidden_cost_learning.csv` | 1,254 | One PPO leader seed at one validation point (`responses` counts seller responses), plus four fixed baseline lines. `consumer_surplus` is the validation score. |
| `hidden_cost_policy.csv` | 225 | One ordered seller-price pair: prices, admission decisions, and indicators marking the low/high-cost outcomes. |
| `atari_e1_responses.csv` | 440 | One evaluation episode for a frozen response and fixed opponent value. Includes transfers, shots, quoted price and payoff. |
| `atari_meta_stackpomdp.csv` | 300 | One leader seed at one of six fixed endpoints, with means over 20 evaluation episodes. The x-axis counts physical transitions. Accepted price is total payments divided by total purchases. |
| `response_credit_ablation.csv` | 50,050 | One seed/method/update: policy value, gradient and counts `n00`–`n11` of the joint response/gameplay actions. |
| `fig_memory_pg.csv` | 15,150 | One commitment-consistency condition, learning rate, seed and evaluation point. The x-axis uses `policy_updates`; episode count, standard deviation and standard error are also retained. |
| `fig_bots_leaderreward.csv` | 1,200 | One reward-timing condition, game, seed and evaluation point. The x-axis uses `global_step`; episode count, standard deviation and standard error are also retained. |

`plots/reproduce.py` runs the full pipeline. `plots/build_figures.py` generates
all nine empirical figures. Its `_support/` folder groups data readers, shared
styles and application-specific drawing/numerical helpers (`mspm.py`,
`platform.py`, `atari.py`, `response_credit.py`).
The architecture drawing is source-native and lives with the manuscript.

## CSVs for tables and comparisons

| CSV in `data/` | Rows | Contents |
| --- | ---: | --- |
| `table_mspm_welfare.csv` | 125 | Per-seed best observed and final greedy welfare gaps, for all five type counts. |
| `table_platform_results.csv` | 1,400 | Individual seller-response outcomes: method, leader seed, cost and seller seed. `cycle_actions` lists recurrent price-index pairs as `i:j;i:j`. Search-selection fields are populated only for the one random-search winner. |
| `table_atari_fresh_final.csv` | 50 | Each final leader's means over 100 fresh evaluation episodes. |
| `table_ablation_endpoints.csv` | 250 | Each matrix-ablation leader's mean over 100 final evaluation episodes. The response checkpoint digest identifies matched comparisons. |
| `table_atari_compute.csv` | 50 | Recorded wall time, CPU use and resource allocations for the Atari leader runs. Times are in seconds. |

`plots/paper_results.py` calculates the numerical claims;
`plots/statistical_comparisons.py` calculates the comparisons. It resamples
platform training seeds and seller seeds separately, keeping both costs together.
Random search is one run: its comparison is conditional on the frozen selected
policy. The full statistical protocol is `plots/_support/statistical_protocol.json`.

These plotting CSVs retain the resolution used in the paper. The additional
records below retain the available observations behind their within-run means.
Seeds and evaluation points are preserved, so uncertainty is recomputed at
the correct sampling-unit level.
The CSV conversion changes storage only; no experiments were rerun or selected
anew. All nine empirical figures and the paper's numerical outputs are checked
against the same frozen figure and numerical reference in `validation/paper_reference.json`.

## Retained observations and run records

| CSV in `data/` | Rows | Contents |
| --- | ---: | --- |
| `mspm_training_updates.csv` | 31,250 | Every recorded PPO update from the 125 MSPM runs: episode counts, training reward, query counts and action standard deviations. The figure CSVs separately retain all 51 scheduled exact evaluations per run. |
| `raw_atari_evaluations.csv` | 11,000 | Individual emulator episodes for all six scheduled endpoints and the fresh final cohort, with payoffs, payments, purchases, shots and checkpoint hashes. |
| `raw_ablation_final_evaluations.csv` | 25,000 | All 100 fresh final episodes for each of the 250 matrix-ablation policies. |
| `raw_platform_validation.csv` | 12,540 | Individual seller responses for every plotted PPO validation point and the four benchmark lines. |
| `platform_random_search_candidates.csv` | 10,000 | Every sampled rule, its seller seed, both cost-specific outcomes, stopping counts and selection score. Threshold entries use the ordered pairs `(i,j)` with `i <= j`, enumerating `i` then `j`; threshold `k` admits sellers whose price index is at most `k` (so minimum-price sellers are always included). |
| `recorded_run_configurations.csv` | 20,867 | Populated settings recorded for the reported runs. Nested parameter names use dots; values retain JSON types inside CSV cells. Absolute paths are reduced to filenames. These historical settings may name retired command flags; use the current root README commands to run the code. |
| `recorded_source_hashes.csv` | 3 | Original response-credit source, protocol and result hashes, which still match the included files. |

The record checks recompute Atari and ablation final means, all platform
validation points, and the selected random-search candidate. They also check
complete MSPM update coverage and the response-credit source hashes when the
experiment source is present. `make check-data` and ordinary figure reproduction
run these checks automatically.

The original intermediate evaluations for the two matrix ablations were saved
only as per-run means, standard deviations, standard errors and episode counts.
Their individual evaluation episodes were not retained. The CSVs preserve those
summaries exactly; the final evaluations do include individual episodes.
The archived progress files match both original Unity rerun cohorts byte for
byte. The current logger also saves `episode_rows` at initialization and every
scheduled evaluation for future runs; this does not reconstruct past episodes.
Full transition traces, exploratory runs and the complete checkpoint collection
remain in the author archive. These exports add no new experiments or outcomes.

`validation/check.py` checks the inputs and generated outputs against that
reference. Statistical-method tests are in the code package's `tests/` folder;
run `make test` or `make test-statistics` from the code root. The single root
Makefile also provides `make reproduce` and `make check-data`.

The CSVs and plots are identical in the author's `JAIR/logs/` folder and the
submission package's `reproduction/` folder. Plotting code uses the MIT license
in `plots/LICENSE`; numerical data use the license in `data/LICENSE`.

To update manuscript assets explicitly, supply both `--install` and
`--paper-dir /path/to/paper` to `plots/reproduce.py`. Ordinary reproduction only
writes generated outputs into the chosen build directory.
