# Stackelberg POMDP: Learning to Lead via Reinforcement Learning

Code, pretrained policies, and recorded results accompanying the manuscript.
The package covers mechanism design (MSPM), platform intervention against
algorithmic collusion, Atari bilateral trade, and three controlled ablations.

This submission is frozen as [jair-2026-v1.0.0](https://github.com/GlcBrero/StackelbergPOMDP/releases/tag/jair-2026-v1.0.0).
Download the complete archive and its SHA-256 from that release, or clone the
repository and check out the tag:

```bash
git clone https://github.com/GlcBrero/StackelbergPOMDP.git
cd StackelbergPOMDP
git checkout jair-2026-v1.0.0
```

- **Reproduce the paper:** generate its nine empirical figures and numerical
  tables from the included CSVs. No training, checkpoints, or Atari ROM is needed.
- **Rerun experiments:** use the commands below with the recorded paper settings.
- **Write a new setting:** start with [experiments/example/](experiments/example/)
  and the [experiment guide](experiments/README.md).

## Reproduce figures and tables

Use Python 3.9. From the package root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install -r reproduction/plots/requirements.txt
make reproduce
```

Outputs go to `build/paper/`. The command checks the inputs and numerical results
against the paper reference. The [data guide](reproduction/README.md) maps each
figure and table to its CSV and describes sampling units and uncertainty.
The 21 CSVs total about 22 MB and include every training seed and the retained
episode evaluations. Intermediate evaluations for two matrix ablations were
saved only as per-run summaries; their final evaluations include individual episodes.

## Install the experiment code

For training and implementation tests, add the full dependencies to the environment
above. Gym 0.21 requires these build-tool versions:

```bash
python -m pip install pip==23.3 setuptools==65.5.1 wheel==0.37.1
python -m pip install -r requirements.txt
python run.py --list
python run.py example --check
make test
make verify-checkpoints
```

Alternatively, use `conda env create -f environment.yml` and
`conda activate stackelberg-pomdp`. MSPM and platform require a C++17 compiler
(`clang++` or `g++`). Atari training and emulator evaluation additionally require
the external Space Invaders ROM described in [pretrained/README.md](pretrained/README.md).
The focused tests use mocked Atari emulators and need no ROM.

Training and evaluation write local logs and checkpoints. Use a fresh output path
for each run; all commands below run locally and require no online logging account.

## Package layout

| Folder | Contents |
| --- | --- |
| `stackelberg_pomdp/` | Shared response/reward executor, private episode commitment, policy and complete-episode rollout collection. |
| `experiments/` | Paper settings and a copyable example; each setting defines its game, response and experiment entry point. |
| `pretrained/` | Six Atari checkpoints, SHA-256 manifest and optional pretraining scripts. |
| `reproduction/` | Recorded CSV data, plotting scripts and result checks. |
| `support/` | Shared training, checkpoint-loading and packaging helpers. |
| `tests/` | Focused implementation and statistical checks. |

## MSPM

```bash
# One full run. Indices 0–124 enumerate types 2–6 and seeds 96–120.
python run.py mspm --index 0 --output results/mspm/t2-s96
# Explicitly select another paper run.
python run.py mspm --types 3 --seed 96 --output results/mspm/t3-s96
```

The common configuration is `experiments/mspm/_support/paper.json`: 10,000
complete commitment episodes, 40 episodes per rollout, initial action std 0.5,
first exact BCCE certification, and exact expected reward evaluation.
The runner records its actual source manifest beside every new run.
New local runs default to `results/`; use a separate `--output` or
`--output-root` when collecting another cohort.

A technical check uses `--episodes 40 --eval-every 40 --sampled-repetitions 1`
and a separate output directory. It is not part of the paper cohort.

## Platform intervention

```bash
python run.py platform train --seed 801 --output results/platform/seed801
python run.py platform confirm --seed 801 --output results/platform/seed801
# Repeat train + confirm for every seed 801–825.
# Select fixed-cap and clipped-minimum baselines on validation seeds,
# then evaluate only their winners on fresh confirmation seeds.
python run.py platform baselines --output results/platform/seed801
```

`experiments/platform/_support/paper.json` fixes the Johnson 15-price grid,
costs, 20,000 physical seller responses, PPO parameters, validation seeds,
and 25 confirmation seeds per cost. It preserves the paper's threshold-network
initialization and uses direct seller-admission masks internally. Seller Q-learning requests
leader actions on demand; response and reward share the same lazy commitment.
The budget counts complete episodes even when the number of queries varies.
Add `--smoke` to `train` for one PPO rollout with short seller responses;
confirmation then reads that same smoke protocol automatically.

The separate uniform-search diagnostic uses the same common follower response:

```bash
python run.py platform-search train --output results/platform-search
python run.py platform-search confirm --output results/platform-search
python run.py platform-search plot-validation --output results/platform-search
```

The paper uses a single-pass search: 10,000 distinct rules,
with one seller response per cost per rule (20,000 responses). Freeze the rule
with the highest single-pass score before evaluating it on 25 fresh seller
seeds per cost. `plot-validation` evaluates that same frozen winner on the
figure's five validation seeds per cost; it cannot change the selection.
This remains one search run, not a 25-run search cohort. `--smoke` uses four
rules but retains the full seller stopping criterion.

## Atari bilateral trade

The [pretrained setup notes](pretrained/README.md) give the required ROM
SHA-256 and checkpoint inventory. Supply the external ROM and verify the included
pretrained checkpoints:

```bash
export STACKPOMDP_SPACE_INVADERS_ROM=/absolute/path/to/space_invaders.bin
python pretrained/atari/checkpoints/download.py --verify-only
```

The paper's leader runs reuse the four included E0/E1 checkpoints; no pretraining
is needed. Optional [pretraining commands](pretrained/README.md#optional-pretraining)
are documented separately. Train one seller leader with:

```bash
python run.py atari-leader --leader-role seller --seed 1 --timesteps 1952420 --num-envs 4 \
  --n-steps 205 --batch-size 820 --actor-loss-mode balanced --response-checkpoint \
  pretrained/atari/checkpoints/meta_buyer.zip --leader-e1-checkpoint \
  pretrained/atari/checkpoints/meta_seller.zip --checkpoint-every 390476 --checkpoint \
  results/atari/leader-seller-s1.zip
```

An Atari leader episode contains 205 policy decisions: five economic
queries and 200 visual decisions. Its five cached trades still execute, for
210 physical transitions. The example retains 2,381 updates with four complete
episodes per update. The completed paper cohort uses this budget.

Evaluate the six prespecified checkpoints (including initialization), then the
final policy on 100 fresh episodes:

```bash
python run.py atari-evaluate-fixed --leader-role seller --checkpoint \
  results/atari/leader-seller-s1.zip --response-checkpoint \
  pretrained/atari/checkpoints/meta_buyer.zip --output results/atari/seller-s1-fixed
python run.py atari-evaluate-fixed --fresh-final --leader-role seller --checkpoint \
  results/atari/leader-seller-s1.zip --response-checkpoint \
  pretrained/atari/checkpoints/meta_buyer.zip --output results/atari/seller-s1-fresh
```

Fixed endpoints use 20 episodes each, starting at seed 4000001. Fresh-final
uses 100 episodes starting at seed 5000001. Every endpoint is retained;
these commands do not select checkpoints or training seeds.

For buyer leaders, change `--leader-role buyer` and swap the two meta-response
checkpoint inputs. The replication leader cohort uses seeds 1–25 for each role (50 policies).
`atari-evaluate-response --help` and `atari-evaluate-leader --help` describe the
screening, independent confirmation, and protocol checks. The gameplay-stage
E0b evaluator is available as `experiments/atari/_support/runners/evaluate_atari_e0b_sb3.py`.
The six included ZIPs contain four reused E0/E1 policies and two historical
seed-1 leader examples. The two leader examples are not the new 25-seed cohort;
current figures use the included rerun data.

The accompanying data contain 25 training seeds per condition, including
25 Atari leaders per role. Uniform random search is the single-run exception.
No pretraining needs to be repeated to reproduce the paper's leader runs.

## Controlled ablations

Response credit is a small self-contained scalar REINFORCE experiment:

```bash
python run.py response-credit --simulate --output-dir results/response-credit
```

Commitment consistency trains a contextual follower, then paired leaders:

```bash
python run.py normal-form meta-follower --certify --algorithm REINFORCE --memory-mode opponent \
  --seed 1 --output-root results/normal-form
python run.py normal-form leader --figure fig_memory_pg --condition visible --memory-mode \
  opponent --response-checkpoint /path/printed/by/follower/run/model.zip --seed 1 --output-root \
  results/normal-form
# Repeat with --condition hidden, sharing that seed's follower checkpoint.
```

This diagnostic uses opponent memory (three observations), the default in the
unified entry point. Full leader budgets
are 200,000 transitions; the learning rates are 0.004, 0.008, 0.015, 0.03 for
visible phase and 0.008, 0.015 for hidden phase, each with seeds 0–24.
Pass `--learning-rate` explicitly when sweeping these six curves. Follower
quality must pass the response gate before scientific leader runs. `--certify`
uses the paper stopping rule: at least 500 updates, exact checks every 100, and
the first maximum regret at most 0.25, capped at 10,000 updates. It preserves
the same learner and optimizer between checks. Failure retains an explicitly
uncertified checkpoint and raises an error. No leader outcome enters this rule.
Omitting `--certify` selects fixed-budget diagnostic training only.

Reward timing uses its experiment-local SimpleQ learner and tabular followers:

```bash
python run.py normal-form leader --figure fig_bots_leaderreward --condition excluded --seed 1 \
  --output-root results/normal-form
```

Repeat with `included`; run both with and without
`--matrix coordination_penalized_miscoordination`, for seeds 0–24.
The default budget is 22,000 transitions. Both completed normal-form cohorts
are included in the paper data; see [reproduction/README.md](reproduction/README.md).
New runs retain individual episodes in `progress.jsonl` at every evaluation
point and in the final `evaluation.json`, alongside their summary statistics.

## Verification and citation

```bash
make check-data          # Verify retained observations and figure inputs.
make verify-checkpoints # Verify all six included model hashes.
make test               # Check the implementation and statistical methods.
```

Both the repository and release archive include a `SHA256SUMS` inventory.
Run `python support/package.py --verify` from the package root. To build an archive locally, use
`make package`; generated outputs, caches, Git history and ROMs are excluded.

Please cite the accompanying manuscript and identify the release version when
using this package; author and software metadata are in [CITATION.cff](CITATION.cff).
Code is [MIT-licensed](LICENSE-CODE). Numerical data and the included checkpoints
use [CC BY 4.0](LICENSE-DATA). The Atari ROM is not redistributed.
