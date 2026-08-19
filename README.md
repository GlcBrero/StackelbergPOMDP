# Stackelberg POMDP

Official research code for *Stackelberg POMDP: Learning to Lead via
Reinforcement Learning*. The project turns a leader's commitment problem into
a partially observable Markov decision process whose response phase executes a
policy-interactive follower algorithm and whose reward phase evaluates the
resulting follower response.

The repository contains maintained implementations and replication interfaces
for the paper's indirect mechanism-design, platform-pricing, matrix-game, and
Atari Space Invaders experiments. Historical cohorts that cannot be regenerated
faithfully with the corrected implementation are labeled as archived data,
rather than silently approximated. The main learning stack uses
Stable-Baselines3; a separate, strictly pinned Ray environment is retained only
for the paper's native ES comparison.

## Installation

The reference environment uses Python 3.9 and pinned package versions:

```bash
conda env create -f environment.yml
conda activate stackelberg-pomdp
```

Ray RLlib 2.0.1 is intentionally isolated from the main environment. Create
its environment only when reproducing the native ES comparison:

```bash
conda env create -f environment-ray-es.yml
conda activate stackelberg-pomdp-ray-es
```

Commands should be run from the repository root. Both environments set
`PYTHONNOUSERSITE=1` to prevent accidental imports from user-level packages.

Atari runs additionally require a locally supplied Space Invaders ROM. The ROM
is not redistributed by this repository; setup and integrity instructions are
in [`replication/atari/README.md`](replication/atari/README.md).

## Reproducing paper experiments

`replication/targets.json` is the machine-readable paper manifest and
`replication/run.py` is its local entry point:

```bash
# Inspect the paper-to-code map.
python replication/run.py --list

# Check every runnable command against its experiment parser without training.
python replication/run.py --validate

# Inspect or execute one target and seed.
python replication/run.py fig_simple_allocation_stackpomdp_mappo --seed 1 --dry-run
python replication/run.py fig_simple_allocation_stackpomdp_mappo --seed 1
```

Some targets expand into named variants. Use `--variant NAME` to run one
variant or omit it to run the complete target. Each manifest entry is labeled
as runnable here, analytic, archived-data-only, or delegated to a specialized
replication workflow; the runner never substitutes a modern implementation for
a historical plotted cohort.

The detailed target map, executed historical rollout conventions, and direct
experiment entry points are documented in
[`replication/README.md`](replication/README.md). Atari's curriculum,
checkpoint requirements, and evaluation gates are documented separately in
[`replication/atari/README.md`](replication/atari/README.md).

## Repository layout

- `stackelberg_pomdp/envs/`: all domain environments, including Atari.
- `stackelberg_pomdp/wrappers/`: generic StackPOMDP/follower wrappers and
  domain-specific Gym adapters.
- `stackelberg_pomdp/policies/`: generic and Atari policy architectures.
- `stackelberg_pomdp/algorithms/`: SB3 algorithm extensions.
- `stackelberg_pomdp/atari/`: Atari protocol, sampling, and training support
  that is neither an environment, wrapper, nor policy.
- `stackelberg_pomdp/experiments/`: maintained experiment entry points.
- `replication/`: final-paper manifests, diagnostics, launchers, and protocol
  documentation.
- `tests/`: fast contract and regression tests; ROM-dependent integration is
  optional.

`stackelberg_pomdp/baselines_utils.py` and
`stackelberg_pomdp/atari/stackpomdp_policy.py` are compatibility imports for
historical serialized policy paths; maintained code does not implement new
functionality in either file.

Generated checkpoints, logs, W&B run directories, and plot outputs are not
versioned. Curated paper data and figure-generation scripts are distributed in
the journal reproducibility artifact rather than mixed with training source.

## Verification

After installing the main environment, validate the public manifest and run
the test suite:

```bash
python replication/run.py --validate
pytest -q
```

The core tests do not require an Atari ROM. See the Atari README for the
optional real-ALE smoke test.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). This software
is released under the [MIT License](LICENSE).
