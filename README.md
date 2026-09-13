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
Stable-Baselines3.

Optimizer settings follow SB3 defaults, with documented exceptions for the
economic objective, episode commitments, and retained paper protocols. See
[training parameters](replication/PARAMETERS.md) for the rules and the effect
of the September defaults cleanup on existing results.

## Installation

The reference environment uses Python 3.9 and pinned package versions:

```bash
conda env create -f environment.yml
conda activate stackelberg-pomdp
```

Commands should be run from the repository root. The environment sets
`PYTHONNOUSERSITE=1` to prevent accidental imports from user-level packages.

Atari runs additionally require a locally supplied Space Invaders ROM. The ROM
is not redistributed by this repository; setup and integrity instructions are
in [`replication/atari/README.md`](replication/atari/README.md).

## Follow one episode

After installation, run this small example from the repository root:

```bash
python -m examples.one_episode
```

[`examples/one_episode.py`](examples/one_episode.py) runs one outer StackPOMDP
episode and one PPO update on CPU. It uses the two-action matrix game
`game_3`, with two response plays followed by two reward plays. It needs no
Atari ROM, pretrained checkpoint, or W&B account.

Follow these five components in the example:

1. **Environment:** `BaseEnvSimpleMatrixGame` in
   [`envs/normal_form.py`](stackelberg_pomdp/envs/normal_form.py) receives the
   leader and follower actions and computes both payoffs. Each play of this
   matrix game finishes in one step.
2. **Follower response:** `MWFollowersWrapper` in
   [`wrappers/core.py`](stackelberg_pomdp/wrappers/core.py) supplies the follower
   action and passes its payoff to `CertifiedMWResponse` in
   [`follower_responses.py`](stackelberg_pomdp/follower_responses.py). The response
   evaluates actions 0 and 1, updates the multiplicative weights, then uses
   the greedy response. Certification is disabled for this tiny walkthrough.
3. **Phase wrapper:** `StackPOMDPWrapper` surrounds that follower wrapper. It
   returns zero leader reward during the two response plays, then passes
   through the leader's payoff during two plays against the selected response.
   It ends the outer episode after the fourth step and sets
   `info["is_reward_phase"]` and `info["exclude_from_buffer"]` on each transition.
4. **Policy/cache:** `CustomPolicy` in
   [`policies/generic.py`](stackelberg_pomdp/policies/generic.py) samples an
   action on the first visit to the actor's observation. In this example that
   observation is always zero, so subsequent visits reuse the same action.
   Critic-only fields do not enter the cache key. `FixPolicyActionsCallback`
   enables caching and clears it when the outer episode ends. Exploration
   comes from sampling the learned action distribution; `ent_coef` controls
   entropy regularization during training.
5. **Rollout collection:** `CustomOnPolicyAlgorithm.collect_rollouts` in
   [`algorithms/on_policy.py`](stackelberg_pomdp/algorithms/on_policy.py) drives
   the loop: call the policy, step the wrapped environment, run callbacks,
   and retain the observation/action/reward/value/log-probability data unless
   `exclude_from_buffer` is true. Once the rollout is full, it computes returns
   and advantages; PPO then updates the policy.

The printed trace with the example's seed is:

| Step | Phase | Leader action | Follower action | Leader reward | Retained in rollout |
| --- | --- | --- | --- | --- | --- |
| 1 | Response | 1 | 0 | 0 | Yes |
| 2 | Response | 1 | 1 | 0 | Yes |
| 3 | Reward | 1 | 1 | 0.667 | Yes |
| 4 | Reward | 1 | 1 | 0.667 | Yes |

The leader reward is rounded here; the game's normalized reward is `2/3`.
The cache has one entry during the episode and zero entries afterward.
One update demonstrates the execution path; it is not a trained paper model.

To see how rollout filtering differs from environment execution, run:

```bash
python -m examples.one_episode --hidden-queries
```

All four steps still execute, and the follower still evaluates both actions.
The first two rows now print `retain=False`; only the two reward rows enter
the PPO buffer. The example sets `n_steps` to four in standard mode and two
in hidden-query mode so each rollout contains exactly one outer episode.

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
[`replication/README.md`](replication/README.md). The
[paper coverage map](replication/PAPER_COVERAGE.md) accounts for every
empirical figure and table. Atari checkpoint requirements and evaluation gates
are documented separately in
[`replication/atari/README.md`](replication/atari/README.md).

## Repository layout

- `stackelberg_pomdp/envs/`: all domain environments, including Atari.
- `stackelberg_pomdp/wrappers/`: generic StackPOMDP/follower wrappers and
  domain-specific Gym adapters.
- `stackelberg_pomdp/policies/`: generic and Atari policy architectures.
- `stackelberg_pomdp/checkpoints/`: checkpoint loading, transfer, provenance,
  and file integrity checks.
- `stackelberg_pomdp/evaluation/`: policy adapters and Atari rollouts, audits,
  selection, and reporting.
- `stackelberg_pomdp/algorithms/`: SB3 algorithm extensions.
- `stackelberg_pomdp/atari/`: Atari protocol, sampling, and training support
  that is neither an environment, wrapper, nor policy.
- `stackelberg_pomdp/experiments/`: maintained experiment entry points.
- `examples/`: small runnable implementation walkthroughs.
- `replication/`: final-paper manifests, diagnostics, launchers, and protocol
  documentation.
- `tests/`: contract and regression tests, including checkpoint round trips.

For a guided reading order and the responsibilities of individual environment
and evaluator modules, see [`CODE_GUIDE.md`](CODE_GUIDE.md).

`stackelberg_pomdp/baselines_utils.py` and
`stackelberg_pomdp/atari/stackpomdp_policy.py` are compatibility imports for
historical serialized policy paths; maintained code does not implement new
functionality in either file.

Six frozen Atari reference checkpoints are included in
[`replication/atari/checkpoints/paper/`](replication/atari/checkpoints/paper/README.md),
with their roles, training steps, and SHA-256 checksums. The bundle contains
both gameplay stages, both meta-responses, and one selected leader per role.
Generated run checkpoints, logs, W&B directories, and plot outputs are not
versioned. The full multi-seed plotting data and figure-generation scripts
remain in the journal reproducibility artifact.

## Verification

After installing the main environment, validate the public manifest and run
the test suite:

```bash
python replication/run.py --validate
pytest -q
```

Most Atari tests mock the emulator, but environment and checkpoint
configuration tests still resolve a ROM file. Set
`STACKPOMDP_SPACE_INVADERS_ROM` to your local ROM before running the full suite;
see the Atari README for setup details.

The [normal-form appendix guide](replication/matrix_ablations/README.md) covers
all three PG/SimpleQ diagnostics; the [rerun plan](replication/RERUN_PLAN.md)
lists the remaining scientific validation.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). This software
is released under the [MIT License](LICENSE).
