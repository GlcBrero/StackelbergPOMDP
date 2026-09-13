# Paper Replication Targets

This directory is the public interface for reproducing paper results owned by
this codebase.

The manifest `targets.json` maps final-paper figures and tables to runnable
targets, analytic results, or explicit archived-data handoffs.  Runnable
targets spell out the executed protocol rather than relying on mutable CLI
defaults.

[Current-paper coverage](PAPER_COVERAGE.md) maps every empirical figure and
table to its training route, retained inputs, and plotting script.

List targets:

```bash
python replication/run.py --list
```

Print a command without running it:

```bash
python replication/run.py fig_collusion_learning_state --seed 1 --dry-run
```

Parser-validate every runnable command in the manifest without training:

```bash
python replication/run.py --validate
```

Run a target:

```bash
python replication/run.py fig_simple_allocation_stackpomdp_mappo --seed 1
```

Override target defaults for local variants:

```bash
python replication/run.py fig_simple_allocation_stackpomdp_mappo --seed 1 --num_messages 2
```

Some targets contain named variants.  A dry run prints every expansion; use
`--variant` to select one:

```bash
python replication/run.py fig_matrix_design_ablation --seed 1 --dry-run
python replication/run.py fig_matrix_design_ablation --variant basic_ppo --seed 1
```

Use `--mw_response_cycles N` to set a fixed number of complete MW updates.
Simple Allocation and Matrix Design now share a default of 33 updates in all
conditions. One update visits every joint follower-action profile once: this
costs 1, 2, or 3 follower games in Simple Allocation and 4 in Matrix Design.
The corresponding response prefixes are 33, 66, 99, and 132 games.

Alternatively, `--tot_num_response_episodes N` supplies a follower-game budget.
MW always rounds it down to complete updates (100 becomes 99 with three
messages; 30 becomes 28 in Matrix Design). A budget shorter than one update
is rejected. These options are mutually exclusive; a replication CLI override
replaces the target's choice of budget unit. Certified MSPM may still extend
the fixed prefix by complete updates to meet its response threshold. Its
existing budgets, learning rate 0.1, and certification settings are unchanged.

Simple-allocation ablations use `stackelberg` and `hidden_queries` POMDP modes.
The response-reward diagnostic is not part of the final Simple Allocation
figure and is therefore absent from the release target set.

The retained Simple Allocation figures came from five-million-step runs,
plotted through one million steps, with 30 sampled reward games and evaluation
records every 5,000 steps.  Their historical protocol executed exactly 100
response queries for every message count.  These contain 100, 50, and 33
complete MW updates for one, two, and three messages, respectively; for three
messages, the last query starts an incomplete update and does not change the
response weights. Matrix Design used 30 queries, seven complete updates, and
two unused queries. The new shared 33-update protocol changes these budgets;
it is not the protocol that generated the retained curves.

The current targets use SB3 optimizer defaults and automatic episode-based
rollout sizing; see [PARAMETERS.md](PARAMETERS.md). The September reruns instead
used ratio-scaled rollouts with large minibatches. Earlier documentation
incorrectly identified those newer settings as the original historical recipe;
archived Simple Allocation policy histories contradict that update cadence.
The original per-run optimizer metadata is incomplete. Future runs and their
figures must identify their actual settings rather than inherit that claim.

The retained mechanism-design cohorts predate two correctness fixes: their
nominal learner seeds did not reach the base environment's private RNG, and the
transition filter did not preserve the outer-episode boundary across an
excluded query prefix. This release corrects both behaviors. Current commands
use a prospective corrected optimizer recipe. The completed September reruns
also completed the final MW sweep despite the former alignment opt-out:
three-message SA executed 102 queries/34 updates, and Matrix Design 32/8.
The maintained wrapper now floors the fixed prefix before execution, including
when constructed directly. The obsolete alignment opt-out is rejected.
Archived run-level data remain the authority for the published curves;
current commands are not exact replays.

The current wrapper also corrects a separate critic-observation off-by-one:
with `critic_obs=flag` or `full`, the observation used for the first
reward-phase action now has `critic:is_reward_step=1`. Transition rewards and
rollout filtering are unchanged. This changes critic inputs in affected
training runs; its effect on retained paper results has not been measured.
The Atari E2 pipeline disables this generic flag (`critic_obs=None`) and
supplies its own critic fields.

Runnable paper targets call experiment-specific modules under
`stackelberg_pomdp.experiments`, so each experiment exposes only the arguments
that matter for that experiment.

Owned targets run from this repository. The three normal-form appendix
diagnostics now include PG/SimpleQ training and all plotted treatment curves;
see [matrix_ablations/README.md](matrix_ablations/README.md). Their historical
curves remain archived data, and new training is not an exact historical
replay. [RERUN_PLAN.md](RERUN_PLAN.md) identifies the paper cohorts that need
new training or a narrower invariance argument.

`specialized_workflow` identifies a maintained, multi-stage
pipeline whose checkpoints, selection audit, and aggregation cannot be reduced
to one manifest command.  The target's `owner` and `entrypoints` provide the
exact handoff.  Both Atari paper figures use this status and are implemented
entirely by the Stable-Baselines3 workflow in `replication/atari/README.md`;
they no longer depend on the historical StackeRLberg repository.

Historical top-level `run_*.sh` launchers were removed from the public
replication surface. They remain recoverable from git history; new local or
cluster launchers should be generated from `targets.json`.
