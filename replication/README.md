# Paper Replication Targets

This directory is the public interface for reproducing paper results owned by
this codebase.

The manifest `targets.json` maps final-paper figures and tables to runnable
targets, analytic results, or explicit archived-data handoffs.  Runnable
targets spell out the executed protocol rather than relying on mutable CLI
defaults.

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

Use `--tot_num_response_episodes N` to change the follower-response horizon
for targets that use the StackPOMDP wrapper.

Simple-allocation ablations use `stackelberg` and `hidden_queries` POMDP modes.
The response-reward diagnostic is not part of the final Simple Allocation
figure and is therefore absent from the release target set.

The retained Simple Allocation figures came from five-million-step runs,
plotted through one million steps, with 30 sampled reward games and evaluation
records every 5,000 steps.  Their historical protocol executed exactly 100
response queries for every message count.  These contain 100, 50, and 33
complete MW updates for one, two, and three messages, respectively; for three
messages, the last query starts an incomplete update and does not change the
response weights.  The paper targets therefore set
`--align_mw_response_phase false`.  Complete-update alignment remains
available for new experiments, but it is not the protocol that generated the
retained curves.

Those two historical cohorts also predate complete-episode PPO batching.  Their
targets opt into `--ppo_rollout_geometry historical_ratio_scaled`, which
preserves the executed ratio-scaled rollout formula.  The maintained
`complete_episodes` geometry remains the CLI default for every new experiment.
For Simple Allocation, the historical stored-transition rollout/minibatch
sizes are 131,072/8,192 for StackPOMDP and 32,768/2,048 for Basic POMDP.  For
Matrix Design they are 1,015,808/63,488 and 32,768/2,048, respectively.

The retained mechanism-design cohorts predate two correctness fixes: their
nominal learner seeds did not reach the base environment's private RNG, and the
transition filter did not preserve the outer-episode boundary across an
excluded query prefix. This release corrects both behaviors. Current commands
therefore reproduce the declared training geometry and sampling protocol, but
are deterministic corrected reruns rather than bitwise replays of historical
random draws or buffer contents. The archived run-level data remain the
authority for the published curves.

Runnable paper targets call experiment-specific modules under
`stackelberg_pomdp.experiments`, so each experiment exposes only the arguments
that matter for that experiment.

Owned targets run from this repository.  `archived_data_only` means that the
plotted legacy cohort is retained in the journal reproducibility bundle but
the maintained implementation intentionally differs from that historical
protocol; the runner does not pretend a modern sensitivity run is an exact
regeneration.  `specialized_workflow` identifies a maintained, multi-stage
pipeline whose checkpoints, selection audit, and aggregation cannot be reduced
to one manifest command.  The target's `owner` and `entrypoints` provide the
exact handoff.  Both Atari paper figures use this status and are implemented
entirely by the Stable-Baselines3 workflow in `replication/atari/README.md`;
they no longer depend on the historical StackeRLberg repository.

Historical top-level `run_*.sh` launchers were removed from the public
replication surface. They remain recoverable from git history; new local or
cluster launchers should be generated from `targets.json`.
