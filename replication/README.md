# Paper Replication Targets

This directory is the public interface for reproducing paper results owned by
this codebase.

The manifest `targets.json` maps paper figures/tables to runnable targets or
explicit TODO/companion-code placeholders.

List targets:

```bash
python replication/run.py --list
```

Print a command without running it:

```bash
python replication/run.py fig_collusion_learning_state --seed 1 --dry-run
```

Run a target:

```bash
python replication/run.py fig_simple_allocation_stackpomdp_mappo --seed 1
```

Override target defaults for local variants:

```bash
python replication/run.py fig_simple_allocation_stackpomdp_mappo --seed 1 --num_messages 2
```

Use `--tot_num_response_episodes N` to change the follower-response horizon
for targets that use the StackPOMDP wrapper.

Simple-allocation ablations use named `pomdp_mode` values:
`stackelberg`, `hidden_queries`, and `reward_during_response`.

Runnable paper targets call experiment-specific modules under
`stackelberg_pomdp.experiments`, so each experiment exposes only the arguments
that matter for that experiment.

Owned targets run from this repository. Targets marked `companion_stackerlberg`
are paper experiments whose current implementation lives in `Code/StackeRLberg`
or `atari experiment/`; their placeholder entries document the gap until the
exact companion command is added or the experiment is ported.

Historical top-level `run_*.sh` launchers were removed from the public
replication surface. They remain recoverable from git history; new local or
cluster launchers should be generated from `targets.json`.
