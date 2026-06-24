# Replication Targets

This directory is the consolidation point for paper replication commands.

The first manifest, `bertrand_targets.json`, covers the Calvano/Bertrand
experiment family:

- `calvano_m4_calibration`
- `calvano_m4_deviation`
- `platform_intervention_state`
- `platform_intervention_no_state`

List targets:

```bash
python replication/run.py --list
```

Print a command without running it:

```bash
python replication/run.py platform_intervention_state --seed 1 --dry-run
```

Run a target:

```bash
python replication/run.py calvano_m4_calibration
```

The old top-level `run_*.sh` files are still present for historical continuity,
but this directory should become the public replication interface.
