# Mechanism-Design Replication

Paper targets owned by this codebase:

- `fig_simple_allocation_stackpomdp_mappo`
- `table_mspm_3types_2messages`
- `table_mspm_4types_2messages`
- `table_mspm_5types_2messages`
- `table_mspm_5types_3messages`
- `baseline_mspm_evolutionary_3types_2messages`
- `baseline_mspm_evolutionary_4types_2messages`

These currently run through `replication/run.py`, which invokes the reusable
StackPOMDP training entrypoint with the paper configuration.

TODO:

- Add aggregation/plotting scripts that regenerate `SA-Ablation_new` and the
  MSPM table from raw logs.
- Confirm final seed ranges and max-step budgets against the submitted paper.

Baseline note:

- The evolutionary MSPM baseline is intentionally non-StackPOMDP. It performs
  black-box evolutionary search over a compact tabular leader mechanism and
  evaluates each candidate by running MW follower response plus the same BCCE
  and welfare diagnostics used for MSPM.
