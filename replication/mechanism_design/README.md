# Mechanism-Design Replication

Paper targets owned by this codebase:

- `fig_simple_allocation_stackpomdp_mappo`
- `fig_simple_allocation_stackpomdp_mappo_m1`
- `fig_simple_allocation_stackpomdp_mappo_m2`
- `fig_simple_allocation_hidden_queries_mappo`
- `fig_simple_allocation_stackpomdp_ppo`
- `fig_simple_allocation_hidden_queries_ppo`
- `fig_pi_mspm_2types_2messages`
- `fig_pi_spm_2types`
- `table_mspm_3types_2messages`
- `table_mspm_4types_2messages`
- `table_mspm_5types_2messages`
- `table_mspm_6types_2messages`
- `table_spm_exact_optima` (analytic; no training command)

These currently run through `replication/run.py`, which invokes the reusable
StackPOMDP training entrypoint with the paper configuration.

The manifest records the 25-seed Simple Allocation/MSPM cohorts, the 10-seed
SPM comparator, exact MW response-prefix schedules, BCCE threshold, and
evaluation cadence.  Figure aggregation is maintained in the journal
reproducibility bundle.
