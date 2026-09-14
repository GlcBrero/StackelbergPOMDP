# Unity reruns

`cohort.py` expands the approved mechanism-design cohort directly from
`../targets.json`: six Simple Allocation configurations × 25 seeds, two
Basic-POMDP Matrix Design configurations × 25 seeds, and ten standard-SPM
seeds. It preserves the paper budgets and evaluation settings. Only the
frequency of printed training-reward diagnostics is reduced.

Run from a dedicated source snapshot, so existing results cannot be replaced:

```bash
python replication/cluster/cohort.py plan --batch-root /path/to/batch
python replication/matrix_ablations/sweep.py plan \
  --results-root /path/to/batch/appendix --sweep-id appendix-v1
mkdir -p /path/to/batch/slurm
```

`run.sbatch` accepts a stage and the batch root. Supply the job name, log paths,
array range, concurrency limit, memory, and wall time to `sbatch`. For example:

```bash
sbatch --job-name=jair-sa --array=0-149%25 \
  --output=/path/to/batch/slurm/%x_%A_%a.out \
  --error=/path/to/batch/slurm/%x_%A_%a.err \
  replication/cluster/run.sbatch simple_allocation /path/to/batch
```

| Stage | Array | Purpose |
|---|---|---|
| `simple_allocation` | 0–149 | Six configurations, 25 seeds each |
| `matrix_design` | 0–49 | Basic MAPPO and Basic PPO, 25 seeds each |
| `spm` | 0–9 | Standard-SPM baseline |
| `meta-follower` | 0–9 | Ten appendix response models |
| `independent` | 0–39 | Reward-timing appendix leaders |
| `gate` | Single job | Certify all ten response models |
| `meta-dependent` | 0–59 | Phase-awareness appendix leaders |
| `plot` | Single job | Check complete appendix cohorts and generate figures |

Use `afterok` dependencies: follower array → gate → dependent leader array.
Plotting requires both leader arrays to succeed. A failed gate must be
investigated; do not bypass it for scientific results. `preflight` is a utility
stage for a batch-local `preflight.py`; its reduced-budget outputs are separate
from scientific runs.

The batch directory stores the exact source manifest, resolved configurations,
plans, scheduler IDs, and environment inventory. Each main run writes
`mechanism_runs/<record>/run.json` and `training.log`; its checkpoints and CSVs
are under the frozen repository's `stackelberg_pomdp/logs/`. Successful run
records include artifact hashes. Appendix runs retain their existing immutable
manifests and checkpoints under `appendix/appendix-v1/`.

The runner refuses existing output directories and changed target manifests.
For interrupted jobs, inspect scheduler accounting as well as the run record:
a forced termination can leave a record marked `running`. Preserve failed
attempts before preparing any retry. The current approved scope ignores the
critic flag as a reason to add more cohorts; it uses the current corrected code.
