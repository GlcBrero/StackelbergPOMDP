#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --no-requeue

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/stackpomdp-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/stackpomdp-cache-${SLURM_JOB_ID}"

cd "${SLURM_SUBMIT_DIR:?submit from the frozen repository root}"
BATCH_ROOT="${2:?provide the batch root}"
APPENDIX_ROOT="$BATCH_ROOT/appendix"
SWEEP_ID="appendix-v1"

case "${1:?provide a stage}" in
  simple_allocation|matrix_design|spm)
    exec python replication/cluster/cohort.py run --batch-root "$BATCH_ROOT" \
      --group "$1" --record-index "${SLURM_ARRAY_TASK_ID:?array task required}" ;;
  meta-follower|independent|meta-dependent)
    STAGE=leader
    GROUP="$1"
    if [[ "$1" == meta-follower ]]; then STAGE=meta-follower; GROUP=all; fi
    exec python replication/matrix_ablations/sweep.py run \
      --results-root "$APPENDIX_ROOT" --sweep-id "$SWEEP_ID" \
      --stage "$STAGE" --record-group "$GROUP" \
      --record-index "${SLURM_ARRAY_TASK_ID:?array task required}" --no-wandb ;;
  gate)
    exec python replication/matrix_ablations/gate_e1.py \
      --sweep-root "$APPENDIX_ROOT/$SWEEP_ID" --max-regret 0.25 ;;
  plot)
    python replication/matrix_ablations/trend.py \
      --results-root "$APPENDIX_ROOT" --sweep-id "$SWEEP_ID"
    exec python replication/matrix_ablations/plot.py \
      --input "$APPENDIX_ROOT/$SWEEP_ID" --figure all \
      --paper-logs-root "$APPENDIX_ROOT/$SWEEP_ID/paper_logs" ;;
  preflight)
    exec python "$BATCH_ROOT/preflight.py" "$BATCH_ROOT" ;;
  *) printf 'Unknown stage: %s\n' "$1" >&2; exit 2 ;;
esac
