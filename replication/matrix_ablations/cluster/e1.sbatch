#!/bin/bash
#SBATCH --job-name=matrix-e1
#SBATCH --partition=cpu
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=replication/matrix_ablations/slurm_logs/%x_%A_%a.out
#SBATCH --error=replication/matrix_ablations/slurm_logs/%x_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WANDB_SILENT=true

PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit this array from the repository root}"
cd "$PROJECT_DIR"
mkdir -p replication/matrix_ablations/slurm_logs

python replication/matrix_ablations/sweep.py run \
  --sweep-id appendix-v1 \
  --stage meta-follower \
  --record-index "$SLURM_ARRAY_TASK_ID" \
  --wandb
