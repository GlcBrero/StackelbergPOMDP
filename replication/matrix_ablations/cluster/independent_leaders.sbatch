#!/bin/bash
#SBATCH --job-name=matrix-q
#SBATCH --partition=cpu
#SBATCH --array=0-59%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/gianluca_brero_uri_edu/StackelbergPOMDP-matrix-qualitative-20260728/replication/matrix_ablations/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/gianluca_brero_uri_edu/StackelbergPOMDP-matrix-qualitative-20260728/replication/matrix_ablations/slurm_logs/%x_%A_%a.err

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

PROJECT_DIR="$HOME/StackelbergPOMDP-matrix-qualitative-20260728"
cd "$PROJECT_DIR"

python replication/matrix_ablations/sweep.py run \
  --sweep-id qualitative-v1 \
  --stage leader \
  --record-group independent \
  --record-index "$SLURM_ARRAY_TASK_ID" \
  --wandb
