#!/bin/bash
#SBATCH --job-name=matrix-meta-ray-free
#SBATCH --partition=cpu
#SBATCH --array=0-59%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=replication/matrix_ablations/slurm_logs/%x_%A_%a.out
#SBATCH --error=replication/matrix_ablations/slurm_logs/%x_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp
set -euo pipefail

# The base-environment sweep contains A2C/PPO records only.  Native ES is
# launched exclusively by hidden_queries_rllib_es_s1to25.sbatch.

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
  --sweep-id qualitative-v1 \
  --stage leader \
  --record-group meta-dependent \
  --record-index "$SLURM_ARRAY_TASK_ID" \
  --wandb
