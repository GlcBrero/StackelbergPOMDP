#!/bin/bash
#SBATCH --job-name=matrix-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=replication/matrix_ablations/slurm_logs/%x_%j.out
#SBATCH --error=replication/matrix_ablations/slurm_logs/%x_%j.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
cd "$PROJECT_DIR"
mkdir -p replication/matrix_ablations/slurm_logs

python replication/matrix_ablations/gate_e1.py \
  --sweep-root replication/matrix_ablations/results/appendix-v1 \
  --max-regret 0.25
