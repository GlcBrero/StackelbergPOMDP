#!/bin/bash
#SBATCH --job-name=matrix-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/home/gianluca_brero_uri_edu/StackelbergPOMDP-matrix-qualitative-20260728/replication/matrix_ablations/slurm_logs/%x_%j.out
#SBATCH --error=/home/gianluca_brero_uri_edu/StackelbergPOMDP-matrix-qualitative-20260728/replication/matrix_ablations/slurm_logs/%x_%j.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_DIR="$HOME/StackelbergPOMDP-matrix-qualitative-20260728"
cd "$PROJECT_DIR"

python replication/matrix_ablations/gate_e1.py \
  --sweep-root replication/matrix_ablations/results/qualitative-v1 \
  --max-regret 0.25
