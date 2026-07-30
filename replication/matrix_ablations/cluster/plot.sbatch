#!/bin/bash
#SBATCH --job-name=matrix-plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"

PROJECT_DIR="$HOME/StackelbergPOMDP-matrix-qualitative-20260728"
SWEEP_ROOT="replication/matrix_ablations/results/qualitative-v1"
cd "$PROJECT_DIR"
mkdir -p "$MPLCONFIGDIR"

python replication/matrix_ablations/trend.py --sweep-id qualitative-v1
python replication/matrix_ablations/plot.py \
  --input "$SWEEP_ROOT" \
  --figure all \
  --paper-logs-root "$SWEEP_ROOT/paper_logs"
