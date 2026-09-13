#!/bin/bash
#SBATCH --job-name=matrix-plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
SWEEP_ROOT="replication/matrix_ablations/results/appendix-v1"
cd "$PROJECT_DIR"
mkdir -p "$MPLCONFIGDIR" replication/matrix_ablations/slurm_logs

python replication/matrix_ablations/trend.py --sweep-id appendix-v1
python replication/matrix_ablations/plot.py \
  --input "$SWEEP_ROOT" \
  --figure all \
  --paper-logs-root "$SWEEP_ROOT/paper_logs"
