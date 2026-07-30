#!/bin/bash
#SBATCH --job-name=matrix-e1-legacy-pg
#SBATCH --partition=cpu
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/home/gianluca_brero_uri_edu/StackelbergPOMDP-matrix-e1-reinforce-diagnostic-20260728/replication/matrix_ablations/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/gianluca_brero_uri_edu/StackelbergPOMDP-matrix-e1-reinforce-diagnostic-20260728/replication/matrix_ablations/slurm_logs/%x_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WANDB_SILENT=true

PROJECT_DIR="$HOME/StackelbergPOMDP-matrix-e1-reinforce-diagnostic-20260728"
SEEDS=(1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
NAME="legacy_opponent_v1-reinforce-lr0.02-seed${SEED}"

cd "$PROJECT_DIR"

python -m stackelberg_pomdp.experiments.matrix_ablations meta-follower \
  --matrix modified_pd \
  --profile-id legacy_opponent_v1 \
  --algorithm REINFORCE \
  --seed "$SEED" \
  --timesteps 32500 \
  --learning-rate 0.02 \
  --output-root replication/matrix_ablations/results/e1_reinforce_legacy_parity \
  --wandb \
  --wandb-project StackPOMDP \
  --wandb-group qualitative-v1-e1-reinforce-legacy-parity \
  --wandb-name "$NAME"
