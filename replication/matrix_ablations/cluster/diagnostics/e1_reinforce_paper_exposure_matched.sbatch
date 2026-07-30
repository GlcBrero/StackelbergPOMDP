#!/bin/bash
#SBATCH --job-name=matrix-e1-paper-exposure
#SBATCH --partition=cpu
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
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
SEEDS=(1 2 1 2)
LEARNING_RATES=(0.01 0.01 0.02 0.02)
INDEX="$SLURM_ARRAY_TASK_ID"
SEED="${SEEDS[$INDEX]}"
LEARNING_RATE="${LEARNING_RATES[$INDEX]}"
NAME="paper_joint_v1-exposure-matched-lr${LEARNING_RATE}-seed${SEED}"

cd "$PROJECT_DIR"

python -m stackelberg_pomdp.experiments.matrix_ablations meta-follower \
  --matrix modified_pd \
  --memory-mode joint \
  --algorithm REINFORCE \
  --seed "$SEED" \
  --timesteps 130000 \
  --learning-rate "$LEARNING_RATE" \
  --output-root replication/matrix_ablations/results/e1_reinforce_paper_exposure_matched \
  --wandb \
  --wandb-project StackPOMDP \
  --wandb-group qualitative-v1-e1-reinforce-paper-exposure-matched \
  --wandb-name "$NAME"
