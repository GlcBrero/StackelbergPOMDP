#!/bin/bash
#SBATCH --job-name=matrix-e1diag
#SBATCH --partition=cpu
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/gianluca_brero_uri_edu/StackelbergPOMDP-matrix-e1-diagnostic-20260728/replication/matrix_ablations/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/gianluca_brero_uri_edu/StackelbergPOMDP-matrix-e1-diagnostic-20260728/replication/matrix_ablations/slurm_logs/%x_%A_%a.err

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

PROJECT_DIR="$HOME/StackelbergPOMDP-matrix-e1-diagnostic-20260728"
ALGORITHMS=(PPO PPO PPO PPO PPO PPO PPO A2C)
TIMESTEPS=(500000 1000000 500000 500000 500000 1000000 500000 500000)
LEARNING_RATES=(0.002 0.002 0.001 0.0005 0.002 0.002 0.001 0.0007)
NET_ARCHES=(32,32 32,32 32,32 32,32 128,128 128,128 128,128 128,128)
INDEX="$SLURM_ARRAY_TASK_ID"
NAME="${ALGORITHMS[$INDEX]}-t${TIMESTEPS[$INDEX]}-lr${LEARNING_RATES[$INDEX]}-net${NET_ARCHES[$INDEX]}"
cd "$PROJECT_DIR"

python -m stackelberg_pomdp.experiments.matrix_ablations meta-follower \
  --matrix modified_pd \
  --memory-mode joint \
  --algorithm "${ALGORITHMS[$INDEX]}" \
  --seed 1 \
  --timesteps "${TIMESTEPS[$INDEX]}" \
  --learning-rate "${LEARNING_RATES[$INDEX]}" \
  --net-arch "${NET_ARCHES[$INDEX]}" \
  --checkpoint-every 100000 \
  --output-root replication/matrix_ablations/results/e1_diagnostic \
  --wandb \
  --wandb-project StackPOMDP \
  --wandb-group qualitative-v1-e1-diagnostic \
  --wandb-name "$NAME"
