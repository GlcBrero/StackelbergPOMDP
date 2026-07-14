#!/bin/bash
#SBATCH --job-name=atari_a3c_ammo10m
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --exclusive
#SBATCH --array=1-3
#SBATCH --output=replication/atari/unity_logs/ammo_aware_a3c_10m_%A_%a.out
#SBATCH --error=replication/atari/unity_logs/ammo_aware_a3c_10m_%A_%a.err

source /etc/profile
set -euo pipefail
module load conda/latest
conda activate stackerlberg_atari

cd "$HOME/StackelbergPOMDP"
export PYTHONPATH="$HOME/StackelbergPOMDP:$HOME/StackeRLberg:${PYTHONPATH:-}"
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray_atari_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p replication/atari/checkpoints/unity replication/atari/unity_logs "$RAY_TMPDIR"

seed="$SLURM_ARRAY_TASK_ID"
checkpoint="replication/atari/checkpoints/unity/space_invaders_5bullets_a3c_ammo_mask_seed${seed}_10m.pkl"

python -u -m replication.atari.train_five_bullet_gameplay \
    --algorithm A3C \
    --ammo-aware \
    --ammo-hidden 32 \
    --seed "$seed" \
    --timesteps 10000000 \
    --num-workers 6 \
    --num-envs-per-worker 1 \
    --timesteps-per-iteration 5000 \
    --learning-rate 1e-4 \
    --entropy-coeff 0.01 \
    --checkpoint "$checkpoint" \
    --checkpoint-every 25 \
    --archive-checkpoints \
    --eval-episodes 20 \
    --wandb \
    --wandb-entity glcbrero \
    --wandb-name "five_bullet_a3c_ammo_mask_seed${seed}_10m_unity"
