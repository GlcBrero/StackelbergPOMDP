#!/bin/bash
#SBATCH --job-name=atari_ppo_ammo
#SBATCH --partition=cpu
#SBATCH --qos=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --exclusive
#SBATCH --array=1-3
#SBATCH --output=replication/atari/unity_logs/ammo_aware_ppo_%A_%a.out
#SBATCH --error=replication/atari/unity_logs/ammo_aware_ppo_%A_%a.err

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
checkpoint="replication/atari/checkpoints/unity/space_invaders_5bullets_ppo_ammo_mask_seed${seed}_10m.pkl"

python -u -m replication.atari.train_five_bullet_gameplay \
    --algorithm PPO \
    --ammo-aware \
    --ammo-hidden 32 \
    --seed "$seed" \
    --timesteps 10000000 \
    --num-workers 6 \
    --num-envs-per-worker 1 \
    --rollout-fragment-length 100 \
    --train-batch-size 5000 \
    --sgd-minibatch-size 100 \
    --num-sgd-iter 10 \
    --learning-rate 2.5e-4 \
    --entropy-coeff 0.01 \
    --clip-param 0.1 \
    --checkpoint "$checkpoint" \
    --checkpoint-every 50 \
    --archive-checkpoints \
    --eval-episodes 20 \
    --wandb \
    --wandb-entity glcbrero \
    --wandb-name "five_bullet_ppo_ammo_mask_seed${seed}_10m_unity"
