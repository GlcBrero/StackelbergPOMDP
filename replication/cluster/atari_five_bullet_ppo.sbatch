#!/bin/bash
#SBATCH --job-name=atari_5bullet_ppo
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=replication/atari/logs/slurm_%j.out
#SBATCH --error=replication/atari/logs/slurm_%j.err

set -euo pipefail

source /etc/profile
module load conda/latest
conda activate stackerlberg_atari

cd "$HOME/StackelbergPOMDP"
mkdir -p replication/atari/checkpoints replication/atari/logs
export PYTHONPATH="$HOME/StackelbergPOMDP:$HOME/StackeRLberg"
export WANDB_START_METHOD=thread

python -u -m replication.atari.train_five_bullet_gameplay \
  --seed 1 \
  --timesteps 10000000 \
  --num-workers 16 \
  --num-envs-per-worker 4 \
  --checkpoint replication/atari/checkpoints/space_invaders_5bullets_ppo_seed1.pkl \
  --checkpoint-every 25 \
  --eval-episodes 20 \
  --wandb \
  --wandb-entity glcbrero \
  --wandb-name five_bullet_ppo_seed1_10m_unity
