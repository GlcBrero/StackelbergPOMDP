#!/bin/bash
#SBATCH --job-name=sa_ablation
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-59
#SBATCH --output=results/simple_allocation_ablation/slurm_%A_%a.out
#SBATCH --error=results/simple_allocation_ablation/slurm_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp

set -euo pipefail

cd ~/StackelbergPOMDP

mkdir -p results/simple_allocation_ablation

targets=(
  fig_simple_allocation_stackpomdp_mappo
  fig_simple_allocation_hidden_queries_mappo
  fig_simple_allocation_reward_during_response_mappo
  fig_simple_allocation_stackpomdp_ppo
  fig_simple_allocation_hidden_queries_ppo
  fig_simple_allocation_reward_during_response_ppo
)

target_idx=$((SLURM_ARRAY_TASK_ID / 10))
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1))
target="${targets[$target_idx]}"

echo "target=${target}"
echo "seed=${seed}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
date

python replication/run.py "${target}" \
  --seed "${seed}" \
  --max_steps 5000000 \
  --progress_freq 100000

date
