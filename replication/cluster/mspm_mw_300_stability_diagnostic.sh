#!/bin/bash
#SBATCH --job-name=mspm_stab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --output=results/mspm_mw_300_stability_diagnostic/slurm_%A_%a.out
#SBATCH --error=results/mspm_mw_300_stability_diagnostic/slurm_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp

set -euo pipefail

cd ~/StackelbergPOMDP

mkdir -p results/mspm_mw_300_stability_diagnostic

configs=(
  "0.02"
  "0.05"
  "0.10"
  "0.20"
)

mw_epsilon="${configs[$SLURM_ARRAY_TASK_ID]}"

echo "mw_epsilon=${mw_epsilon}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
date

python -m stackelberg_pomdp.experiments.mspm \
  --setting MSGSpace \
  --num_types 3 \
  --num_messages 2 \
  --seed 1 \
  --max_steps 2000000 \
  --algorithm PPO \
  --mw_epsilon "${mw_epsilon}" \
  --reward_phase_mode exact_expectation \
  --tot_num_response_episodes 300 \
  --tot_num_reward_episodes 100 \
  --response_diagnostic_freq 1 \
  --reward_print_freq 1 \
  --eval_freq 0 \
  --progress_freq 100000

date
