#!/bin/bash
#SBATCH --job-name=mspm_last_det
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --output=simulation_logs/cluster_mspm_msgspace/slurm_%A_%a.out
#SBATCH --error=simulation_logs/cluster_mspm_msgspace/slurm_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP
mkdir -p simulation_logs/cluster_mspm_msgspace

CONFIGS=(
  "3 2"
  "3 3"
  "4 2"
  "4 3"
  "5 2"
  "5 3"
)

read NUM_TYPES NUM_MESSAGES <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stackelberg-pomdp-matplotlib
export XDG_CACHE_HOME=/tmp/stackelberg-pomdp-cache

python -u -m stackelberg_pomdp.experiments.mspm \
  --setting MSGSpace \
  --num_types "$NUM_TYPES" \
  --num_messages "$NUM_MESSAGES" \
  --seed 1 \
  --max_steps 1000000 \
  --tot_num_response_episodes 300 \
  --tot_num_reward_episodes 30 \
  --response_bcce_threshold 0.05 \
  --response_bcce_min_records 1 \
  --response_bcce_check_freq 1 \
  --critic_obs full \
  --ent_coef 0.01 \
  --progress_freq 50000 \
  --reward_print_freq 1 \
  --response_diagnostic_freq 1 \
  --ppo_log_interval 10 \
  --use_wandb
