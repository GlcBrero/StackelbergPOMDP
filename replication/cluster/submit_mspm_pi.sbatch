#!/bin/bash
#SBATCH --job-name=mspm_pi
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH --output=simulation_logs/cluster_mspm_pi/slurm_%A_%a.out
#SBATCH --error=simulation_logs/cluster_mspm_pi/slurm_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP
mkdir -p simulation_logs/cluster_mspm_pi

SEEDS=(1 2 3 4 5)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stackelberg-pomdp-matplotlib
export XDG_CACHE_HOME=/tmp/stackelberg-pomdp-cache

python -u -m stackelberg_pomdp.experiments.mspm \
  --setting PI \
  --num_types 2 \
  --num_messages 2 \
  --seed "$SEED" \
  --max_steps 40000000 \
  --tot_num_response_episodes 300 \
  --tot_num_reward_episodes 30 \
  --response_bcce_threshold 0.05 \
  --response_bcce_min_records 1 \
  --response_bcce_check_freq 1 \
  --critic_obs full \
  --ent_coef 0.01 \
  --progress_freq 500000 \
  --reward_print_freq 1 \
  --response_diagnostic_freq 1 \
  --ppo_log_interval 10 \
  --use_wandb
