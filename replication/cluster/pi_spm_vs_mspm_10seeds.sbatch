#!/bin/bash
#SBATCH --job-name=pi_spm_mspm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=simulation_logs/cluster_pi_spm_vs_mspm_10seeds/slurm_%A_%a.out
#SBATCH --error=simulation_logs/cluster_pi_spm_vs_mspm_10seeds/slurm_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp

set -euo pipefail

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP

OUTPUT_DIR="simulation_logs/cluster_pi_spm_vs_mspm_10seeds"
mkdir -p "${OUTPUT_DIR}"

if (( SLURM_ARRAY_TASK_ID < 10 )); then
    METHOD="mspm"
    SEED=$((SLURM_ARRAY_TASK_ID + 1))
else
    METHOD="spm"
    SEED=$((SLURM_ARRAY_TASK_ID - 9))
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/stackelberg-pomdp-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/stackelberg-pomdp-cache-${SLURM_JOB_ID}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_RUN_GROUP="pi_spm_vs_mspm_10seeds_20260713"
export WANDB_JOB_TYPE="${METHOD}"
export WANDB_TAGS="PI,${METHOD},seed_${SEED},working_mspm_d8a709a"

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${WANDB_DIR}"

echo "source_commit=d8a709a"
echo "method=${METHOD}"
echo "seed=${SEED}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
date

if [[ "${METHOD}" == "mspm" ]]; then
    python -u -m stackelberg_pomdp.experiments.mspm \
        --setting PI \
        --num_types 2 \
        --num_messages 2 \
        --seed "${SEED}" \
        --max_steps 10000000 \
        --tot_num_response_episodes 100 \
        --tot_num_reward_episodes 4 \
        --critic_obs full \
        --learning_rate 3e-4 \
        --ent_coef 0 \
        --ppo_episodes_per_batch 1 \
        --ppo_batch_size 64 \
        --ppo_n_epochs 10 \
        --reward_print_freq 10 \
        --progress_freq 50000 \
        --eval_freq 50000 \
        --eval_episodes 10 \
        --use_wandb
else
    python -u -m stackelberg_pomdp.experiments.spm_baseline \
        --setting PI \
        --num_types 2 \
        --seed "${SEED}" \
        --max_steps 10000000 \
        --discrete_prices false \
        --spm_exact_eval_freq 100000000 \
        --learning_rate 3e-4 \
        --ent_coef 0.01 \
        --reward_print_freq 1000 \
        --progress_freq 50000 \
        --use_wandb
fi

date
