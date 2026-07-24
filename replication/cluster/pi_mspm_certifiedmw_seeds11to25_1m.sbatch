#!/bin/bash
#SBATCH --job-name=pi_mspm_s11_25
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=simulation_logs/cluster_pi_mspm_certifiedmw_bestdiag_seeds11to25_1m_20260724/slurm_%A_%a.out
#SBATCH --error=simulation_logs/cluster_pi_mspm_certifiedmw_bestdiag_seeds11to25_1m_20260724/slurm_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp

set -euo pipefail

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP

OUTPUT_DIR="simulation_logs/cluster_pi_mspm_certifiedmw_bestdiag_seeds11to25_1m_20260724"
mkdir -p "${OUTPUT_DIR}"

SEED=$((SLURM_ARRAY_TASK_ID + 11))
MAX_STEPS=1000000
EVAL_FREQ=10000
WANDB_GROUP="pi_mspm_certifiedmw_bestdiag_seeds11to25_1m_20260724"
WANDB_COHORT="pi_mspm_certifiedmw_bestdiag_25seeds_1m"
JOB_TYPE="pi_mspm_certified_argmax_eval"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/stackelberg-pomdp-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/stackelberg-pomdp-cache-${SLURM_JOB_ID}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_JOB_TYPE="${JOB_TYPE}"
export WANDB_TAGS="PI,mspm,seed_${SEED},paper_figure,${WANDB_COHORT},certified_mw,exact_mw_actions,fixed_type_seed_0,bcce005,exact_argmax,exact_type_profiles,best_reward,response_diagnostics,eval_metrics,1m"

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${WANDB_DIR}"

echo "source_base_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
echo "source_patch=certified_mw_best_response_diagnostics_20260715"
echo "cohort=${WANDB_COHORT}"
echo "group=${WANDB_GROUP}"
echo "job_type=${JOB_TYPE}"
echo "setting=PI"
echo "method=mspm"
echo "num_types=2"
echo "num_messages=2"
echo "seed=${SEED}"
echo "mw_fixed_seed=0"
echo "response_prefix_games=100"
echo "reward_profiles=4"
echo "bcce_threshold=0.05"
echo "max_steps=${MAX_STEPS}"
echo "eval_freq=${EVAL_FREQ}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
date

python -u -m stackelberg_pomdp.experiments.mspm \
    --setting PI \
    --num_types 2 \
    --num_messages 2 \
    --seed "${SEED}" \
    --mw_fixed_seed 0 \
    --max_steps "${MAX_STEPS}" \
    --tot_num_response_episodes 100 \
    --tot_num_reward_episodes 4 \
    --response_bcce_threshold 0.05 \
    --response_bcce_min_records 1 \
    --response_bcce_check_freq 1 \
    --response_bcce_max_extra_updates 10000 \
    --critic_obs full \
    --learning_rate 3e-4 \
    --ent_coef 0 \
    --ppo_episodes_per_batch 1 \
    --ppo_batch_size 64 \
    --ppo_n_epochs 10 \
    --reward_print_freq 100 \
    --response_diagnostic_freq 0 \
    --progress_freq 50000 \
    --eval_freq "${EVAL_FREQ}" \
    --eval_episodes 1 \
    --eval_reward_episodes 4 \
    --use_wandb

date
