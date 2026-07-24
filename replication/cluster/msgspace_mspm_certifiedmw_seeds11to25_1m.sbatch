#!/bin/bash
#SBATCH --job-name=msg_mspm_s11_25
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-59
#SBATCH --output=simulation_logs/cluster_msgspace_mspm_certifiedmw_bestdiag_seeds11to25_1m_20260724/slurm_%A_%a.out
#SBATCH --error=simulation_logs/cluster_msgspace_mspm_certifiedmw_bestdiag_seeds11to25_1m_20260724/slurm_%A_%a.err

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp

set -euo pipefail

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP

OUTPUT_DIR="simulation_logs/cluster_msgspace_mspm_certifiedmw_bestdiag_seeds11to25_1m_20260724"
mkdir -p "${OUTPUT_DIR}"

TYPE_COUNTS=(3 4 5 6)
RESPONSE_GAMES=(300 532 832 1200)
TYPE_INDEX=$((SLURM_ARRAY_TASK_ID / 15))
SEED=$((SLURM_ARRAY_TASK_ID % 15 + 11))
NUM_TYPES="${TYPE_COUNTS[TYPE_INDEX]}"
RESPONSE_EPISODES="${RESPONSE_GAMES[TYPE_INDEX]}"
REWARD_PROFILES=$((NUM_TYPES * NUM_TYPES))

MAX_STEPS=1000000
EVAL_FREQ=10000
WANDB_COHORT="msgspace_mspm_certifiedmw_bestdiag_25seeds_1m"
WANDB_GROUP="msgspace_mspm_certifiedmw_bestdiag_types${NUM_TYPES}_seeds11to25_1m_20260724"
JOB_TYPE="msgspace_mspm_certified_argmax_eval"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/stackelberg-pomdp-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/stackelberg-pomdp-cache-${SLURM_JOB_ID}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_JOB_TYPE="${JOB_TYPE}"
export WANDB_TAGS="MSGSpace,mspm,types_${NUM_TYPES},messages_2,seed_${SEED},paper_scaling,${WANDB_COHORT},certified_mw,exact_mw_actions,fixed_type_seed_0,bcce005,exact_argmax,exact_type_profiles,best_reward,response_diagnostics,eval_metrics,1m"

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${WANDB_DIR}"

echo "source_base_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"
echo "source_patch=certified_mw_best_response_diagnostics_20260716"
echo "cohort=${WANDB_COHORT}"
echo "group=${WANDB_GROUP}"
echo "job_type=${JOB_TYPE}"
echo "setting=MSGSpace"
echo "method=mspm"
echo "num_types=${NUM_TYPES}"
echo "num_messages=2"
echo "seed=${SEED}"
echo "mw_fixed_seed=0"
echo "response_prefix_games=${RESPONSE_EPISODES}"
echo "reward_profiles=${REWARD_PROFILES}"
echo "bcce_threshold=0.05"
echo "max_steps=${MAX_STEPS}"
echo "eval_freq=${EVAL_FREQ}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
date

python -u -m stackelberg_pomdp.experiments.mspm \
    --setting MSGSpace \
    --num_types "${NUM_TYPES}" \
    --num_messages 2 \
    --seed "${SEED}" \
    --mw_fixed_seed 0 \
    --max_steps "${MAX_STEPS}" \
    --tot_num_response_episodes "${RESPONSE_EPISODES}" \
    --tot_num_reward_episodes "${REWARD_PROFILES}" \
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
    --eval_reward_episodes "${REWARD_PROFILES}" \
    --use_wandb

date
