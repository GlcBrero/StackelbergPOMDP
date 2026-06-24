#!/bin/bash
#SBATCH --job-name=ws_swp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/warmstart_sweep/slurm_%A_%a.out
#SBATCH --error=results/warmstart_sweep/slurm_%A_%a.err
#SBATCH --array=1-40

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 2 (warm/no) x 4 (eq steps) x 5 seeds = 40 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 5) + 1 ))
TASK=$((TASK / 5))
EQ_IDX=$((TASK % 4))
WARM_IDX=$((TASK / 4))

EQ_STEPS=(1000 5000 10000 20000)
EQ=${EQ_STEPS[$EQ_IDX]}

WARM_FLAGS=("" "--warm_start_q --q_tables_path ../results/warm_start_qtables/seed_${SEED}.pkl")
WARM_LABEL=("cold" "warm")
WARM=${WARM_FLAGS[$WARM_IDX]}
LABEL=${WARM_LABEL[$WARM_IDX]}

echo "${LABEL}, eq=${EQ}, Seed=$SEED"

python -u main_args.py \
    --experiment_type bertrand \
    --platform_intervention learn_threshold \
    --platform_observation_space price_profile \
    --price_grid_length 5 \
    --tot_num_eq_episodes $EQ \
    --tot_num_reward_episodes 30 \
    --algorithm A2C \
    --max_steps 50000000 \
    --critic_obs full \
    --fix_episode_actions True \
    --followers_algorithm Qlearning \
    --q_restart_rate -1 \
    --seed $SEED \
    --learning_method RL:Standard \
    --response_phase_prob 1.0 \
    --follower_beta 4e-5 \
    $WARM
