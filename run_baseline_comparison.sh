#!/bin/bash
#SBATCH --job-name=basecmp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/baseline_comparison/slurm_%A_%a.out
#SBATCH --error=results/baseline_comparison/slurm_%A_%a.err
#SBATCH --array=1-30

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# Decode: 3 conditions x 2 obs x 5 seeds = 30 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 5) + 1 ))
TASK=$((TASK / 5))
OBS_IDX=$((TASK % 2))
COND=$((TASK / 2))

OBS_TYPES=(price_profile no_observation)
OBS=${OBS_TYPES[$OBS_IDX]}

# Common args
COMMON="--experiment_type bertrand \
    --platform_intervention learn_threshold \
    --platform_observation_space $OBS \
    --price_grid_length 5 \
    --tot_num_eq_episodes 50000 \
    --tot_num_reward_episodes 30 \
    --algorithm A2C \
    --max_steps 50000000 \
    --critic_obs full \
    --fix_episode_actions True \
    --followers_algorithm Qlearning \
    --q_restart_rate -1 \
    --seed $SEED \
    --learning_method RL:Standard \
    --response_phase_prob 1.0"

if [ $COND -eq 0 ]; then
    echo "Condition: OLD_GRID, Obs=$OBS, Seed=$SEED"
    python -u main_args.py $COMMON \
        --grid_lower_bound 0.95 --grid_upper_bound 2.1
elif [ $COND -eq 1 ]; then
    echo "Condition: NEW_GRID, Obs=$OBS, Seed=$SEED"
    python -u main_args.py $COMMON
elif [ $COND -eq 2 ]; then
    echo "Condition: NEW_GRID+WARMSTART, Obs=$OBS, Seed=$SEED"
    python -u main_args.py $COMMON --warm_start_q
fi
