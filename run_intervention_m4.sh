#!/bin/bash
#SBATCH --job-name=int_m4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/intervention_m4/slurm_%A_%a.out
#SBATCH --error=results/intervention_m4/slurm_%A_%a.err
#SBATCH --array=1-10

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 2 obs x 5 seeds = 10 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 5) + 1 ))
OBS_IDX=$((TASK / 5))

OBS_TYPES=(price_profile no_observation)
OBS=${OBS_TYPES[$OBS_IDX]}

echo "Obs=$OBS, Seed=$SEED, m=4, eq=100k"

python -u main_args.py \
    --experiment_type bertrand \
    --platform_intervention learn_threshold \
    --platform_observation_space $OBS \
    --price_grid_length 4 \
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
    --response_phase_prob 1.0 \
    --follower_beta 0.01
