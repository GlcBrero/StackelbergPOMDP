#!/bin/bash
#SBATCH --job-name=int25s
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/intervention_m4_25seeds/slurm_%A_%a.out
#SBATCH --error=results/intervention_m4_25seeds/slurm_%A_%a.err
#SBATCH --array=1-40

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 2 obs x 20 seeds (6-25) = 40 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 20) + 6 ))
OBS_IDX=$((TASK / 20))

OBS_TYPES=(price_profile no_observation)
OBS=${OBS_TYPES[$OBS_IDX]}

echo "m=4, Obs=$OBS, Seed=$SEED, a=0.25, b=1e-4, eq=50k"

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
    --follower_alpha 0.25 \
    --follower_beta 1e-4
