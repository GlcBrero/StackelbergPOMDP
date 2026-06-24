#!/bin/bash
#SBATCH --job-name=mib_gr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/mibuari_grid/slurm_%A_%a.out
#SBATCH --error=results/mibuari_grid/slurm_%A_%a.err
#SBATCH --array=1-20

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 2 grids x 2 obs x 5 seeds = 20 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 5) + 1 ))
TASK=$((TASK / 5))
OBS_IDX=$((TASK % 2))
GRID_IDX=$((TASK / 2))

OBS_TYPES=(price_profile no_observation)
OBS=${OBS_TYPES[$OBS_IDX]}

# Grid configs: m=4 or m=5
GRID_SIZES=(4 5)
BETAS=(0.01 4e-5)
EQ_STEPS=(50000 50000)
M=${GRID_SIZES[$GRID_IDX]}
BETA=${BETAS[$GRID_IDX]}
EQ=${EQ_STEPS[$GRID_IDX]}

echo "m=$M, Obs=$OBS, Seed=$SEED, beta=$BETA, eq=$EQ"

python -u main_args.py \
    --experiment_type bertrand \
    --platform_intervention learn_threshold \
    --platform_observation_space $OBS \
    --price_grid_length $M \
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
    --follower_beta $BETA
