#!/bin/bash
#SBATCH --job-name=bert_nos
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/bertrand_nostate/slurm_%A_%a.out
#SBATCH --error=results/bertrand_nostate/slurm_%A_%a.err
#SBATCH --array=1-3

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

python -u main_args.py \
    --experiment_type bertrand \
    --platform_intervention learn_threshold \
    --platform_observation_space no_observation \
    --price_grid_length 5 \
    --marginal_cost 1.0 \
    --num_pricing_agents 2 \
    --tot_num_eq_episodes 50000 \
    --tot_num_reward_episodes 30 \
    --algorithm A2C \
    --max_steps 50000000 \
    --critic_obs full \
    --fix_episode_actions True \
    --followers_algorithm Qlearning \
    --q_restart_rate -1 \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --learning_method RL:Standard \
    --response_phase_prob 1.0
