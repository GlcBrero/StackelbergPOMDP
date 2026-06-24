#!/bin/bash
#SBATCH --job-name=bert_vno
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=results/bertrand_verify/slurm_%A_%a.out
#SBATCH --error=results/bertrand_verify/slurm_%A_%a.err
#SBATCH --array=1-2

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

case ${SLURM_ARRAY_TASK_ID} in
    1) ALG=A2C ;;
    2) ALG=PPO ;;
esac

echo "Config: obs=no_observation, alg=${ALG}"

python -u main_args.py \
    --experiment_type bertrand \
    --platform_intervention learn_threshold \
    --platform_observation_space no_observation \
    --price_grid_length 5 \
    --marginal_cost 1.0 \
    --num_pricing_agents 2 \
    --tot_num_eq_episodes 1000 \
    --tot_num_reward_episodes 10 \
    --algorithm ${ALG} \
    --max_steps 505000 \
    --critic_obs full \
    --fix_episode_actions True \
    --followers_algorithm Qlearning \
    --q_restart_rate -1 \
    --seed 1 \
    --learning_method RL:Standard \
    --response_phase_prob 1.0
