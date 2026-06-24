#!/bin/bash
#SBATCH --job-name=bert_vfy
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=results/bertrand_verify/slurm_%A_%a.out
#SBATCH --error=results/bertrand_verify/slurm_%A_%a.err
#SBATCH --array=1-4

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 4 configurations: (obs_space, algorithm)
# 1: price_profile, A2C
# 2: price_profile, PPO
# 3: no_observation, A2C
# 4: no_observation, PPO

case ${SLURM_ARRAY_TASK_ID} in
    1) OBS=price_profile; ALG=A2C ;;
    2) OBS=price_profile; ALG=PPO ;;
    3) OBS=no_observation; ALG=A2C ;;
    4) OBS=no_observation; ALG=PPO ;;
esac

echo "Config: obs=${OBS}, alg=${ALG}"

python -u main_args.py \
    --experiment_type bertrand \
    --platform_intervention learn_threshold \
    --platform_observation_space ${OBS} \
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
