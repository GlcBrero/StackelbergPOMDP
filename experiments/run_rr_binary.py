"""Round-robin followers with binary threshold, lambda=0.5.

Usage: python run_rr_binary.py <seed> <obs_type> <sort_obs>
"""
import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, '..')
sys.path.insert(0, os.path.join(base_dir, 'stackelberg_pomdp'))
sys.path.insert(0, base_dir)
from run_setups import train_run

seed = int(sys.argv[1])
obs = sys.argv[2]
sort_obs = sys.argv[3] == 'sorted'

config = {
    'learning_method': 'RL:Standard',
    'experiment_type': 'bertrand',
    'max_steps': 50_000_000,
    'algorithm': 'A2C',
    'seed': seed,
    'training_seed': seed,
    'tot_num_reward_episodes': 30,
    'tot_num_response_episodes': 4,
    'critic_obs': 'full',
    'fix_episode_actions': 'True',
    'followers_algorithm': 'RoundRobin',
    'platform_observation_space': obs,
    'platform_intervention': 'learn_binary_threshold',
    'price_grid_length': 4,
    'price_min': 1.3,
    'price_max': 1.7,
    'leader_k': 1,
    'intervention_lambda': 0.5,
    'use_wandb': False,
    'cost_perturbation': False,
    'warm_start_q': False,
    'sort_obs': sort_obs,
    'ent_coef': 0.01,
}

train_run(config)
