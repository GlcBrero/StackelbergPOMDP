import sys, os
# Add both the stackelberg_pomdp dir and its parent to path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, '..')
sys.path.insert(0, os.path.join(base_dir, 'stackelberg_pomdp'))
sys.path.insert(0, base_dir)
from run_setups import train_run

seed = int(sys.argv[1])
lam = float(sys.argv[2])
obs = sys.argv[3]

config = {
    'learning_method': 'RL:Standard',
    'experiment_type': 'bertrand',
    'max_steps': 50_000_000,
    'algorithm': 'A2C',
    'seed': seed,
    'training_seed': seed,
    'tot_num_reward_episodes': 30,
    'tot_num_eq_episodes': 50000,
    'critic_obs': 'full',
    'fix_episode_actions': 'True',
    'followers_algorithm': 'Qlearning',
    'response_phase_prob': 1.0,
    'platform_observation_space': obs,
    'platform_intervention': 'learn_threshold',
    'price_grid_length': 4,
    'price_min': 1.3,
    'price_max': 1.7,
    'intervention_lambda': lam,
    'use_wandb': False,
    'cost_perturbation': False,
    'warm_start_q': False,
    'ent_coef': 0.01,
}

train_run(config)
