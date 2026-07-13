"""
Evaluate a trained leader policy: run Q-learners under the learned intervention
and check if they still collude.

Loads a trained A2C model, runs Q-learners for many steps with the leader
setting thresholds from its policy, and measures follower convergence.
"""
import sys, os, argparse, glob
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from replication.bertrand.price_collusion import (
    PAPER_INTERVENTION_PRICE_COLLUSION,
    average_symmetric_gain_index,
)

ALPHA = PAPER_INTERVENTION_PRICE_COLLUSION.alpha
BETA = PAPER_INTERVENTION_PRICE_COLLUSION.beta
DELTA = PAPER_INTERVENTION_PRICE_COLLUSION.discount
EVAL_STEPS = 250000


def eval_model(model_path, config):
    """Load a trained model and run Q-learners under its intervention."""
    from stable_baselines3.common import logger
    from stackelberg_pomdp.baselines_utils import CustomA2C
    from stackelberg_pomdp.env_setups import get_bertrand_env

    log = logger.configure(format_strings=[])
    config['logger'] = log

    # Create env
    env = get_bertrand_env(config)
    env.is_eval = True

    # Load model
    model = CustomA2C.load(model_path, env=env)

    # Run evaluation episodes
    obs = env.reset()
    cs_values = []
    prices_history = []

    for step in range(EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if info.get('is_reward_phase', False) and info.get('reward_generated', False):
            cs = info.get('surplus', 0)
            cs_values.append(cs)
            if 'followers_actions' in info:
                prices_history.append(list(info['followers_actions'].values()))

        if done:
            obs = env.reset()

    return cs_values, prices_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to log directory containing trained model')
    parser.add_argument('--obs_type', type=str, default='no_observation')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Find latest model
    models = sorted(glob.glob(os.path.join(args.log_dir, 'rl_model_*_steps.zip')))
    if not models:
        print(f"No models found in {args.log_dir}")
        sys.exit(1)
    model_path = models[-1]

    config = {
        'seed': args.seed,
        'price_grid_length': 4,
        'platform_intervention': 'learn_threshold',
        'platform_observation_space': args.obs_type,
        'tot_num_response_episodes': 50000,
        'tot_num_reward_episodes': 30,
        'critic_obs': 'full',
        'intervention_lambda': 0.0,
        'follower_alpha': ALPHA,
        'follower_beta': BETA,
        'fix_episode_actions': 'True',
        'followers_algorithm': 'Qlearning',
        'q_restart_rate': -1,
    }

    print(f"Evaluating: {model_path}")
    print(f"Obs: {args.obs_type}, Seed: {args.seed}")

    cs_values, prices_history = eval_model(model_path, config)

    if cs_values:
        print(f"\nResults ({len(cs_values)} reward steps):")
        print(f"  Mean CS: {np.mean(cs_values):.3f}")
        print(f"  Last 100 CS: {np.mean(cs_values[-100:]):.3f}")

    if prices_history:
        last_prices = prices_history[-100:]
        price_counts = {}
        for p in last_prices:
            key = tuple(p)
            price_counts[key] = price_counts.get(key, 0) + 1
        print(f"  Last 100 price pairs:")
        for k, v in sorted(price_counts.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v} times")

    # Compute Delta
    from stackelberg_pomdp.gym_envs.envs.base_envs import BertrandCompetitionEnv

    env = BertrandCompetitionEnv(m=4)
    prices = env.action_price_space
    p_N, p_M = env.p_nash, env.p_monopoly

    if prices_history:
        last_prices_flat = [prices[p] for pair in last_prices for p in pair]
        avg_price = np.mean(last_prices_flat)
        delta = average_symmetric_gain_index(last_prices_flat, p_N, p_M)
        print(f"\n  Avg price: {avg_price:.3f}")
        print(f"  Delta: {delta:.3f} ({'COLLUSION' if delta > 0.5 else 'COMPETITIVE'})")
