"""
Run the price-deviation punishment diagnostic.

Loads converged Q-tables from calibrate_price_learners.py output, then:
1. Force one agent to deviate to static best response for 1 period
2. Let both agents play their learned strategies afterwards
3. Track price dynamics for ~20 periods
"""
import sys, os, argparse, pickle, glob
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from replication.bertrand.price_collusion import DEFAULT_PRICE_COLLUSION

POST_DEVIATION_PERIODS = 20


def build_parser():
    """Build the CLI parser used by both the manifest runner and this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Directory with saved Q-tables from calibrate_price_learners.py',
    )
    parser.add_argument(
        '--post_deviation_periods', type=int, default=POST_DEVIATION_PERIODS,
        help='Number of periods after the pre-deviation observation.',
    )
    return parser


def run_deviation(state, post_deviation_periods=POST_DEVIATION_PERIODS):
    """Run deviation experiment from a saved converged state."""
    from stable_baselines3.common import logger
    from stackelberg_pomdp.gym_envs.envs.base_envs import BertrandCompetitionEnv
    from stackelberg_pomdp.gym_envs.envs.wrappers import QLearningFollowersWrapper

    m = state['m']
    seed = state['seed']

    log = logger.configure(format_strings=[])
    env = BertrandCompetitionEnv(m=m, platform_intervention='no_intervene', seed=seed, logger=log)
    prices = env.action_price_space
    agents = env.followers_list
    wrapper = QLearningFollowersWrapper(
        env,
        alpha=DEFAULT_PRICE_COLLUSION.alpha,
        delta=DEFAULT_PRICE_COLLUSION.discount,
        beta=DEFAULT_PRICE_COLLUSION.beta,
    )
    wrapper.reset()

    # Load converged Q-tables and actions
    for agent in agents:
        wrapper.q_tables[agent] = state['q_tables'][agent]
    for agent in agents:
        wrapper.current_actions[agent] = state['current_actions'][agent]

    converged_actions = dict(state['current_actions'])
    deviator = agents[0]
    non_deviator = agents[1]

    # Compute static best response for deviator
    rival_action = converged_actions[non_deviator]
    rival_price = prices[rival_action]
    all_agents_list = list(range(env.num_agents))

    best_dev_action = converged_actions[deviator]
    best_dev_profit = -1
    for a_idx in range(m):
        p_dev = prices[a_idx]
        p_arr = np.array([p_dev, rival_price])
        demand = env._demand(env.a, p_arr, env.mu, 0, all_agents_list)
        profit = (p_dev - env.c_i) * demand
        if profit > best_dev_profit:
            best_dev_profit = profit
            best_dev_action = a_idx

    # Record trajectory
    traj_dev = [prices[converged_actions[deviator]]]     # tau=0: pre-deviation
    traj_non = [prices[converged_actions[non_deviator]]]

    # tau=1: force deviation, non-deviator plays greedy
    obs_key = tuple(converged_actions[a] for a in agents)

    # Set actions for this step
    actions_for_step = {}
    actions_for_step[deviator] = best_dev_action
    if obs_key in wrapper.q_tables[non_deviator]:
        actions_for_step[non_deviator] = int(np.argmax(wrapper.q_tables[non_deviator][obs_key]))
    else:
        actions_for_step[non_deviator] = converged_actions[non_deviator]

    # Execute step by setting current_actions and stepping
    wrapper.current_actions[deviator] = best_dev_action
    wrapper.current_actions[non_deviator] = actions_for_step[non_deviator]

    # Step env with no_intervene (platform action doesn't matter)
    all_actions = {a: wrapper.current_actions[a] for a in agents}
    all_actions[env.leader] = 0
    env.step(all_actions)

    traj_dev.append(prices[best_dev_action])
    traj_non.append(prices[actions_for_step[non_deviator]])

    # tau=2+: both play greedy from Q-tables
    for t in range(post_deviation_periods - 1):
        # Current state = last actions played
        obs_key = tuple(wrapper.current_actions[a] for a in agents)

        # Each agent plays greedy
        for agent in agents:
            if obs_key in wrapper.q_tables[agent]:
                wrapper.current_actions[agent] = int(np.argmax(wrapper.q_tables[agent][obs_key]))

        # Step env
        all_actions = {a: wrapper.current_actions[a] for a in agents}
        all_actions[env.leader] = 0
        env.step(all_actions)

        traj_dev.append(prices[wrapper.current_actions[deviator]])
        traj_non.append(prices[wrapper.current_actions[non_deviator]])

    return {
        'seed': seed,
        'converged_price': prices[converged_actions[deviator]],
        'deviation_price': prices[best_dev_action],
        'traj_dev': traj_dev,
        'traj_non': traj_non,
    }


if __name__ == '__main__':
    args = build_parser().parse_args()

    # Load all converged sessions
    files = sorted(glob.glob(os.path.join(args.input_dir, 'seed_*.pkl')))
    if not files:
        print(f"No Q-table files found in {args.input_dir}")
        sys.exit(1)

    states = []
    for f in files:
        with open(f, 'rb') as fh:
            s = pickle.load(fh)
        if s['converged']:
            states.append(s)

    from stackelberg_pomdp.gym_envs.envs.base_envs import BertrandCompetitionEnv

    m = states[0]['m']
    env = BertrandCompetitionEnv(m=m)
    prices = env.action_price_space
    p_N, p_M = env.p_nash, env.p_monopoly

    print(f"Price-deviation punishment diagnostic: m={m}, {len(states)} converged sessions")
    print(f"p^N={p_N:.4f}, p^M={p_M:.4f}")
    print(f"Grid: {[round(p,4) for p in prices]}")
    print()

    # Only use sessions that converged to symmetric collusive prices (above Nash)
    results = []
    for s in states:
        cp = s['prices']
        if abs(cp[0] - cp[1]) > 0.01:  # skip asymmetric
            continue
        if cp[0] <= p_N + 0.01:  # skip competitive
            continue
        res = run_deviation(
            s, post_deviation_periods=args.post_deviation_periods
        )
        print(f"  seed={res['seed']}: collusive={res['converged_price']:.4f}, "
              f"deviation={res['deviation_price']:.4f}")
        results.append(res)

    if not results:
        print("No suitable sessions found!")
        sys.exit(1)

    # Average trajectory
    T = args.post_deviation_periods + 1
    avg_dev = np.mean([r['traj_dev'][:T] for r in results], axis=0)
    avg_non = np.mean([r['traj_non'][:T] for r in results], axis=0)

    print(f"\nAverage trajectory ({len(results)} sessions):")
    print(f"{'tau':>4} {'Deviator':>10} {'Non-Dev':>10}")
    for t in range(T):
        print(f"{t:>4} {avg_dev[t]:>10.4f} {avg_non[t]:>10.4f}")

    print(f"\nNash={p_N:.4f}, Monopoly={p_M:.4f}")
    print(f"Long-run (collusive) price={avg_dev[0]:.4f}")
