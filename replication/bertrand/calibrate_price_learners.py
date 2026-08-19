"""
Verify Q-learning price collusion following the prior algorithmic-collusion literature.
Runs Q-learners through BertrandCompetitionEnv with no platform intervention.

Saves converged Q-tables to --output_dir for use by punishment_diagnostic.py.
"""
import sys, os, argparse, pickle
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from replication.bertrand.price_collusion import (
    DEFAULT_PRICE_COLLUSION,
    average_symmetric_gain_index,
    symmetric_profit,
)

ALPHA = DEFAULT_PRICE_COLLUSION.alpha
BETA = DEFAULT_PRICE_COLLUSION.beta
DELTA = DEFAULT_PRICE_COLLUSION.discount
CONVERGENCE_WINDOW = DEFAULT_PRICE_COLLUSION.convergence_window
MAX_STEPS = DEFAULT_PRICE_COLLUSION.max_steps


def build_parser():
    """Build the CLI parser used by both the manifest runner and this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--beta', type=float, default=4e-6)
    parser.add_argument('--platform_intervention', type=str, default='no_intervene')
    parser.add_argument('--n_sessions', type=int, default=100)
    parser.add_argument('--seed_offset', type=int, default=0)
    parser.add_argument('--convergence_window', type=int, default=CONVERGENCE_WINDOW)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save converged Q-tables. Default: results/price_collusion_m{m}/',
    )
    return parser


def run_session(
        m,
        seed,
        alpha=None,
        beta=None,
        platform_intervention='no_intervene',
        convergence_window=CONVERGENCE_WINDOW,
        max_steps=MAX_STEPS,
):
    from stable_baselines3.common import logger
    from stackelberg_pomdp.gym_envs.envs.base_envs import BertrandCompetitionEnv
    from stackelberg_pomdp.gym_envs.envs.wrappers import QLearningFollowersWrapper

    alpha = alpha or ALPHA
    beta = beta or BETA
    log = logger.configure(format_strings=[])
    env = BertrandCompetitionEnv(m=m, platform_intervention=platform_intervention, seed=seed, logger=log)
    wrapper = QLearningFollowersWrapper(env, alpha=alpha, delta=DELTA, beta=beta)

    wrapper.reset()
    wrapper.this_step_mode = "standard"

    prev_greedy = None
    stable_count = 0

    for step in range(max_steps):
        wrapper.step(0)

        current_greedy = {
            (agent, sk): int(np.argmax(qv))
            for agent, qt in wrapper.q_tables.items()
            for sk, qv in qt.items()
        }

        if current_greedy == prev_greedy:
            stable_count += 1
        else:
            stable_count = 0
            prev_greedy = current_greedy

        if stable_count >= convergence_window:
            converged = True
            break
    else:
        converged = False

    prices = env.action_price_space
    cp = [prices[wrapper.current_actions[a]] for a in env.followers_list]

    # Package converged state for saving
    state = {
        'q_tables': {a: dict(qt) for a, qt in wrapper.q_tables.items()},
        'current_actions': dict(wrapper.current_actions),
        'converged': converged,
        'steps': step + 1 if converged else max_steps,
        'prices': cp,
        'seed': seed,
        'm': m,
    }
    return state


if __name__ == '__main__':
    args = build_parser().parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), f'results/price_collusion_m{args.m}')
    os.makedirs(output_dir, exist_ok=True)

    from stackelberg_pomdp.gym_envs.envs.base_envs import BertrandCompetitionEnv

    env = BertrandCompetitionEnv(m=args.m)
    prices = env.action_price_space
    p_N, p_M = env.p_nash, env.p_monopoly

    pi_N = symmetric_profit(p_N)
    pi_M = symmetric_profit(p_M)

    print(f"Price-collusion calibration: m={args.m}, alpha={args.alpha}, beta={args.beta}, intervention={args.platform_intervention}")
    print(f"p^N={p_N:.4f}, p^M={p_M:.4f}, pi^N={pi_N:.6f}, pi^M={pi_M:.6f}")
    print(f"Grid: {[round(p,4) for p in prices]}")
    print(f"Output: {output_dir}")
    print(f"Running {args.n_sessions} sessions...")
    print()

    results = []
    for i in range(args.n_sessions):
        seed = args.seed_offset + i + 1
        state = run_session(
            args.m,
            seed,
            alpha=args.alpha,
            beta=args.beta,
            platform_intervention=args.platform_intervention,
            convergence_window=args.convergence_window,
            max_steps=args.max_steps,
        )

        # Save Q-tables
        save_path = os.path.join(output_dir, f'seed_{seed}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        cp = state['prices']
        delta = average_symmetric_gain_index(cp, p_N, p_M)
        tag = "CONV" if state['converged'] else "MAX"
        print(f"  {i+1:3d}: Delta={delta:.3f} prices=[{cp[0]:.4f},{cp[1]:.4f}] "
              f"steps={state['steps']:,} [{tag}] -> {save_path}")
        results.append({'delta': delta, 'conv': state['converged'], 'steps': state['steps'],
                        'prices': cp, 'seed': seed})

    deltas = [r['delta'] for r in results]
    conv_rate = np.mean([r['conv'] for r in results])
    print(f"\n{'='*50}")
    print(f"Mean Delta: {np.mean(deltas):.3f} +/- {np.std(deltas)/np.sqrt(len(deltas)):.3f}")
    print(f"Convergence: {conv_rate:.0%}")
    print(f"Prior algorithmic-collusion benchmarks typically find Delta around 0.7-0.9")
    print(f"Q-tables saved to {output_dir}")
