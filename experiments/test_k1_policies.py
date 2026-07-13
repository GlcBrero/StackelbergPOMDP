"""
Systematic test of k=1 leader policies against Calvano-converged Q-learners.
For each policy (mapping 16 current price pairs to threshold), run Q-learning
to true convergence and measure CS and exclusion.

Grid: p_min=1.3, p_max=1.7, m=4, c=1.0
Prices: [1.300, 1.433, 1.567, 1.700]

Usage: python test_k1_policies.py <policy_idx> <seed>
  policy_idx: 0-11 (which policy to test)
  seed: random seed
"""
import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, '..')
sys.path.insert(0, os.path.join(base_dir, 'stackelberg_pomdp'))
sys.path.insert(0, base_dir)

import numpy as np
import json
from stable_baselines3.common import logger
from stackelberg_pomdp.gym_envs.envs.base_envs import BertrandCompetitionEnv
from stackelberg_pomdp.gym_envs.envs.wrappers import QLearningFollowersWrapper

# ── Parameters ──
M = 4
PRICE_MIN, PRICE_MAX = 1.3, 1.7
C = 1.0
A, A0, MU = 2.0, 0.0, 0.25
ALPHA, BETA, DELTA = 0.15, 4e-6, 0.95
CONVERGENCE_WINDOW = 100_000
MAX_STEPS = 10_000_000

PRICES = np.linspace(PRICE_MIN, PRICE_MAX, M)

def q_sym(p):
    x = np.exp((A - p) / MU)
    return x / (2 * x + np.exp(A0 / MU))

def compute_cs(prices_pair, bbx_idx):
    if len(bbx_idx) == 0:
        return MU * np.log(np.exp(A0 / MU))
    bb = np.array(bbx_idx)
    exp_vals = np.exp((A - prices_pair[bb]) / MU)
    denom = exp_vals.sum() + np.exp(A0 / MU)
    return MU * np.log(denom)

def run_with_policy(policy_fn, seed):
    log = logger.configure(format_strings=[])
    env = BertrandCompetitionEnv(m=M, price_min=PRICE_MIN, price_max=PRICE_MAX,
                                  platform_intervention='learn_threshold',
                                  seed=seed, logger=log)
    wrapper = QLearningFollowersWrapper(env, alpha=ALPHA, delta=DELTA, beta=BETA)
    wrapper.reset()
    wrapper.this_step_mode = "standard"

    prev_greedy = None
    stable_count = 0

    for step in range(MAX_STEPS):
        p0 = wrapper.current_actions.get('agent_0', 0)
        p1 = wrapper.current_actions.get('agent_1', 0)
        threshold = policy_fn(p0, p1)
        wrapper.step(threshold)

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

        if stable_count >= CONVERGENCE_WINDOW:
            break

    converged = stable_count >= CONVERGENCE_WINDOW

    # Measure converged behavior: run 10k greedy steps
    wrapper.this_step_mode = "argmax"
    cs_vals = []
    excl_vals = []
    for _ in range(10_000):
        p0 = wrapper.current_actions.get('agent_0', 0)
        p1 = wrapper.current_actions.get('agent_1', 0)
        threshold = policy_fn(p0, p1)
        obs, rew, done, info = wrapper.step(threshold)
        bbx = info.get('bbx_idx', list(range(2)))
        prices_pair = np.array([PRICES[p0], PRICES[p1]])
        cs_vals.append(compute_cs(prices_pair, bbx))
        excl_vals.append(1.0 - len(bbx) / 2.0)

    return np.mean(cs_vals), np.mean(excl_vals), step + 1, converged

# ── Define k=1 policies ──
POLICIES = [
    ("uniform_t=0", lambda p0, p1: 0),
    ("uniform_t=1", lambda p0, p1: 1),
    ("uniform_t=2", lambda p0, p1: 2),
    ("uniform_t=3", lambda p0, p1: 3),
    ("close_if_both>=2", lambda p0, p1: 0 if (p0 >= 2 and p1 >= 2) else 3),
    ("close_if_both>=1", lambda p0, p1: 0 if (p0 >= 1 and p1 >= 1) else 3),
    ("close_if_both>=3", lambda p0, p1: 0 if (p0 >= 3 and p1 >= 3) else 3),
    ("anti_collusion_diag", lambda p0, p1: 0 if (p0 == p1 and p0 >= 1) else 3),
    ("close_near_diag", lambda p0, p1: 0 if (abs(p0-p1) <= 1 and p0 >= 2 and p1 >= 2) else 3),
    ("open_if_competitive", lambda p0, p1: 3 if (p0 == 0 or p1 == 0) else 0),
    ("reward_undercutting", lambda p0, p1: 3 if abs(p0-p1) >= 2 else 0),
    ("graduated_min", lambda p0, p1: min(p0, p1)),
    ("partial_t2_if_both>=2", lambda p0, p1: 2 if (p0 >= 2 and p1 >= 2) else 3),
    ("partial_t1_if_both>=2", lambda p0, p1: 1 if (p0 >= 2 and p1 >= 2) else 3),
    ("close_if_any>=2", lambda p0, p1: 0 if (p0 >= 2 or p1 >= 2) else 3),
]

if __name__ == '__main__':
    policy_idx = int(sys.argv[1])
    seed = int(sys.argv[2])

    name, policy_fn = POLICIES[policy_idx]
    cs, excl, steps, converged = run_with_policy(policy_fn, seed)

    result = {
        'policy': name,
        'policy_idx': policy_idx,
        'seed': seed,
        'cs': cs,
        'exclusion': excl,
        'steps': steps,
        'converged': converged,
    }
    # Write to stdout as JSON for easy collection
    print(json.dumps(result))
