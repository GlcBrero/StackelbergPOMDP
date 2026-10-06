"""Demand, profits, consumer surplus and admission-rule encodings."""
import itertools
import numpy as np


class Market:
    def __init__(self, m, c_i, platform_intervention, price_min, price_max, seed=0):
        self.m, self.c_i = m, c_i
        self.num_agents, self.a_0, self.mu = 2, 0, .25
        self.a = np.array([2, 2])
        self._exp_a0_mu = np.exp(self.a_0 / self.mu)
        self.action_price_space = np.linspace(price_min, price_max, m)
        self.platform_intervention = platform_intervention

    def _demand(self, a, p, mu, agent_idx, bb_idx):
        if bb_idx is None:
            return np.exp((a[agent_idx] - p[agent_idx]) / mu) / (
                np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu)
            )
        if agent_idx not in bb_idx:
            return 0
        denom = np.sum([np.exp((a[idx] - p[idx]) / mu) for idx in bb_idx]) + np.exp(self.a_0 / mu)
        return np.exp((a[agent_idx] - p[agent_idx]) / mu) / denom

    def _compute_surplus(self, prices, bbx_idx):
        val = np.sum([np.exp((self.a[i] - float(prices[i])) / self.mu) for i in bbx_idx])
        val += np.exp(self.a_0 / self.mu)
        return self.mu * np.log(val)

    def compute_q_init_entry(self, delta):
        """Compute initial Q-values à la Calvano et al. (2019), eq. 8.

        Q_{i,0}(s, a_i) = sum_{a_{-i}} pi_i(a_i, a_{-i}) / ((1-delta) * |A|^{n-1})

        Assumes uniform opponent play and computes expected discounted profit
        for each action. When platform_intervention != 'no_intervene', uses
        buy-box demand (Johnson et al. 2021 heuristic).
        """
        n = self.m
        all_agents = list(range(self.num_agents))
        entry = np.empty(n)
        for i in range(n):
            avg_reward = 0
            for j in range(n):
                price_i = self.action_price_space[i]
                price_j = self.action_price_space[j]
                prices = np.array([price_i, price_j])

                if self.platform_intervention == 'no_intervene':
                    # Calvano: all agents always displayed
                    demand_i = self._demand(self.a, prices, self.mu, 0, all_agents)
                else:
                    # Johnson et al.: buy-box winner gets displayed
                    bbx_demand = self._demand(self.a, prices, self.mu, 0, [0])
                    non_bbx_demand = self._demand(self.a, prices, self.mu, 0, [1])

                    if price_i < price_j:
                        demand_i = bbx_demand
                    elif price_i > price_j:
                        demand_i = non_bbx_demand
                    else:
                        sigma = 0.01
                        numer = np.exp(-price_i / sigma)
                        denom = np.sum([np.exp(-p / sigma) for p in prices])
                        demand_i = (numer / denom) * bbx_demand + (1 - numer / denom) * non_bbx_demand

                avg_reward += (price_i - self.c_i) * demand_i / n
            entry[i] = avg_reward / (1 - delta)
        return entry


def admissions_from_rule(rule):
    """Read a direct [price 1, price 2, seller] display table.

    Two-dimensional threshold tables are accepted only as a compatibility
    input for archived policies and the fixed/clipped benchmark families.
    """
    rule = np.asarray(rule)
    if rule.ndim not in (2, 3) or rule.shape[0] != rule.shape[1]:
        raise ValueError("Expected a square price-pair rule")
    m = rule.shape[0]
    if rule.ndim == 3:
        if rule.shape != (m, m, 2) or not np.all((rule == 0) | (rule == 1)):
            raise ValueError("Direct admissions must contain two Boolean seller decisions")
        return rule.astype(bool)
    if not np.all(np.isfinite(rule)) or not np.all((rule >= 0) & (rule < m)):
        raise ValueError("Invalid archived threshold table")
    if not np.all(rule == rule.astype(int)):
        raise ValueError("Threshold indices must be integers")
    first, second = np.indices((m, m))
    return np.stack((first <= rule, second <= rule), axis=-1)


def admission_payoff_tables(cost, prices=np.linspace(1.05, 1.7, 4)):
    """Payoffs for each displayed set, without an intermediate price ceiling.

    The first axis encodes the two seller inclusion bits: 0 is neither,
    1 is seller 1, 2 is seller 2, and 3 is both. Policy restrictions belong
    to the admission-rule constructor, not to the market payoff calculation.
    """
    prices = np.asarray(prices, dtype=float)
    m = len(prices)
    env = Market(
        m=m, c_i=float(cost), platform_intervention='no_intervene',
        price_min=float(prices[0]), price_max=float(prices[-1]), seed=0,
    )
    np.testing.assert_allclose(env.action_price_space, prices, rtol=0, atol=1e-12)
    # Match the base market's grid exactly, including its floating-point values.
    prices = env.action_price_space
    utility = np.zeros((4, m, m, 2))
    surplus = np.zeros((4, m, m))
    for displayed, first, second in itertools.product(range(4), range(m), range(m)):
        occupants = [seller for seller in range(2) if displayed & (1 << seller)]
        pair = prices[[first, second]]
        demands = [env._demand(env.a, pair, env.mu, seller, occupants)
                   for seller in range(2)]
        utility[displayed, first, second] = (pair - float(cost)) * demands
        surplus[displayed, first, second] = env._compute_surplus(pair, occupants)
    return utility, surplus


def admission_game(tables, rule):
    """Select market payoffs directly from a rule's two inclusion decisions."""
    utility, surplus = tables
    admitted = admissions_from_rule(rule)
    m = len(admitted)
    if utility.shape != (4, m, m, 2) or surplus.shape != (4, m, m):
        raise ValueError("Admission rule and payoff grid do not match")
    displayed = admitted[..., 0].astype(int) + 2 * admitted[..., 1]
    first, second = np.indices((m, m))
    return utility[displayed, first, second], surplus[displayed, first, second]


def initial_q(cost, memory, delta=.95, prices=np.linspace(1.05, 1.7, 4), *,
              initialization="buybox"):
    """Initialize against uniform opponent prices under a specified reference game.

    ``no_intervention`` is Calvano's full-market initialization. ``buybox``
    preserves the historical prominence-based initialization. Both references
    are independent of the candidate platform commitment being evaluated.
    """
    if initialization not in ("buybox", "no_intervention"):
        raise ValueError("Unknown seller Q initialization: " + str(initialization))
    prices = np.asarray(prices, dtype=float)
    m = len(prices)
    env = Market(
        m=m, c_i=cost, platform_intervention=(
            'no_intervene' if initialization == "no_intervention" else 'learn_threshold'),
        price_min=prices[0], price_max=prices[-1], seed=0,
    )
    np.testing.assert_allclose(env.action_price_space, prices, rtol=0, atol=1e-12)
    entry = env.compute_q_init_entry(delta)
    return np.tile(entry, (2, m * m if memory else 1, 1))


def payoff_tables(cost, prices=np.linspace(1.05, 1.7, 4)):
    tables = admission_payoff_tables(cost, prices)
    m = len(prices)
    pairs = [admission_game(tables, np.full((m, m), k)) for k in range(m)]
    return np.stack([x[0] for x in pairs]), np.stack([x[1] for x in pairs])


def visible_queries(mode, m=4):
    pairs = list(itertools.combinations_with_replacement(range(m), 2))
    if mode == "constant":
        return np.zeros((len(pairs), 2), dtype=np.float32)
    if mode == "prices":
        return (np.asarray(pairs, dtype=np.float32) / ((m - 1) / 2) - 1).astype(np.float32)
    raise ValueError(mode)


def table_from_actions(actions, m=4, ordered=False):
    pairs = (list(itertools.product(range(m), repeat=2)) if ordered else
             list(itertools.combinations_with_replacement(range(m), 2)))
    if len(actions) != len(pairs):
        raise ValueError("Wrong commitment size")
    table = np.zeros((m, m), dtype=np.int32)
    for (a, b), action in zip(pairs, actions):
        table[a, b] = int(action)
        if not ordered:
            table[b, a] = int(action)
    return table


def admission_table_from_actions(actions, m=4, ordered=False):
    """Build direct seller-inclusion decisions from three admission actions.

    0 excludes sellers except those at the minimum grid price; 1 admits
    every lowest-price seller (both at a tie); 2 admits both sellers.
    These boundary conventions preserve the reported experiment's policy
    space and action aliases. No mask or threshold conversion is used.
    """
    actions = np.asarray(actions)
    pairs = (list(itertools.product(range(m), repeat=2)) if ordered else
             list(itertools.combinations_with_replacement(range(m), 2)))
    if actions.shape != (len(pairs),):
        raise ValueError("Wrong commitment size")
    if not np.all(np.isin(actions, (0, 1, 2))):
        raise ValueError("Admission actions must be 0, 1, or 2")
    rule = np.zeros((m, m, 2), dtype=bool)
    for (first, second), action in zip(pairs, actions):
        if action == 0:
            admitted = (first == 0, second == 0)
        elif action == 1:
            admitted = (first <= second, second <= first)
        else:
            admitted = (True, True)
        rule[first, second] = admitted
        if not ordered:
            rule[second, first] = admitted[::-1]
    return rule
