"""Two sellers choose prices, then the platform chooses who is displayed.

A round has two joint-action steps. In the prices step only seller actions
matter; in the display step only the platform action matters. Profits and
consumer surplus are paid after display. Seller costs are private to sellers.
"""
import itertools
import gym
import numpy as np
from stackelberg_pomdp import POMG
from ._support._critic import observation_space
from ._support.market import (
    Market, admission_game, admission_payoff_tables, admissions_from_rule,
    admission_table_from_actions, table_from_actions, initial_q, payoff_tables, visible_queries,
)


class Game(POMG):
    leader = "platform"
    sellers = ("seller_0", "seller_1")

    def __init__(self, prices, costs, *, admission_classes=True, ordered=False, rounds=1, seed=0):
        self.prices, self.costs = np.asarray(prices), tuple(costs)
        self.m, self.rounds = len(self.prices), int(rounds)
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        self.horizon = 2 * self.rounds
        self.admission_classes, self.ordered = admission_classes, ordered
        self.pairs = (list(itertools.product(range(self.m), repeat=2)) if ordered else
                      list(itertools.combinations_with_replacement(range(self.m), 2)))
        self.n_queries = len(self.pairs)
        self.queries = (np.asarray(self.pairs, dtype=np.float32) / ((self.m - 1) / 2) - 1).astype(np.float32)
        self.action_spaces = {self.leader: gym.spaces.Discrete(3 if admission_classes else self.m),
                              **{seller: gym.spaces.Discrete(self.m) for seller in self.sellers}}
        private = gym.spaces.Dict({"cost": gym.spaces.Discrete(len(self.costs)),
                                   "prices": gym.spaces.MultiDiscrete([self.m, self.m]),
                                   "stage": gym.spaces.Discrete(2)})
        self.observation_spaces = {self.leader: gym.spaces.Dict({"base_environment":
            gym.spaces.Box(-1., 1., (2,), dtype=np.float32)}),
            **{seller: private for seller in self.sellers}}
        self.leader_observation_space = observation_space(self.n_queries, self.m)
        self.payoffs = {cost: admission_payoff_tables(cost, self.prices) for cost in self.costs}
        self.seed(seed)

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, *, cost_index=None, price_state=0):
        self.cost_index = int(self.rng.integers(len(self.costs)) if cost_index is None else cost_index)
        self.current_prices = np.asarray(divmod(int(price_state), self.m), dtype=np.int64)
        self.stage, self.completed_rounds = "prices", 0
        return self._observations()

    def observation(self, index):
        return {"base_environment": self.queries[index].copy()}

    def _observations(self):
        pair = tuple(self.current_prices if self.ordered else sorted(self.current_prices))
        return {self.leader: self.observation(self.pairs.index(pair)),
                **{seller: {"cost": self.cost_index, "prices": self.current_prices.copy(),
                            "stage": int(self.stage == "display")} for seller in self.sellers}}

    def leader_observation(self, history):
        return history[-1]

    def step(self, actions):
        if set(actions) != set(self.action_spaces):
            raise ValueError("Supply platform and both seller actions")
        if self.completed_rounds >= self.rounds:
            raise RuntimeError("Reset before playing another market episode")
        rewards = {agent: 0. for agent in self.action_spaces}
        if self.stage == "prices":
            if any(not self.action_spaces[seller].contains(actions[seller]) for seller in self.sellers):
                raise ValueError("Invalid seller price index")
            self.current_prices = np.asarray([actions[seller] for seller in self.sellers])
            self.stage = "display"
        else:
            action = int(actions[self.leader])
            if not self.action_space.contains(action):
                raise ValueError("Invalid platform admission action")
            first, second = map(int, self.current_prices)
            if self.admission_classes:
                admitted = ((first == 0, second == 0) if action == 0 else
                            (first <= second, second <= first) if action == 1 else (True, True))
            else:
                admitted = (first <= action, second <= action)
            displayed = int(admitted[0]) + 2 * int(admitted[1])
            profits, surplus = self.payoffs[self.costs[self.cost_index]]
            rewards[self.leader] = float(surplus[displayed, first, second])
            for index, seller in enumerate(self.sellers):
                rewards[seller] = float(profits[displayed, first, second, index])
            self.completed_rounds += 1
            self.stage = "prices"
        return self._observations(), rewards, self.completed_rounds == self.rounds, {}
