"""Two buyers send messages; the leader chooses offer order and prices.

Buyers privately observe their types. Their strategic actions are messages;
an offered item is accepted exactly when value > price. The message round
ignores the leader action; offer rounds ignore the buyers' message actions.
"""
from collections import OrderedDict
from itertools import product
import random
import gym
import numpy as np
from stackelberg_pomdp import POMG, LeaderQuery
from ._support._critic import observation_space
from ._support.mechanism import play_profile


class Game(POMG):
    leader = "leader"
    buyers = ("buyer_0", "buyer_1")
    horizon = 3  # One message round, followed by at most two offers.

    def __init__(self, num_types=2, num_messages=2, *, seed=0):
        self.num_types, self.num_messages = int(num_types), int(num_messages)
        if self.num_types < 2 or self.num_messages < 1:
            raise ValueError("MSPM requires at least two types and one message")
        if self.num_types == 2:
            self.values = np.asarray([[.2, 1.], [0., .4]])
            self.type_probabilities = np.asarray([[.8, .2], [.5, .5]])
        else:
            self.values = np.tile(np.linspace(0, 1, self.num_types), (2, 1))
            self.type_probabilities = np.full((2, self.num_types), 1 / self.num_types)
        self._rng = random.Random(seed)
        self.action_spaces = {self.leader: gym.spaces.Box(0., 1., (3,), dtype=np.float32),
                              **{buyer: gym.spaces.Discrete(self.num_messages) for buyer in self.buyers}}
        self.leader_observation_space = observation_space(self.num_types, self.num_messages)
        public = OrderedDict((key, space) for key, space in self.leader_observation_space.spaces.items()
                             if not key.startswith("critic:"))
        self.observation_spaces = {self.leader: gym.spaces.Dict(public),
                                   **{buyer: gym.spaces.Discrete(self.num_types) for buyer in self.buyers}}
        self.types = np.zeros(2, dtype=int)
        self.last_observation = None

    def seed(self, seed=None):
        self._rng.seed(seed)

    def sample_types(self):
        if self.num_types == 2:
            return tuple(self._rng.choices(range(self.num_types), weights=p)[0]
                         for p in self.type_probabilities)
        return tuple(self._rng.choice(range(self.num_types)) for _ in range(2))

    def type_profiles(self):
        for types in product(range(self.num_types), repeat=2):
            yield types, float(np.prod(self.type_probabilities[np.arange(2), types]))

    def reset(self, *, types=None):
        self.types = np.asarray(self.sample_types() if types is None else types, dtype=np.int64)
        if self.types.shape != (2,) or np.any(self.types < 0) or np.any(self.types >= self.num_types):
            raise ValueError("Need one valid private type per buyer")
        self.messages = np.zeros(2, dtype=np.int64)
        self.remaining = np.ones(2, dtype=bool)
        self.allocation, self.prices, self.utilities = np.zeros(2), np.zeros(2), np.zeros(2)
        self.welfare, self.offers = 0., 0
        self.stage = "messages"
        return self._observations()

    def _observations(self):
        self.last_observation = OrderedDict([
            ("base_environment", np.r_[self.remaining, 1 - self.allocation.sum(),
                                      1 - self.allocation, self.prices].astype(np.float32)),
            ("base:follower_actions", self.messages.copy()),
        ])
        return {self.leader: self.last_observation,
                **{buyer: int(self.types[i]) for i, buyer in enumerate(self.buyers)}}

    def leader_observation(self, history):
        return history[-1]

    def leader_query(self, observation):
        remaining = observation["base_environment"][:2]
        mask = int(remaining[0]) + 2 * int(remaining[1])
        return LeaderQuery(observation, ("mspm", *map(int, observation["base:follower_actions"]), mask))

    def step(self, actions):
        if set(actions) != set(self.action_spaces):
            raise ValueError("Supply the leader and both buyers' actions")
        if self.stage == "done":
            raise RuntimeError("Reset before playing another mechanism")
        rewards = {agent: 0. for agent in self.action_spaces}
        if self.stage == "messages":
            for i, buyer in enumerate(self.buyers):
                if not self.action_spaces[buyer].contains(actions[buyer]):
                    raise ValueError("Invalid buyer message")
                self.messages[i] = actions[buyer]
            self.stage = "offers"
            return self._observations(), rewards, False, {}
        action = np.clip(np.asarray(actions[self.leader], dtype=float), 0., 1.)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("MSPM action is two buyer scores and one price")
        buyer = int(np.argmax(np.where(self.remaining, action[:2], -np.inf)))
        price = float(action[2])
        values = self.values[np.arange(2), self.types]
        self.remaining[buyer], self.prices[buyer] = False, price
        self.offers += 1
        if values[buyer] > price:
            self.allocation[buyer] = 1
            self.welfare = float(values[buyer])
            self.utilities[buyer] = values[buyer] - price
            rewards[self.buyers[buyer]] = float(self.utilities[buyer])
        done = bool(self.allocation.any() or not self.remaining.any())
        if done:
            self.stage = "done"
            rewards[self.leader] = self.welfare - float(values.max())
        return self._observations(), rewards, done, {"welfare": self.welfare}

    def play_profile(self, types, messages):
        """Exact response/evaluation path, using the same joint-action game."""
        return play_profile(self, types, messages)
