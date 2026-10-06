"""Compose the platform game, seller Q-learning response and surplus reward.

Read game.py for the market, response.py for the learning procedure, and
_support/reward.py for evaluation. The inner Q update loop lives in q_learning.cpp.
The paper runner supplies the learner's configuration and fixed seed schedule.
"""
from stackelberg_pomdp import StackelbergPOMDPEnv

from .game import Game
from .response import QLearningResponse
from ._support.reward import PlatformReward


class PlatformEnv(StackelbergPOMDPEnv):
    """Small accessors used by the platform training and evaluation runner."""

    @property
    def episodes(self):
        return self.episode_count

    @property
    def n_queries(self):
        return self.game.n_queries

    @property
    def seller_learning(self):
        return self.response.seller_learning

    def observation(self, index):
        return self.observations.encode(self.game.observation(index), self.commitment, self.phase)


def build(seller_learning, seeds, seed=0, offset=0, *, lazy=True):
    if not lazy:
        raise ValueError("Platform's native response requires lazy=True; a fresh-sampling response is not implemented")
    game = Game(seller_learning.prices, seller_learning.costs,
                admission_classes=hasattr(seller_learning, "n_actions"),
                ordered=seller_learning.ordered, seed=seed)
    response = QLearningResponse(seller_learning, seeds, offset, seed)
    return PlatformEnv(game, response, PlatformReward(response),
                       lazy=lazy, add_phase_observation=False)
