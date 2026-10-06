"""Encode private training state for the centralized critic, never the actor."""
from collections import OrderedDict

import gym
import numpy as np


def observation_space(n_queries, n_prices):
    return gym.spaces.Dict(OrderedDict([
        ("base_environment", gym.spaces.Box(-1., 1., (2,), dtype=np.float32)),
        ("critic:context", gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)),
        ("critic:commitment", gym.spaces.Box(-1., 1., (n_queries,), dtype=np.float32)),
        ("critic:stage", gym.spaces.Box(0., 1., (2,), dtype=np.float32)),
        ("critic:followers", gym.spaces.Box(0., 1., (2 * n_prices,), dtype=np.float32)),
    ]))


def encode_critic_state(response, game, phase):
    learner = response.seller_learning
    state = response.state
    distributions = (
        state.seller_response["final_distributions"] if state.seller_response
        else np.full((2, learner.m), 1 / learner.m)
    )
    return OrderedDict([
        ("critic:context", np.asarray([learner.costs[state.cost_index]], dtype=np.float32)),
        ("critic:commitment", (
            state.leader_actions / max(1, game.action_space.n - 1)
        ).astype(np.float32)),
        ("critic:stage", np.asarray([
            state.query_index / game.n_queries, phase == "reward"
        ], dtype=np.float32)),
        ("critic:followers", np.asarray(distributions, dtype=np.float32).ravel()),
    ])
