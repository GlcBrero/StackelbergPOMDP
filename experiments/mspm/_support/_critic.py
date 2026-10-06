"""Observation-space declarations and private centralized-critic encoding."""
from collections import OrderedDict
import gym
import numpy as np


def observation_space(t, m):
    return gym.spaces.Dict(OrderedDict([
        ("base_environment", gym.spaces.Box(0., 1., (7,), dtype=np.float32)),
        ("base:follower_actions", gym.spaces.MultiDiscrete([m, m])),
        ("critic:types", gym.spaces.MultiDiscrete([t, t])),
        ("critic:weights", gym.spaces.Box(0., 1., (2, t, m), dtype=np.float32)),
        ("critic:progress", gym.spaces.Box(0., np.inf, (2,), dtype=np.float32)),
        ("critic:deviations", gym.spaces.Box(0., 1., (2, m), dtype=np.float32)),
        ("critic:empirical_joint", gym.spaces.Box(0., 1., (t, t, m, m), dtype=np.float32)),
        ("critic:payoffs", gym.spaces.Box(0., 1., (t, t, m, m, 4), dtype=np.float32)),
        ("critic:commitment", gym.spaces.Box(0., 1., (m * m * 3, 4), dtype=np.float32)),
    ]))


def encode_critic_state(state, types, commitment, phase):
    del phase
    entries = np.zeros((state.num_messages ** 2 * 3, 4), dtype=np.float32)
    for key, action in commitment.snapshot().items():
        if isinstance(key, tuple) and len(key) == 4 and key[0] == "mspm":
            _, first, second, remaining = key
            index = (first * state.num_messages + second) * 3 + remaining - 1
            entries[index] = np.r_[1., np.asarray(action).ravel()]
    return {
        "critic:types": types.copy(),
        "critic:weights": state.weights.astype(np.float32),
        "critic:progress": state.progress.astype(np.float32),
        "critic:deviations": state.deviations.astype(np.float32),
        "critic:empirical_joint": state.empirical_joint.astype(np.float32),
        "critic:payoffs": state.payoff_state.astype(np.float32),
        "critic:commitment": entries,
    }
