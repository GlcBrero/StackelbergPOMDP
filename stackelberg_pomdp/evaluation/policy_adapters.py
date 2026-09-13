"""Adapters that expose SB3 and deterministic baselines to evaluators."""

import numpy as np
import gym


def _zero_observation(space):
    if isinstance(space, gym.spaces.Dict):
        return {
            key: _zero_observation(subspace)
            for key, subspace in space.spaces.items()
        }
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return 0
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros(space.nvec.shape, dtype=np.int64)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.zeros(space.n, dtype=np.int8)
    raise NotImplementedError(f"Unsupported observation space: {space}")


class BaselinePolicyWrapper:
    def __init__(self, baselines_policy, env, deterministic=True):
        self.baselines_policy = baselines_policy
        self.env = env
        self.deterministic = deterministic

    def get_action(self, observation):
        obs_full = _zero_observation(self.env.observation_space)
        for key, value in observation.items():
            if key in obs_full and not key.startswith("critic:"):
                obs_full[key] = value
        return self.baselines_policy.predict(obs_full, deterministic=self.deterministic)[0]


__all__ = ["BaselinePolicyWrapper"]
