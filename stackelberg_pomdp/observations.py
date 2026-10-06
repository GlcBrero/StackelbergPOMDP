"""Add centralized-critic inputs without changing leader queries or cache keys."""
from collections import OrderedDict
from copy import deepcopy

import gym


class CriticObservation:
    """Adapt an emitted observation and its space for centralized training.

    Apply this when the executor emits an observation, before the phase closes.
    This also preserves the critic state attached to a terminal observation.
    The actor's policy must ignore all fields prefixed with ``critic:``.
    """

    def __init__(self, game, response, *, add_phase=True):
        self.game = game
        self.response = response
        self.add_phase = bool(add_phase)
        self.space = game.observation_space
        if self.add_phase:
            if not isinstance(self.space, gym.spaces.Dict):
                raise TypeError("A privileged phase observation requires a Dict space")
            spaces = OrderedDict(self.space.spaces)
            spaces["critic:is_reward_step"] = gym.spaces.Discrete(2)
            self.space = gym.spaces.Dict(spaces)

    def encode(self, observation, commitment, phase):
        observation = deepcopy(observation)
        if hasattr(self.game, "critic_observation"):
            observation.update(self.game.critic_observation(commitment, phase))
        if hasattr(self.response, "critic_observation"):
            observation.update(self.response.critic_observation(self.game, commitment, phase))
        if self.add_phase:
            observation["critic:is_reward_step"] = int(phase == "reward")
        return observation
