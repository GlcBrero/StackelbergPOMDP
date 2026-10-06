"""Drive the POMG at a fixed type/message profile for exact payoff integration."""
from dataclasses import dataclass
import numpy as np


@dataclass
class MechanismOutcome:
    utilities: np.ndarray
    welfare: float
    efficient_welfare: float
    observation: dict
    transitions: int

    @property
    def reward(self):
        return self.welfare - self.efficient_welfare


def play_profile(game, types, messages):
    game.reset(types=types)
    actions = {game.leader: np.zeros(3), **dict(zip(game.buyers, messages))}
    observations, _, done, _ = game.step(actions)
    while not done:
        observation = game.leader_observation((observations[game.leader],))
        actions[game.leader] = yield game.leader_query(observation)
        observations, _, done, _ = game.step(actions)
    values = game.values[np.arange(2), game.types]
    return MechanismOutcome(game.utilities.copy(), game.welfare, float(values.max()),
                            game.last_observation, game.offers)
