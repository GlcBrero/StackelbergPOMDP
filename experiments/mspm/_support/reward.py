"""Exact welfare relative to efficient allocation, averaged over buyer types."""
from itertools import product
import numpy as np
from stackelberg_pomdp import Reward, RewardDone, RewardPhase


class MSPMReward(RewardPhase):
    """Exact expected W-W*, using the returned joint message distribution."""

    def run(self, game, response_result):
        profile = response_result.followers
        welfare, efficient, mass = 0., 0., 0.
        for types, type_probability in game.type_profiles():
            for messages in product(range(game.num_messages), repeat=2):
                weight = type_probability * profile.probability(types, messages)
                if weight == 0:
                    continue
                result = yield from game.play_profile(types, messages)
                welfare += weight * result.welfare
                efficient += weight * result.efficient_welfare
                mass += weight
                yield Reward(weight * result.reward, elapsed_steps=result.transitions)
        if not np.isclose(mass, 1., rtol=0, atol=1e-10):
            raise RuntimeError(f"Reward profile probabilities sum to {mass}")
        return RewardDone(game.last_observation, {
            "welfare": welfare, "efficient_welfare": efficient,
            "unconstrained_reward": welfare - efficient,
            "reward_profile_mass": mass,
        })
