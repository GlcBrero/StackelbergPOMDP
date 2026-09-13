from collections import OrderedDict
import copy
from itertools import product

"""Sequential posted-price mechanism environments."""

from gym.spaces import Box, Dict, Discrete
import numpy as np

from .base import BaseEnv


class BaseSPM(BaseEnv):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
            discrete_prices=False,
    ):
        super().__init__(game=game, logger=logger, seed=seed)
        self.discrete_prices = discrete_prices
        self.discrete_price_vec = (
            self.game.discrete_price_vector()
            if self.discrete_prices
            else None
        )

        state_low, state_high = self.game.mechanism_state_bounds()
        self.observation_space = Dict({
            'base_environment': Box(
                low=state_low,
                high=state_high,
                dtype=np.float32,
            )
        })

        num_price_actions = (
            self.game.num_diff_items * len(self.discrete_price_vec)
            if self.discrete_prices
            else self.game.num_diff_items
        )
        self.action_space = Box(
            low=np.zeros(self.num_followers + num_price_actions, dtype=np.float32),
            high=np.ones(self.num_followers + num_price_actions, dtype=np.float32),
            dtype=np.float32,
        )

    def leader_action(self, action):
        """Translate PPO's SPM action into game-level scores and prices."""
        if not self.discrete_prices:
            return action

        agent_scores = np.asarray(action[:self.num_followers])
        price_logits = np.asarray(action[self.num_followers:])
        prices = []
        num_prices = len(self.discrete_price_vec)
        for item_idx in range(self.game.num_diff_items):
            start = item_idx * num_prices
            stop = start + num_prices
            price_idx = int(np.argmax(price_logits[start:stop]))
            prices.append(self.discrete_price_vec[price_idx])
        return np.concatenate((agent_scores, np.asarray(prices, dtype=np.float32)))

    def reset(self):
        self.types = self.reset_types()
        self.mechanism_episode = self.game.new_episode(self.types)
        return OrderedDict({
            "base_environment": self.game.mechanism_state_vector(
                self.mechanism_episode.mechanism_state
            )
        })

    def step(self, action):
        """Run one truthful-buyer step of the standard SPM.

        This is the Brero-style baseline: the leader policy directly controls
        the sequential posted-price mechanism, with no follower messages and no
        StackPOMDP response phase.
        """
        leader_action = self.leader_action(action)
        actions_dict = {
            self.leader: leader_action,
            **{follower: 0 for follower in self.followers_list},
        }
        result = self.game.step_episode(self.mechanism_episode, actions_dict)
        info = dict(result["info"])
        info["is_reward_phase"] = True
        info["leader_action"] = leader_action
        info["reward"] = result["reward"]
        info["reward_generated"] = result["done"]

        if result["done"]:
            self.max_social_welfare = info["max_social_welfare"]
            self.allocative_efficiency = info["efficiency"]

        return (
            OrderedDict({"base_environment": result["observation"]}),
            result["reward"],
            result["done"],
            info,
        )

    def max_episode_transitions(self):
        return self.max_subepisode_transitions()

    def max_subepisode_transitions(self):
        # A sequential mechanism can visit each buyer at most once. It may end
        # earlier when inventory is exhausted.
        return len(self.followers_list)

    def log_info(self, info):
        self.game.log_episode(
            self.mechanism_episode,
            self.logger,
            self.allocative_efficiency,
            self.max_social_welfare,
        )


class BaseMessageSPM(BaseSPM):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
    ):
        super().__init__(game=game, logger=logger, seed=seed)

        # Followers observe private types and submit messages/bids.
        self.followers_observation_space = {follower: Discrete(self.game.num_types) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(self.game.num_messages) for follower in self.followers_list}
        self.reward_phase_profiles = []
        self.reward_phase_profile_idx = 0
        self.include_zero_weight_reward_profiles = True

    def _build_reward_phase_profiles(self):
        profiles = []
        for type_values in product(range(self.game.num_types), repeat=len(self.followers_list)):
            types = {
                follower: type_value
                for follower, type_value in zip(self.followers_list, type_values)
            }
            weight = self.game.type_profile_probability(types)
            if self.include_zero_weight_reward_profiles or weight > 0:
                profiles.append({"types": types, "weight": weight})
        return profiles

    def start_reward_phase(self):
        self.set_reward_phase_profiles(self._build_reward_phase_profiles())

    def set_reward_phase_profiles(self, profiles):
        self.reward_phase_profiles = list(profiles)
        self.reward_phase_profile_idx = 0
        if self.reward_phase_profiles:
            self.types = dict(self.reward_phase_profiles[0]["types"])

    def end_reward_phase(self):
        self.reward_phase_profiles = []
        self.reward_phase_profile_idx = 0

    def reward_phase_length(self, default_length):
        return (
            len(self.reward_phase_profiles)
            or len(self._build_reward_phase_profiles())
            or default_length
        )

    def max_reward_phase_length(self, default_length):
        return len(self._build_reward_phase_profiles()) or default_length

    def advance_reward_phase_profile(self):
        if not self.reward_phase_profiles:
            return False

        self.reward_phase_profile_idx += 1
        if self.reward_phase_profile_idx >= len(self.reward_phase_profiles):
            return False

        self.types = dict(self.reward_phase_profiles[self.reward_phase_profile_idx]["types"])
        return True

    def current_reward_phase_profile(self):
        if not self.reward_phase_profiles:
            return None
        if self.reward_phase_profile_idx >= len(self.reward_phase_profiles):
            return None
        return self.reward_phase_profiles[self.reward_phase_profile_idx]

    def reset(self):
        profile = self.current_reward_phase_profile()
        if profile is not None:
            self.types = dict(profile["types"])
        else:
            self.types = self.reset_types()
        self.mechanism_episode = self.game.new_episode(self.types)
        return {**self.types, self.leader: 0}

    def step(self, actions_dict):
        """Step the sequential mechanism. Follower bids read from actions_dict on first call."""
        actions_dict = dict(actions_dict)
        actions_dict[self.leader] = self.leader_action(actions_dict[self.leader])
        result = self.game.step_episode(self.mechanism_episode, actions_dict)
        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = result["observation"]

        if result["done"]:
            self.max_social_welfare = result["info"]["max_social_welfare"]
            self.allocative_efficiency = result["info"]["efficiency"]
            profile = self.current_reward_phase_profile()
            if profile is not None:
                weight = profile["weight"]
                result["info"]["unweighted_reward"] = result["reward"]
                result["info"]["exact_profile_weight"] = weight
                result["info"]["weighted_efficiency"] = result["info"]["efficiency"] * weight
                # Exact profiles contribute directly to the expected return.
                # Do not multiply by the number of profiles: doing so preserves
                # a reporting average but changes PPO's reward/entropy scale.
                result["reward"] = result["reward"] * weight
                result["info"]["reward"] = result["reward"]
                result["info"]["surplus"] = result["reward"]
                if "response_action_weight" in profile:
                    result["info"]["response_action_weight"] = profile[
                        "response_action_weight"
                    ]
                    result["info"]["response_candidate"] = profile.get(
                        "response_candidate"
                    )
            result["info"]["type_profile"] = dict(self.types)
            result["info"]["mechanism_outcome"] = copy.deepcopy(
                self.mechanism_episode.outcome
            )

        return observation, result["reward"], result["done"], result["info"]

    # Observation/state representation.
    def leader_observation(self, follower_actions=None):
        return self.leader_state_observation()

    def leader_state_observation_space(self):
        return self.observation_space["base_environment"]

    def leader_state_observation(self):
        return self.game.mechanism_state_vector(self.mechanism_episode.mechanism_state)

    def threshold_domain_diagnostics(self, policy, followers_strategy):
        return OrderedDict({
            "mechanism_outcome": self.game.describe_policy_outcomes(self, policy),
        })

    def log_info(self, info):
        self.game.log_episode(
            self.mechanism_episode,
            self.logger,
            self.allocative_efficiency,
            self.max_social_welfare,
        )
