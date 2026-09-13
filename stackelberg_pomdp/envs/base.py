"""Shared base interface for multi-agent environments."""

from collections import OrderedDict
import random

import gym
from gym.spaces import Dict, Discrete, MultiDiscrete
import numpy as np


_LEGACY_ENVIRONMENT_MODULES = {
    "BaseEnvSimpleMatrixGame": "normal_form",
    "BaseEnvMatrixDesignGame": "matrix_design",
    "BaseSimpleAllocation": "simple_allocation",
    "BaseSPM": "spm",
    "BaseMessageSPM": "spm",
    "BertrandCompetitionEnv": "bertrand",
}


def __getattr__(name):
    """Load moved environments for legacy imports and pickle globals."""
    try:
        module_name = _LEGACY_ENVIRONMENT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        ) from exc
    module = __import__(
        "{}.{}".format(__package__, module_name), fromlist=[name]
    )
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(_LEGACY_ENVIRONMENT_MODULES))


__all__ = ("BaseEnv",) + tuple(_LEGACY_ENVIRONMENT_MODULES)


class BaseEnv(gym.Env):
    """Base class for all multi-agent environments.

    Provides: leader, followers_list, num_followers, _rng, logger.
    Subclasses implement step(all_actions_dict) and reset().
    """

    def __init__(self, game=None, leader=None, followers_list=None, logger=None, seed=None):
        super().__init__()
        self.logger = logger
        self._rng = random.Random()
        self._rng.seed(seed)
        self.freeze_types = False

        if game is not None:
            self.game = game
            self.leader = game.leader
            self.followers_list = game.followers_list
            self.num_followers = len(self.followers_list)
            self.num_types = game.num_types
            self.game.set_seed(seed)
        else:
            self.leader = leader or "platform"
            self.followers_list = followers_list or []
            self.num_followers = len(self.followers_list)

    def run_episode(self, policy, types, bids):
        missing = object()
        previous_freeze_types = self.freeze_types
        previous_types = getattr(self, "types", missing)
        previous_mechanism_episode = getattr(self, "mechanism_episode", missing)
        previous_max_social_welfare = getattr(self, "max_social_welfare", missing)
        previous_allocative_efficiency = getattr(self, "allocative_efficiency", missing)

        try:
            # Counterfactual diagnostics must evaluate the requested type
            # profile, regardless of whether the live response phase is
            # currently freezing types across MW deviations.
            self.types = dict(types)
            self.freeze_types = True
            self.reset()
            observation = self.reactive_leader_observation(bids)
            done = False
            while not done:
                action = policy.get_action(observation)
                actions = dict(bids)
                actions[self.leader] = action
                _, reward, done, info = self.step(actions)
                observation = self.reactive_leader_observation(bids)
        finally:
            self.freeze_types = previous_freeze_types
            for attribute, value in (
                    ("types", previous_types),
                    ("mechanism_episode", previous_mechanism_episode),
                    ("max_social_welfare", previous_max_social_welfare),
                    ("allocative_efficiency", previous_allocative_efficiency),
            ):
                if value is missing:
                    if hasattr(self, attribute):
                        delattr(self, attribute)
                else:
                    setattr(self, attribute, value)
        return info

    def leader_state_observation_space(self):
        return Discrete(1)

    def leader_state_observation(self):
        return 0

    def follower_action_observation_space(self):
        if len(self.followers_list) == 1:
            return self.followers_action_space[self.followers_list[0]]
        return MultiDiscrete([
            self.followers_action_space[follower].n
            for follower in self.followers_list
        ])

    def follower_action_observation(self, follower_actions):
        if follower_actions is None:
            return 0
        values = [follower_actions[follower] for follower in self.followers_list]
        if len(values) == 1:
            return values[0]
        return np.array(values, dtype=np.int64)

    def combine_leader_observation(self, leader_state, follower_action_obs):
        if leader_state is None:
            return follower_action_obs
        return np.concatenate((np.atleast_1d(leader_state), np.atleast_1d(follower_action_obs)))

    def reactive_observation_space(self):
        return Dict(OrderedDict({
            "base_environment": self.leader_state_observation_space(),
            "base:follower_actions": self.follower_action_observation_space(),
        }))

    def reactive_leader_observation(self, follower_actions):
        """Return the structured reactive leader observation.

        The actor sees both non-critic keys. Keeping follower actions in their
        own field makes the wrapper logic explicit while leaving domain-specific
        state construction in the base environment.
        """
        return OrderedDict({
            "base_environment": self.leader_state_observation(),
            "base:follower_actions": self.follower_action_observation(follower_actions),
        })

    def leader_observation(self, follower_actions=None):
        return 0

    def reset_types(self):
        """Sample or reuse private types for follower counterfactual evaluation."""
        if not self.freeze_types or not hasattr(self, "types"):
            self.types = self.game.sample_types()
        return self.types

    def reward_phase_length(self, default_length):
        """Return the reward-phase length for this environment.

        Most environments use the configured Monte Carlo length. Environments
        with a finite exact reward distribution can override this and enumerate
        the profiles directly.
        """
        return default_length

    def max_reward_phase_length(self, default_length):
        return self.reward_phase_length(default_length)

    def max_subepisode_transitions(self):
        """Maximum leader transitions in one generated response/reward game."""
        return 1

    def start_reward_phase(self):
        """Initialize any environment-specific exact reward-phase schedule."""
        return

    def end_reward_phase(self):
        """Clear any environment-specific reward schedule."""
        return

    def advance_reward_phase_profile(self):
        """Advance an exact reward-phase schedule, if one is active."""
        return False

    def current_reward_phase_profile(self):
        return None

    def log_info(self, info):
        return

    def threshold_domain_diagnostics(self, policy, followers_strategy):
        return OrderedDict()
