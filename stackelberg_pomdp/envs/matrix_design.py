"""Matrix-design game environment."""

from gym.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np

from .base import BaseEnv


class BaseEnvMatrixDesignGame(BaseEnv):
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

        # Leader observes follower actions and chooses compensation transfers.
        self.observation_space = Dict({'base_environment': MultiDiscrete([self.game.follower_action_space(agent) for agent in self.followers_list])})
        self.action_space = Box(low=0, high=10, shape=(game.num_agents,), dtype=np.float32)

        # Followers observe no state and choose matrix actions.
        self.followers_observation_space = {follower: Discrete(1) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(self.game.follower_action_space(follower)) for follower in self.followers_list}

    def leader_observation(self, follower_actions=None):
        if follower_actions is None:
            return np.zeros(len(self.followers_list), dtype=int)
        return np.array([follower_actions[follower] for follower in self.followers_list], dtype=int)


    def reset(self):
        self.followers_obs = {follower: 0 for follower in self.followers_list}
        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = 0
        return observation

    def step(self, actions_dict):
        """Step the multi-agent game. actions_dict includes leader + all followers."""
        info = {}
        leader_action = actions_dict[self.leader]
        followers_actions = {a: actions_dict[a] for a in self.followers_list}

        game_utilities = {agent: self.game.follower_payoff(followers_actions, agent) for agent in self.followers_list}
        reward = sum(game_utilities.values())

        # Adjust follower utilities with leader's compensation
        adjusted_utilities = {
            follower: self.game.follower_payoff(actions_dict, follower) + leader_action[i]
            for i, follower in enumerate(self.followers_list)
        }
        adjusted_utilities[self.leader] = reward

        info["reward_generated"] = True
        info["utilities"] = adjusted_utilities
        info["surplus"] = reward

        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = self.leader_observation(followers_actions)
        return observation, reward, True, info
