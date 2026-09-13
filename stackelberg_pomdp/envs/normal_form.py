"""Normal-form game environment."""

from gym.spaces import Box, Dict, Discrete
import numpy as np

import stackelberg_pomdp.utils as utils

from .base import BaseEnv


class BaseEnvSimpleMatrixGame(BaseEnv):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
            randomized=False,
            randomization_type="linear",
    ):
        super().__init__(game=game, logger=logger, seed=seed)

        self.randomized = randomized
        self.randomization_type = randomization_type

        # Leader observes no state and chooses a row or mixed row.
        self.observation_space = Dict({'base_environment': Discrete(1)})
        if self.randomized == True:
            self.action_space = Box(
                low=np.array([0 for _ in range(self.game.action_space(self.leader))]),
                high=np.array([1 for _ in range(self.game.action_space(self.leader))]),
                dtype=np.float32,
            )
        else:
            self.action_space = Discrete(self.game.action_space(self.leader))

        # Follower observes no state and chooses a matrix column/action.
        self.followers_observation_space = {follower: Discrete(1) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(self.game.action_space(follower)) for follower in self.followers_list}


    def reset(self):
        self.followers_obs = {follower: 0 for follower in self.followers_list}
        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = 0
        return observation

    def step(self, actions_dict):
        """Step the multi-agent game. actions_dict includes leader + all followers."""
        info = {}
        leader_action = actions_dict[self.leader]
        follower_actions = {a: actions_dict[a] for a in self.followers_list}

        if self.randomized:
            action_probs = utils.weights_to_probs(leader_action)
            utilities = {agent: 0 for agent in self.game.list_of_agents}
            for curr_action in range(self.game.action_space(self.leader)):
                curr_actions = actions_dict.copy()
                curr_actions[self.leader] = curr_action
                curr_utilities = {agent: self.game.payoff(curr_actions, agent) for agent in self.game.list_of_agents}
                for agent in self.game.list_of_agents:
                    utilities[agent] += action_probs[curr_action] * curr_utilities[agent]
        else:
            utilities = {agent: self.game.payoff(actions_dict, agent) for agent in self.game.list_of_agents}

        observation = {agent: 0 for agent in self.followers_list}
        observation[self.leader] = self.leader_observation(follower_actions)
        reward = utilities[self.leader]

        info["reward_generated"] = True
        info["utilities"] = utilities
        info["surplus"] = reward

        return observation, reward, True, info
