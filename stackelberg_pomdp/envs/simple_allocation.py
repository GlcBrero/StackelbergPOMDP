"""Simple allocation mechanism environment."""

from gym.spaces import Dict, Discrete

from .base import BaseEnv


class BaseSimpleAllocation(BaseEnv):
    def __init__(
            self,
            game,
            logger=None,
            seed=None,
    ):
        super().__init__(game=game, logger=logger, seed=seed)

        # Leader observes only the reported message and chooses an item/type.
        self.observation_space = Dict({'base_environment': Discrete(game.num_messages)})
        self.action_space = Discrete(self.num_types)

        # Follower observes its private type and reports one of the available messages.
        self.followers_observation_space = {follower: Discrete(self.num_types) for follower in self.followers_list}
        self.followers_action_space = {follower: Discrete(game.num_messages) for follower in self.followers_list}

    def leader_observation(self, follower_actions=None):
        if follower_actions is None:
            return 0
        return follower_actions[self.followers_list[0]]


    def reset(self):
        return {**self.reset_types(), self.leader: 0}

    def step(self, actions_dict):
        """Allocate the item selected by the leader after the follower message."""
        leader_action = actions_dict[self.leader]

        # Welfare is one iff the allocated item matches the follower's type.
        follower_rewards = {
            agent: int(leader_action == self.types[agent])
            for agent in self.followers_list
        }
        reward = follower_rewards[self.followers_list[0]]
        utilities = {self.leader: reward, **follower_rewards}

        info = {
            "reward_generated": True,
            "utilities": utilities,
            "surplus": reward,
        }

        follower_actions = {agent: actions_dict[agent] for agent in self.followers_list}
        return {**self.types, self.leader: self.leader_observation(follower_actions)}, reward, True, info

    def log_info(self, info):
        self.logger.record("sampled_type", list(self.types.values()))
