"""Both players choose 0 or 1. The follower wants to match the leader.

The leader earns 1 when both choose 1, and 0 otherwise. The follower earns
1 for matching. Neither player observes the other's final action before acting.
"""
import gym
from stackelberg_pomdp import POMG


class Game(POMG):
    horizon = 1

    def __init__(self):
        self.action_spaces = {agent: gym.spaces.Discrete(2)
                              for agent in ("leader", "follower")}
        self.observation_spaces = {agent: gym.spaces.Discrete(1)
                                   for agent in self.action_spaces}
        self.leader_observation_space = gym.spaces.Dict(
            {"base_environment": gym.spaces.Discrete(1)})

    def leader_observation(self, history):
        return {"base_environment": history[-1]}  # One state, so no memory needed.

    def seed(self, seed=None):
        for space in self.action_spaces.values():
            space.seed(seed)
        return [seed]

    def reset(self):
        return {"leader": 0, "follower": 0}

    def step(self, actions):
        leader_action = int(actions["leader"])
        follower_action = int(actions["follower"])
        rewards = {"leader": float(leader_action == follower_action == 1),
                   "follower": float(leader_action == follower_action)}
        info = {"follower_action": follower_action,
                "follower_reward": rewards["follower"]}
        return {"leader": 0, "follower": 0}, rewards, True, info
