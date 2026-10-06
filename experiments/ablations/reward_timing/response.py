"""Learn a follower strategy with tabular Q-learning in the one-shot game."""
from dataclasses import dataclass
import numpy as np
from stackelberg_pomdp import LeaderQuery, PolicyProfile, ResponsePhase, ResponseReady, Reward
from experiments.ablations._support.game import FOLLOWER
from ._support.q_learning import TabularQLearner


@dataclass(frozen=True)
class FollowerPolicy:
    action: int

    def act(self, history):
        return self.action


class QResponse(ResponsePhase):
    allow_rewards = True  # Only the included-response-reward condition uses this.

    def __init__(self, spec, *, response_episodes=10, include_response_reward=False, **learning_options):
        if spec.episode_length != 1 or response_episodes < 1:
            raise ValueError("Use a one-shot game and a positive response episode count")
        self.response_episodes = int(response_episodes)
        self.include_response_reward = bool(include_response_reward)
        self.learner = TabularQLearner(spec.num_follower_actions, **learning_options)

    def seed(self, seed=None):
        self.learner.seed(seed)

    def run(self, game):
        learner, replay = self.learner, []
        learner.reset()
        for index in range(self.response_episodes):
            observations = game.reset()
            observation = game.leader_observation((observations[game.leader],))
            action = int((yield LeaderQuery(observation)))
            follower_action = learner.action()
            observations, rewards, _, _ = game.step({game.leader: action, FOLLOWER: follower_action})
            follower_reward, previous = rewards[FOLLOWER], float(learner.q[follower_action])
            learner.q[follower_action] += learner.alpha * (follower_reward - learner.q[follower_action])
            reward = rewards[game.leader] if self.include_response_reward else 0.
            next_observation = game.leader_observation((observations[game.leader],))
            replay.append((observation, action, reward, next_observation, False))
            yield Reward(reward, info=dict(exclude_from_buffer=False, is_query=True,
                is_reward_phase=False, response_phase_done=index + 1 == self.response_episodes,
                response_index=index, leader_action=action, follower_action=follower_action,
                follower_reward=follower_reward, q_previous=previous, q_target=follower_reward,
                q_values=learner.q.copy(), reward_generated=self.include_response_reward))
        strategy = FollowerPolicy(int(np.argmax(learner.q)))
        return ResponseReady(PolicyProfile({FOLLOWER: strategy}), evaluation_context={
            "replay": replay, "q_values": learner.q.copy(), "response_episodes": self.response_episodes})

    def max_episode_transitions(self, game):
        return self.response_episodes + game.horizon

    max_executed_transitions = max_episode_transitions
