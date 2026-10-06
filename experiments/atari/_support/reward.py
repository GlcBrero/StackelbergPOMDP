"""Play both Atari games, reusing the five committed economic actions."""
from stackelberg_pomdp import LeaderQuery, RewardPhase, Reward, RewardDone
from experiments.atari._support.wrappers.meta_follower import record_transition


class AtariReward(RewardPhase):
    """Live gameplay draws and canonical economic replays use one executor."""
    def __init__(self, response):
        self.response = response

    def run(self, game, response_result):
        runtime = self.response.runtime
        runtime.followers = response_result.followers
        runtime.on_step_mode_changed("reward")
        observation = runtime.current_leader_observation()
        while True:
            key = ("trade", int(game.core.next_event)) if game.core.at_event else None
            action = yield LeaderQuery(observation, key=key)
            previous = observation
            observation, reward, done, info = runtime._reward_step(action)
            record_transition(game, previous, action, reward, done, info, reward_phase=True)
            yield Reward(reward, info=info)
            if done:
                return RewardDone(observation, info)
