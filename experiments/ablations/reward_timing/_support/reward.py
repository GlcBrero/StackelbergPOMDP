"""Fresh reward gameplay, with replay records for the paper's SimpleQ learner."""
from stackelberg_pomdp import LeaderQuery, RewardPhase, Reward, RewardDone
from experiments.ablations._support.game import FOLLOWER


class QReward(RewardPhase):
    def run(self, game, response_result):
        context = response_result.evaluation_context
        observations = game.reset()
        observation = game.leader_observation((observations[game.leader],))
        action = int((yield LeaderQuery(observation)))
        actions = response_result.followers.act({FOLLOWER: (observations[FOLLOWER],)})
        observations, rewards, _, _ = game.step({game.leader: action, **actions})
        reward = rewards[game.leader]
        next_observation = game.leader_observation((observations[game.leader],))
        replay = context["replay"]
        replay.append((observation, action, reward, next_observation, True))
        info = dict(exclude_from_buffer=False, is_query=False, is_reward_phase=True,
            leader_action=action, follower_action=actions[FOLLOWER], q_values=context["q_values"].copy(),
            reward_generated=True, episode={"r": reward, "l": context["response_episodes"] + 1})
        yield Reward(reward, info=info)
        return RewardDone(next_observation, dict(info, reward_phase_return=reward,
            reward_phase_steps=1, replay_transitions=replay))
