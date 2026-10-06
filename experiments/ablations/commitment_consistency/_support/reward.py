"""Fresh joint-action gameplay, retaining this ablation's consistency diagnostics."""
from stackelberg_pomdp import LeaderQuery, RewardPhase, Reward, RewardDone
from experiments.ablations._support.game import FOLLOWER


class MetaReward(RewardPhase):
    def run(self, game, response_result):
        followers = response_result.followers
        commitment = followers.policies[FOLLOWER].commitment
        observations = game.reset()
        histories = {agent: (observations[agent],) for agent in game.action_spaces}
        total, steps, consistent = 0., 0, True
        while True:
            state = int(observations[game.leader])
            observation = game.leader_observation(histories[game.leader])
            action = int((yield LeaderQuery(observation)))
            actions = followers.act({FOLLOWER: histories[FOLLOWER]})
            observations, rewards, done, info = game.step({game.leader: action, **actions})
            for agent in histories:
                histories[agent] += (observations[agent],)
            reward = rewards[game.leader]
            total, steps = total + reward, steps + 1
            info.update(exclude_from_buffer=False, is_query=False, is_reward_phase=True,
                leader_commitment=commitment, committed_action_at_state=commitment[state],
                commitment_consistent=bool(action == commitment[state]),
                meta_follower_action=actions[FOLLOWER], reward_generated=True)
            consistent = consistent and info["commitment_consistent"]
            if done:
                info["episode"] = {"r": float(total), "l": len(commitment) + steps}
            yield Reward(reward, info=info)
            if done:
                return RewardDone(game.leader_observation(histories[game.leader]), dict(info,
                    reward_phase_return=total, reward_phase_steps=steps,
                    physical_episode_steps=len(commitment) + steps, commitment_consistent=consistent))
