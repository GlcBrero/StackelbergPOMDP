"""Trace one matrix-game episode and one PPO update; see README.md."""

import argparse

from stackelberg_pomdp.algorithms.on_policy import CustomPPO
from stackelberg_pomdp.callbacks import FixPolicyActionsCallback
from stackelberg_pomdp.envs.normal_form import BaseEnvSimpleMatrixGame
from stackelberg_pomdp.games import get_normal_form_game
from stackelberg_pomdp.policies.generic import CustomPolicy
from stackelberg_pomdp.rl_trainer_setup import get_observation_split
from stackelberg_pomdp.wrappers.core import (
    MWFollowersWrapper,
    StackPOMDPWrapper,
)


class PrintEpisode(FixPolicyActionsCallback):
    """Print executed transitions while retaining the real cache lifecycle."""

    def _on_step(self):
        info = self.locals["infos"][0]
        phase = "reward" if info["is_reward_phase"] else "response"
        leader = int(self.locals["actions"][0])
        follower = info["followers_actions"]["follower_0"]
        retained = not info["exclude_from_buffer"]
        print(
            f"step={self.num_timesteps} phase={phase} "
            f"leader={leader} follower={follower} "
            f"reward={float(self.locals['rewards'][0]):.3f} "
            f"retain={retained} cache_entries={len(self.model.policy.obs_action_map)}"
        )
        # In particular, clear the action map when the outer episode ends.
        return super()._on_step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-queries", action="store_true")
    args = parser.parse_args()

    game = get_normal_form_game("game_3")  # Two leader and two follower actions.
    base = BaseEnvSimpleMatrixGame(game, seed=0)
    response = MWFollowersWrapper(base, fixed_seed=0)
    env = StackPOMDPWrapper(
        response,
        tot_num_response_episodes=2,  # One complete two-action MW update.
        tot_num_reward_episodes=2,   # Play twice against the greedy MW response.
        critic_obs="full",
        response_variant="hidden_queries" if args.hidden_queries else "stackelberg",
    )
    cutoff, actor_keys = get_observation_split(env)
    stored_steps = 2 if args.hidden_queries else 4
    model = CustomPPO(
        CustomPolicy, env, seed=0, device="cpu",
        n_steps=stored_steps, batch_size=stored_steps, n_epochs=1,
        gamma=1.0, gae_lambda=1.0, ent_coef=0.01,
        policy_kwargs={"cutoff_entry": cutoff, "actor_obs_keys": actor_keys},
        verbose=0,
    )
    try:
        model.learn(total_timesteps=4, callback=PrintEpisode())
        print(f"Executed steps: {model.num_timesteps}")
        print(f"Stored rewards: {model.rollout_buffer.rewards.ravel().tolist()}")
        print(f"Cache entries after episode: {len(model.policy.obs_action_map)}")
    finally:
        model.get_env().close()


if __name__ == "__main__":
    main()
