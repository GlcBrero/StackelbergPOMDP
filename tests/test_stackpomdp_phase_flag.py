"""Observations describe the next action; info describes the completed step."""

import pytest

from stackelberg_pomdp.envs.normal_form import BaseEnvSimpleMatrixGame
from stackelberg_pomdp.games import get_normal_form_game
from stackelberg_pomdp.wrappers.core import (
    MWFollowersWrapper,
    StackPOMDPWrapper,
)


@pytest.mark.parametrize("critic_obs", ["flag", "full", None])
@pytest.mark.parametrize("response_variant", ["stackelberg", "hidden_queries"])
def test_critic_phase_flag_is_ready_before_first_reward_action(
        critic_obs, response_variant,
):
    env = StackPOMDPWrapper(
        MWFollowersWrapper(
            BaseEnvSimpleMatrixGame(get_normal_form_game("game_3"), seed=0),
            fixed_seed=0,
        ),
        tot_num_response_episodes=2,
        tot_num_reward_episodes=2,
        critic_obs=critic_obs,
        response_variant=response_variant,
    )
    try:
        # Repeat to verify that resetting restores the response-phase flag.
        for _ in range(2):
            observation = env.reset()
            action_phase_flags, rewards, dones, infos = [], [], [], []
            for _ in range(4):
                assert observation["base_environment"] == 0
                action_phase_flags.append(observation.get("critic:is_reward_step"))
                observation, reward, done, info = env.step(1)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

            assert rewards == pytest.approx([0, 0, 2 / 3, 2 / 3])
            assert dones == [False, False, False, True]
            assert [info["is_reward_phase"] for info in infos] == [
                False, False, True, True,
            ]
            assert infos[1]["response_phase_done"] is True
            hidden = response_variant == "hidden_queries"
            assert [info["exclude_from_buffer"] for info in infos] == [
                hidden, hidden, False, False,
            ]
            if critic_obs is None:
                assert action_phase_flags == [None, None, None, None]
                assert "critic:is_reward_step" not in observation
            else:
                assert action_phase_flags == [0, 0, 1, 1]
                assert observation["critic:is_reward_step"] == 1
    finally:
        env.close()


def test_unknown_response_mode_is_rejected():
    response = MWFollowersWrapper(
        BaseEnvSimpleMatrixGame(get_normal_form_game("game_3"), seed=0)
    )
    try:
        with pytest.raises(ValueError, match="Unsupported response_variant"):
            StackPOMDPWrapper(response, response_variant="misspelled_mode")
    finally:
        response.close()
