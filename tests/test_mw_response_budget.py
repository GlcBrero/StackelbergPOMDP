"""A response prefix ends after complete MW updates, including direct wrappers."""

import pytest

from stackelberg_pomdp.env_setups import (
    get_matrix_design_env, get_mspm_env, get_simple_allocation_env,
)
from stackelberg_pomdp.envs.normal_form import BaseEnvSimpleMatrixGame
from stackelberg_pomdp.experiments import matrix_design, simple_allocation
from stackelberg_pomdp.experiments.common import finalized_config
from stackelberg_pomdp.games import get_normal_form_game
from stackelberg_pomdp.wrappers.core import MWFollowersWrapper, StackPOMDPWrapper


@pytest.mark.parametrize("mode", ["stackelberg", "hidden_queries"])
@pytest.mark.parametrize("factory,extra,requested,effective,updates", [
    (get_simple_allocation_env, {"num_messages": 3}, 100, 99, 33),
    (get_matrix_design_env, {}, 30, 28, 7),
    (get_mspm_env, {"setting": "PI", "num_types": 2, "num_messages": 2}, 7, 4, 1),
])
def test_factory_executes_only_complete_fixed_prefixes(mode, factory, extra,
                                                      requested, effective, updates):
    config = dict(logger=None, seed=1, followers_algorithm="MW", critic_obs="full",
                  pomdp_mode=mode, tot_num_response_episodes=requested,
                  tot_num_reward_episodes=2, **extra)
    env = factory(config)
    try:
        assert config["effective_tot_num_response_episodes"] == effective
        for _ in range(2):
            observation = env.reset()
            response_games = 0
            done = False
            for _ in range(1000):
                previous_flag = observation["critic:is_reward_step"]
                observation, reward, done, info = env.step(env.action_space.sample())
                if not info["is_reward_phase"]:
                    response_games += int(info.get("reward_generated", False))
                    assert previous_flag == 0
                    assert reward == 0
                    assert info["exclude_from_buffer"] == (mode == "hidden_queries")
                    if info["response_phase_done"]:
                        assert response_games == effective
                        assert info["response_updates"] == updates
                        assert info["requested_response_games"] == requested
                        assert info["response_prefix_games"] == effective
                        assert observation["critic:is_reward_step"] == 1
                else:
                    assert previous_flag == 1
                    assert not info["exclude_from_buffer"]
                if done:
                    break
            assert done
            assert response_games == effective
    finally:
        env.close()


@pytest.mark.parametrize("requested,effective", [(2, 2), (3, 2), (4, 4), (5, 4)])
def test_direct_stack_wrapper_rounds_down_without_factory(requested, effective):
    follower = MWFollowersWrapper(BaseEnvSimpleMatrixGame(get_normal_form_game("game_3"), seed=0))
    env = StackPOMDPWrapper(follower, tot_num_response_episodes=requested,
                           tot_num_reward_episodes=2)
    try:
        env.reset()
        infos = [env.step(1)[3] for _ in range(effective + 2)]
        assert [row["is_reward_phase"] for row in infos] == [False] * effective + [True] * 2
        assert infos[effective - 1]["response_updates"] == effective // 2
    finally:
        env.close()


@pytest.mark.parametrize("requested", [0, 1, -1])
def test_prefix_shorter_than_one_update_is_rejected(requested):
    follower = MWFollowersWrapper(BaseEnvSimpleMatrixGame(get_normal_form_game("game_3"), seed=0))
    try:
        with pytest.raises(ValueError, match="shorter than one MW"):
            StackPOMDPWrapper(follower, tot_num_response_episodes=requested)
    finally:
        follower.close()


@pytest.mark.parametrize("messages", [1, 2, 3, None])
def test_shared_cycle_default_reaches_environment(messages):
    module = simple_allocation if messages else matrix_design
    argv = ["--num_messages", str(messages)] if messages else []
    experiment = f"simple_allocation:{messages}" if messages else "matrix_design"
    config = finalized_config(module.build_parser().parse_args(argv), experiment)
    config["logger"] = None
    factory = get_simple_allocation_env if messages else get_matrix_design_env
    env = factory(config)
    try:
        assert config["mw_response_cycles"] == 33
        assert env.tot_num_response_episodes == 33 * (messages or 4)
        assert config["effective_tot_num_response_episodes"] == env.tot_num_response_episodes
    finally:
        env.close()


def test_explicit_game_budget_overrides_default_cycles():
    args = simple_allocation.build_parser().parse_args(["--tot_num_response_episodes", "100"])
    config = finalized_config(args, "simple_allocation:3")
    assert config["mw_response_cycles"] is None
    config["logger"] = None
    env = get_simple_allocation_env(config)
    try:
        assert env.tot_num_response_episodes == 99
    finally:
        env.close()


def test_conflicting_cli_budgets_are_rejected():
    with pytest.raises(SystemExit):
        simple_allocation.build_parser().parse_args([
            "--tot_num_response_episodes", "100", "--mw_response_cycles", "33",
        ])


@pytest.mark.parametrize("argv", [["--mw_response_cycles", "0"],
                                 ["--mw_response_cycles", "33", "--followers_algorithm", "Qlearning"]])
def test_invalid_cycle_options_are_rejected(argv):
    args = simple_allocation.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match="requires MW followers and a positive cycle count"):
        finalized_config(args, "simple_allocation:3")


def test_programmatic_cycle_count_and_conflict_validation():
    config = dict(logger=None, seed=1, num_messages=3, followers_algorithm="MW",
                  mw_response_cycles=33)
    env = get_simple_allocation_env(config)
    try:
        assert env.tot_num_response_episodes == 99
        assert config["tot_num_response_episodes"] == 99
    finally:
        env.close()
    config["tot_num_response_episodes"] = 100
    with pytest.raises(ValueError, match="conflicts"):
        get_simple_allocation_env(config)
