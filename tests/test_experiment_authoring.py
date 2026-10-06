"""A reader can run the example and extend it through the shared runner."""
import json

import pytest

from experiments.example.experiment import build


@pytest.mark.parametrize("action", [0, 1])
def test_matching_game_executes_the_strategy_under_the_same_commitment(action):
    env = build()
    observation = env.reset()
    assert observation["base_environment"] == 0
    _, reward, done, info = env.step(action)
    assert done and reward == float(action)
    assert env.response_result.followers.act({"follower": (0,)}) == {"follower": action}
    assert info["follower_action"] == action and info["follower_reward"] == 1
    assert info["policy_queries"] == 1
    assert info["reward_policy_queries"] == 0 and info["cache_hits"] == 1
    # A new episode selects a new strategy and a new commitment.
    env.reset()
    _, reward, done, info = env.step(1 - action)
    assert done and reward == float(1 - action)
    assert info["follower_action"] == 1 - action
    env.close()


@pytest.mark.parametrize("lazy", [True, False])
def test_example_trains_saves_and_reloads_with_default_policy(tmp_path, lazy):
    from support.custom import main
    from stable_baselines3 import PPO
    from stackelberg_pomdp.training import EpisodePPO

    output = tmp_path / "run"
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"lazy": lazy}))
    main("example", ["--steps", "16", "--seed", "3", "--config", str(config),
                     "--output", str(output)])
    summary = json.loads((output / "run.json").read_text())
    assert summary["completed_episodes"] == 16
    assert summary["policy_decisions"] == (16 if lazy else 32)
    env = build(lazy=lazy)
    model = EpisodePPO.load(output / "model.zip", env=env)
    assert EpisodePPO.train is PPO.train
    observation, done = env.reset(), False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, _ = env.step(action)
    assert reward in (0.0, 1.0)
    env.close()


def test_standard_gameplay_executes_the_selected_follower_strategy():
    env = build(lazy=False)
    env.reset()
    # The response selects the follower strategy "always play 1".
    _, reward, done, _ = env.step(1)
    assert not done and reward == 0
    assert env.response_result.followers.act({"follower": (0,)}) == {"follower": 1}
    # A fresh leader action must not overwrite the follower's chosen strategy.
    _, reward, done, info = env.step(0)
    assert done and reward == 0
    assert info["follower_action"] == 1 and info["follower_reward"] == 0
    assert info["policy_queries"] == 2 and info["cache_hits"] == 0
    assert info["commitment_entries"] == 0
    env.close()


def test_fresh_gameplay_uses_private_histories_and_joint_actions():
    import gym
    import numpy as np
    from stackelberg_pomdp import (POMG, PolicyProfile, ResponsePhase,
                                  ResponseReady, LeaderQuery, StackelbergPOMDPEnv)

    class HiddenGame(POMG):
        horizon = 2
        action_spaces = {"leader": gym.spaces.Discrete(2),
                         "follower": gym.spaces.Discrete(2)}
        observation_spaces = {"leader": gym.spaces.Discrete(1),
                              "follower": gym.spaces.Discrete(10)}
        leader_observation_space = gym.spaces.Dict({"base_environment":
            gym.spaces.Box(-1, 1, (2,), dtype=np.float32)})

        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1
            self.private_state = 3 if self.resets == 1 else 7
            self.steps = 0
            self.joint_actions = []
            return {"leader": 0, "follower": self.private_state}

        def leader_observation(self, history):
            # Equal current observations with different histories need distinct keys.
            return {"base_environment": np.asarray(
                [0, -1 if len(history) == 1 else 0], dtype=np.float32)}

        def step(self, actions):
            self.steps += 1
            self.joint_actions.append(actions)
            return ({"leader": 0, "follower": self.private_state},
                    {"leader": 1., "follower": 100.}, False, {})

    class RememberingPolicy:
        def __init__(self):
            self.seen = []

        def act(self, history):
            self.seen.append(history)
            return int(history[-1] == 7)

    follower = RememberingPolicy()

    class Response(ResponsePhase):
        def run(self, game):
            observations = game.reset()
            yield LeaderQuery(game.leader_observation((observations[game.leader],)))
            game.step({"leader": 0, "follower": 0})
            game.step({"leader": 0, "follower": 0})
            return ResponseReady(PolicyProfile({"follower": follower}))

    game = HiddenGame()
    env = StackelbergPOMDPEnv(game, Response(), lazy=True)
    env.reset()
    _, reward, done, _ = env.step(1)
    assert not done and reward == 1
    _, reward, done, info = env.step(0)
    assert done and reward == 1 and info["episode_return"] == 2
    assert game.resets == 2 and game.steps == 2
    assert follower.seen == [(7,), (7, 7)]
    assert game.joint_actions == [{"follower": 1, "leader": 1},
                                  {"follower": 1, "leader": 0}]
    assert info["cache_hits"] == 1 and info["policy_queries"] == 2
    env.close()


def test_custom_profile_preserves_correlated_follower_actions():
    from experiments.example.game import Game
    from stackelberg_pomdp import (PolicyProfile, ResponsePhase, ResponseReady,
                                  LeaderQuery, StackelbergPOMDPEnv)

    class TwoFollowers(Game):
        def __init__(self):
            super().__init__()
            self.action_spaces["second"] = self.action_spaces["follower"]
            self.observation_spaces["second"] = self.observation_spaces["follower"]

        def reset(self):
            return {"leader": 0, "follower": 0, "second": 0}

        def step(self, actions):
            assert actions["follower"] == actions["second"] == 1
            return self.reset(), {"leader": 1, "follower": 0, "second": 0}, True, {}

    class JointProfile(PolicyProfile):
        def __init__(self):
            super().__init__({})

        def act(self, histories):
            assert histories == {"follower": (0,), "second": (0,)}
            return {"follower": 1, "second": 1}

    class Response(ResponsePhase):
        def run(self, game):
            yield LeaderQuery(game.leader_observation((0,)))
            return ResponseReady(JointProfile())

    env = StackelbergPOMDPEnv(TwoFollowers(), Response())
    env.reset()
    assert env.step(1)[1:3] == (1., True)
    env.close()


def test_default_gameplay_rejects_response_dependent_initialization():
    from experiments.example.game import Game
    from stackelberg_pomdp import (PolicyProfile, ResponsePhase, ResponseReady,
                                  LeaderQuery, StackelbergPOMDPEnv)

    class Response(ResponsePhase):
        def run(self, game):
            yield LeaderQuery(game.leader_observation((0,)))
            return ResponseReady(PolicyProfile({}), evaluation_context={"state": 3})

    env = StackelbergPOMDPEnv(Game(), Response())
    env.reset()
    with pytest.raises(ValueError, match="custom RewardPhase"):
        env.step(0)
    env.close()
