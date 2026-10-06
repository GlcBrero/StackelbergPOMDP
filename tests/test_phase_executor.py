"""Scientific invariants of the common response/reward query executor."""
from collections import OrderedDict

import gym
import numpy as np
import pytest

from stackelberg_pomdp.stackelberg import (
    Commitment, LeaderQuery, ResponsePhase, ResponseReady, Reward, RewardDone,
    RewardPhase, StackelbergPOMDPEnv,
)


def observation(value, private=0):
    return OrderedDict(actor=np.asarray([value], dtype=np.float32),
                       **{"critic:private": np.asarray([private], dtype=np.float32)})


class Game:
    action_space = gym.spaces.Box(-10, 10, (1,), dtype=np.float32)
    observation_space = gym.spaces.Dict(OrderedDict([
        ("actor", gym.spaces.Box(0, 10, (1,), dtype=np.float32)),
        ("critic:private", gym.spaces.Box(0, 10, (1,), dtype=np.float32)),
    ]))


class Response(ResponsePhase):
    def run(self, game):
        first = yield LeaderQuery(observation(0))
        replay = yield LeaderQuery(observation(0, private=5))
        np.testing.assert_array_equal(first, replay)
        # Mutating a returned action must not mutate the commitment.
        first[:] = -9
        return ResponseReady("followers")


class Evaluation(RewardPhase):
    def run(self, game, response_result):
        assert response_result.followers == "followers"
        replay = yield LeaderQuery(observation(0, private=7))
        yield Reward(float(replay[0]), elapsed_steps=2)
        fresh = yield LeaderQuery(observation(1))
        yield Reward(float(fresh[0]))
        replay = yield LeaderQuery(observation(0))
        yield Reward(float(replay[0]))
        return RewardDone(observation(0), {"finished": True})


def test_cache_spans_phases_and_new_reward_query_extends_it():
    env = StackelbergPOMDPEnv(Game(), Response(), Evaluation(), gamma=.5)
    obs = env.reset()
    assert obs["critic:is_reward_step"] == 0
    obs, reward, done, info = env.step(np.asarray([2.]))
    assert obs["actor"][0] == 1 and obs["critic:is_reward_step"] == 1
    assert not done and reward == 2 and info["continuation_discount"] == .25
    _, reward, done, info = env.step(np.asarray([4.]))
    assert done and reward == 5  # 4 + .5 * 2, including the cached tail
    assert info["policy_queries"] == 2
    assert info["cache_hits"] == 3
    assert info["commitment_entries"] == 2
    assert info["episode_return"] == 8
    env.reset()
    assert len(env.commitment) == 0


def test_complete_reward_phase_with_no_new_actor_decision():
    class CachedReward(RewardPhase):
        def run(self, game, result):
            action = yield LeaderQuery(observation(0, private=8))
            yield Reward(float(action[0]))
            yield Reward(3.)
            return RewardDone(observation(0))
    env = StackelbergPOMDPEnv(Game(), Response(), CachedReward())
    env.reset()
    _, reward, done, info = env.step(np.asarray([2.]))
    assert done and reward == 5 and info["reward_policy_queries"] == 0
    assert info["response_policy_queries"] == 1


@pytest.mark.parametrize("explicit_terminal", [True, False])
def test_terminal_critic_state_and_episode_diagnostics_survive_reset(explicit_terminal):
    class Respond(ResponsePhase):
        certificate = {"gap": .01}

        def critic_observation(self, game, commitment, phase):
            return {"critic:private": np.asarray([7 if phase == "reward" else 2], dtype=np.float32)}

        def run(self, game):
            yield LeaderQuery(observation(0))
            return ResponseReady(None, self.certificate, {"diagnostics": {"values": [1]}})

    class Play(RewardPhase):
        def run(self, game, response):
            yield Reward(2., elapsed_steps=2)
            return RewardDone(observation(0) if explicit_terminal else None)

    response = Respond()
    env = StackelbergPOMDPEnv(Game(), response, Play(), gamma=.5)
    env.reset()
    terminal, reward, done, info = env.step(np.asarray([1.]))
    assert done and env.phase == "done" and reward == 2.
    # Explicit terminal observations use the final reward state; absent ones
    # retain the last emitted observation, including its original critic state.
    assert terminal["critic:is_reward_step"] == int(explicit_terminal)
    assert terminal["critic:private"][0] == (7 if explicit_terminal else 2)
    assert env.observation_space.contains(terminal)
    assert info["continuation_discount"] == .25 and info["elapsed_steps"] == 2
    info["diagnostics"]["values"].append(9)
    response.certificate["gap"] = .5
    assert env.last_episode["diagnostics"]["values"] == [1]
    assert env.last_episode["response_certificate"]["gap"] == .01
    completed = env.last_episode
    env.reset()
    assert env.episode_count == 1 and env.last_episode is completed
    _, _, done, info = env.step(np.asarray([0.]))
    assert done and env.episode_count == 2
    assert info["episode_return"] == 2. and info["policy_queries"] == 1
    assert info["commitment_entries"] == 1
    env.close()


@pytest.mark.parametrize("lazy", [True, False])
def test_setting_selects_reuse_or_fresh_actions_in_both_phases(lazy):
    class RepeatedQueries(ResponsePhase):
        def run(self, game):
            first = yield LeaderQuery(observation(0))
            second = yield LeaderQuery(observation(0))
            return ResponseReady((float(first[0]), float(second[0])))
    class Play(RewardPhase):
        def run(self, game, response):
            action = yield LeaderQuery(observation(0))
            yield Reward(float(action[0]))
            return RewardDone(observation(0))
    env = StackelbergPOMDPEnv(Game(), RepeatedQueries(), Play(), lazy=lazy)
    env.reset()
    done, calls = False, 0
    while not done:
        calls += 1
        _, reward, done, info = env.step(np.asarray([calls], dtype=np.float32))
    assert calls == (1 if lazy else 3)
    assert env.response_result.followers == ((1., 1.) if lazy else (1., 2.))
    assert reward == (1. if lazy else 3.)
    assert info["commitment_entries"] == (1 if lazy else 0)
    assert info["cache_hits"] == (2 if lazy else 0)
    env.close()


@pytest.mark.parametrize("setting,kwargs", [
    ("mspm", {}),
    ("platform", {"seller_learning": None, "seeds": [1]}),
    ("atari", {"leader_role": "seller"}),
])
def test_fixed_commitment_paper_backends_reject_unsupported_fresh_queries(setting, kwargs):
    from importlib import import_module
    build = import_module(f"experiments.{setting}.experiment").build
    with pytest.raises(ValueError, match="requires lazy=True"):
        build(lazy=False, **kwargs)


def test_live_decisions_are_not_economic_commitment_replays():
    class LiveReward(RewardPhase):
        def run(self, game, result):
            for _ in range(2):
                action = yield LeaderQuery(observation(0), key=None)
                yield Reward(float(action[0]))
            return RewardDone(observation(0))
    env = StackelbergPOMDPEnv(Game(), Response(), LiveReward())
    env.reset()
    assert not env.step(np.asarray([2.]))[2]
    assert not env.step(np.asarray([3.]))[2]
    _, reward, done, info = env.step(np.asarray([4.]))
    assert done and reward == 4
    assert info["episode_return"] == 7
    assert info["commitment_entries"] == 1 and info["policy_queries"] == 3


def test_response_rewards_are_rejected_and_keys_cannot_leak_private_state():
    class InvalidResponse(ResponsePhase):
        def run(self, game):
            yield LeaderQuery(observation(0))
            yield Reward(1)
            return ResponseReady(None)
    env = StackelbergPOMDPEnv(Game(), InvalidResponse(), Evaluation())
    env.reset()
    with pytest.raises(ValueError, match="Response interactions"):
        env.step(np.asarray([1.]))
    cache = Commitment()
    cache.record(LeaderQuery(observation(0), key="same"), np.asarray([1.]))
    with pytest.raises(ValueError, match="different actor observations"):
        cache.lookup(LeaderQuery(observation(1), key="same"))
    with pytest.raises(RuntimeError, match="overwritten"):
        cache.record(LeaderQuery(observation(0), key="same"), np.asarray([2.]))


def test_mspm_paper_mechanism_certifies_and_uses_only_actor_observations():
    from experiments.mspm.experiment import build
    env = build(num_types=2)
    obs = env.reset()
    total, done = 0., False
    while not done:
        state, messages = obs["base_environment"], obs["base:follower_actions"]
        if state[:2].sum() == 1:
            action = [0, 0, 0]
        elif messages[1] == 0:
            action = [1, 0, 0]
        elif messages[0] == 0:
            action = [0, 1, .2]
        else:
            action = [1, 0, .4]
        assert env.observation_space.contains(obs)
        obs, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
        total += reward
    assert info["response_certificate"]["gap"] <= .05
    from stackelberg_pomdp import POMG
    assert isinstance(env.game, POMG)
    assert info["policy_queries"] <= 8 and info["reward_policy_queries"] == 0
    assert info["reward_profile_mass"] == pytest.approx(1.)
    assert info["efficient_welfare"] == pytest.approx(.44)
    assert total == pytest.approx(info["welfare"] - .44)


def test_bcce_uses_joint_distribution_and_conditional_type_weights():
    from experiments.mspm.response import bcce_gap
    # Correlated matching yields no profitable deviation; independent
    # averaging would destroy the perfect coordination of this response.
    utility = np.zeros((2, 2, 2, 2, 2))
    utility[:, :, 0, 0, :] = 1
    utility[:, :, 1, 1, :] = 1
    snapshots = np.zeros((2, 2, 2, 2))
    snapshots[0, :, :, 0] = 1
    snapshots[1, :, :, 1] = 1
    assert bcce_gap(snapshots, utility, np.asarray([[.8, .2], [.5, .5]])) == 0
    # Force miscoordination: deviating to the opponent's fixed label gains 1.
    snapshots[:] = 0
    snapshots[:, 0, :, 0] = 1
    snapshots[:, 1, :, 1] = 1
    assert bcce_gap(snapshots, utility, np.asarray([[.8, .2], [.5, .5]])) == 1


def test_collector_preserves_segment_discounts_and_complete_episodes():
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stackelberg_pomdp.policy import DefaultPolicy
    from stackelberg_pomdp.training import CustomPPO
    class Callback(BaseCallback):
        def _on_step(self):
            return True
    env = StackelbergPOMDPEnv(Game(), Response(), Evaluation(), gamma=.5)
    model = CustomPPO(DefaultPolicy, env, n_steps=2, batch_size=2,
                      gamma=.5, gae_lambda=1., episodes_per_rollout=2, seed=5)
    _, callback = model._setup_learn(4, Callback(), True, "test", False)
    before = [p.detach().clone() for p in model.policy.parameters()]
    assert model.collect_rollouts(model.env, callback, model.rollout_buffer, 2)
    buffer = model.rollout_buffer
    assert buffer.buffer_size == 4
    np.testing.assert_array_equal(buffer.episode_starts[:, 0], [1, 0, 1, 0])
    np.testing.assert_array_equal(buffer.actor_credit, 1)
    for start in (0, 2):
        assert buffer.returns[start, 0] == pytest.approx(
            buffer.rewards[start, 0] + .25 * buffer.rewards[start + 1, 0], abs=1e-6)
        assert buffer.returns[start + 1, 0] == pytest.approx(buffer.rewards[start + 1, 0], abs=1e-6)
    assert all(torch.equal(a, b) for a, b in zip(before, model.policy.parameters()))
    assert model.num_episodes == 2 and not hasattr(model.policy, "obs_action_map")
    model.env.close()


def test_default_actor_is_independent_of_private_state():
    import torch
    from stackelberg_pomdp.policy import DefaultPolicy
    env = StackelbergPOMDPEnv(Game(), Response(), Evaluation())
    policy = DefaultPolicy(env.observation_space, env.action_space, lambda _: 0.)
    first = env.reset()
    second = {key: np.copy(value) for key, value in first.items()}
    second["critic:private"][:] = 9
    second["critic:is_reward_step"] = 1
    a = policy.get_distribution(policy.obs_to_tensor(first)[0]).distribution.mean
    b = policy.get_distribution(policy.obs_to_tensor(second)[0]).distribution.mean
    assert torch.equal(a, b)


@pytest.mark.parametrize("included", [False, True])
def test_normal_form_q_phases_keep_one_action_and_reward_timing(included):
    from experiments.ablations._support.game import get_matrix_game, make_tabular_q_leader_env
    env = make_tabular_q_leader_env(get_matrix_game("coordination_zero_miscoordination"),
        response_episodes=10, include_response_reward=included,
        q_init="zero", exploration="epsilon_greedy", q_epsilon=0., q_alpha=.2, seed=2)
    env.reset()
    _, reward, done, info = env.step(0)
    assert done and info["policy_queries"] == 1 and info["cache_hits"] == 10
    assert info["underlying_steps"] == 11
    expected = env.game.game_spec.centered_payoffs[0, 0, 0]
    assert reward == pytest.approx(expected * (11 if included else 1))
    assert info["reward_phase_return"] == expected


@pytest.mark.parametrize("visible", [False, True])
def test_normal_form_visibility_condition_is_experiment_owned(visible):
    from experiments.ablations._support.game import get_matrix_game, make_meta_leader_env
    class Follower:
        def predict(self, observation, deterministic=True):
            return 0, None
    env = make_meta_leader_env(get_matrix_game("prisoners_dilemma", memory_mode="opponent"),
                              response_model=Follower(), phase_observable=visible)
    obs, done = env.reset(), False
    while not done:
        # Phase-conditioned behavior is intentionally possible only in the
        # visible-phase ablation; the generic core has no special-case switch.
        action = int(obs.get("base:is_reward_phase", 0))
        obs, reward, done, info = env.step(action)
    assert info["response_policy_queries"] == 3
    assert info["reward_phase_steps"] == env.game.game_spec.episode_length
    assert info["commitment_consistent"] is (not visible)
    assert (info["reward_policy_queries"] > 0) is visible


def test_independent_response_credit_ablation_has_the_paper_gradients():
    from experiments.ablations.response_credit.experiment import exact_moments
    for p in (.1, .3, .7, .9):
        value, full_gradient = exact_moments(p, "full")
        other_value, gameplay_gradient = exact_moments(p, "gameplay_only")
        assert value == pytest.approx(p)
        assert other_value == pytest.approx(p)
        assert full_gradient == pytest.approx(p * (1-p))
        assert gameplay_gradient == pytest.approx(-p * (1-p))
        z, epsilon = np.log(p / (1-p)), 1e-5
        above = exact_moments(1 / (1 + np.exp(-z-epsilon)), "full")[0]
        below = exact_moments(1 / (1 + np.exp(-z+epsilon)), "full")[0]
        assert full_gradient == pytest.approx((above-below) / (2*epsilon), abs=1e-9)


def test_mspm_matches_archived_scientific_response_and_exact_reward():
    import json
    from pathlib import Path
    from experiments.mspm.experiment import build
    cases = json.loads((Path(__file__).parent / "fixtures/mspm_reference.json").read_text())["cases"]
    for expected in cases:
        env = build(num_types=expected["types"])
        rules = np.random.RandomState(expected["seed"]).uniform(0, 1, (4, 3))
        obs, total, done = env.reset(), 0., False
        while not done:
            first, second = map(int, obs["base:follower_actions"])
            action = rules[2 * first + second].astype(np.float32).copy()
            if np.sum(obs["base_environment"][:2]) == 1:
                action[2] *= .5
            obs, reward, done, info = env.step(action)
            total += reward
        assert info["response_updates"] == expected["updates"]
        assert info["response_candidate"] == expected["candidate"]
        assert info["response_bcce_gap"] == pytest.approx(expected["gap"], abs=1e-12)
        np.testing.assert_allclose(env.response.state.weights, expected["weights"], atol=1e-12, rtol=0)
        assert total == pytest.approx(expected["reward"], abs=1e-12)
        env.close()


def test_simple_q_keeps_physical_replay_and_budget_after_query_compression():
    from experiments.ablations._support.game import make_tabular_q_leader_env, get_matrix_game
    from experiments.ablations._support.simple_q import SimpleQ
    env = make_tabular_q_leader_env(get_matrix_game("coordination_zero_miscoordination"),
                                   include_response_reward=True, response_episodes=10)
    model = SimpleQ(env=env, seed=2)
    model.learn(121)
    assert model.num_timesteps == 121 and env.episode_count == 11
    assert model.replay_buffer.size() == 121
    assert model._n_calls == 121
    assert model._n_updates == 2
    terminals = model.replay_buffer.dones[:121, 0]
    np.testing.assert_array_equal(np.flatnonzero(terminals), np.arange(10, 121, 11))
    env.close()
