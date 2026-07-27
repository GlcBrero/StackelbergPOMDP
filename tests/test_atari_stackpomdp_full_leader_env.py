from collections import OrderedDict

import gym
import numpy as np
import pytest

from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
)
from stackelberg_pomdp.atari.stackpomdp_full_leader_env import (
    FullTraceStackPOMDPAtariLeaderEnv,
    StepwiseDualAtariTradeCore,
    actor_observation,
)
from stackelberg_pomdp.atari.query_trace import LeaderQueryTrace


GAME_ACTION = 2


class _FrozenController:
    def __call__(self, observation):
        del observation
        return GAME_ACTION


class _FakeGameplaySide:
    """Deterministic one-point-per-bullet Atari stand-in."""

    def __init__(self, *, seed, config, controller, env_factory):
        del seed, config, env_factory
        self.controller = controller
        self.game_action_count = 6
        self.observation_space = gym.spaces.Dict(OrderedDict([
            (
                "image",
                gym.spaces.Box(0, 255, shape=(1, 2, 2), dtype=np.uint8),
            ),
            (
                "ammo_fraction",
                gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "projectile_active",
                gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "action_mask",
                gym.spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
            ),
            (
                "offer_active",
                gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "opportunities_remaining",
                gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "critic:price",
                gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
        ]))
        self._ammo = 0
        self.observation = None
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0
        self.step_calls = 0
        self.executed_actions = []

    @property
    def ammo(self):
        return int(self._ammo)

    def _refreshed_observation(self):
        image_value = min(255, self.step_calls + 10 * self._ammo)
        mask = np.ones(6, dtype=np.float32)
        return OrderedDict([
            (
                "image",
                np.full((1, 2, 2), image_value, dtype=np.uint8),
            ),
            (
                "ammo_fraction",
                np.array([self._ammo / 5.0], dtype=np.float32),
            ),
            ("projectile_active", np.array([0.0], dtype=np.float32)),
            ("action_mask", mask),
            ("offer_active", np.array([0.0], dtype=np.float32)),
            (
                "opportunities_remaining",
                np.array([1.0], dtype=np.float32),
            ),
            ("critic:price", np.array([0.0], dtype=np.float32)),
        ])

    def reset(self):
        self._ammo = 0
        self.game_reward = 0.0
        self.shots_fired = 0
        self.step_calls = 0
        self.executed_actions = []
        self.observation = self._refreshed_observation()
        return self.observation

    def grant(self, amount=1):
        if self._ammo + amount > 5:
            return 0
        self._ammo += int(amount)
        self.observation = self._refreshed_observation()
        return int(amount)

    def consume(self, amount=1):
        if amount > self._ammo:
            raise RuntimeError("fake ammo underflow")
        self._ammo -= int(amount)
        self.observation = self._refreshed_observation()

    def step(self, action=None):
        if int(action) != GAME_ACTION:
            raise RuntimeError("fake gameplay did not use frozen action")
        self.executed_actions.append(int(action))
        self.step_calls += 1
        fired = int(self._ammo > 0)
        if fired:
            self.consume(1)
        reward = float(fired)
        self.shots_fired += fired
        self.game_reward += reward
        self.observation = self._refreshed_observation()
        return reward, fired, {"shots_fired_this_step": fired}

    def close(self):
        return


class _FakeMetaFollower:
    def __init__(self, role, economic_action):
        self.policy = type("PolicyMetadata", (), {"economic_role": role})()
        self.economic_action = float(economic_action)
        self.observations = []
        self.query_traces = []

    def predict_from_query_trace(
            self, observation, query_trace, deterministic=True
    ):
        assert deterministic
        assert isinstance(query_trace, LeaderQueryTrace)
        self.observations.append(OrderedDict(
            (key, np.array(value, copy=True))
            for key, value in observation.items()
        ))
        self.query_traces.append(query_trace)
        return np.array([4.0, self.economic_action], dtype=np.float32), None


class _LegacyFiveContextFollower:
    """Stand-in for the already-trained five-economic-input checkpoint."""

    def __init__(self, role):
        self.policy = type("PolicyMetadata", (), {"economic_role": role})()
        self.observations = []

    def predict(self, observation, deterministic=True):
        assert deterministic
        self.observations.append(OrderedDict(
            (key, np.array(value, copy=True))
            for key, value in observation.items()
        ))
        return np.array([4.0, 1.0], dtype=np.float32), None


def _config():
    return BilateralAtariConfig(
        seed=17,
        gameplay_horizon=7,
        event_tail_steps=2,
        fixed_event_steps=(0, 1, 2, 3, 4),
    )


def _controller_factory():
    return _FrozenController()


def _core_factory(config, *, game_checkpoint, controller_factory):
    del game_checkpoint
    return StepwiseDualAtariTradeCore(
        config,
        controller_factory=controller_factory,
        side_factory=_FakeGameplaySide,
        env_factory=None,
    )


def _make_env(*, leader_role=SELLER, follower_action=0.25):
    follower_role = BUYER if leader_role == SELLER else SELLER
    response = _FakeMetaFollower(follower_role, follower_action)
    env = FullTraceStackPOMDPAtariLeaderEnv(
        leader_role=leader_role,
        response_checkpoint="unused.zip",
        game_checkpoint="unused.zip",
        config=_config(),
        core_factory=_core_factory,
        controller_factory=_controller_factory,
        response_model_factory=lambda path, device: response,
    )
    return env, response


def _assert_mapping_equal(left, right):
    assert tuple(left.keys()) == tuple(right.keys())
    for key in left:
        assert np.array_equal(left[key], right[key]), key


def test_stepwise_episode_exposes_full_trace_and_every_reward_transition():
    env, response = _make_env()
    prices = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    query_actions = np.column_stack([
        np.arange(5, dtype=np.float32),
        prices,
    ])

    observation = env.reset()
    query_observations = []
    transitions = []
    for event_index, action in enumerate(query_actions):
        query_observations.append({
            key: np.array(value, copy=True)
            for key, value in observation.items()
        })
        observation, reward, done, info = env.step(action)
        transitions.append((reward, done, dict(info)))
        assert reward == 0.0
        assert not done
        assert info["is_query"]
        assert info["substep_type"] == "query"
        assert not info["exclude_from_buffer"]
        if event_index < 4:
            assert not info["reward_game_started"]
        else:
            assert info["reward_game_started"]

    assert isinstance(env.leader_query_trace, LeaderQueryTrace)
    assert len(env.leader_query_trace.queries) == 5
    assert observation["event_active"].item() == 1.0
    assert observation["event_one_hot"][0] == 1.0

    reward_observations = {}
    paused_step_counts = []
    done = False
    while not done:
        if observation["event_active"].item() == 1.0:
            event_index = int(np.argmax(observation["event_one_hot"]))
            reward_observations[event_index] = {
                key: np.array(value, copy=True)
                for key, value in observation.items()
            }
            action = query_actions[event_index]
            before = (
                env.core.seller.step_calls,
                env.core.buyer.step_calls,
            )
        else:
            action = np.array([GAME_ACTION, 0.99], dtype=np.float32)
            before = None

        observation, reward, done, info = env.step(action)
        transitions.append((reward, done, dict(info)))
        assert info["is_reward_phase"]
        assert not info["exclude_from_buffer"]
        if info["substep_type"] == "trade":
            after = (
                env.core.seller.step_calls,
                env.core.buyer.step_calls,
            )
            paused_step_counts.append((before, after))
            assert before == after
            assert not info["emulator_advanced"]
            assert info["cache_hit"]
        else:
            assert info["substep_type"] == "gameplay"
            assert info["emulator_advanced"]

    horizon = _config().gameplay_horizon
    assert len(transitions) == 5 + horizon + 5
    assert [item[0] for item in transitions[:5]] == [0.0] * 5
    assert sum(item[2]["substep_type"] == "query" for item in transitions) == 5
    assert sum(item[2]["substep_type"] == "trade" for item in transitions) == 5
    assert (
        sum(item[2]["substep_type"] == "gameplay" for item in transitions)
        == horizon
    )
    assert all(before == after for before, after in paused_step_counts)

    # Query and deployed trade observations reveal no phase to the actor.  The
    # phase differs only in the critic-only state.
    for event_index in range(5):
        _assert_mapping_equal(
            actor_observation(query_observations[event_index]),
            actor_observation(reward_observations[event_index]),
        )
        assert query_observations[event_index]["critic:state"][0] == 0.0
        assert reward_observations[event_index]["critic:state"][0] == 1.0
        assert not any(
            "phase" in key
            for key in actor_observation(query_observations[event_index])
        )

    assert len(response.observations) == 5
    assert len(response.query_traces) == 5
    assert len({trace.sha256 for trace in response.query_traces}) == 1
    for follower_observation, trace in zip(
            response.observations, response.query_traces
    ):
        assert np.array_equal(
            trace.full_action_matrix(), query_actions
        )
        reconstructed = trace.reconstruct()
        stacked_events = np.stack([
            item[0]["event_one_hot"] for item in reconstructed
        ])
        assert np.array_equal(stacked_events, np.eye(5, dtype=np.float32))
        stacked_images = np.stack([
            item[0]["image"] for item in reconstructed
        ])
        assert stacked_images.shape == (5, 1, 2, 2)
        assert stacked_images.sum() == 0
        # The already-trained five-context follower remains usable through the
        # explicitly verified sufficient-statistic projection.
        assert np.array_equal(
            follower_observation["opponent_context"], prices
        )
        assert follower_observation["critic:state"].shape == (12,)

    terminal = transitions[-1][2]
    # The follower threshold 0.25 buys at 0.1 and 0.2.  The buyer fires those
    # two bullets; the seller fires the three rejected bullets.
    assert terminal["purchases"] == 2
    assert np.isclose(terminal["payments"], 0.3)
    assert terminal["buyer_shots_fired"] == 2
    assert terminal["seller_shots_fired"] == 3
    assert terminal["seller_final_ammo"] == 0
    assert terminal["buyer_final_ammo"] == 0
    assert terminal["cache_hits"] == 5
    assert terminal["outer_transition_count"] == 5 + horizon + 5
    assert terminal["seller_bullet_error"] == 0
    assert terminal["buyer_bullet_error"] == 0
    assert np.isclose(terminal["seller_reward"], 0.3 + 0.1 * 3)
    assert np.isclose(terminal["buyer_reward"], 2.0 - 0.3)
    assert np.isclose(
        sum(item[0] for item in transitions), terminal["leader_reward"]
    )


def test_reward_trade_requires_a_full_policy_cache_hit():
    env, _ = _make_env()
    query_actions = np.column_stack([
        np.arange(5, dtype=np.float32),
        np.linspace(0.1, 0.5, 5, dtype=np.float32),
    ])
    env.reset()
    for action in query_actions:
        observation, _, _, _ = env.step(action)
    assert observation["event_one_hot"][0] == 1.0

    wrong_full_action = np.array(query_actions[0], copy=True)
    wrong_full_action[0] += 1.0
    with pytest.raises(RuntimeError, match="policy cache miss"):
        env.step(wrong_full_action)
    assert env.core.seller.step_calls == 0
    assert env.core.buyer.step_calls == 0


def test_gameplay_action_cannot_depart_from_frozen_e0():
    env, _ = _make_env()
    query_actions = np.column_stack([
        np.arange(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
    ])
    env.reset()
    for action in query_actions:
        observation, _, _, _ = env.step(action)
    # Resolve event zero using the required query cache hit.
    observation, _, done, _ = env.step(query_actions[0])
    assert not done
    assert observation["event_active"].item() == 0.0

    with pytest.raises(RuntimeError, match="frozen deterministic E0"):
        env.step(np.array([GAME_ACTION + 1, 0.0], dtype=np.float32))
    assert env.core.seller.step_calls == 0
    assert env.core.buyer.step_calls == 0


def test_existing_five_context_follower_uses_verified_trace_projection():
    response = _LegacyFiveContextFollower(BUYER)
    env = FullTraceStackPOMDPAtariLeaderEnv(
        leader_role=SELLER,
        response_checkpoint="unused.zip",
        game_checkpoint="unused.zip",
        config=_config(),
        core_factory=_core_factory,
        controller_factory=_controller_factory,
        response_model_factory=lambda path, device: response,
    )
    prices = np.linspace(0.1, 0.5, 5, dtype=np.float32)
    actions = np.column_stack([
        np.arange(5, dtype=np.float32),
        prices,
    ])
    env.reset()
    for action in actions:
        observation, _, _, _ = env.step(action)
    assert isinstance(env.leader_query_trace, LeaderQueryTrace)

    env.step(actions[0])
    assert len(response.observations) == 1
    follower_observation = response.observations[0]
    assert np.array_equal(follower_observation["opponent_context"], prices)
    assert follower_observation["critic:state"].shape == (12,)
    assert not any(key.startswith("leader_trace:") for key in follower_observation)
