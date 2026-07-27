from types import SimpleNamespace

import gym
import numpy as np

from stackelberg_pomdp.atari.curriculum_env import (
    AtariCurriculumConfig,
    AtariCurriculumEnv,
)
from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTOR_STATE,
    CACHED_TRADE_REPLAY,
    FOLLOWER_TRADE,
    GAMEPLAY,
    LEADER_QUERY,
    actor_observation,
)
from stackelberg_pomdp.atari.query_trace import LeaderQueryTrace
from stackelberg_pomdp.atari.meta_response import (
    AtariMetaFollowerWrapper,
    make_stackpomdp_atari_leader_env,
)
from stackelberg_pomdp.atari.stackpomdp_env import (
    AtariFixedCommitmentResponseWrapper,
    BUYER,
    SELLER,
    BilateralAtariConfig,
    BilateralAtariRewardEnv,
    make_atari_meta_response_env,
)
from stackelberg_pomdp.gym_envs.envs.wrappers import StackPOMDPWrapper


FIRE = 1


class _FakeAmmoWrapper:
    fire_action_indices = (FIRE,)


class _FakeSide:
    def __init__(self, *, seed, initial_ammo, capacity, **kwargs):
        del seed, kwargs
        self.initial_ammo = int(initial_ammo)
        self.capacity = int(capacity)
        self.game_action_count = 6
        self.image_space = gym.spaces.Box(
            0, 255, shape=(84, 84, 4), dtype=np.uint8
        )
        self.ammo_wrapper = _FakeAmmoWrapper()
        self._ammo = self.initial_ammo
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0
        self.step_calls = 0
        self.env = SimpleNamespace(
            render=lambda mode="rgb_array": np.zeros(
                (84, 84, 3), dtype=np.uint8
            )
        )

    @property
    def ammo(self):
        return self._ammo

    @property
    def projectile_active(self):
        return False

    @property
    def image(self):
        return np.full(
            self.image_space.shape,
            min(255, self.step_calls + 10 * self._ammo),
            dtype=np.uint8,
        )

    @property
    def action_mask(self):
        result = np.ones(self.game_action_count, dtype=np.float32)
        if self._ammo == 0:
            result[FIRE] = 0.0
        return result

    def reset(self):
        self._ammo = self.initial_ammo
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0
        self.step_calls = 0

    def grant(self, amount=1):
        amount = min(int(amount), self.capacity - self._ammo)
        self._ammo += amount
        return amount

    def consume(self, amount=1):
        amount = int(amount)
        if amount > self._ammo:
            raise RuntimeError("fake ammo underflow")
        self._ammo -= amount

    def step(self, action):
        self.step_calls += 1
        fired = int(int(np.rint(action)) == FIRE and self._ammo > 0)
        if fired:
            self.consume(1)
        reward = float(fired)
        self.shots_fired += fired
        self.game_reward += reward
        return reward, fired, {"shots_fired_this_step": fired}

    def close(self):
        return


def _e0_config(stage):
    return AtariCurriculumConfig(
        stage=stage,
        seed=7,
        gameplay_horizon=7,
        event_tail_steps=2,
        fixed_event_steps=(0, 1, 2, 3, 4),
    )


def _bilateral_config():
    return BilateralAtariConfig(
        seed=11,
        gameplay_horizon=7,
        event_tail_steps=2,
        fixed_event_steps=(0, 1, 2, 3, 4),
    )


class _ZeroGameController:
    def __call__(self, observation):
        assert observation[ACTOR_STATE].shape == (14,)
        return 0


class _FrozenFollower:
    def __init__(self, role, economic_action=1.0):
        self.policy = SimpleNamespace(
            economic_role=role,
            economic_input_mode="full",
            set_training_mode=lambda mode: None,
            parameters=lambda: (),
        )
        self.economic_action = float(economic_action)
        self.observations = []

    def predict(self, observation, deterministic=True):
        assert deterministic
        self.observations.append({
            key: np.array(value, copy=True)
            for key, value in observation.items()
        })
        return np.array([FIRE, self.economic_action], dtype=np.float32), None


def test_e0a_and_e0b_use_one_interface_and_exact_branch_credit():
    e0a = AtariCurriculumEnv(
        _e0_config("e0a"), side_factory=_FakeSide
    )
    e0b = AtariCurriculumEnv(
        _e0_config("e0b"), side_factory=_FakeSide
    )
    try:
        first_a = e0a.reset()
        first_b = e0b.reset()
        assert e0a.observation_space == e0b.observation_space
        assert e0a.action_space == e0b.action_space
        assert first_a[ACTOR_STATE].shape == first_b[ACTOR_STATE].shape == (14,)
        np.testing.assert_array_equal(first_a[ACTION_CREDIT], [1, 0])
        np.testing.assert_array_equal(first_b[ACTION_CREDIT], [0, 0])

        total_a = 0.0
        done = False
        while not done:
            _, reward, done, info_a = e0a.step([FIRE, 0.5])
            total_a += reward
        assert total_a == 5.0
        assert info_a["shots_fired"] == 5
        assert info_a["outer_transition_count"] == 7

        total_b = 0.0
        done = False
        credit_by_substep = []
        while not done:
            observation, reward, done, info_b = e0b.step([FIRE, 0.5])
            total_b += reward
            credit_by_substep.append(
                (info_b["substep_type"], observation[ACTION_CREDIT].copy())
            )
        assert total_b == 5.0
        assert info_b["free_transfers"] == 5
        assert info_b["shots_fired"] == 5
        assert info_b["outer_transition_count"] == 12
        assert e0b.side.step_calls == 7
    finally:
        e0a.close()
        e0b.close()


def test_e1_exposes_all_gameplay_and_trade_steps_with_common_reward():
    prices = np.full(5, 0.2, dtype=np.float32)
    env = make_atari_meta_response_env(
        controlled_role=BUYER,
        config=_bilateral_config(),
        context_sampler=lambda rng: prices,
        controller_factory=_ZeroGameController,
        side_factory=_FakeSide,
    )
    try:
        assert isinstance(env, AtariFixedCommitmentResponseWrapper)
        assert isinstance(env.env, BilateralAtariRewardEnv)
        assert env.env.leader == BUYER
        assert env.env.followers_list == [SELLER]
        observation = env.reset()
        rewards = []
        substeps = []
        done = False
        while not done:
            if np.array_equal(observation[ACTION_CREDIT], [0, 1]):
                substeps.append(FOLLOWER_TRADE)
            else:
                substeps.append(GAMEPLAY)
            observation, reward, done, info = env.step([FIRE, 0.5])
            rewards.append(reward)
        assert substeps.count(FOLLOWER_TRADE) == 5
        assert substeps.count(GAMEPLAY) == 7
        assert info["purchases"] == 5
        assert info["buyer_shots_fired"] == 5
        assert np.isclose(sum(rewards), 4.0)
        assert info["outer_transition_count"] == 12
        assert info["buyer_payoff_error"] == 0.0
    finally:
        env.close()


def test_bilateral_base_uses_the_common_leader_follower_contract():
    env = BilateralAtariRewardEnv(
        leader_role=SELLER,
        config=_bilateral_config(),
        side_factory=_FakeSide,
    )
    try:
        observations = env.reset(seed=123)
        assert tuple(observations) == (SELLER, BUYER)
        assert env.leader == SELLER
        assert env.followers_list == [BUYER]
        assert env.action_space == env.followers_action_space[BUYER]

        _, reward, done, info = env.step({
            SELLER: np.array([0.0, 0.4], dtype=np.float32),
            BUYER: np.array([0.0, 0.6], dtype=np.float32),
        })
        assert not done
        assert np.isclose(reward, 0.4)
        assert np.isclose(info["surplus"], reward)
        assert np.isclose(info["utilities"][SELLER], 0.4)
        assert np.isclose(info["utilities"][BUYER], -0.4)
        assert info["reward_generated"]
        assert not info["emulator_advanced"]
    finally:
        env.close()


def test_e1_seller_response_preserves_role_reversal_and_reward_owner():
    thresholds = np.full(5, 0.3, dtype=np.float32)
    env = make_atari_meta_response_env(
        controlled_role=SELLER,
        config=_bilateral_config(),
        context_sampler=lambda rng: thresholds,
        controller_factory=_ZeroGameController,
        side_factory=_FakeSide,
    )
    try:
        observation = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            observation, reward, done, info = env.step([FIRE, 0.8])
            total_reward += reward

        assert info["controlled_role"] == SELLER
        assert info["leader_role"] == SELLER
        assert info["follower_role"] == BUYER
        assert info["purchases"] == 0
        assert info["seller_shots_fired"] == 5
        assert np.isclose(total_reward, 0.5)
        assert np.isclose(info["seller_reward"], total_reward)
        assert info["seller_payoff_error"] == 0.0
    finally:
        env.close()


def test_e2_query_and_cached_trade_are_actor_identical_but_credit_differs():
    follower = _FrozenFollower(BUYER, economic_action=1.0)
    env = make_stackpomdp_atari_leader_env(
        leader_role=SELLER,
        response_checkpoint="unused.zip",
        config=_bilateral_config(),
        response_model_factory=lambda path, device: follower,
        side_factory=_FakeSide,
    )
    try:
        assert isinstance(env, StackPOMDPWrapper)
        assert isinstance(env.follower_wrapper, AtariMetaFollowerWrapper)
        assert env.tot_num_response_episodes == 5
        observation = env.reset()
        assert env.current_reward_phase_episodes == 12
        query_observations = []
        query_actions = []
        for event in range(5):
            np.testing.assert_array_equal(observation[ACTION_CREDIT], [0, 1])
            query_observations.append(actor_observation(observation))
            action = np.array([0.0, 0.1 * (event + 1)], dtype=np.float32)
            query_actions.append(action)
            observation, reward, done, info = env.step(action)
            assert reward == 0.0 and not done

        transitions = 5
        done = False
        while not done:
            if np.array_equal(observation[ACTION_CREDIT], [0, 0]):
                event_index = env.follower_wrapper.core.next_event
                current_actor = actor_observation(observation)
                for key in current_actor:
                    np.testing.assert_array_equal(
                        current_actor[key], query_observations[event_index][key]
                    )
                action = query_actions[event_index]
            else:
                np.testing.assert_array_equal(observation[ACTION_CREDIT], [1, 0])
                action = np.array([FIRE, 0.5], dtype=np.float32)
            observation, _, done, info = env.step(action)
            transitions += 1

        assert transitions == 17  # five queries + seven gameplay + five trades
        assert info["cache_hits"] == 5
        assert info["purchases"] == 5
        assert info["outer_transition_count"] == 17
        assert isinstance(
            env.follower_wrapper.leader_query_trace, LeaderQueryTrace
        )
        np.testing.assert_allclose(
            env.follower_wrapper.leader_query_trace.economic_commitment,
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )
        restored = LeaderQueryTrace.from_json_bytes(
            env.follower_wrapper.leader_query_trace.to_json_bytes()
        )
        assert restored.sha256 == env.follower_wrapper.leader_query_trace.sha256
        assert len(follower.observations) == 12
    finally:
        env.close()


def test_e2_buyer_leader_uses_seller_response_and_buyer_reward():
    follower = _FrozenFollower(SELLER, economic_action=0.0)
    env = make_stackpomdp_atari_leader_env(
        leader_role=BUYER,
        response_checkpoint="unused.zip",
        config=_bilateral_config(),
        response_model_factory=lambda path, device: follower,
        side_factory=_FakeSide,
    )
    try:
        observation = env.reset()
        query_action = np.array([0.0, 1.0], dtype=np.float32)
        for _ in range(5):
            observation, reward, done, _ = env.step(query_action)
            assert reward == 0.0 and not done

        total_reward = 0.0
        done = False
        while not done:
            action = (
                query_action
                if np.array_equal(observation[ACTION_CREDIT], [0, 0])
                else np.array([FIRE, 0.5], dtype=np.float32)
            )
            observation, reward, done, info = env.step(action)
            total_reward += reward

        assert info["leader_role"] == BUYER
        assert info["follower_role"] == SELLER
        assert info["purchases"] == 5
        assert info["buyer_shots_fired"] == 5
        assert np.isclose(total_reward, 5.0)
        assert np.isclose(info["leader_reward"], total_reward)
        assert info["cache_hits"] == 5
        assert len(follower.observations) == 12
    finally:
        env.close()


def test_e2_rejects_a_reward_trade_cache_miss():
    follower = _FrozenFollower(BUYER, economic_action=1.0)
    env = make_stackpomdp_atari_leader_env(
        leader_role=SELLER,
        response_checkpoint="unused.zip",
        config=_bilateral_config(),
        response_model_factory=lambda path, device: follower,
        side_factory=_FakeSide,
    )
    try:
        observation = env.reset()
        for _ in range(5):
            observation, _, _, _ = env.step([0.0, 0.25])
        assert np.array_equal(observation[ACTION_CREDIT], [0, 0])
        try:
            env.step([0.0, 0.75])
        except RuntimeError as error:
            assert "cache miss" in str(error)
        else:
            raise AssertionError("reward trade accepted a non-cached action")
    finally:
        env.close()
