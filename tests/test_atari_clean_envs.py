from types import SimpleNamespace

import gym
import numpy as np

from replication.atari.evaluate_atari_stackpomdp_leader_sb3 import (
    apply_economic_commitment_override,
)
from stackelberg_pomdp.envs.atari.curriculum import (
    AtariCurriculumConfig,
    AtariCurriculumEnv,
)
from stackelberg_pomdp.atari.sampling import (
    ALL_EQUAL_E1_SAMPLER,
    CONTEXT_STRATA,
    SCHEDULE_STRATA,
    TEMPORAL_MIX_E1_SAMPLER,
    UNIFORM_E1_SAMPLER,
)
from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTOR_STATE,
    CACHED_TRADE_REPLAY,
    FOLLOWER_TRADE,
    GAMEPLAY,
    ACTOR_STATE_DIM,
    LEADER_QUERY,
    OPPONENT_COMMITMENT_SLICE,
    actor_observation,
)
from stackelberg_pomdp.atari.query_trace import LeaderQueryTrace
from stackelberg_pomdp.atari.sampling import ExactFiveEventSchedule
from stackelberg_pomdp.wrappers.atari.meta_follower import (
    AtariMetaFollowerWrapper,
    make_stackpomdp_atari_leader_env,
)
from stackelberg_pomdp.envs.atari.bilateral import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
    BilateralAtariRewardEnv,
)
from stackelberg_pomdp.wrappers.atari.fixed_commitment import (
    AtariFixedCommitmentResponseWrapper,
    make_atari_meta_response_env,
)
from stackelberg_pomdp.wrappers.core import StackPOMDPWrapper


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
        self.real_terminal_resets = 0
        self.real_terminal_reset_steps = []
        self.true_game_over_resets = 0
        self.true_game_over_reset_steps = []
        self.time_limit_resets = 0
        self.time_limit_reset_steps = []
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
        self.real_terminal_resets = 0
        self.real_terminal_reset_steps = []
        self.true_game_over_resets = 0
        self.true_game_over_reset_steps = []
        self.time_limit_resets = 0
        self.time_limit_reset_steps = []

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
        return reward, fired, {
            "shots_fired_this_step": fired,
            "emulator_advanced": True,
            "real_terminal_reset": False,
            "real_game_over_reset": False,
            "time_limit_reset": False,
        }

    def close(self):
        return


class _EarlyResetSide(_FakeSide):
    """Fake side that locally resets on its second gameplay decision."""

    def step(self, action):
        reward, fired, info = super().step(action)
        if self.step_calls == 2:
            self.real_terminal_resets += 1
            self.real_terminal_reset_steps.append(2)
            self.true_game_over_resets += 1
            self.true_game_over_reset_steps.append(2)
            info.update({
                "real_terminal_reset": True,
                "real_game_over_reset": True,
            })
        return reward, fired, info


class _SellerEarlyResetSide(_FakeSide):
    """Only the lower-seeded seller locally resets once."""

    def __init__(self, *, seed, **kwargs):
        self._terminate_early = int(seed) < 100_000
        super().__init__(seed=seed, **kwargs)

    def step(self, action):
        reward, fired, info = super().step(action)
        if self._terminate_early and self.step_calls == 2:
            self.real_terminal_resets += 1
            self.real_terminal_reset_steps.append(2)
            self.true_game_over_resets += 1
            self.true_game_over_reset_steps.append(2)
            info.update({
                "real_terminal_reset": True,
                "real_game_over_reset": True,
            })
        return reward, fired, info


def _e0_config(stage):
    return AtariCurriculumConfig(
        stage=stage,
        seed=7,
        gameplay_horizon=7,
        event_tail_steps=0,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )


def _bilateral_config():
    return BilateralAtariConfig(
        seed=11,
        gameplay_horizon=7,
        event_tail_steps=0,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )


class _ZeroGameController:
    def __call__(self, observation):
        assert observation[ACTOR_STATE].shape == (ACTOR_STATE_DIM,)
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


class _ContextFollower(_FrozenFollower):
    """Make the economic response visibly depend on the delivered omega."""

    def predict(self, observation, deterministic=True):
        assert deterministic
        self.observations.append({
            key: np.array(value, copy=True)
            for key, value in observation.items()
        })
        context = np.asarray(
            observation[ACTOR_STATE][OPPONENT_COMMITMENT_SLICE],
            dtype=np.float32,
        )
        return np.array([FIRE, context[0]], dtype=np.float32), None


class _ChooseLastCandidates:
    def choice(self, candidates, *, size, replace):
        assert not replace
        return np.asarray(candidates[-size:], dtype=np.int64)


def test_event_schedule_default_covers_the_complete_gameplay_horizon():
    schedule = ExactFiveEventSchedule(gameplay_horizon=200)
    assert schedule.tail_steps == 0
    assert schedule.event_stop == 200
    assert schedule.sample(_ChooseLastCandidates()) == (195, 196, 197, 198, 199)

    minimal = ExactFiveEventSchedule(gameplay_horizon=5)
    assert minimal.sample(np.random.default_rng(1)) == (0, 1, 2, 3, 4)


def test_event_schedule_fixed_override_is_sorted_distinct_and_can_be_last_step():
    schedule = ExactFiveEventSchedule(
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )
    assert schedule.sample(np.random.default_rng(1)) == (0, 1, 2, 3, 6)

    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        ExactFiveEventSchedule(
            gameplay_horizon=7,
            fixed_event_steps=(0, 1, 1, 3, 6),
        )
    with np.testing.assert_raises_regex(ValueError, "at least five"):
        ExactFiveEventSchedule(gameplay_horizon=4)


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
        assert (
            first_a[ACTOR_STATE].shape
            == first_b[ACTOR_STATE].shape
            == (ACTOR_STATE_DIM,)
        )
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


def test_e1_default_sampler_preserves_legacy_rng_draw_order_exactly():
    config = BilateralAtariConfig(
        seed=29,
        gameplay_horizon=7,
        event_tail_steps=0,
        fixed_event_steps=None,
    )
    legacy_rng = np.random.default_rng(config.seed + 74_711)
    expected = []
    for _ in range(3):
        context = np.asarray(
            legacy_rng.uniform(0.0, 1.0, size=5), dtype=np.float32
        )
        inner_seed = int(legacy_rng.integers(0, 2 ** 31 - 1))
        steps = ExactFiveEventSchedule(
            gameplay_horizon=config.gameplay_horizon,
            tail_steps=config.event_tail_steps,
        ).sample(np.random.default_rng(inner_seed))
        expected.append((context, steps))

    env = make_atari_meta_response_env(
        controlled_role=BUYER,
        config=config,
        controller_factory=_ZeroGameController,
        side_factory=_FakeSide,
    )
    try:
        for expected_context, expected_steps in expected:
            env.reset()
            assert env.e1_sampler_mode == UNIFORM_E1_SAMPLER
            np.testing.assert_array_equal(
                env.opponent_commitment, expected_context
            )
            assert env.core.event_steps == expected_steps
    finally:
        env.close()


def test_e1_temporal_sampler_reports_terminal_strata_and_per_env_counts():
    config = BilateralAtariConfig(
        seed=31,
        gameplay_horizon=200,
        event_tail_steps=0,
        fixed_event_steps=None,
    )
    env = make_atari_meta_response_env(
        controlled_role=BUYER,
        config=config,
        e1_sampler_mode=TEMPORAL_MIX_E1_SAMPLER,
        controller_factory=_ZeroGameController,
        side_factory=_FakeSide,
    )
    try:
        terminal_infos = []
        for episode_number in (1, 2):
            env.reset()
            assert len(env.core.event_steps) == 5
            assert env.core.event_steps[-1] < 200
            done = False
            while not done:
                _, _, done, info = env.step([0.0, 0.5])
            terminal_infos.append(info)
            assert info["e1_sampler_mode"] == TEMPORAL_MIX_E1_SAMPLER
            assert info["e1_sampler_episode_count_per_env"] == episode_number
            assert sum(
                info[f"e1_schedule_stratum_one_hot_{name}"]
                for name in SCHEDULE_STRATA
            ) == 1
            assert sum(
                info[f"e1_context_stratum_one_hot_{name}"]
                for name in CONTEXT_STRATA
            ) == 1
            assert sum(
                info[f"e1_schedule_stratum_per_env_count_{name}"]
                for name in SCHEDULE_STRATA
            ) == episode_number
            assert sum(
                info[f"e1_context_stratum_per_env_count_{name}"]
                for name in CONTEXT_STRATA
            ) == episode_number

        assert terminal_infos[0]["outer_transition_count"] == 205
        assert terminal_infos[1]["outer_transition_count"] == 205
    finally:
        env.close()


def test_e1_all_equal_sampler_reports_exact_205_step_episode_provenance():
    config = BilateralAtariConfig(
        seed=37,
        gameplay_horizon=200,
        event_tail_steps=0,
        fixed_event_steps=None,
    )
    env = make_atari_meta_response_env(
        controlled_role=SELLER,
        config=config,
        e1_sampler_mode=ALL_EQUAL_E1_SAMPLER,
        controller_factory=_ZeroGameController,
        side_factory=_FakeSide,
    )
    try:
        shared_values = []
        for episode_number in (1, 2, 3):
            env.reset()
            commitment = np.array(env.opponent_commitment, copy=True)
            assert np.all(commitment == commitment[0])
            shared_values.append(float(commitment[0]))

            transition_count = 0
            done = False
            while not done:
                _, _, done, info = env.step([0.0, 0.5])
                transition_count += 1

            assert transition_count == 205
            assert info["outer_transition_count"] == 205
            assert info["e1_sampler_mode"] == ALL_EQUAL_E1_SAMPLER
            assert info["e1_sampler_episode_count_per_env"] == episode_number
            assert info["e1_context_stratum"] == "all_equal"
            assert info["e1_context_stratum_one_hot_all_equal"] == 1
            assert info["e1_context_entries_all_equal"] == 1
            assert info["e1_context_shared_value"] == float(commitment[0])
            assert tuple(commitment) == info["opponent_commitment"]
            assert sum(
                info[f"e1_context_stratum_per_env_count_{name}"]
                for name in CONTEXT_STRATA
            ) == episode_number

        assert len(set(shared_values)) == len(shared_values)
    finally:
        env.close()


def test_e1_all_equal_sampler_owns_its_context():
    config = BilateralAtariConfig(
        seed=37,
        gameplay_horizon=200,
        event_tail_steps=0,
    )
    with np.testing.assert_raises_regex(ValueError, "context_sampler"):
        make_atari_meta_response_env(
            controlled_role=SELLER,
            config=config,
            context_sampler=lambda rng: np.zeros(5, dtype=np.float32),
            e1_sampler_mode=ALL_EQUAL_E1_SAMPLER,
            controller_factory=_ZeroGameController,
            side_factory=_FakeSide,
        )


def test_e1_temporal_sampler_owns_both_episode_marginals():
    config = BilateralAtariConfig(
        seed=31,
        gameplay_horizon=200,
        event_tail_steps=0,
    )
    with np.testing.assert_raises_regex(ValueError, "context_sampler"):
        make_atari_meta_response_env(
            controlled_role=BUYER,
            config=config,
            context_sampler=lambda rng: np.zeros(5, dtype=np.float32),
            e1_sampler_mode=TEMPORAL_MIX_E1_SAMPLER,
            controller_factory=_ZeroGameController,
            side_factory=_FakeSide,
        )


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


def test_e2_endpoint_override_replays_exactly_and_changes_full_response_context():
    follower = _ContextFollower(BUYER)
    env = make_stackpomdp_atari_leader_env(
        leader_role=SELLER,
        response_checkpoint="unused.zip",
        config=_bilateral_config(),
        response_model_factory=lambda path, device: follower,
        side_factory=_FakeSide,
    )
    forced = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    raw_query_action = np.asarray([0.0, 0.9], dtype=np.float32)
    effective_queries = []
    try:
        observation = env.reset()
        for event in range(5):
            effective, tag = apply_economic_commitment_override(
                observation, raw_query_action, forced
            )
            assert tag == (LEADER_QUERY, event)
            assert effective[0] == raw_query_action[0]
            effective_queries.append(effective)
            observation, reward, done, _ = env.step(effective)
            assert reward == 0.0 and not done

        replay_count = 0
        done = False
        while not done:
            if np.array_equal(observation[ACTION_CREDIT], [0, 0]):
                effective, tag = apply_economic_commitment_override(
                    observation, raw_query_action, forced
                )
                assert tag == (CACHED_TRADE_REPLAY, replay_count)
                assert effective == effective_queries[replay_count]
                replay_count += 1
            else:
                raw_gameplay = np.asarray([FIRE, 0.9], dtype=np.float32)
                effective, tag = apply_economic_commitment_override(
                    observation, raw_gameplay, forced
                )
                assert tag is None
                np.testing.assert_array_equal(effective, raw_gameplay)
            observation, _, done, info = env.step(effective)

        assert replay_count == 5
        np.testing.assert_allclose(
            env.follower_wrapper.leader_query_trace.economic_commitment,
            forced,
        )
        assert info["cache_hits"] == 5
        # The buyer response reads omega_1=0.1 as its threshold at every
        # trade, so only the first forced price is accepted.  This proves the
        # altered context is not merely recorded: it recomputes behavior.
        assert info["purchases"] == 1
        np.testing.assert_allclose(
            np.asarray(info["follower_actions"], dtype=np.float32)[:, 1],
            forced[0],
        )
        assert follower.observations
        for values in follower.observations:
            np.testing.assert_allclose(
                values[ACTOR_STATE][OPPONENT_COMMITMENT_SLICE], forced
            )
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


def test_e0b_true_game_over_resets_locally_and_horizon_transfers_continue():
    env = AtariCurriculumEnv(
        _e0_config("e0b"), side_factory=_EarlyResetSide
    )
    try:
        env.reset()
        gameplay_steps = 0
        done = False
        while not done:
            _, reward, done, info = env.step([0.0, 0.5])
            assert reward == 0.0
            if info["substep_type"] == GAMEPLAY:
                gameplay_steps += 1
                assert info["emulator_advanced"]

        assert info["gameplay_steps"] == 7
        assert gameplay_steps == 7
        assert info["trade_transitions"] == 5
        assert info["outer_transition_count"] == 12
        assert info["free_transfers"] == 5
        assert info["final_ammo"] == 5
        assert info["bullet_accounting_error"] == 0
        assert info["emulator_step_calls"] == 7
        assert info["true_game_over_resets"] == 1
        assert info["true_game_over_reset_steps"] == (2,)
        assert info["true_game_over_before_fifth_event"]
    finally:
        env.close()


def test_e1_asymmetric_game_over_keeps_other_game_and_all_trades_live():
    env = make_atari_meta_response_env(
        controlled_role=BUYER,
        config=_bilateral_config(),
        context_sampler=lambda rng: np.full(5, 0.2, dtype=np.float32),
        controller_factory=_ZeroGameController,
        side_factory=_SellerEarlyResetSide,
    )
    try:
        observation = env.reset()
        done = False
        while not done:
            observation, _, done, info = env.step([FIRE, 1.0])

        assert info["reward_transition_count"] == 12
        assert info["trade_transitions"] == 5
        assert info["purchases"] == 5
        assert info["buyer_shots_fired"] == 5
        assert info["seller_true_game_over_resets"] == 1
        assert info["buyer_true_game_over_resets"] == 0
        assert info["seller_true_game_over_reset_steps"] == (2,)
        assert info["seller_true_game_over_before_fifth_event"]
        assert info["any_true_game_over_before_fifth_event"]
        assert info["seller_emulator_step_calls"] == 7
        assert info["buyer_emulator_step_calls"] == 7
        assert all(event["accepted"] for event in info["events"])
        assert [
            event["seller_true_game_over_resets"]
            for event in info["events"]
        ] == [
            0,
            0,
            1,
            1,
            1,
        ]
        assert info["seller_bullet_error"] == 0
        assert info["buyer_bullet_error"] == 0
        assert np.isclose(info["seller_payoff_error"], 0.0)
        assert np.isclose(info["buyer_payoff_error"], 0.0)
    finally:
        env.close()


def test_e1_controlled_side_keeps_same_actor_schema_after_local_reset():
    env = make_atari_meta_response_env(
        controlled_role=SELLER,
        config=_bilateral_config(),
        context_sampler=lambda rng: np.ones(5, dtype=np.float32),
        controller_factory=_ZeroGameController,
        side_factory=_SellerEarlyResetSide,
    )
    try:
        observation = env.reset()
        trade_state_shapes = []
        done = False
        while not done:
            if np.array_equal(observation[ACTION_CREDIT], [0, 1]):
                trade_state_shapes.append(observation[ACTOR_STATE].shape)
            observation, _, done, info = env.step([0.0, 0.5])

        assert trade_state_shapes == [(ACTOR_STATE_DIM,)] * 5
        assert info["trade_transitions"] == 5
        assert info["seller_true_game_over_resets"] == 1
        assert info["seller_emulator_step_calls"] == 7
    finally:
        env.close()


def test_e2_local_leader_reset_keeps_exact_query_reward_and_cache_protocol():
    follower = _FrozenFollower(BUYER, economic_action=1.0)
    env = make_stackpomdp_atari_leader_env(
        leader_role=SELLER,
        response_checkpoint="unused.zip",
        config=_bilateral_config(),
        response_model_factory=lambda path, device: follower,
        side_factory=_SellerEarlyResetSide,
    )
    try:
        observation = env.reset()
        query_action = np.array([0.0, 0.2], dtype=np.float32)
        for _ in range(5):
            observation, _, done, _ = env.step(query_action)
            assert not done

        transitions = 5
        done = False
        while not done:
            action = (
                query_action
                if observation[ACTOR_STATE][3] == 1.0
                else np.array([0.0, 0.5], dtype=np.float32)
            )
            observation, _, done, info = env.step(action)
            transitions += 1

        assert transitions == 17
        assert info["outer_transition_count"] == 17
        assert info["query_transitions"] == 5
        assert info["reward_transition_count"] == 12
        assert info["cache_hits"] == 5
        assert info["purchases"] == 5
        assert info["seller_true_game_over_resets"] == 1
        assert info["seller_true_game_over_reset_steps"] == (2,)
        assert info["seller_emulator_step_calls"] == 7
        assert info["seller_bullet_error"] == 0
        assert info["buyer_bullet_error"] == 0
    finally:
        env.close()
