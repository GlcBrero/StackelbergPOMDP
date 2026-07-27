from collections import OrderedDict

import gym
import numpy as np

from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
    DualAtariTradeCore,
    ExactFiveEventSchedule,
    MetaEconomicResponseEnv,
    NativeGameplaySide,
)
from stackelberg_pomdp.atari.stackpomdp_leader_env import (
    StackPOMDPAtariLeaderEnv,
)


class _FakeSide:
    """Tiny deterministic stand-in for one frozen Atari controller/game."""

    def __init__(self, *, seed, config, controller, env_factory):
        del seed, config, env_factory
        self.controller = controller
        self.game_action_count = 6
        self.observation_space = gym.spaces.Dict(OrderedDict([
            (
                "image",
                gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8),
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
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0
        self.step_calls = 0

    @property
    def ammo(self):
        return self._ammo

    def reset(self):
        self._ammo = 0
        self.game_reward = 0.0
        self.shots_fired = 0
        self.step_calls = 0

    def grant(self, amount=1):
        if self._ammo + amount > 5:
            return 0
        self._ammo += amount
        return amount

    def consume(self, amount=1):
        if amount > self._ammo:
            raise RuntimeError("fake ammo underflow")
        self._ammo -= amount

    def step(self):
        self.step_calls += 1
        fired = int(self._ammo > 0)
        if fired:
            self.consume(1)
        reward = float(fired)
        self.shots_fired += fired
        self.game_reward += reward
        return reward, fired, {}

    def close(self):
        return


class _ResettingLedger:
    def __init__(self):
        self.value = 0
        self.capacity = 5

    @property
    def fraction(self):
        return self.value / self.capacity

    def grant(self, amount=1):
        granted = min(amount, self.capacity - self.value)
        self.value += granted
        return granted

    def consume(self, amount=1):
        self.value -= amount


class _ResettingAmmoWrapper:
    def __init__(self, ledger):
        self.ledger = ledger

    def projectile_active(self):
        return False

    def action_mask(self):
        mask = np.ones(6, dtype=np.float32)
        if self.ledger.value == 0:
            mask[1:] = 0.0
        return mask


class _LifeEndingEnv:
    def __init__(self):
        self.ammo_ledger = _ResettingLedger()
        self.ammo_wrapper = _ResettingAmmoWrapper(self.ammo_ledger)
        self.observation_space = _FakeSide(
            seed=0,
            config=None,
            controller=lambda observation: 0,
            env_factory=None,
        ).observation_space
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.reset_calls = 0

    def _observation(self):
        return OrderedDict([
            ("image", np.zeros((4, 84, 84), dtype=np.uint8)),
            ("ammo_fraction", np.array([0.0], dtype=np.float32)),
            ("projectile_active", np.array([0.0], dtype=np.float32)),
            ("action_mask", np.ones(6, dtype=np.float32)),
            ("offer_active", np.array([0.0], dtype=np.float32)),
            ("opportunities_remaining", np.array([1.0], dtype=np.float32)),
            ("critic:price", np.array([0.0], dtype=np.float32)),
        ])

    def reset(self):
        self.reset_calls += 1
        self.ammo_ledger.value = 0
        return self._observation()

    def step(self, action):
        del action
        return self._observation(), 0.0, True, {
            "shots_fired_this_step": 0,
        }

    def close(self):
        return


def _config(seed=3):
    return BilateralAtariConfig(
        seed=seed,
        gameplay_horizon=7,
        event_tail_steps=2,
        fixed_event_steps=(0, 1, 2, 3, 4),
    )


def _controller_factory():
    return lambda observation: 0


def _core(config=None):
    return DualAtariTradeCore(
        config or _config(),
        controller_factory=_controller_factory,
        side_factory=_FakeSide,
        env_factory=None,
    )


def _core_factory(config, *, game_checkpoint, controller_factory):
    del game_checkpoint, controller_factory
    return _core(config)


class _FakeResponseModel:
    def __init__(self, role, economic_action):
        self.policy = type("PolicyMetadata", (), {"economic_role": role})()
        self.economic_action = float(economic_action)
        self.observations = []

    def predict(self, observation, deterministic=True):
        assert deterministic
        self.observations.append({
            key: np.array(value, copy=True)
            for key, value in observation.items()
        })
        return np.array([0.0, self.economic_action], dtype=np.float32), None


def test_exact_five_schedule_is_seeded_and_reserves_tail():
    config = BilateralAtariConfig(
        seed=7,
        gameplay_horizon=100,
        event_tail_steps=20,
    )
    sampler = ExactFiveEventSchedule(config)
    first = sampler.sample(np.random.default_rng(11))
    second = sampler.sample(np.random.default_rng(11))
    assert first == second
    assert len(first) == len(set(first)) == 5
    assert tuple(sorted(first)) == first
    assert first[-1] < 80


def test_schedule_rejects_window_with_fewer_than_five_positions():
    try:
        BilateralAtariConfig(
            gameplay_horizon=7,
            event_tail_steps=3,
        ).resolved()
    except ValueError as error:
        assert "at least five" in str(error)
    else:
        raise AssertionError("invalid exact-five event window was accepted")


def test_response_env_construction_seed_reproduces_context_and_schedule():
    config = BilateralAtariConfig(
        seed=19,
        gameplay_horizon=20,
        event_tail_steps=5,
    )
    environments = [
        MetaEconomicResponseEnv(
            controlled_role=BUYER,
            game_checkpoint="unused.zip",
            config=config,
            core_factory=_core_factory,
            controller_factory=_controller_factory,
        )
        for _ in range(2)
    ]
    observations = [environment.reset() for environment in environments]
    assert np.array_equal(
        observations[0]["opponent_context"],
        observations[1]["opponent_context"],
    )
    assert environments[0].core.event_steps == environments[1].core.event_steps


def test_trade_is_paused_and_transfer_accounting_is_atomic():
    core = _core()
    core.reset()
    core.advance_to_event_or_end()
    assert core.at_event
    assert core.seller.ammo == 1
    assert core.seller.step_calls == 0
    assert core.buyer.step_calls == 0

    event = core.trade(price=0.4, threshold=0.4)
    assert event["accepted"]
    assert core.seller.step_calls == 0
    assert core.buyer.step_calls == 0
    assert core.seller.ammo == 0
    assert core.buyer.ammo == 1
    assert core.transfers == 1
    assert core.payments == 0.4


def test_episodic_life_reset_preserves_outer_market_ammo():
    created = []

    def env_factory(config):
        del config
        env = _LifeEndingEnv()
        created.append(env)
        return env

    side = NativeGameplaySide(
        seed=1,
        config=_config(),
        controller=lambda observation: 0,
        env_factory=env_factory,
    )
    side.reset()
    side.grant(2)
    side.step(action=0)
    assert side.life_resets == 1
    assert side.ammo == 2
    assert created[0].reset_calls == 2
    assert np.isclose(side.observation["ammo_fraction"].item(), 0.4)


def test_meta_buyer_gets_full_price_context_and_uses_five_bullets():
    prices = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    env = MetaEconomicResponseEnv(
        controlled_role=BUYER,
        game_checkpoint="unused.zip",
        config=_config(),
        context_sampler=lambda rng: prices,
        core_factory=_core_factory,
        controller_factory=_controller_factory,
    )
    observation = env.reset()
    assert np.array_equal(observation["opponent_context"], prices)
    assert np.array_equal(
        observation["event_one_hot"],
        np.array([1, 0, 0, 0, 0], dtype=np.float32),
    )
    assert observation["event_active"].item() == 1.0
    assert observation["image"].sum() == 0
    assert observation["ammo_fraction"].item() == 0.0

    rewards = []
    done = False
    for event_index in range(5):
        observation, reward, done, info = env.step(
            np.array([5.0, 1.0], dtype=np.float32)
        )
        rewards.append(reward)
        if event_index < 4:
            assert not done
            assert observation["event_one_hot"][event_index + 1] == 1.0
    assert done
    assert info["trade_opportunities"] == 5
    assert info["bullets_arrived"] == 5
    assert info["purchases"] == 5
    assert info["buyer_shots_fired"] == 5
    assert info["buyer_final_ammo"] == 0
    assert info["seller_shots_fired"] == 0
    assert info["seller_bullet_error"] == 0
    assert info["buyer_bullet_error"] == 0
    assert np.isclose(sum(rewards), 5.0 - prices.sum())


def test_meta_seller_sees_arrived_bullet_and_rejected_bullets_are_used():
    thresholds = np.zeros(5, dtype=np.float32)
    env = MetaEconomicResponseEnv(
        controlled_role=SELLER,
        game_checkpoint="unused.zip",
        config=_config(),
        context_sampler=lambda rng: thresholds,
        core_factory=_core_factory,
        controller_factory=_controller_factory,
    )
    observation = env.reset()
    assert np.isclose(observation["ammo_fraction"].item(), 0.2)

    done = False
    rewards = []
    while not done:
        observation, reward, done, info = env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )
        rewards.append(reward)
    assert info["purchases"] == 0
    assert info["seller_shots_fired"] == 5
    assert info["seller_final_ammo"] == 0
    assert info["buyer_shots_fired"] == 0
    assert info["seller_bullet_error"] == 0
    assert info["buyer_bullet_error"] == 0
    assert np.isclose(sum(rewards), 0.5)


def test_seller_leader_has_five_zero_reward_queries_then_cached_reward_game():
    response = _FakeResponseModel(BUYER, economic_action=1.0)
    env = StackPOMDPAtariLeaderEnv(
        leader_role=SELLER,
        response_checkpoint="unused.zip",
        game_checkpoint="unused.zip",
        config=_config(),
        core_factory=_core_factory,
        controller_factory=_controller_factory,
        response_model_factory=lambda path, device: response,
    )
    observation = env.reset()
    actor_keys = [key for key in observation if not key.startswith("critic:")]
    for event_index in range(5):
        canonical = env.canonical_reward_observation(event_index)
        for key in actor_keys:
            assert np.array_equal(observation[key], canonical[key])
        assert env.core.seller.step_calls == 0
        assert env.core.buyer.step_calls == 0
        observation, reward, done, info = env.step(
            np.array([3.0, 0.1 * (event_index + 1)], dtype=np.float32)
        )
        if event_index < 4:
            assert reward == 0.0
            assert not done
            assert not info["reward_game_started"]

    assert done
    assert info["reward_game_started"]
    assert info["trade_opportunities"] == 5
    assert info["purchases"] == 5
    assert info["buyer_shots_fired"] == 5
    assert np.allclose(
        info["leader_context"], [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    assert np.allclose(
        np.asarray(info["leader_full_actions"])[:, 0],
        np.full(5, 3.0),
    )
    assert np.allclose(
        np.asarray(info["leader_full_actions"])[:, 1],
        info["leader_context"],
    )
    assert np.isclose(reward, 1.5)
    assert len(response.observations) == 5
    for event_index, response_observation in enumerate(response.observations):
        assert np.allclose(
            response_observation["opponent_context"],
            info["leader_context"],
        )
        assert response_observation["event_one_hot"][event_index] == 1.0


def test_buyer_leader_uses_frozen_meta_seller_response():
    response = _FakeResponseModel(SELLER, economic_action=0.2)
    env = StackPOMDPAtariLeaderEnv(
        leader_role=BUYER,
        response_checkpoint="unused.zip",
        game_checkpoint="unused.zip",
        config=_config(),
        core_factory=_core_factory,
        controller_factory=_controller_factory,
        response_model_factory=lambda path, device: response,
    )
    env.reset()
    done = False
    for _ in range(5):
        _, reward, done, info = env.step(
            np.array([0.0, 0.3], dtype=np.float32)
        )
    assert done
    assert info["purchases"] == 5
    assert info["buyer_shots_fired"] == 5
    assert np.isclose(info["payments"], 1.0)
    assert np.isclose(reward, 4.0)
