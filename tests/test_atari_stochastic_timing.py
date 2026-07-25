import copy

import gym
import numpy as np
import torch
from torch import nn

from stable_baselines3 import PPO

from replication.atari.train_price_aware_atari_sb3 import (
    copy_pretrained_gameplay,
    evaluation_key,
)
from stackelberg_pomdp.atari.policy import PriceAwareAtariPolicy
from stackelberg_pomdp.atari.wrappers import (
    AmmoLedger,
    AsymmetricBuyerObservationWrapper,
    AtariEpisodeMetricsWrapper,
    BernoulliOfferTimingProcess,
    BulletMarketWrapper,
    FixedPriceProcess,
    ImmediateOfferTimingProcess,
)


class _ImageGameEnv(gym.Env):
    """Small old-Gym-API game stub; no ALE or Space Invaders ROM needed."""

    metadata = {}

    def __init__(self, *, done_after=10_000, game_actions=3):
        super().__init__()
        self.done_after = int(done_after)
        self.action_space = gym.spaces.Discrete(game_actions)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 4),
            dtype=np.uint8,
        )
        self.steps = 0

    def _observation(self):
        return np.full(
            self.observation_space.shape,
            self.steps % 256,
            dtype=np.uint8,
        )

    def reset(self):
        self.steps = 0
        return self._observation()

    def step(self, action):
        assert self.action_space.contains(int(action))
        self.steps += 1
        return (
            self._observation(),
            0.0,
            self.steps >= self.done_after,
            {
                "shot_did_fire": False,
                "shots_fired_this_step": 0,
            },
        )


class _AmmoSensor:
    """Observation-only interface normally supplied by ScarceAmmoWrapper."""

    def __init__(self, game_actions=3):
        self.action_space = gym.spaces.Discrete(game_actions)

    def projectile_active(self):
        return False

    def action_mask(self):
        return np.ones(self.action_space.n, dtype=np.float32)


def _make_market(
        *,
        timing_process,
        episode_horizon=64,
        offer_chances=5,
        max_purchases=5,
        price=0.5,
):
    ledger = AmmoLedger(initial_ammo=0, capacity=5)
    market = BulletMarketWrapper(
        _ImageGameEnv(),
        ledger=ledger,
        price_process=FixedPriceProcess(price),
        timing_process=timing_process,
        trade_enabled=True,
        offer_chances=offer_chances,
        max_purchases=max_purchases,
        price_max=1.0,
        episode_horizon=episode_horizon,
    )
    return market, ledger


def _rejected_offer_trace(seed):
    market, _ = _make_market(
        timing_process=BernoulliOfferTimingProcess(0.25, seed=seed),
    )
    market.reset()
    trace = []
    for _ in range(64):
        _, _, _, info = market.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        if info["trade_event"]:
            trace.append((
                info["offer_step"],
                info["normalized_timestep"],
                info["time_bin"],
            ))
    market.close()
    return trace


def test_bernoulli_offer_timing_is_seeded_and_reproducible():
    first = _rejected_offer_trace(17)
    second = _rejected_offer_trace(17)
    different_seed = _rejected_offer_trace(18)

    assert first == second
    assert first != different_seed
    assert 0 < len(first) <= 5
    assert [event[0] for event in first] == sorted(event[0] for event in first)


def test_rejected_bernoulli_offers_consume_the_five_opportunities():
    market, ledger = _make_market(
        timing_process=BernoulliOfferTimingProcess(1.0, seed=19),
        episode_horizon=8,
    )
    market.reset()
    infos = []
    for _ in range(8):
        _, _, _, info = market.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        infos.append(info)

    offers = [info for info in infos if info["trade_event"]]
    assert [info["offer_step"] for info in offers] == [0, 1, 2, 3, 4]
    assert [
        info["opportunities_remaining_before"] for info in offers
    ] == [5, 4, 3, 2, 1]
    assert [
        info["opportunities_remaining_after"] for info in offers
    ] == [4, 3, 2, 1, 0]
    assert all(not info["trade_this_step"] for info in offers)
    assert all(info["ammo_before_trade"] == 0 for info in offers)
    assert all(info["ammo_after_trade"] == 0 for info in offers)
    assert sum(info["trade_event"] for info in infos) == 5
    assert market.state.opportunities_used == 5
    assert market.state.accepted_trades == 0
    assert market.remaining_opportunities == 0
    assert ledger.value == 0
    market.close()


def _make_context_env(
        *, episode_horizon=6, max_steps=None, actor_economic_context=True
):
    market, ledger = _make_market(
        timing_process=ImmediateOfferTimingProcess(),
        episode_horizon=episode_horizon,
        price=0.5,
    )
    env = AsymmetricBuyerObservationWrapper(
        market,
        ledger=ledger,
        ammo_wrapper=_AmmoSensor(),
        market_wrapper=market,
        actor_economic_context=actor_economic_context,
    )
    if max_steps is not None:
        env = AtariEpisodeMetricsWrapper(
            env,
            ledger=ledger,
            max_steps=max_steps,
        )
    return env


def test_context_observation_tracks_monotonic_normalized_time():
    env = _make_context_env(episode_horizon=4)
    observation = env.reset()
    assert set(("price", "normalized_timestep", "time_remaining")) <= set(
        observation
    )
    assert observation["price"].item() == np.float32(0.5)

    observation_times = [observation["normalized_timestep"].item()]
    observation_remaining = [observation["time_remaining"].item()]
    decision_times = []
    decision_remaining = []
    decision_bins = []
    for _ in range(4):
        observation, _, _, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        observation_times.append(observation["normalized_timestep"].item())
        observation_remaining.append(observation["time_remaining"].item())
        decision_times.append(info["normalized_timestep"])
        decision_remaining.append(info["time_remaining"])
        decision_bins.append(info["time_bin"])

    np.testing.assert_allclose(
        observation_times, [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    np.testing.assert_allclose(
        observation_remaining, [1.0, 0.75, 0.5, 0.25, 0.0]
    )
    np.testing.assert_allclose(decision_times, [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(
        np.asarray(decision_times) + np.asarray(decision_remaining),
        np.ones(4),
    )
    assert decision_bins == ["early", "early", "middle", "late"]
    env.close()


def test_episode_metrics_reveal_a_last_minute_rejection():
    env = _make_context_env(episode_horizon=6, max_steps=6)
    env.reset()
    terminal_info = None
    for step in range(6):
        # Accept the first four offers, then reject the fifth (late) offer.
        threshold = 1.0 if step < 4 else 0.0
        _, _, done, terminal_info = env.step(
            np.array([0.0, threshold], dtype=np.float32)
        )
    assert done
    assert terminal_info is not None
    assert len(terminal_info["trade_events"]) == 5
    assert terminal_info["trade_events"][-1]["time_bin"] == "late"
    assert not terminal_info["trade_events"][-1]["accepted"]

    episode = terminal_info["episode"]
    assert episode["offer_count_early"] == 2
    assert episode["purchase_count_early"] == 2
    assert episode["offer_count_middle"] == 2
    assert episode["purchase_count_middle"] == 2
    assert episode["offer_count_late"] == 1
    assert episode["purchase_count_late"] == 0
    assert episode["last_offer_accepted"] == 0.0
    assert episode["late_rejection_rate"] == 1.0
    env.close()


def test_actor_price_and_time_change_threshold_but_not_game_logits():
    env = _make_context_env(episode_horizon=8)
    try:
        model = PPO(
            PriceAwareAtariPolicy,
            env,
            policy_kwargs={
                "stage": "priced",
                "actor_economic_context": True,
            },
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            seed=23,
            device="cpu",
        )
        policy = model.policy
        feature_dim = policy.features_extractor.features_dim
        linears = [
            module for module in policy.threshold_net
            if isinstance(module, nn.Linear)
        ]
        assert linears[0].in_features == feature_dim + 2

        # Make the fresh threshold head explicitly sensitive to both context
        # coordinates. Its normal initialization deliberately starts at 0.5.
        with torch.no_grad():
            for layer in linears:
                layer.weight.zero_()
                layer.bias.zero_()
            linears[0].weight[0, feature_dim] = 1.0
            linears[0].weight[0, feature_dim + 1] = 2.0
            linears[1].weight[0, 0] = 1.0
            linears[2].weight[0, 0] = 1.0
            linears[2].weight[1, 0] = -1.0

        base = env.reset()
        base["offer_active"] = np.array([1.0], dtype=np.float32)
        base["price"] = np.array([0.2], dtype=np.float32)
        base["critic:price"] = np.array([0.2], dtype=np.float32)
        base["normalized_timestep"] = np.array([0.2], dtype=np.float32)
        base["time_remaining"] = np.array([0.8], dtype=np.float32)
        price_changed = copy.deepcopy(base)
        price_changed["price"] = np.array([0.8], dtype=np.float32)
        price_changed["critic:price"] = np.array([0.8], dtype=np.float32)
        time_changed = copy.deepcopy(base)
        time_changed["normalized_timestep"] = np.array(
            [0.8], dtype=np.float32
        )
        time_changed["time_remaining"] = np.array([0.2], dtype=np.float32)

        outputs = []
        for observation in (base, price_changed, time_changed):
            tensor, _ = policy.obs_to_tensor(observation)
            with torch.no_grad():
                features = policy._features(tensor)
                distribution = policy._distribution(tensor, features)
            outputs.append((
                distribution.game.logits.detach().clone(),
                distribution.threshold.mean.detach().clone(),
            ))

        base_logits, base_threshold = outputs[0]
        for game_logits, _ in outputs[1:]:
            torch.testing.assert_close(
                game_logits, base_logits, rtol=0.0, atol=0.0
            )
        assert not torch.equal(outputs[1][1], base_threshold)
        assert not torch.equal(outputs[2][1], base_threshold)
    finally:
        env.close()


def test_e0_migration_copies_gameplay_exactly_and_leaves_new_head_fresh():
    source_env = _make_context_env(
        episode_horizon=8,
        actor_economic_context=False,
    )
    target_env = _make_context_env(episode_horizon=8)
    try:
        source = PPO(
            PriceAwareAtariPolicy,
            source_env,
            policy_kwargs={
                "stage": "gameplay",
                "actor_economic_context": False,
            },
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            seed=29,
            device="cpu",
        )
        target = PPO(
            PriceAwareAtariPolicy,
            target_env,
            policy_kwargs={
                "stage": "priced",
                "actor_economic_context": True,
            },
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            seed=31,
            device="cpu",
        )
        fresh_threshold = target.policy.threshold_net[0].weight.detach().clone()

        copied = copy_pretrained_gameplay(source.policy, target.policy)

        assert copied
        source_state = source.policy.state_dict()
        target_state = target.policy.state_dict()
        for name in copied:
            torch.testing.assert_close(
                target_state[name], source_state[name], rtol=0.0, atol=0.0
            )
        torch.testing.assert_close(
            target.policy.threshold_net[0].weight,
            fresh_threshold,
            rtol=0.0,
            atol=0.0,
        )

        contextual_observation = target_env.reset()
        legacy_observation = {
            key: contextual_observation[key]
            for key in source_env.observation_space.spaces
        }
        source_tensor, _ = source.policy.obs_to_tensor(legacy_observation)
        target_tensor, _ = target.policy.obs_to_tensor(contextual_observation)
        with torch.no_grad():
            source_distribution = source.policy._distribution(source_tensor)
            target_distribution = target.policy._distribution(target_tensor)
        torch.testing.assert_close(
            target_distribution.game.logits,
            source_distribution.game.logits,
            rtol=0.0,
            atol=0.0,
        )
    finally:
        source_env.close()
        target_env.close()


def test_contextual_selector_does_not_penalize_rational_late_rejection():
    result = {
        "summary": {
            "pass_condition": False,
            "random_mean_net_reward": 0.8,
            "random_positive_net_reward_rate": 0.75,
            "random_fired_fraction_of_purchases": 1.0,
            "zero_price_acceptance_rate": 1.0,
        },
        "fixed_price_table": [{
            "buying_profitable": True,
            "aggregate_acceptance_rate": 0.6,
        }],
    }
    contextual = evaluation_key(
        result, "priced", contextual_timing=True
    )
    legacy = evaluation_key(
        result, "priced", contextual_timing=False
    )

    assert contextual == (0.8, 0.75, 1.0, 1.0)
    assert legacy[0] == 0.0
