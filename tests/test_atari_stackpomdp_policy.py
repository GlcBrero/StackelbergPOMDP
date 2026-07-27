from collections import OrderedDict
import copy

import gym
import numpy as np
import pytest
import torch

from stable_baselines3 import PPO

from stackelberg_pomdp.atari.policy import PriceAwareAtariPolicy
from stackelberg_pomdp.atari.stackpomdp_policy import (
    StackPOMDPAtariEconomicPolicy,
)


GAME_ACTIONS = 3
TRADE_EVENTS = 5


def _legacy_observation_space():
    return gym.spaces.Dict(OrderedDict([
        (
            "image",
            gym.spaces.Box(
                0, 255, shape=(84, 84, 4), dtype=np.uint8
            ),
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
            gym.spaces.Box(
                0.0, 1.0, shape=(GAME_ACTIONS,), dtype=np.float32
            ),
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


def _stackpomdp_observation_space():
    spaces = OrderedDict(_legacy_observation_space().spaces)
    spaces.update([
        (
            "event_active",
            gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        ),
        (
            "event_one_hot",
            gym.spaces.Box(
                0.0, 1.0, shape=(TRADE_EVENTS,), dtype=np.float32
            ),
        ),
        (
            "opponent_context",
            gym.spaces.Box(
                0.0, 1.0, shape=(TRADE_EVENTS,), dtype=np.float32
            ),
        ),
        (
            "critic:state",
            gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
        ),
    ])
    return gym.spaces.Dict(spaces)


def _action_space():
    return gym.spaces.Box(
        low=np.array([0.0, 0.0], dtype=np.float32),
        high=np.array([GAME_ACTIONS - 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )


class _PolicyTestEnv(gym.Env):
    metadata = {}

    def __init__(self, *, stackpomdp):
        super().__init__()
        self.stackpomdp = bool(stackpomdp)
        self.observation_space = (
            _stackpomdp_observation_space()
            if self.stackpomdp
            else _legacy_observation_space()
        )
        self.action_space = _action_space()
        self.steps = 0

    def _observation(self):
        observation = OrderedDict([
            (
                "image",
                np.full((84, 84, 4), self.steps % 256, dtype=np.uint8),
            ),
            ("ammo_fraction", np.array([0.4], dtype=np.float32)),
            ("projectile_active", np.array([0.0], dtype=np.float32)),
            (
                "action_mask",
                np.ones(GAME_ACTIONS, dtype=np.float32),
            ),
            ("offer_active", np.array([0.0], dtype=np.float32)),
            (
                "opportunities_remaining",
                np.array([1.0], dtype=np.float32),
            ),
            ("critic:price", np.array([0.0], dtype=np.float32)),
        ])
        if self.stackpomdp:
            observation.update([
                ("event_active", np.array([1.0], dtype=np.float32)),
                (
                    "event_one_hot",
                    np.array([0, 0, 1, 0, 0], dtype=np.float32),
                ),
                (
                    "opponent_context",
                    np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
                ),
                ("critic:state", np.zeros(7, dtype=np.float32)),
            ])
        return observation

    def reset(self):
        self.steps = 0
        return self._observation()

    def step(self, action):
        assert self.action_space.contains(
            np.asarray(action, dtype=np.float32)
        )
        self.steps += 1
        return self._observation(), 0.0, self.steps >= 8, {}


def _make_source(checkpoint):
    model = PPO(
        PriceAwareAtariPolicy,
        _PolicyTestEnv(stackpomdp=False),
        policy_kwargs={
            "stage": "gameplay",
            "visual_features": 32,
            "ammo_features": 4,
            "market_features": 4,
            "threshold_hidden": 8,
        },
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=71,
        device="cpu",
    )
    model.save(checkpoint)
    return model


def _make_composite():
    return PPO(
        StackPOMDPAtariEconomicPolicy,
        _PolicyTestEnv(stackpomdp=True),
        policy_kwargs={
            "economic_role": "buyer",
            "trade_events": TRADE_EVENTS,
            "visual_features": 32,
            "ammo_features": 4,
            "market_features": 4,
            "economic_hidden": 8,
            "critic_hidden": 16,
        },
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=73,
        device="cpu",
    )


def _matching_tensors(source, composite):
    target_observation = composite.get_env().reset()
    source_observation = {
        key: target_observation[key]
        for key in source.policy.observation_space.spaces
    }
    source_tensor, _ = source.policy.obs_to_tensor(source_observation)
    target_tensor, _ = composite.policy.obs_to_tensor(target_observation)
    return source_tensor, target_tensor


def test_e0_copy_is_exact_frozen_and_standalone_after_reload(tmp_path):
    source_checkpoint = tmp_path / "e0.zip"
    source = _make_source(source_checkpoint)
    composite = _make_composite()

    with pytest.raises(RuntimeError, match="frozen gameplay is uninitialized"):
        composite.predict(composite.get_env().reset(), deterministic=True)

    provenance = composite.policy.load_frozen_gameplay_checkpoint(
        source_checkpoint
    )
    assert provenance["sha256"] == composite.policy.gameplay_fingerprint

    source_tensor, target_tensor = _matching_tensors(source, composite)
    with torch.no_grad():
        source_logits = source.policy._distribution(source_tensor).game.logits
        target_logits = composite.policy._distribution(target_tensor).game.logits
    torch.testing.assert_close(target_logits, source_logits, rtol=0.0, atol=0.0)

    protected_modules = (
        composite.policy.features_extractor,
        composite.policy.game_action_net,
    )
    assert all(
        not parameter.requires_grad
        for module in protected_modules
        for parameter in module.parameters()
    )
    protected_before = {
        name: tensor.detach().clone()
        for name, tensor in composite.policy.state_dict().items()
        if name.startswith((
            "features_extractor.",
            "game_action_net.",
        ))
    }

    _, values, log_prob = composite.policy.forward(target_tensor)
    composite.policy.optimizer.zero_grad()
    (-(log_prob.mean()) + values.square().mean()).backward()
    assert all(
        parameter.grad is None
        for module in protected_modules
        for parameter in module.parameters()
    )
    composite.policy.optimizer.step()
    for name, before in protected_before.items():
        torch.testing.assert_close(
            composite.policy.state_dict()[name], before, rtol=0.0, atol=0.0
        )

    composite_checkpoint = tmp_path / "composite.zip"
    composite.save(composite_checkpoint)
    reloaded = PPO.load(composite_checkpoint, device="cpu")
    assert bool(reloaded.policy.gameplay_ready.item())
    assert reloaded.policy.gameplay_fingerprint == provenance["sha256"]
    reloaded_tensor, _ = reloaded.policy.obs_to_tensor(
        composite.get_env().reset()
    )
    with torch.no_grad():
        reloaded_logits = reloaded.policy._distribution(
            reloaded_tensor
        ).game.logits
    torch.testing.assert_close(
        reloaded_logits, target_logits, rtol=0.0, atol=0.0
    )


def test_only_active_economic_action_contributes_policy_loss(tmp_path):
    source_checkpoint = tmp_path / "e0.zip"
    _make_source(source_checkpoint)
    composite = _make_composite()
    composite.policy.load_frozen_gameplay_checkpoint(source_checkpoint)

    observation = composite.get_env().reset()
    batched = {
        key: np.repeat(value, 2, axis=0)
        for key, value in observation.items()
    }
    batched["event_active"] = np.array([[1.0], [0.0]], dtype=np.float32)
    tensor, _ = composite.policy.obs_to_tensor(batched)
    distribution = composite.policy._distribution(tensor)

    first_actions = torch.tensor(
        [[0.0, 0.2], [0.0, 0.8]], dtype=torch.float32
    )
    different_game_actions = first_actions.clone()
    different_game_actions[:, 0] = torch.tensor([2.0, 1.0])
    first_log_prob = distribution.log_prob(first_actions)
    second_log_prob = distribution.log_prob(different_game_actions)

    torch.testing.assert_close(
        first_log_prob, second_log_prob, rtol=0.0, atol=0.0
    )
    assert first_log_prob[1].item() == 0.0
    assert distribution.entropy()[1].item() == 0.0

    composite.policy.optimizer.zero_grad()
    (-first_log_prob.sum()).backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in composite.policy.economic_head.parameters()
    )
    assert all(
        parameter.grad is None
        for module in (
            composite.policy.features_extractor,
            composite.policy.game_action_net,
        )
        for parameter in module.parameters()
    )


def test_actor_is_phase_blind_and_full_cache_reuses_query_action(tmp_path):
    source_checkpoint = tmp_path / "e0.zip"
    _make_source(source_checkpoint)
    composite = _make_composite()
    policy = composite.policy
    policy.load_frozen_gameplay_checkpoint(source_checkpoint)

    response_observation = composite.get_env().reset()
    reward_observation = copy.deepcopy(response_observation)
    reward_observation["critic:state"][:, 0] = 1.0
    response_tensor, _ = policy.obs_to_tensor(response_observation)
    reward_tensor, _ = policy.obs_to_tensor(reward_observation)

    with torch.no_grad():
        response_distribution = policy._distribution(response_tensor)
        reward_distribution = policy._distribution(reward_tensor)
        response_value = policy._values(response_tensor)
        reward_value = policy._values(reward_tensor)
    torch.testing.assert_close(
        response_distribution.game.logits,
        reward_distribution.game.logits,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        response_distribution.economic.mean,
        reward_distribution.economic.mean,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(response_value, reward_value)

    policy.fix_policy_actions()
    first_action = policy._predict(response_tensor, deterministic=False)
    second_action = policy._predict(reward_tensor, deterministic=False)
    torch.testing.assert_close(
        first_action, second_action, rtol=0.0, atol=0.0
    )
    assert len(policy.obs_action_map) == 1
    cached_action = next(iter(policy.obs_action_map.values()))
    assert tuple(cached_action.shape) == (2,)
    torch.testing.assert_close(
        cached_action, first_action[0].cpu(), rtol=0.0, atol=0.0
    )

    policy.clear_obs_action_map()
    assert policy.obs_action_map == {}
