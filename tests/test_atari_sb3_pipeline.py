import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from stackelberg_pomdp.atari import make_atari_buyer_env
from stackelberg_pomdp.atari.policy import PriceAwareAtariPolicy
from replication.atari.train_price_aware_atari_sb3 import build_or_load_model


def _space_signature(space):
    return tuple(
        (key, value.shape, value.dtype)
        for key, value in space.spaces.items()
    )


def test_all_curriculum_stages_have_one_observation_and_action_interface():
    environments = [
        make_atari_buyer_env(stage=stage, seed=101, max_steps=2)
        for stage in ("gameplay", "free_trade", "priced", "fixed_price")
    ]
    try:
        reference_observation = _space_signature(environments[0].observation_space)
        reference_action = environments[0].action_space
        for env in environments[1:]:
            assert _space_signature(env.observation_space) == reference_observation
            assert env.action_space.shape == reference_action.shape == (2,)
            np.testing.assert_array_equal(env.action_space.low, reference_action.low)
            np.testing.assert_array_equal(env.action_space.high, reference_action.high)

        observation = environments[0].reset()
        assert "price" not in observation
        assert observation["critic:price"].item() == 0.0
        assert observation["offer_active"].item() == 0.0
        assert observation["ammo_fraction"].item() == 1.0
    finally:
        for env in environments:
            env.close()


def test_free_trade_transfers_five_bullets_without_payments():
    env = make_atari_buyer_env(
        stage="free_trade", seed=102, max_steps=8
    )
    try:
        observation = env.reset()
        assert observation["ammo_fraction"].item() == 0.0
        assert observation["offer_active"].item() == 1.0
        last_info = None
        for _ in range(5):
            observation, reward, done, last_info = env.step(
                np.array([0.0, 0.0], dtype=np.float32)
            )
            assert reward == 0.0
            assert not done
        assert last_info["purchases"] == 5
        assert last_info["episode_payments"] == 0.0
        assert last_info["final_ammo"] == 5
        assert observation["offer_active"].item() == 0.0
        assert observation["opportunities_remaining"].item() == 0.0
    finally:
        env.close()


def test_fixed_price_comparison_and_immediate_payment():
    env = make_atari_buyer_env(
        stage="fixed_price", fixed_price=0.2, seed=103, max_steps=4
    )
    try:
        observation = env.reset()
        assert observation["critic:price"].item() == np.float32(0.2)

        _, rejected_reward, _, rejected_info = env.step(
            np.array([0.0, 0.1], dtype=np.float32)
        )
        assert rejected_info["trade_event"]
        assert not rejected_info["trade_this_step"]
        assert rejected_reward == 0.0
        assert rejected_info["final_ammo"] == 0

        _, accepted_reward, _, accepted_info = env.step(
            np.array([0.0, 0.3], dtype=np.float32)
        )
        assert accepted_info["trade_this_step"]
        assert np.isclose(accepted_reward, -0.2)
        assert np.isclose(accepted_info["episode_payments"], 0.2)
        assert accepted_info["final_ammo"] == 1
    finally:
        env.close()


def test_actor_is_exactly_price_invariant_but_critic_is_not():
    env = make_atari_buyer_env(stage="priced", seed=104, max_steps=4)
    try:
        model = PPO(
            PriceAwareAtariPolicy,
            env,
            policy_kwargs={"stage": "priced"},
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            seed=104,
            device="cpu",
        )
        observation = env.reset()
        low_price = copy.deepcopy(observation)
        high_price = copy.deepcopy(observation)
        low_price["critic:price"] = np.array([0.0], dtype=np.float32)
        high_price["critic:price"] = np.array([1.0], dtype=np.float32)
        low_tensor, _ = model.policy.obs_to_tensor(low_price)
        high_tensor, _ = model.policy.obs_to_tensor(high_price)

        with torch.no_grad():
            low_features = model.policy._features(low_tensor)
            high_features = model.policy._features(high_tensor)
            low_actions = model.policy._distribution(
                low_tensor, low_features
            ).mode()
            high_actions = model.policy._distribution(
                high_tensor, high_features
            ).mode()
            low_value = model.policy._values(low_tensor, low_features)
            high_value = model.policy._values(high_tensor, high_features)

        torch.testing.assert_close(low_features, high_features, rtol=0, atol=0)
        torch.testing.assert_close(low_actions, high_actions, rtol=0, atol=0)
        assert not torch.equal(low_value, high_value)
    finally:
        env.close()


def test_stage_freezing_matches_curriculum():
    env = make_atari_buyer_env(stage="gameplay", seed=105, max_steps=4)
    try:
        model = PPO(
            PriceAwareAtariPolicy,
            env,
            policy_kwargs={"stage": "gameplay"},
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            seed=105,
            device="cpu",
        )
        policy = model.policy
        assert all(parameter.requires_grad for parameter in policy.features_extractor.parameters())
        assert all(parameter.requires_grad for parameter in policy.game_action_net.parameters())
        assert all(not parameter.requires_grad for parameter in policy.threshold_net.parameters())

        policy.set_stage("priced", rebuild_optimizer=True)
        assert all(not parameter.requires_grad for parameter in policy.features_extractor.parameters())
        assert all(not parameter.requires_grad for parameter in policy.game_action_net.parameters())
        assert all(parameter.requires_grad for parameter in policy.threshold_net.parameters())
        assert all(parameter.requires_grad for parameter in policy.value_net.parameters())
    finally:
        env.close()


def test_resume_rebuilds_rollout_buffer_with_undiscounted_e1_parameters():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "source.zip"
        source_env = make_atari_buyer_env(
            stage="gameplay", seed=106, max_steps=4
        )
        try:
            source = PPO(
                PriceAwareAtariPolicy,
                source_env,
                policy_kwargs={"stage": "gameplay"},
                n_steps=4,
                batch_size=4,
                n_epochs=1,
                gamma=0.99,
                gae_lambda=0.95,
                seed=106,
                device="cpu",
            )
            source.save(checkpoint)
        finally:
            source_env.close()

        vec_env = DummyVecEnv([
            lambda: make_atari_buyer_env(
                stage="priced", seed=107, max_steps=4
            )
        ])
        args = SimpleNamespace(
            resume=str(checkpoint),
            device="cpu",
            stage="priced",
            learning_rate=5.0e-5,
            n_steps=8,
            batch_size=8,
            n_epochs=2,
            gamma=1.0,
            gae_lambda=1.0,
            entropy_coeff=0.0,
            clip_range=0.1,
            value_coefficient=0.5,
            max_grad_norm=0.5,
        )
        try:
            resumed = build_or_load_model(args, vec_env)
            assert resumed.gamma == 1.0
            assert resumed.gae_lambda == 1.0
            assert resumed.rollout_buffer.gamma == 1.0
            assert resumed.rollout_buffer.gae_lambda == 1.0
            assert resumed.rollout_buffer.buffer_size == 8
            assert resumed.policy.optimizer.param_groups[0]["lr"] == 5.0e-5
            assert resumed.policy.stage == "priced"
            assert all(
                not parameter.requires_grad
                for parameter in resumed.policy.features_extractor.parameters()
            )
            assert all(
                parameter.requires_grad
                for parameter in resumed.policy.threshold_net.parameters()
            )
        finally:
            vec_env.close()
