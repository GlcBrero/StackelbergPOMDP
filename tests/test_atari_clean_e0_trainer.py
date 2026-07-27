from types import SimpleNamespace

import gym
import numpy as np
import pytest
import torch

from stable_baselines3.common.vec_env import DummyVecEnv

from replication.atari import train_atari_curriculum_sb3 as trainer
from replication.atari.sb3_common import ScaledLearningRatePPO
from stackelberg_pomdp.atari.protocol import (
    GAMEPLAY,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


class _StableEnv(gym.Env):
    def __init__(self):
        super().__init__()
        image = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype=np.uint8)
        self.observation_space = observation_space(image, 6)
        self.action_space = action_space(6)

    def _obs(self):
        return observation(
            image=np.zeros((84, 84, 4), dtype=np.uint8),
            state=actor_state(
                ammo_fraction=1.0,
                projectile_active=0.0,
                normalized_time=0.0,
                trade_mode=0.0,
                event_index=None,
                opponent_commitment=np.zeros(5),
            ),
            action_mask=np.ones(6, dtype=np.float32),
            decision_kind=GAMEPLAY,
        )

    def reset(self):
        return self._obs()

    def step(self, action):
        del action
        return self._obs(), 0.0, True, {}


def _args(stage, *, init_checkpoint=None, resume=None):
    return SimpleNamespace(
        stage=stage,
        seed=1,
        init_checkpoint=init_checkpoint,
        resume=resume,
        learning_rate=2.5e-4,
        pretrained_lr_scale=0.1,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        clip_range=0.1,
        entropy_coeff=0.01,
        value_coefficient=0.5,
        max_grad_norm=0.5,
        device="cpu",
    )


def _source_model(vec_env, *, scale=1.0):
    return ScaledLearningRatePPO(
        StackPOMDPAtariPolicy,
        vec_env,
        policy_kwargs={
            "economic_role": "gameplay",
            "economic_input_mode": "full",
            "pretrained_lr_scale": scale,
        },
        learning_rate=2.5e-4,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=1.0,
        seed=9,
        device="cpu",
        verbose=0,
    )


def _module_equal(left, right):
    return all(
        torch.equal(left.state_dict()[key], right.state_dict()[key])
        for key in left.state_dict()
    )


def test_e0_parser_separates_stage_initialization_from_true_resume(tmp_path):
    with pytest.raises(SystemExit):
        trainer.parse_args(["--stage", "e0b", "--no-wandb"])

    e0b = trainer.parse_args([
        "--stage", "e0b",
        "--init-checkpoint", str(tmp_path / "e0a.zip"),
        "--no-wandb",
    ])
    assert e0b.n_steps == 205
    assert e0b.batch_size == 820

    with pytest.raises(SystemExit):
        trainer.parse_args([
            "--stage", "e0b",
            "--init-checkpoint", str(tmp_path / "e0a.zip"),
            "--resume", str(tmp_path / "e0b.zip"),
            "--no-wandb",
        ])


def test_e0b_transfers_actor_but_not_critic_and_uses_scaled_groups(tmp_path):
    vec_env = DummyVecEnv([_StableEnv])
    try:
        source = _source_model(vec_env, scale=1.0)
        with torch.no_grad():
            for parameter in source.policy.value_net.parameters():
                parameter.fill_(0.314159)
        checkpoint = tmp_path / "e0a.zip"
        source.save(checkpoint)

        target = trainer._new_model(
            _args("e0b", init_checkpoint=str(checkpoint)), vec_env
        )
        assert _module_equal(
            source.policy.features_extractor,
            target.policy.features_extractor,
        )
        assert _module_equal(
            source.policy.game_action_net,
            target.policy.game_action_net,
        )
        assert not _module_equal(source.policy.value_net, target.policy.value_net)
        assert [
            group["lr_scale"] for group in target.policy.optimizer.param_groups
        ] == [0.1, 1.0]
    finally:
        vec_env.close()


def test_e0_resume_restores_the_complete_stage_checkpoint(tmp_path):
    vec_env = DummyVecEnv([_StableEnv])
    try:
        source = _source_model(vec_env, scale=1.0)
        with torch.no_grad():
            for parameter in source.policy.value_net.parameters():
                parameter.fill_(0.271828)
            for parameter in source.policy.economic_head.parameters():
                parameter.fill_(-0.161803)
        checkpoint = tmp_path / "e0a_resume.zip"
        source.save(checkpoint)

        restored = trainer._resumed_model(
            _args("e0a", resume=str(checkpoint)), vec_env
        )
        assert _module_equal(source.policy.value_net, restored.policy.value_net)
        assert _module_equal(
            source.policy.economic_head,
            restored.policy.economic_head,
        )
        assert [
            group["lr_scale"]
            for group in restored.policy.optimizer.param_groups
        ] == [1.0, 1.0]
    finally:
        vec_env.close()
