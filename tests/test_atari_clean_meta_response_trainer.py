from types import SimpleNamespace

import gym
import numpy as np
import pytest
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.sb3_common import (
    EpisodeCheckpointCallback,
    ScaledLearningRatePPO,
)
from stackelberg_pomdp.atari.protocol import (
    ACTOR_STATE_DIM,
    GAMEPLAY,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.stackpomdp_env import BUYER, SELLER
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


class _StableProtocolEnv(gym.Env):
    """Small no-ALE environment used only to construct and save clean PPOs."""

    def __init__(self):
        super().__init__()
        image_space = gym.spaces.Box(
            0, 255, shape=(84, 84, 4), dtype=np.uint8
        )
        self.observation_space = observation_space(image_space, 6)
        self.action_space = action_space(6)

    def _observation(self):
        return observation(
            image=np.zeros((84, 84, 4), dtype=np.uint8),
            state=actor_state(
                ammo_fraction=0.0,
                projectile_active=0.0,
                normalized_time=0.0,
                trade_mode=0.0,
                event_index=None,
                opponent_commitment=np.zeros(5, dtype=np.float32),
            ),
            action_mask=np.ones(6, dtype=np.float32),
            decision_kind=GAMEPLAY,
        )

    def reset(self):
        return self._observation()

    def step(self, action):
        del action
        return self._observation(), 0.0, True, {}


def _source_e0b_checkpoint(tmp_path, vec_env):
    model = ScaledLearningRatePPO(
        StackPOMDPAtariPolicy,
        vec_env,
        policy_kwargs={
            "economic_role": "gameplay",
            "economic_input_mode": "full",
            "pretrained_lr_scale": 0.1,
        },
        learning_rate=2.5e-4,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=1.0,
        seed=123,
        device="cpu",
        verbose=0,
    )
    with torch.no_grad():
        for parameter in model.policy.value_net.parameters():
            parameter.fill_(0.314159)
        for parameter in model.policy.economic_head.parameters():
            parameter.fill_(-0.271828)
    checkpoint = tmp_path / "clean_e0b.zip"
    model.save(checkpoint)
    return model, checkpoint


def _trainer_args(*, role, checkpoint):
    return SimpleNamespace(
        role=role,
        seed=7,
        resume=None,
        e0b_checkpoint=str(checkpoint),
        learning_rate=1.0e-4,
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


def _module_equal(first, second):
    first_state = first.state_dict()
    second_state = second.state_dict()
    return all(
        torch.equal(first_state[key], second_state[key])
        for key in first_state
    )


def test_parser_aligns_rollout_with_one_full_h_plus_five_episode(tmp_path):
    args = trainer.parse_args([
        "--role",
        BUYER,
        "--e0b-checkpoint",
        str(tmp_path / "e0b.zip"),
        "--checkpoint",
        str(tmp_path / "e1.zip"),
        "--gameplay-horizon",
        "17",
        "--num-envs",
        "3",
        "--fixed-event-steps",
        "0,2,4,6,8",
        "--no-wandb",
    ])

    assert args.n_steps == 22
    assert args.batch_size == 66
    assert args.fixed_event_steps == (0, 2, 4, 6, 8)
    assert args.event_tail_steps == 0
    assert args.wandb_project == "StackPOMDP"
    assert not args.wandb


def test_parser_rejects_a_rollout_that_ends_mid_episode(tmp_path):
    with pytest.raises(SystemExit):
        trainer.parse_args([
            "--role",
            BUYER,
            "--e0b-checkpoint",
            str(tmp_path / "e0b.zip"),
            "--gameplay-horizon",
            "17",
            "--n-steps",
            "17",
            "--no-wandb",
        ])


def test_actor_source_role_requires_gameplay_e0b():
    base = {
        "source_economic_role": "gameplay",
        "source_economic_input_mode": "full",
    }
    trainer._validate_e0b_source(base)
    with pytest.raises(ValueError, match="E0b checkpoint"):
        trainer._validate_e0b_source(
            {**base, "source_economic_role": SELLER}
        )


@pytest.mark.parametrize(
    ("role", "expected_economic_mean"),
    ((BUYER, 0.95), (SELLER, 0.5)),
)
def test_e1_build_transfers_only_actor_and_uses_role_specific_economic_start(
        tmp_path, role, expected_economic_mean
):
    vec_env = DummyVecEnv([_StableProtocolEnv])
    try:
        source, checkpoint = _source_e0b_checkpoint(tmp_path, vec_env)
        model = trainer.build_model(
            _trainer_args(role=role, checkpoint=checkpoint), vec_env
        )

        assert isinstance(model.policy, StackPOMDPAtariPolicy)
        assert model.policy.economic_role == role
        assert model.policy.economic_input_mode == "full"
        assert model.policy.observation_space["actor_state"].shape == (
            ACTOR_STATE_DIM,
        )
        assert model.gamma == 1.0
        assert model.gae_lambda == 1.0

        # E0b visual/state/game actor modules transfer exactly.
        assert _module_equal(
            source.policy.features_extractor,
            model.policy.features_extractor,
        )
        assert _module_equal(
            source.policy.game_action_net,
            model.policy.game_action_net,
        )

        # The E1 critic is newly initialized, not copied from E0b.
        assert not _module_equal(source.policy.value_net, model.policy.value_net)

        tensor_observation, _ = model.policy.obs_to_tensor(
            _StableProtocolEnv().reset()
        )
        with torch.no_grad():
            economic_mean = float(
                model.policy.get_distribution(tensor_observation)
                .economic_mean.item()
            )
        assert economic_mean == pytest.approx(
            expected_economic_mean, abs=2.0e-5
        )

        groups = model.policy.optimizer.param_groups
        assert [group["lr_scale"] for group in groups] == [0.1, 1.0]
        assert [group["lr"] for group in groups] == pytest.approx(
            [1.0e-5, 1.0e-4]
        )

        # This is a regression test for the two-group optimizer checkpoint bug:
        # an E1 save must reconstruct the same groups under normal SB3 loading.
        saved = tmp_path / f"e1_{role}.zip"
        model.save(saved)
        restored = PPO.load(saved, env=vec_env, device="cpu")
        assert restored.policy.pretrained_lr_scale == pytest.approx(0.1)
        assert [
            group["lr_scale"]
            for group in restored.policy.optimizer.param_groups
        ] == [0.1, 1.0]
    finally:
        vec_env.close()


def test_eval_only_requires_an_e1_actor_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        trainer.parse_args([
            "--role",
            BUYER,
            "--e0b-checkpoint",
            str(tmp_path / "e0b.zip"),
            "--eval-only",
            "--no-wandb",
        ])


def test_e1_resume_restores_economic_head_critic_and_optimizer(tmp_path):
    vec_env = DummyVecEnv([_StableProtocolEnv])
    try:
        _, e0b_checkpoint = _source_e0b_checkpoint(tmp_path, vec_env)
        args = _trainer_args(role=BUYER, checkpoint=e0b_checkpoint)
        source = trainer._new_model(args, vec_env)
        with torch.no_grad():
            for parameter in source.policy.economic_head.parameters():
                parameter.fill_(0.123456)
            for parameter in source.policy.value_net.parameters():
                parameter.fill_(-0.654321)
        checkpoint = tmp_path / "buyer_e1_resume.zip"
        source.save(checkpoint)

        args.resume = str(checkpoint)
        restored = trainer._resumed_model(args, vec_env)
        assert _module_equal(
            source.policy.economic_head,
            restored.policy.economic_head,
        )
        assert _module_equal(source.policy.value_net, restored.policy.value_net)
        assert [
            group["lr_scale"]
            for group in restored.policy.optimizer.param_groups
        ] == [0.1, 1.0]
    finally:
        vec_env.close()


def test_episode_wandb_metrics_follow_the_controlled_seller_role(tmp_path):
    class _Run:
        def __init__(self):
            self.rows = []

        def log(self, payload, step):
            self.rows.append((dict(payload), int(step)))

    run = _Run()
    callback = EpisodeCheckpointCallback(
        checkpoint=tmp_path / "e1.zip",
        checkpoint_every=1_000_000,
        seed=3,
        wandb_run=run,
    )
    callback.model = SimpleNamespace(
        lr_schedule=lambda _: 1.0e-4,
        _current_progress_remaining=0.5,
    )
    callback.num_timesteps = 205
    callback.locals = {
        "dones": np.array([True]),
        "infos": [{
            "episode": {
                "r": 2.2,
                "l": 205,
                "controlled_role": SELLER,
                "seller_game_reward": 1.2,
                "buyer_game_reward": 4.0,
                "seller_shots_fired": 3,
                "buyer_shots_fired": 5,
                "seller_final_ammo": 2,
                "buyer_final_ammo": 0,
                "seller_life_resets": 4,
                "seller_real_terminal_resets": 2,
                "seller_true_game_over_resets": 2,
                "seller_true_game_over_reset_rate": 0.01,
                "seller_time_limit_resets": 0,
                "seller_true_game_over_before_fifth_event": True,
            }
        }],
    }
    callback._init_callback()

    assert callback._on_step()
    payload, step = run.rows[0]
    assert step == 205
    assert payload["train/game_reward"] == pytest.approx(1.2)
    assert payload["train/shots_fired"] == 3
    assert payload["train/final_ammo"] == 2
    assert payload["train/reward_per_bullet"] == pytest.approx(0.4)
    assert payload["train/life_resets"] == 4
    assert payload["train/real_terminal_resets"] == 2
    assert payload["train/true_game_over_resets"] == 2
    assert payload["train/true_game_over_reset_rate"] == pytest.approx(0.01)
    assert payload["train/time_limit_resets"] == 0
    assert payload["train/true_game_over_before_fifth_event"] == 1


def test_episode_wandb_aggregates_simultaneous_vector_completions(tmp_path):
    class _Run:
        def __init__(self):
            self.rows = []

        def log(self, payload, step):
            self.rows.append((dict(payload), int(step)))

    run = _Run()
    callback = EpisodeCheckpointCallback(
        checkpoint=tmp_path / "e0a.zip",
        checkpoint_every=1_000_000,
        seed=3,
        wandb_run=run,
    )
    callback.model = SimpleNamespace(
        lr_schedule=lambda _: 1.0e-4,
        _current_progress_remaining=0.5,
    )
    callback.num_timesteps = 400
    callback.locals = {
        "dones": np.array([True, True]),
        "infos": [
            {"episode": {
                "r": 4.0,
                "l": 200,
                "game_reward": 4.0,
                "shots_fired": 5,
                "final_ammo": 0,
            }},
            {"episode": {
                "r": 2.0,
                "l": 200,
                "game_reward": 2.0,
                "shots_fired": 3,
                "final_ammo": 2,
            }},
        ],
    }
    callback._init_callback()

    assert callback._on_step()
    assert len(run.rows) == 1
    payload, step = run.rows[0]
    assert step == 400
    assert payload["train/vector_episodes"] == 2
    assert payload["train/episode"] == 2
    assert payload["train/game_reward"] == pytest.approx(3.0)
    assert payload["train/shots_fired"] == pytest.approx(4.0)
    assert payload["train/final_ammo"] == pytest.approx(1.0)
