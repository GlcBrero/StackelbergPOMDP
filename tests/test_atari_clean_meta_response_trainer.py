import json
from types import SimpleNamespace

import gym
import numpy as np
import pytest
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from replication.atari import train_atari_meta_response_sb3 as trainer
from stackelberg_pomdp.atari.training import (
    EpisodeCheckpointCallback,
    PHASE_BALANCED_ACTOR_LOSS_MODE,
    ScaledLearningRatePPO,
    STANDARD_ACTOR_LOSS_MODE,
    _episode_trade_time_metrics,
    model_actor_loss_mode,
    model_economic_initialization,
)
from stackelberg_pomdp.atari.protocol import (
    ACTOR_STATE_DIM,
    GAMEPLAY,
    OPPONENT_COMMITMENT_SLICE,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.envs.atari.bilateral import BUYER, SELLER
from stackelberg_pomdp.policies.atari import StackPOMDPAtariPolicy


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
    assert args.actor_loss_mode == STANDARD_ACTOR_LOSS_MODE
    assert args.buyer_init_mean == pytest.approx(0.95)
    assert args.buyer_init_concentration == pytest.approx(10.0)
    assert args.wandb_project == "StackPOMDP"
    assert not args.wandb
    assert args.e1_sampler_mode == trainer.UNIFORM_E1_SAMPLER


def test_release_cli_rejects_superseded_seller_recovery_variants(tmp_path):
    common = [
        "--role", SELLER,
        "--e0b-checkpoint", str(tmp_path / "e0b.zip"),
        "--no-wandb",
    ]
    with pytest.raises(SystemExit):
        trainer.parse_args(common + ["--economic-threshold-residual"])
    with pytest.raises(SystemExit):
        trainer.parse_args(common + [
            "--economic-architecture", "seller_two_branch_beta_v4",
        ])


def test_temporal_sampler_parser_is_explicit_and_horizon_bound(tmp_path):
    common = [
        "--role",
        BUYER,
        "--e0b-checkpoint",
        str(tmp_path / "e0b.zip"),
        "--e1-sampler-mode",
        trainer.TEMPORAL_MIX_E1_SAMPLER,
        "--no-wandb",
    ]
    args = trainer.parse_args(common)
    assert args.e1_sampler_mode == trainer.TEMPORAL_MIX_E1_SAMPLER
    assert args.gameplay_horizon == 200
    assert args.event_tail_steps == 0
    assert args.fixed_event_steps is None
    assert args.n_steps == 205
    assert "_temporal_mix_v1_seed1.zip" in args.checkpoint

    for incompatible in (
            ["--gameplay-horizon", "199"],
            ["--event-tail-steps", "1"],
            ["--fixed-event-steps", "0,1,2,3,4"],
            ["--eval-only", "--resume", str(tmp_path / "e1.zip")],
    ):
        with pytest.raises(SystemExit):
            trainer.parse_args([*common, *incompatible])


def test_all_equal_sampler_parser_slug_and_eval_contract(tmp_path):
    common = [
        "--role",
        SELLER,
        "--e0b-checkpoint",
        str(tmp_path / "e0b.zip"),
        "--e1-sampler-mode",
        trainer.ALL_EQUAL_E1_SAMPLER,
        "--gameplay-horizon",
        "17",
        "--fixed-event-steps",
        "0,2,4,6,8",
        "--no-wandb",
    ]
    args = trainer.parse_args(common)

    assert args.e1_sampler_mode == trainer.ALL_EQUAL_E1_SAMPLER
    assert args.gameplay_horizon == 17
    assert args.fixed_event_steps == (0, 2, 4, 6, 8)
    assert args.n_steps == 22
    assert "_all_equal_v1_seed1.zip" in args.checkpoint

    with pytest.raises(SystemExit):
        trainer.parse_args([
            *common,
            "--eval-only",
            "--resume",
            str(tmp_path / "e1.zip"),
        ])


def test_canonical_evaluation_args_never_reuse_temporal_training_sampler():
    marker = object()
    source = SimpleNamespace(
        e1_sampler_mode=trainer.TEMPORAL_MIX_E1_SAMPLER,
        marker=marker,
    )
    result = trainer.canonical_evaluation_args(source)
    assert result is not source
    assert result.e1_sampler_mode == trainer.UNIFORM_E1_SAMPLER
    assert result.marker is marker
    assert source.e1_sampler_mode == trainer.TEMPORAL_MIX_E1_SAMPLER


def test_canonical_evaluation_args_never_reuse_all_equal_training_sampler():
    source = SimpleNamespace(
        e1_sampler_mode=trainer.ALL_EQUAL_E1_SAMPLER,
        marker=object(),
    )
    result = trainer.canonical_evaluation_args(source)

    assert result is not source
    assert result.e1_sampler_mode == trainer.UNIFORM_E1_SAMPLER
    assert source.e1_sampler_mode == trainer.ALL_EQUAL_E1_SAMPLER


def test_phase_balanced_parser_requires_a_full_rollout_batch(tmp_path):
    common = [
        "--role",
        BUYER,
        "--e0b-checkpoint",
        str(tmp_path / "e0b.zip"),
        "--gameplay-horizon",
        "17",
        "--num-envs",
        "1",
        "--n-steps",
        "22",
        "--actor-loss-mode",
        PHASE_BALANCED_ACTOR_LOSS_MODE,
        "--no-wandb",
    ]
    args = trainer.parse_args([*common, "--batch-size", "22"])
    assert args.actor_loss_mode == PHASE_BALANCED_ACTOR_LOSS_MODE
    assert "_balanced_seed1.zip" in args.checkpoint

    with pytest.raises(SystemExit):
        trainer.parse_args([*common, "--batch-size", "11"])
    with pytest.raises(SystemExit):
        trainer.parse_args([*common, "--batch-size", "22", "--target-kl", "0"])
    with pytest.raises(SystemExit):
        trainer.parse_args([
            *common,
            "--batch-size",
            "22",
            "--target-kl",
            "nan",
        ])
    with pytest.raises(SystemExit):
        trainer.parse_args([
            *common,
            "--batch-size",
            "22",
            "--buyer-init-concentration",
            "nan",
        ])

    custom_init = trainer.parse_args([
        "--role",
        BUYER,
        "--e0b-checkpoint",
        str(tmp_path / "e0b.zip"),
        "--buyer-init-mean",
        "0.9",
        "--buyer-init-concentration",
        "8",
        "--no-wandb",
    ])
    assert "_initm0p9c8_seed1.zip" in custom_init.checkpoint


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
        "source_pretrained_lr_scale": 0.1,
    }
    trainer._validate_e0b_source(base)
    with pytest.raises(ValueError, match="E0b checkpoint"):
        trainer._validate_e0b_source(
            {**base, "source_economic_role": SELLER}
        )
    with pytest.raises(ValueError, match="pretrained_lr_scale=0.1"):
        trainer._validate_e0b_source(
            {**base, "source_pretrained_lr_scale": 1.0}
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
        assert model.atari_e1_source_provenance["sha256"] == (
            trainer._checkpoint_sha256(checkpoint)
        )
        assert model.policy.observation_space["actor_state"].shape == (
            ACTOR_STATE_DIM,
        )
        assert model.gamma == 1.0
        assert model.gae_lambda == 1.0
        assert model_actor_loss_mode(model) == STANDARD_ACTOR_LOSS_MODE
        expected_concentration = 10.0 if role == BUYER else 2.0
        assert model_economic_initialization(
            model,
            default_mean=0.0,
            default_concentration=1.0,
        ) == {
            "mean": expected_economic_mean,
            "concentration": expected_concentration,
        }

        # E0b visual/game modules transfer exactly.  The state encoder differs
        # only in the five previously unseen opponent-commitment columns,
        # which start at zero so random E1 context cannot arbitrarily perturb
        # the certified E0b gameplay policy.
        assert _module_equal(
            source.policy.features_extractor.visual,
            model.policy.features_extractor.visual,
        )
        source_state = source.policy.features_extractor.state_encoder[0]
        e1_state = model.policy.features_extractor.state_encoder[0]
        assert torch.equal(source_state.bias, e1_state.bias)
        assert torch.equal(
            source_state.weight[:, :OPPONENT_COMMITMENT_SLICE.start],
            e1_state.weight[:, :OPPONENT_COMMITMENT_SLICE.start],
        )
        assert torch.count_nonzero(
            e1_state.weight[:, OPPONENT_COMMITMENT_SLICE]
        ) == 0
        assert e1_state.weight.requires_grad
        assert model.atari_e1_source_provenance[
            "zero_initialized_actor_state_indices"
        ] == [9, 10, 11, 12, 13]
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
        assert restored.atari_e1_source_provenance == (
            model.atari_e1_source_provenance
        )
        assert restored.policy.pretrained_lr_scale == pytest.approx(0.1)
        assert model_actor_loss_mode(restored) == STANDARD_ACTOR_LOSS_MODE
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


def test_fixed_context_evaluation_is_paired_and_retains_event_rows(monkeypatch):
    seeds_by_value = {}
    sampler_modes = []

    def fake_make_env(args, *, seed, context_sampler=None):
        sampler_modes.append(args.e1_sampler_mode)
        if context_sampler is None:
            value = -1.0
        else:
            context = context_sampler(np.random.default_rng(0))
            value = float(context[0])
            seeds_by_value.setdefault(value, []).append(int(seed))
        return SimpleNamespace(value=value, seed=int(seed))

    def fake_evaluate_model(model, env_factory, *, episodes):
        del model
        rows = []
        for episode in range(episodes):
            env = env_factory(episode)
            rows.append({
                "evaluation_return": env.value,
                "evaluation_steps": 205,
                "events": [{
                    "event_index": event_index,
                    "game_step": 10 * (event_index + 1),
                    "price": env.value,
                    "threshold": 0.5,
                    "accepted": float(env.value <= 0.5),
                } for event_index in range(5)],
            })
        return {
            "summary": {"episodes": episodes},
            "episode_rows": rows,
        }

    monkeypatch.setattr(trainer, "make_env", fake_make_env)
    monkeypatch.setattr(trainer, "evaluate_model", fake_evaluate_model)
    args = SimpleNamespace(
        seed=17,
        eval_episodes=2,
        fixed_eval_episodes=3,
        fixed_eval_values=(0.0, 0.5, 1.0),
        gameplay_horizon=200,
        event_tail_steps=0,
        e1_sampler_mode=trainer.TEMPORAL_MIX_E1_SAMPLER,
    )

    evaluation = trainer.evaluate_response(object(), args)

    expected_seeds = [400_017, 400_018, 400_019]
    assert seeds_by_value == {
        0.0: expected_seeds,
        0.5: expected_seeds,
        1.0: expected_seeds,
    }
    assert len(evaluation["fixed_contexts"]) == 3
    assert len(evaluation["fixed_context_evaluations"]) == 3
    for fixed in evaluation["fixed_context_evaluations"]:
        assert len(fixed["episode_rows"]) == 3
        assert all(len(row["events"]) == 5 for row in fixed["episode_rows"])
        assert fixed["summary"]["trade_events"] == 15
    assert sampler_modes
    assert set(sampler_modes) == {trainer.UNIFORM_E1_SAMPLER}


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
        source.num_timesteps = 205
        # Model an E1 checkpoint produced before sampler provenance existed.
        # It is conservatively treated as canonical uniform training.
        del source.atari_e1_sampler_provenance
        del source.atari_e1_sampler_history
        del source.atari_e1_resume_source_provenance
        checkpoint = tmp_path / "buyer_e1_resume.zip"
        source.save(checkpoint)

        args.resume = str(checkpoint)
        args.e1_sampler_mode = trainer.TEMPORAL_MIX_E1_SAMPLER
        args.gameplay_horizon = 200
        args.event_tail_steps = 0
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
        history = restored.atari_e1_sampler_history
        assert len(history) == 2
        assert history[0]["sampler"]["mode"] == trainer.UNIFORM_E1_SAMPLER
        assert history[0]["inferred_for_legacy_checkpoint"]
        assert history[1]["sampler"]["mode"] == (
            trainer.TEMPORAL_MIX_E1_SAMPLER
        )
        assert history[1]["start_total_timesteps"] == 205
        assert len(history[1]["resume_sources"]) == 1
        resume_record = history[1]["resume_sources"][0]
        assert resume_record["path"] == str(checkpoint.resolve())
        assert resume_record["sha256"] == trainer._checkpoint_sha256(checkpoint)
        assert resume_record["resume_total_timesteps"] == 205

        # An interrupted temporal run resumes under the same sampler stage;
        # it records a new byte-bound parent without pretending that the
        # scientific sampling regime changed.
        interrupted = tmp_path / "buyer_e1_temporal_interrupted.zip"
        restored.save(interrupted)
        args.resume = str(interrupted)
        resumed_again = trainer._resumed_model(args, vec_env)
        repeated_history = resumed_again.atari_e1_sampler_history
        assert len(repeated_history) == 2
        assert len(repeated_history[1]["resume_sources"]) == 2
        assert repeated_history[1]["resume_sources"][-1]["path"] == (
            str(interrupted.resolve())
        )
        assert repeated_history[1]["resume_sources"][-1]["sha256"] == (
            trainer._checkpoint_sha256(interrupted)
        )

        # Eval-only loads use canonical uniform environments, but that must
        # not rewrite a temporal checkpoint's training history as though a
        # new uniform training stage occurred.
        args.eval_only = True
        args.e1_sampler_mode = trainer.UNIFORM_E1_SAMPLER
        evaluated = trainer._resumed_model(args, vec_env)
        assert len(evaluated.atari_e1_sampler_history) == 2
        assert evaluated.atari_e1_sampler_provenance["mode"] == (
            trainer.TEMPORAL_MIX_E1_SAMPLER
        )
        assert evaluated.atari_e1_sampler_history[-1]["sampler"]["mode"] == (
            trainer.TEMPORAL_MIX_E1_SAMPLER
        )
        args.eval_only = False
        args.e1_sampler_mode = trainer.TEMPORAL_MIX_E1_SAMPLER

        args.resume = str(checkpoint)

        args.actor_loss_mode = PHASE_BALANCED_ACTOR_LOSS_MODE
        with pytest.raises(ValueError, match="must match the saved E1"):
            trainer._resumed_model(args, vec_env)
        args.actor_loss_mode = STANDARD_ACTOR_LOSS_MODE

        args.buyer_init_mean = 0.9
        with pytest.raises(ValueError, match="initialization arguments"):
            trainer._resumed_model(args, vec_env)
        args.buyer_init_mean = 0.95

        args.target_kl = 0.01
        with pytest.raises(ValueError, match="must match the saved E1"):
            trainer._resumed_model(args, vec_env)
        args.target_kl = None

        equivalent_source = tmp_path / "same_e0b_bytes.zip"
        equivalent_source.write_bytes(e0b_checkpoint.read_bytes())
        args.e0b_checkpoint = str(equivalent_source)
        trainer._resumed_model(args, vec_env)

        different_source = tmp_path / "different_e0b_bytes.zip"
        different_source.write_bytes(b"different checkpoint bytes")
        args.e0b_checkpoint = str(different_source)
        with pytest.raises(ValueError, match="differs from the source bound"):
            trainer._resumed_model(args, vec_env)
    finally:
        vec_env.close()


def test_e1_all_equal_to_uniform_resume_preserves_optimizer_and_clock(
        tmp_path,
):
    vec_env = DummyVecEnv([_StableProtocolEnv])
    try:
        _, e0b_checkpoint = _source_e0b_checkpoint(tmp_path, vec_env)
        args = _trainer_args(role=SELLER, checkpoint=e0b_checkpoint)
        args.e1_sampler_mode = trainer.ALL_EQUAL_E1_SAMPLER
        args.gameplay_horizon = 200
        args.event_tail_steps = 0
        args.eval_only = False
        source = trainer._new_model(args, vec_env)

        source.policy.optimizer.zero_grad()
        loss = sum(
            parameter.square().sum()
            for parameter in source.policy.parameters()
            if parameter.requires_grad
        )
        loss.backward()
        source.policy.optimizer.step()
        source.num_timesteps = 400_160

        source_parameters = dict(source.policy.named_parameters())
        source_optimizer = {
            name: {
                key: (
                    value.detach().clone()
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in source.policy.optimizer.state[
                    parameter
                ].items()
            }
            for name, parameter in source_parameters.items()
        }
        checkpoint = tmp_path / "seller_e1_all_equal_stage.zip"
        source.save(checkpoint)

        args.resume = str(checkpoint)
        args.e1_sampler_mode = trainer.UNIFORM_E1_SAMPLER
        restored = trainer._resumed_model(args, vec_env)

        assert restored.num_timesteps == 400_160
        assert restored.atari_e1_sampler_provenance["mode"] == (
            trainer.UNIFORM_E1_SAMPLER
        )
        history = restored.atari_e1_sampler_history
        assert [stage["sampler"]["mode"] for stage in history] == [
            trainer.ALL_EQUAL_E1_SAMPLER,
            trainer.UNIFORM_E1_SAMPLER,
        ]
        assert history[0]["start_total_timesteps"] == 0
        assert history[1]["start_total_timesteps"] == 400_160
        assert history[1]["resume_sources"][0]["path"] == str(
            checkpoint.resolve()
        )

        restored_parameters = dict(restored.policy.named_parameters())
        assert restored_parameters.keys() == source_parameters.keys()
        for name, source_parameter in source_parameters.items():
            restored_parameter = restored_parameters[name]
            assert torch.equal(source_parameter, restored_parameter)
            restored_state = restored.policy.optimizer.state[
                restored_parameter
            ]
            assert restored_state.keys() == source_optimizer[name].keys()
            for key, source_value in source_optimizer[name].items():
                restored_value = restored_state[key]
                if torch.is_tensor(source_value):
                    assert torch.equal(source_value, restored_value)
                else:
                    assert source_value == restored_value
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
                "gameplay_transitions": 200,
                "events": [
                    {
                        "event_index": event_index,
                        "game_step": game_step,
                        "price": price,
                        "threshold": 0.5,
                        "accepted": accepted,
                    }
                    for event_index, (game_step, price, accepted) in enumerate((
                        (0, 0.1, 1),
                        (66, 0.9, 0),
                        (67, 0.2, 1),
                        (133, 0.8, 0),
                        (134, 0.7, 0),
                    ))
                ],
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
    assert payload["train/early_trade_events"] == 2
    assert payload["train/middle_trade_events"] == 2
    assert payload["train/late_trade_events"] == 1
    assert payload["train/early_acceptance_rate"] == pytest.approx(0.5)
    assert payload["train/middle_acceptance_rate"] == pytest.approx(0.5)
    assert payload["train/late_acceptance_rate"] == pytest.approx(0.0)
    assert payload["train/early_mean_price"] == pytest.approx(0.5)
    assert payload["train/middle_mean_price"] == pytest.approx(0.5)
    assert payload["train/late_mean_price"] == pytest.approx(0.7)
    local = json.loads(callback.training_log.read_text().splitlines()[-1])
    for key in (
            "train/early_trade_events",
            "train/middle_trade_events",
            "train/late_trade_events",
            "train/early_acceptance_rate",
            "train/middle_acceptance_rate",
            "train/late_acceptance_rate",
    ):
        assert local[key] == payload[key]


def test_trade_time_metrics_omit_undefined_empty_bin_means():
    metrics = _episode_trade_time_metrics({
        "gameplay_transitions": 200,
        "events": [
            {
                "game_step": step,
                "price": 0.2,
                "threshold": 0.4,
                "accepted": True,
            }
            for step in range(5)
        ],
    })

    assert metrics["early_trade_events"] == 5
    assert metrics["middle_trade_events"] == 0
    assert metrics["late_trade_events"] == 0
    assert "middle_acceptance_rate" not in metrics
    assert "middle_mean_price" not in metrics
    assert "middle_mean_threshold" not in metrics
    assert "late_acceptance_rate" not in metrics


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
                "e1_sampler_mode": trainer.TEMPORAL_MIX_E1_SAMPLER,
                "e1_schedule_stratum": "unconditional",
                "e1_context_stratum": "uniform",
                "e1_schedule_stratum_one_hot_unconditional": 1,
                "e1_schedule_stratum_one_hot_late_fifth": 0,
                "e1_context_stratum_one_hot_uniform": 1,
                "e1_context_stratum_one_hot_low_prefix": 0,
                "e1_schedule_stratum_per_env_count_unconditional": 3,
            }},
            {"episode": {
                "r": 2.0,
                "l": 200,
                "game_reward": 2.0,
                "shots_fired": 3,
                "final_ammo": 2,
                "e1_sampler_mode": trainer.TEMPORAL_MIX_E1_SAMPLER,
                "e1_schedule_stratum": "late_fifth",
                "e1_context_stratum": "low_prefix",
                "e1_schedule_stratum_one_hot_unconditional": 0,
                "e1_schedule_stratum_one_hot_late_fifth": 1,
                "e1_context_stratum_one_hot_uniform": 0,
                "e1_context_stratum_one_hot_low_prefix": 1,
                "e1_schedule_stratum_per_env_count_unconditional": 1,
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
    assert payload[
        "train/e1_schedule_stratum_one_hot_unconditional"
    ] == pytest.approx(0.5)
    assert payload[
        "train/e1_schedule_stratum_vector_count_unconditional"
    ] == pytest.approx(1.0)
    assert payload[
        "train/e1_context_stratum_one_hot_low_prefix"
    ] == pytest.approx(0.5)
    assert payload[
        "train/e1_context_stratum_vector_count_low_prefix"
    ] == pytest.approx(1.0)
    assert payload[
        "train/e1_schedule_stratum_per_env_count_unconditional"
    ] == pytest.approx(2.0)
    local_rows = [
        json.loads(line)
        for line in callback.training_log.read_text().splitlines()
    ]
    assert [
        row["train/e1_schedule_stratum"] for row in local_rows
    ] == ["unconditional", "late_fifth"]
