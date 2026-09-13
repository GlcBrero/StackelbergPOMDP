import hashlib
from pathlib import Path

import gym
import numpy as np
import pytest

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from replication.atari import train_atari_stackpomdp_leader_sb3 as trainer
from stackelberg_pomdp.atari.training import (
    PHASE_BALANCED_ACTOR_LOSS_MODE,
    STANDARD_ACTOR_LOSS_MODE,
    ScaledLearningRatePPO,
    model_actor_loss_mode,
    model_economic_initialization,
)
from stackelberg_pomdp.atari.protocol import (
    GAMEPLAY,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.policies.atari import StackPOMDPAtariPolicy
from stackelberg_pomdp.callbacks import FixPolicyActionsCallback


class _StableProtocolEnv(gym.Env):
    """Small no-ALE environment used to exercise the real SB3 save path."""

    def __init__(self):
        super().__init__()
        image = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype=np.uint8)
        self.observation_space = observation_space(image, 6)
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


def _checkpoint_metadata(
        path,
        *,
        role,
        mode,
        digest=None,
        manifest=None,
        gameplay_actor_frozen=False,
):
    policy = {
        "policy_class": (
            "stackelberg_pomdp.policies.atari.composite."
            "StackPOMDPAtariPolicy"
        ),
        "economic_role": role,
        "economic_input_mode": mode,
        "visual_features": 512,
        "state_features": 64,
        "economic_hidden": 64,
        "critic_hidden": 256,
        "pretrained_lr_scale": 0.1,
        "game_action_count": 6,
        "actor_loss_mode": STANDARD_ACTOR_LOSS_MODE,
        "economic_head_initialization": {
            "mean": 0.95 if role == "buyer" and mode == "full" else 0.5,
            "concentration": (
                10.0 if role == "buyer" and mode == "full" else 2.0
            ),
        },
    }
    if mode == "event_only":
        policy["gameplay_actor_frozen"] = bool(gameplay_actor_frozen)
    result = {
        "path": str(path),
        "sha256": digest or ("a" * 64 if role == "buyer" else "b" * 64),
        "economic_role": role,
        "economic_input_mode": mode,
        "policy_metadata": policy,
    }
    if manifest is not None:
        result["e2_provenance_manifest"] = manifest
    return result


def _args(tmp_path, *extra):
    return trainer.parse_args([
        "--leader-role",
        "seller",
        "--response-checkpoint",
        str(tmp_path / "buyer_e1.zip"),
        "--leader-e1-checkpoint",
        str(tmp_path / "seller_e1.zip"),
        "--checkpoint",
        str(tmp_path / "seller_e2.zip"),
        "--no-wandb",
        *extra,
    ])


def test_rollout_is_exactly_queries_gameplay_and_cached_trades(tmp_path):
    args = _args(tmp_path)

    assert args.n_steps == 5 + 200 + 5
    assert args.batch_size == args.n_steps * args.num_envs
    assert args.event_tail_steps == 0
    assert args.actor_loss_mode == STANDARD_ACTOR_LOSS_MODE
    assert args.target_kl is None
    assert args.wandb_project == "StackPOMDP"

    with pytest.raises(SystemExit):
        _args(tmp_path, "--n-steps", "200")


def test_phase_balanced_leader_requires_a_full_rollout_batch(tmp_path):
    args = _args(
        tmp_path,
        "--actor-loss-mode",
        PHASE_BALANCED_ACTOR_LOSS_MODE,
    )
    assert args.actor_loss_mode == PHASE_BALANCED_ACTOR_LOSS_MODE
    assert args.target_kl is None

    with pytest.raises(SystemExit):
        _args(
            tmp_path,
            "--actor-loss-mode",
            PHASE_BALANCED_ACTOR_LOSS_MODE,
            "--batch-size",
            "105",
        )
    with pytest.raises(SystemExit):
        _args(tmp_path, "--target-kl", "nan")


def test_checkpoint_contract_is_same_role_init_and_opposite_response(
        monkeypatch, tmp_path):
    args = _args(tmp_path)

    def metadata(path, *, device, label):
        del device
        if label == "frozen E1 response":
            return _checkpoint_metadata(path, role="buyer", mode="full")
        return _checkpoint_metadata(path, role="seller", mode="full")

    monkeypatch.setattr(trainer, "checkpoint_policy_metadata", metadata)
    result = trainer.validate_stage_checkpoints(args)

    assert result["response"]["economic_role"] == "buyer"
    assert result["leader"]["economic_role"] == "seller"

    def wrong_response(path, *, device, label):
        values = metadata(path, device=device, label=label)
        if label == "frozen E1 response":
            values["economic_role"] = "seller"
        return values

    monkeypatch.setattr(trainer, "checkpoint_policy_metadata", wrong_response)
    with pytest.raises(ValueError, match="requires a buyer E1 response"):
        trainer.validate_stage_checkpoints(args)


def test_resume_contract_requires_event_only_e2_checkpoint(monkeypatch, tmp_path):
    args = _args(
        tmp_path,
        "--resume",
        str(tmp_path / "seller_e2_resume.zip"),
        "--eval-only",
    )

    response = _checkpoint_metadata(
        tmp_path / "copied_buyer_e1.zip", role="buyer", mode="full"
    )
    leader_e1 = _checkpoint_metadata(
        tmp_path / "seller_e1.zip", role="seller", mode="full"
    )
    manifest = trainer.build_e2_provenance_manifest(
        args, response=response, leader_e1=leader_e1
    )

    def metadata(path, *, device, label):
        del device
        if label == "frozen E1 response":
            # Identical bytes at a different path are explicitly compatible.
            return response
        return _checkpoint_metadata(
            path,
            role="seller",
            mode="event_only",
            digest="c" * 64,
            manifest=manifest,
        )

    monkeypatch.setattr(trainer, "checkpoint_policy_metadata", metadata)
    result = trainer.validate_stage_checkpoints(args)

    assert result["leader"]["economic_input_mode"] == "event_only"
    assert result["manifest"]["fingerprint_sha256"] == (
        manifest["fingerprint_sha256"]
    )


def test_checkpoint_metadata_binds_exact_bytes_and_policy(monkeypatch, tmp_path):
    checkpoint = tmp_path / "response.zip"
    checkpoint.write_bytes(b"exact frozen response bytes")

    class FakePolicy:
        economic_role = "buyer"
        economic_input_mode = "full"
        visual_features = 512
        state_features = 64
        economic_hidden = 64
        critic_hidden = 256
        pretrained_lr_scale = 0.1
        game_action_count = 6

    class FakeModel:
        policy = FakePolicy()

    monkeypatch.setattr(trainer, "StackPOMDPAtariPolicy", FakePolicy)
    monkeypatch.setattr(
        trainer.PPO, "load", lambda *args, **kwargs: FakeModel()
    )

    result = trainer.checkpoint_policy_metadata(checkpoint)

    assert result["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert result["policy_metadata"]["economic_role"] == "buyer"
    assert result["policy_metadata"]["game_action_count"] == 6
    assert result["actor_loss_mode"] == STANDARD_ACTOR_LOSS_MODE
    assert result["economic_head_initialization"] == {
        "mean": 0.95,
        "concentration": 10.0,
    }


def test_provenance_accepts_relocated_bytes_and_rejects_any_change(tmp_path):
    args = _args(tmp_path)
    response = _checkpoint_metadata(
        tmp_path / "buyer_e1.zip", role="buyer", mode="full"
    )
    leader_e1 = _checkpoint_metadata(
        tmp_path / "seller_e1.zip", role="seller", mode="full"
    )
    manifest = trainer.build_e2_provenance_manifest(
        args, response=response, leader_e1=leader_e1
    )
    relocated = {
        **response,
        "path": str(tmp_path / "relocated" / "buyer_e1.zip"),
    }

    validated = trainer.require_compatible_e2_provenance(
        manifest, args, response=relocated
    )
    assert validated["fingerprint_sha256"] == manifest["fingerprint_sha256"]

    changed = {**relocated, "sha256": "f" * 64}
    with pytest.raises(ValueError, match="exact frozen response"):
        trainer.require_compatible_e2_provenance(
            manifest, args, response=changed
        )


def test_provenance_separates_scientific_identity_from_run_lineage(tmp_path):
    args = _args(tmp_path)
    response = _checkpoint_metadata(
        tmp_path / "buyer_e1.zip", role="buyer", mode="full"
    )
    leader_e1 = _checkpoint_metadata(
        tmp_path / "seller_e1.zip", role="seller", mode="full"
    )
    first = trainer.build_e2_provenance_manifest(
        args, response=response, leader_e1=leader_e1
    )
    second = trainer.build_e2_provenance_manifest(
        args, response=response, leader_e1=leader_e1
    )

    assert first["scientific_identity_sha256"] == (
        second["scientific_identity_sha256"]
    )
    assert first["run_lineage_id"] != second["run_lineage_id"]
    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]


def test_scientific_config_binds_rom_bytes_not_machine_path(tmp_path):
    first_rom = tmp_path / "machine_a" / "space_invaders.bin"
    second_rom = tmp_path / "machine_b" / "renamed.bin"
    first_rom.parent.mkdir()
    second_rom.parent.mkdir()
    first_rom.write_bytes(b"same rom bytes")
    second_rom.write_bytes(first_rom.read_bytes())

    first = trainer.e2_scientific_config(_args(
        tmp_path, "--rom-path", str(first_rom)
    ))
    second = trainer.e2_scientific_config(_args(
        tmp_path, "--rom-path", str(second_rom)
    ))

    assert first["environment"] == second["environment"]
    assert "rom_path" not in first["environment"]
    second_rom.write_bytes(b"changed rom bytes")
    changed = trainer.e2_scientific_config(_args(
        tmp_path, "--rom-path", str(second_rom)
    ))
    assert changed["environment"]["rom_sha256"] != (
        first["environment"]["rom_sha256"]
    )


def test_scientific_config_distinguishes_actor_loss_mode(tmp_path):
    standard = trainer.e2_scientific_config(_args(tmp_path))
    balanced = trainer.e2_scientific_config(_args(
        tmp_path,
        "--actor-loss-mode",
        PHASE_BALANCED_ACTOR_LOSS_MODE,
    ))

    assert standard["optimization"]["actor_loss_mode"] == (
        STANDARD_ACTOR_LOSS_MODE
    )
    assert balanced["optimization"]["actor_loss_mode"] == (
        PHASE_BALANCED_ACTOR_LOSS_MODE
    )
    assert standard != balanced

    target = trainer.e2_scientific_config(_args(
        tmp_path,
        "--target-kl",
        "0.01",
    ))
    assert target["optimization"]["target_kl"] == pytest.approx(0.01)



def test_provenance_rejects_config_or_manifest_tampering(tmp_path):
    args = _args(tmp_path)
    response = _checkpoint_metadata(
        tmp_path / "buyer_e1.zip", role="buyer", mode="full"
    )
    leader_e1 = _checkpoint_metadata(
        tmp_path / "seller_e1.zip", role="seller", mode="full"
    )
    manifest = trainer.build_e2_provenance_manifest(
        args, response=response, leader_e1=leader_e1
    )
    changed_args = _args(tmp_path, "--event-tail-steps", "1")
    with pytest.raises(ValueError, match="scientific config"):
        trainer.require_compatible_e2_provenance(
            manifest, changed_args, response=response
        )

    tampered = {
        **manifest,
        "scientific_config": {
            **manifest["scientific_config"],
            "leader_role": "buyer",
        },
    }
    with pytest.raises(ValueError, match="fingerprint is invalid"):
        trainer.validate_e2_provenance_manifest(tampered)


def test_attached_provenance_cannot_be_silently_replaced(tmp_path):
    args = _args(tmp_path)
    response = _checkpoint_metadata(
        tmp_path / "buyer_e1.zip", role="buyer", mode="full"
    )
    leader_e1 = _checkpoint_metadata(
        tmp_path / "seller_e1.zip", role="seller", mode="full"
    )
    manifest = trainer.build_e2_provenance_manifest(
        args, response=response, leader_e1=leader_e1
    )

    class Model:
        pass

    model = Model()
    trainer.attach_e2_provenance(model, manifest)
    manifest["schema"] = "mutated caller copy"
    attached = getattr(model, trainer.E2_PROVENANCE_ATTRIBUTE)
    assert attached["schema"] == trainer.E2_PROVENANCE_SCHEMA

    other_response = {**response, "sha256": "e" * 64}
    other = trainer.build_e2_provenance_manifest(
        args, response=other_response, leader_e1=leader_e1
    )
    with pytest.raises(ValueError, match="refusing to replace"):
        trainer.attach_e2_provenance(model, other)


def test_real_sb3_checkpoint_round_trip_retains_manifest(tmp_path):
    args = _args(tmp_path, "--num-envs", "1")
    response = _checkpoint_metadata(
        tmp_path / "buyer_e1.zip", role="buyer", mode="full"
    )
    leader_e1 = _checkpoint_metadata(
        tmp_path / "seller_e1.zip", role="seller", mode="full"
    )
    manifest = trainer.build_e2_provenance_manifest(
        args, response=response, leader_e1=leader_e1
    )
    vec_env = DummyVecEnv([_StableProtocolEnv])
    try:
        model = ScaledLearningRatePPO(
            StackPOMDPAtariPolicy,
            vec_env,
            policy_kwargs={
                "economic_role": "seller",
                "economic_input_mode": "event_only",
                "pretrained_lr_scale": 0.1,
            },
            learning_rate=1.0e-4,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            gamma=1.0,
            gae_lambda=1.0,
            device="cpu",
            verbose=0,
        )
        trainer.attach_e2_provenance(model, manifest)
        trainer.attach_atari_training_contract(
            model,
            actor_loss_mode=STANDARD_ACTOR_LOSS_MODE,
            economic_init_mean=0.5,
            economic_init_concentration=2.0,
        )
        checkpoint = tmp_path / "e2_with_provenance.zip"
        model.save(checkpoint)
        restored = ScaledLearningRatePPO.load(checkpoint, device="cpu")
        assert restored.e2_provenance_manifest == manifest
        assert model_actor_loss_mode(restored) == STANDARD_ACTOR_LOSS_MODE
        assert model_economic_initialization(
            restored, default_mean=0.1, default_concentration=1.0
        ) == {"mean": 0.5, "concentration": 2.0}
    finally:
        vec_env.close()


def test_e2_rejects_frozen_gameplay_actor():
    vec_env = DummyVecEnv([_StableProtocolEnv])
    model = None
    try:
        model = ScaledLearningRatePPO(
            StackPOMDPAtariPolicy,
            vec_env,
            policy_kwargs={
                "economic_role": "buyer",
                "economic_input_mode": "event_only",
                "pretrained_lr_scale": 0.1,
                "gameplay_actor_frozen": True,
            },
            learning_rate=1.0e-4,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            gamma=1.0,
            gae_lambda=1.0,
            device="cpu",
            verbose=0,
        )
        with pytest.raises(ValueError, match="trainable gameplay actor"):
            trainer.validate_e2_gameplay_actor(model)
    finally:
        del model
        vec_env.close()


def test_new_e2_checkpoint_is_provenance_compatible_and_resumable(tmp_path):
    args = _args(
        tmp_path,
        "--num-envs",
        "1",
        "--actor-loss-mode",
        PHASE_BALANCED_ACTOR_LOSS_MODE,
    )
    response = _checkpoint_metadata(
        tmp_path / "buyer_e1.zip", role="buyer", mode="full"
    )
    leader_e1 = _checkpoint_metadata(
        tmp_path / "seller_e1.zip", role="seller", mode="full"
    )
    manifest = trainer.build_e2_provenance_manifest(
        args, response=response, leader_e1=leader_e1
    )
    vec_env = DummyVecEnv([_StableProtocolEnv])
    model = None
    resumed = None
    try:
        algorithm_class = trainer.ppo_class_for_actor_loss_mode(
            args.actor_loss_mode
        )
        model = algorithm_class(
            StackPOMDPAtariPolicy,
            vec_env,
            policy_kwargs={
                "economic_role": args.leader_role,
                "economic_input_mode": "event_only",
                "visual_features": 512,
                "state_features": 64,
                "economic_hidden": 64,
                "critic_hidden": 256,
                "pretrained_lr_scale": args.pretrained_lr_scale,
            },
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=1.0,
            gae_lambda=1.0,
            clip_range=args.clip_range,
            ent_coef=args.entropy_coeff,
            vf_coef=args.value_coefficient,
            max_grad_norm=args.max_grad_norm,
            normalize_advantage=True,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=args.target_kl,
            stats_window_size=100,
            seed=args.seed,
            device=args.device,
            verbose=0,
        )
        model.policy.reset_economic_head(mean=0.5, concentration=2.0)
        trainer.attach_atari_training_contract(
            model,
            actor_loss_mode=PHASE_BALANCED_ACTOR_LOSS_MODE,
            economic_init_mean=0.5,
            economic_init_concentration=2.0,
        )
        trainer.attach_e2_provenance(model, manifest)

        checkpoint = tmp_path / "new_balanced_e2.zip"
        model.save(checkpoint)
        metadata = trainer.checkpoint_policy_metadata(checkpoint)

        assert metadata["policy_metadata"] == (
            manifest["scientific_config"]["leader_policy"]
        )
        trainer.require_compatible_e2_provenance(
            manifest,
            args,
            response=response,
            resumed_leader=metadata,
        )

        args.resume = str(checkpoint)
        resumed = trainer._resumed_model(
            args, vec_env, provenance_manifest=manifest
        )
        assert model_actor_loss_mode(resumed) == (
            PHASE_BALANCED_ACTOR_LOSS_MODE
        )
        assert resumed.e2_provenance_manifest == manifest
    finally:
        del resumed
        del model
        vec_env.close()


def test_new_leader_transfers_actor_only_and_starts_fresh_economic_critic(
        monkeypatch, tmp_path):
    args = _args(tmp_path, "--num-envs", "1")
    manifest = trainer.build_e2_provenance_manifest(
        args,
        response=_checkpoint_metadata(
            tmp_path / "buyer_e1.zip", role="buyer", mode="full"
        ),
        leader_e1=_checkpoint_metadata(
            tmp_path / "seller_e1.zip", role="seller", mode="full"
        ),
    )
    captured = {}

    class FakePolicy:
        def load_actor_checkpoint(
                self, checkpoint, *, include_economic, device):
            captured["transfer"] = (checkpoint, include_economic, device)
            return {
                "checkpoint": str(checkpoint),
                "sha256": "b" * 64,
                "source_economic_role": "seller",
                "source_economic_input_mode": "full",
                "critic_transferred": False,
                "modules": trainer.E2_ACTOR_TRANSFER_MODULES,
            }

        def reset_economic_head(self, *, mean, concentration):
            captured["reset"] = (mean, concentration)

        def clear_obs_action_map(self):
            captured["cache_cleared"] = True

    class FakePPO:
        def __init__(self, policy_class, env, **kwargs):
            captured["policy_class"] = policy_class
            captured["env"] = env
            captured["kwargs"] = kwargs
            self.policy = FakePolicy()

    monkeypatch.setattr(
        trainer,
        "ppo_class_for_actor_loss_mode",
        lambda mode: FakePPO,
    )

    model = trainer._new_model(
        args, vec_env="vec", provenance_manifest=manifest
    )

    assert captured["transfer"][1] is False
    assert captured["reset"] == (0.5, 2.0)
    assert captured["kwargs"]["policy_kwargs"]["economic_input_mode"] == "event_only"
    assert captured["kwargs"]["policy_kwargs"]["pretrained_lr_scale"] == 0.1
    assert captured["kwargs"]["policy_kwargs"]["gameplay_actor_frozen"] is False
    assert captured["kwargs"]["gamma"] == 1.0
    assert captured["kwargs"]["gae_lambda"] == 1.0
    assert captured["cache_cleared"] is True
    assert model_actor_loss_mode(model) == STANDARD_ACTOR_LOSS_MODE
    assert model_economic_initialization(
        model, default_mean=0.1, default_concentration=1.0
    ) == {"mean": 0.5, "concentration": 2.0}
    assert model.e2_provenance_manifest["fingerprint_sha256"] == (
        manifest["fingerprint_sha256"]
    )


def test_eval_only_reports_the_checkpoint_that_was_loaded(tmp_path):
    args = _args(
        tmp_path,
        "--resume", str(tmp_path / "actual_e2.zip"),
        "--eval-only",
    )
    assert trainer.result_checkpoint(args) == (
        tmp_path / "actual_e2.zip"
    ).resolve()

    training = _args(tmp_path)
    assert trainer.result_checkpoint(training) == Path(
        training.checkpoint
    ).resolve()


def test_environment_uses_only_frozen_composite_response(monkeypatch, tmp_path):
    args = _args(tmp_path)
    captured = {}

    def fake_env(**kwargs):
        captured.update(kwargs)
        return "leader-env"

    monkeypatch.setattr(trainer, "make_stackpomdp_atari_leader_env", fake_env)
    result = trainer.make_env(args, seed=37)

    assert result == "leader-env"
    assert captured["leader_role"] == "seller"
    assert captured["response_checkpoint"] == args.response_checkpoint
    assert captured["config"].seed == 37
    assert "game_checkpoint" not in captured


def test_environment_rechecks_frozen_response_bytes_before_loading(
        monkeypatch, tmp_path):
    args = _args(tmp_path)
    response = Path(args.response_checkpoint)
    response.write_bytes(b"validated response")
    args.validated_response_sha256 = trainer._sha256_file(response)
    captured = {}

    def fake_env(**kwargs):
        captured.update(kwargs)
        return "leader-env"

    monkeypatch.setattr(trainer, "make_stackpomdp_atari_leader_env", fake_env)
    loaded = object()
    monkeypatch.setattr(
        trainer.PPO, "load", lambda *args, **kwargs: loaded
    )
    assert trainer.make_env(args, seed=37) == "leader-env"
    factory = captured["response_model_factory"]
    assert factory(response, device="cpu") is loaded

    response.write_bytes(b"mutated response")
    with pytest.raises(RuntimeError, match="changed after provenance"):
        factory(response, device="cpu")


def test_training_always_enables_policy_cache(tmp_path):
    args = _args(tmp_path)
    callbacks = trainer.make_training_callback(args)

    assert isinstance(callbacks, CallbackList)
    assert isinstance(callbacks.callbacks[0], FixPolicyActionsCallback)
    assert isinstance(
        callbacks.callbacks[1], trainer.EpisodeCheckpointCallback
    )


def test_evaluation_is_deterministic_and_cache_enabled(monkeypatch, tmp_path):
    args = _args(tmp_path, "--eval-episodes", "7")
    captured = {}

    def fake_evaluate(model, env_factory, *, episodes, use_action_cache):
        captured.update({
            "model": model,
            "env_factory": env_factory,
            "episodes": episodes,
            "use_action_cache": use_action_cache,
        })
        return {"summary": {"episodes": episodes}, "episode_rows": []}

    monkeypatch.setattr(trainer, "evaluate_model", fake_evaluate)
    result = trainer.evaluate_leader("model", args)

    assert result["summary"]["episodes"] == 7
    assert captured["model"] == "model"
    assert captured["use_action_cache"] is True


def test_clean_checkpoint_default_stays_under_clean_directory(tmp_path):
    args = trainer.parse_args([
        "--leader-role",
        "buyer",
        "--response-checkpoint",
        str(tmp_path / "seller_e1.zip"),
        "--leader-e1-checkpoint",
        str(tmp_path / "buyer_e1.zip"),
        "--no-wandb",
    ])

    checkpoint = Path(args.checkpoint)
    assert checkpoint.parent.name == "clean"
    assert checkpoint.name == "leader_buyer_e2_ppo_seed1.zip"

    balanced = trainer.parse_args([
        "--leader-role",
        "buyer",
        "--response-checkpoint",
        str(tmp_path / "seller_e1.zip"),
        "--leader-e1-checkpoint",
        str(tmp_path / "buyer_e1.zip"),
        "--actor-loss-mode",
        PHASE_BALANCED_ACTOR_LOSS_MODE,
        "--target-kl",
        "0.01",
        "--no-wandb",
    ])
    assert Path(balanced.checkpoint).name == (
        "leader_buyer_e2_ppo_balanced_kl0p01_seed1.zip"
    )
