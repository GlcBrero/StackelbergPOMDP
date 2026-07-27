from pathlib import Path

import pytest

from stable_baselines3.common.callbacks import CallbackList

from replication.atari import train_atari_stackpomdp_leader_sb3 as trainer
from stackelberg_pomdp.callbacks import FixPolicyActionsCallback


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
    assert args.wandb_project == "StackPOMDP"

    with pytest.raises(SystemExit):
        _args(tmp_path, "--n-steps", "200")


def test_checkpoint_contract_is_same_role_init_and_opposite_response(
        monkeypatch, tmp_path):
    args = _args(tmp_path)

    def metadata(path, *, device, label):
        del path, device
        if label == "frozen E1 response":
            return {
                "path": "buyer.zip",
                "economic_role": "buyer",
                "economic_input_mode": "full",
            }
        return {
            "path": "seller.zip",
            "economic_role": "seller",
            "economic_input_mode": "full",
        }

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

    def metadata(path, *, device, label):
        del path, device
        if label == "frozen E1 response":
            role, mode = "buyer", "full"
        else:
            role, mode = "seller", "event_only"
        return {"path": label, "economic_role": role, "economic_input_mode": mode}

    monkeypatch.setattr(trainer, "checkpoint_policy_metadata", metadata)
    result = trainer.validate_stage_checkpoints(args)

    assert result["leader"]["economic_input_mode"] == "event_only"


def test_new_leader_transfers_actor_only_and_starts_fresh_economic_critic(
        monkeypatch, tmp_path):
    args = _args(tmp_path, "--num-envs", "1")
    captured = {}

    class FakePolicy:
        def load_actor_checkpoint(
                self, checkpoint, *, include_economic, device):
            captured["transfer"] = (checkpoint, include_economic, device)
            return {"critic_transferred": False, "modules": ("actor",)}

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

    monkeypatch.setattr(trainer, "ScaledLearningRatePPO", FakePPO)

    model = trainer._new_model(args, vec_env="vec")

    assert captured["transfer"][1] is False
    assert captured["reset"] == (0.5, 2.0)
    assert captured["kwargs"]["policy_kwargs"]["economic_input_mode"] == "event_only"
    assert captured["kwargs"]["policy_kwargs"]["pretrained_lr_scale"] == 0.1
    assert captured["kwargs"]["gamma"] == 1.0
    assert captured["kwargs"]["gae_lambda"] == 1.0
    assert captured["cache_cleared"] is True


def test_environment_uses_only_frozen_composite_response(monkeypatch, tmp_path):
    args = _args(tmp_path)
    captured = {}

    def fake_env(**kwargs):
        captured.update(kwargs)
        return "leader-env"

    monkeypatch.setattr(trainer, "FullTraceStackPOMDPAtariLeaderEnv", fake_env)
    result = trainer.make_env(args, seed=37)

    assert result == "leader-env"
    assert captured["leader_role"] == "seller"
    assert captured["response_checkpoint"] == args.response_checkpoint
    assert captured["config"].seed == 37
    assert "game_checkpoint" not in captured


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
