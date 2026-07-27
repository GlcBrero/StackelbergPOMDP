from pathlib import Path

import pytest

from stable_baselines3.common.callbacks import CallbackList

from replication.atari import train_atari_stackpomdp_leader_sb3 as trainer
from stackelberg_pomdp.atari.stackpomdp_full_leader_env import (
    FullTraceStackPOMDPAtariLeaderEnv,
)
from stackelberg_pomdp.callbacks import FixPolicyActionsCallback


def _args(tmp_path, *extra):
    game = tmp_path / "game.zip"
    response = tmp_path / "response.zip"
    game.write_bytes(b"game")
    response.write_bytes(b"response")
    return trainer.parse_args([
        "--leader-role",
        "seller",
        "--game-checkpoint",
        str(game),
        "--response-checkpoint",
        str(response),
        "--no-wandb",
        *extra,
    ])


def _terminal_info():
    steps = (10, 50, 80, 120, 140)
    accepted = (True, False, True, False, True)
    events = tuple({
        "event_index": index,
        "game_step": step,
        "price": 0.1 * (index + 1),
        "threshold": 0.5,
        "accepted": outcome,
    } for index, (step, outcome) in enumerate(zip(steps, accepted)))
    info = {
        "leader_role": "seller",
        "follower_role": "buyer",
        "query_trace_sha256": "a" * 64,
        "query_actions": tuple((1.0, 0.1 * (index + 1)) for index in range(5)),
        "follower_actions": tuple((1.0, 0.5) for _ in range(5)),
        "events": events,
        "event_steps": steps,
        "outer_transition_count": 210,
        "query_transitions": 5,
        "gameplay_transitions": 200,
        "trade_transitions": 5,
        "cache_hits": 5,
    }
    for field in (
            "trade_opportunities",
            "bullets_arrived",
            "purchases",
            "payments",
            "seller_game_reward",
            "buyer_game_reward",
            "seller_reward",
            "buyer_reward",
            "leader_reward",
            "seller_shots_fired",
            "buyer_shots_fired",
            "seller_final_ammo",
            "buyer_final_ammo",
            "seller_bullet_error",
            "buyer_bullet_error",
            "seller_payoff_error",
            "buyer_payoff_error",
    ):
        info[field] = 0.0
    return info


def test_default_rollout_is_exactly_one_full_no_skipping_episode(tmp_path):
    args = _args(tmp_path)
    trainer.validate_args(args)

    assert trainer.full_episode_transitions(args.gameplay_horizon) == 210
    assert args.n_steps == 210
    assert args.batch_size == 210


def test_rollout_length_must_match_queries_gameplay_and_trades(tmp_path):
    args = _args(tmp_path, "--n-steps", "250")

    with pytest.raises(ValueError, match="five queries.*five trades = 210"):
        trainer.validate_args(args)


def test_short_horizon_smoke_rollout_can_still_end_at_episode_boundary(tmp_path):
    args = _args(
        tmp_path,
        "--gameplay-horizon",
        "7",
        "--event-tail-steps",
        "2",
        "--n-steps",
        "17",
        "--batch-size",
        "17",
    )
    trainer.validate_args(args)

    assert trainer.full_episode_transitions(args.gameplay_horizon) == 17
    assert args.n_steps == 17


def test_trainer_constructs_the_full_trace_environment(monkeypatch, tmp_path):
    args = _args(tmp_path)
    trainer.validate_args(args)
    captured = {}

    def fake_full_trace_env(**kwargs):
        captured.update(kwargs)
        return "full-trace-env"

    monkeypatch.setattr(
        trainer,
        "FullTraceStackPOMDPAtariLeaderEnv",
        fake_full_trace_env,
    )

    result = trainer.make_leader_env(args, seed=37)

    assert result == "full-trace-env"
    assert captured["leader_role"] == "seller"
    assert captured["config"].seed == 37
    assert captured["config"].gameplay_horizon == 200


def test_training_callbacks_include_the_mandatory_policy_cache(tmp_path):
    args = _args(tmp_path)
    checkpoint = Path(tmp_path) / "leader.zip"

    leader_callback, callbacks = trainer.make_training_callback(
        args,
        checkpoint,
        "frozen-fingerprint",
    )

    assert isinstance(callbacks, CallbackList)
    assert isinstance(callbacks.callbacks[0], FixPolicyActionsCallback)
    assert callbacks.callbacks[1] is leader_callback
    callbacks.callbacks[0].locals = {"dones": [False]}
    assert callbacks.callbacks[0]._on_step() is True


def test_wandb_metadata_records_full_trajectory_and_policy_cache(tmp_path):
    args = _args(tmp_path)
    config = trainer._wandb_config(
        args,
        Path(tmp_path) / "leader.zip",
        {},
    )

    assert config["query_steps_in_rollout_buffer"] is True
    assert config["gameplay_steps_in_rollout_buffer"] == 200
    assert config["trade_steps_in_rollout_buffer"] == 5
    assert config["outer_episode_transitions"] == 210
    assert config["exclude_from_buffer"] is False
    assert config["policy_action_cache"] is True
    assert config["leader_action_cached_by_environment"] is False
    assert FullTraceStackPOMDPAtariLeaderEnv is not None


def test_trade_timing_diagnostics_cover_event_and_acceptance_timing():
    events = _terminal_info()["events"]

    diagnostics = trainer.trade_timing_diagnostics(events, 200)

    assert diagnostics["mean_event_1_step"] == pytest.approx(10.0)
    assert diagnostics["mean_event_5_normalized_time"] == pytest.approx(0.7)
    assert diagnostics["accepted_event_count"] == 3
    assert diagnostics["rejected_event_count"] == 2
    assert diagnostics["mean_accepted_normalized_game_time"] == pytest.approx(
        (10 + 80 + 140) / 3 / 200
    )
    assert diagnostics["mean_rejected_normalized_game_time"] == pytest.approx(
        (50 + 120) / 2 / 200
    )
    assert diagnostics["early_acceptance_rate"] == pytest.approx(0.5)
    assert diagnostics["middle_acceptance_rate"] == pytest.approx(0.5)
    assert diagnostics["late_acceptance_rate"] == pytest.approx(1.0)


def test_training_and_evaluation_expose_trade_timing_metrics():
    info = _terminal_info()

    training = trainer._episode_metrics([info], gameplay_horizon=200)
    row = trainer._scalar_episode_row(info, episode_index=0)
    evaluation = trainer.summarize_evaluation([row], gameplay_horizon=200)

    assert training["train/mean_event_3_step"] == pytest.approx(80.0)
    assert training[
        "train/mean_accepted_normalized_game_time"
    ] == pytest.approx((10 + 80 + 140) / 3 / 200)
    assert training["train/late_acceptance_rate"] == pytest.approx(1.0)
    assert evaluation["mean_event_3_step"] == pytest.approx(80.0)
    assert evaluation[
        "mean_rejected_normalized_game_time"
    ] == pytest.approx((50 + 120) / 2 / 200)
    assert evaluation["middle_acceptance_rate"] == pytest.approx(0.5)


def test_reproducibility_hashes_cover_every_uncommitted_leader_source(tmp_path):
    metadata = trainer.reproducibility_source_metadata()
    hashes = metadata["source_files_sha256"]

    assert tuple(hashes) == trainer.REPRODUCIBILITY_SOURCE_FILES
    assert all(len(value) == 64 for value in hashes.values())
    trainer_relative_path = (
        "replication/atari/train_atari_stackpomdp_leader_sb3.py"
    )
    assert hashes[trainer_relative_path] == trainer.file_sha256(
        trainer.REPOSITORY_ROOT / trainer_relative_path
    )
    assert metadata["source_bundle_sha256"] == trainer.source_bundle_sha256(
        dict(reversed(tuple(hashes.items())))
    )

    args = _args(tmp_path)
    config = trainer._wandb_config(args, tmp_path / "leader.zip", {})
    assert config["source_files_sha256"] == hashes
    assert config["source_bundle_sha256"] == metadata[
        "source_bundle_sha256"
    ]
