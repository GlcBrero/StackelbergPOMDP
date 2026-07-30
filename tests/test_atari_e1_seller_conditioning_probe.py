from types import SimpleNamespace

import gym
import numpy as np
import pytest

from replication.atari import probe_atari_e1_seller_conditioning as probe
from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTION_MASK,
    ACTOR_STATE,
    CRITIC_STATE,
    EVENT_SLICE,
    IMAGE,
    OPPONENT_COMMITMENT_SLICE,
    TIME_INDEX,
    TRADE_MODE_INDEX,
    observation_space,
)


class _ProtocolPolicy:
    def __init__(self):
        image_space = gym.spaces.Box(
            0, 255, shape=(84, 84, 4), dtype=np.uint8
        )
        self.observation_space = observation_space(image_space, 6)


def test_canonical_probe_observation_is_exact_valid_trade_protocol():
    policy = _ProtocolPolicy()
    context = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    values = probe.canonical_trade_observation(
        policy, event_index=2, thresholds=context
    )

    assert policy.observation_space.contains(values)
    assert values[IMAGE].shape == (84, 84, 4)
    assert np.count_nonzero(values[IMAGE]) == 0
    assert np.array_equal(values[ACTION_MASK], np.ones(6, dtype=np.float32))
    assert np.count_nonzero(values[CRITIC_STATE]) == 0
    assert np.array_equal(
        values[ACTION_CREDIT], np.array([0.0, 1.0], dtype=np.float32)
    )
    state = values[ACTOR_STATE]
    assert state.shape == (14,)
    assert state[0] == pytest.approx(0.2)
    assert state[1] == 0.0
    assert state[TIME_INDEX] == pytest.approx(80 / 200)
    assert state[TRADE_MODE_INDEX] == 1.0
    assert np.array_equal(
        state[EVENT_SLICE], np.array([0, 0, 1, 0, 0], dtype=np.float32)
    )
    assert np.array_equal(state[OPPONENT_COMMITMENT_SLICE], context)


def test_conditioning_report_measures_event_and_current_coordinate_response(
        monkeypatch,
):
    model = SimpleNamespace(policy=_ProtocolPolicy())

    def threshold_tracking_mean(received_model, values):
        assert received_model is model
        state = values[ACTOR_STATE]
        event_index = int(np.argmax(state[EVENT_SLICE]))
        current_threshold = state[OPPONENT_COMMITMENT_SLICE][event_index]
        return 0.1 + 0.7 * float(current_threshold)

    monkeypatch.setattr(probe, "economic_beta_mean", threshold_tracking_mean)
    result = probe.collect_conditioning_report(model)

    assert result["warmup_gate"]["passed"]
    assert result["all_equal_thresholds"][0][
        "event_beta_mean_prices"
    ] == pytest.approx([0.1] * 5)
    assert result["all_equal_thresholds"][-1][
        "event_beta_mean_prices"
    ] == pytest.approx([0.8] * 5)
    assert result["low_to_high_response"]["per_event"] == pytest.approx(
        [0.7] * 5
    )
    coordinate = result["current_coordinate_only_sensitivity"]
    assert coordinate["mean"] == pytest.approx(0.7)
    assert [row["sensitivity"] for row in coordinate["rows"]] == pytest.approx(
        [0.7] * 5
    )
    for row in coordinate["rows"]:
        event = row["event_index"]
        assert row["low_context"][event] == 0.0
        assert row["high_context"][event] == 1.0
        assert all(
            value == pytest.approx(0.5)
            for index, value in enumerate(row["low_context"])
            if index != event
        )


def test_warmup_gate_predeclares_endpoint_monotonicity_and_range():
    passing = np.repeat(
        np.array([[0.1], [0.3], [0.5], [0.7], [0.9]]), 5, axis=1
    )
    gate = probe.warmup_diagnostic_gate(
        passing, np.full(5, 0.1), np.full(5, 0.9)
    )
    assert gate["predeclared"]
    assert gate["passed"]
    assert gate["checks"][
        "minimum_all_one_minus_all_zero_beta_mean_price"
    ]["required"] == 0.25
    assert gate["checks"][
        "largest_adjacent_threshold_price_reversal"
    ]["required"] == 0.15

    reversal = np.repeat(
        np.array([[0.1], [0.5], [0.2], [0.7], [0.9]]), 5, axis=1
    )
    gate = probe.warmup_diagnostic_gate(
        reversal, np.full(5, 0.1), np.full(5, 0.9)
    )
    assert not gate["passed"]
    assert not gate["checks"][
        "largest_adjacent_threshold_price_reversal"
    ]["passed"]
    assert gate["checks"][
        "largest_adjacent_threshold_price_reversal"
    ]["actual"] == pytest.approx(0.3)

    invalid_high = np.full(5, 0.9)
    invalid_high[0] = np.nan
    gate = probe.warmup_diagnostic_gate(
        passing, np.full(5, 0.1), invalid_high
    )
    assert not gate["passed"]
    assert not gate["checks"][
        "all_outputs_finite_and_in_unit_interval"
    ]["passed"]


def test_loaded_seller_validation_requires_full_role_and_bound_metadata(
        monkeypatch,
):
    class FakePolicy:
        economic_role = "seller"
        economic_input_mode = "full"
        pretrained_lr_scale = 0.1

    monkeypatch.setattr(probe, "StackPOMDPAtariPolicy", FakePolicy)
    model = SimpleNamespace(
        policy=FakePolicy(),
        num_timesteps=410,
        n_steps=205,
        gamma=1.0,
        gae_lambda=1.0,
    )
    metadata = {
        "role": "seller",
        "economic_input_mode": "full",
        "sha256": "a" * 64,
        "training_timesteps": 410,
        "training_config": {"algorithm": "PPO"},
        "e0b_source_provenance": {
            "sha256": "b" * 64,
            "zero_initialized_actor_state_indices": [9, 10, 11, 12, 13],
        },
        "atari_e1_sampler_provenance": {"mode": "uniform"},
        "atari_e1_sampler_history": [{}],
    }
    e0b = {"sha256": "b" * 64}

    result = probe.validate_loaded_seller(model, metadata, e0b)
    assert result["passed"]
    assert all(result["checks"].values())

    model.policy.economic_input_mode = "event_only"
    with pytest.raises(ValueError, match="full_economic_input"):
        probe.validate_loaded_seller(model, metadata, e0b)
    model.policy.economic_input_mode = "full"
    metadata["e0b_source_provenance"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="e0b_source_bytes_match"):
        probe.validate_loaded_seller(model, metadata, e0b)


def test_cli_emits_json_and_can_fail_closed(monkeypatch, tmp_path):
    report = {
        "checkpoint": {"sha256": "d" * 64},
        "warmup_gate": {"passed": False},
    }
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(probe, "run_probe_from_checkpoints", fake_run)
    output = tmp_path / "conditioning.json"
    status = probe.main([
        "--checkpoint",
        "seller.zip",
        "--e0b-checkpoint",
        "e0b.zip",
        "--device",
        "cpu",
        "--output",
        str(output),
        "--require-pass",
    ])

    assert status == 1
    assert calls == [{
        "checkpoint": "seller.zip",
        "e0b_checkpoint": "e0b.zip",
        "device": "cpu",
    }]
    assert output.is_file()
    assert output.read_text().endswith("\n")
    assert __import__("json").loads(output.read_text()) == report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        probe.main([
            "--checkpoint",
            "seller.zip",
            "--e0b-checkpoint",
            "e0b.zip",
            "--output",
            str(output),
        ])
