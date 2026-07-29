from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari.automation import release_atari_e1_primary_economic as release
from stackelberg_pomdp.atari.stackpomdp_env import BUYER


def _fixed(value, *, purchases, shots, payoff, mechanics=True):
    return {
        "opponent_value": float(value),
        "summary": {
            "mean_purchases": float(purchases),
            "mean_buyer_shots_fired": float(shots),
            "mean_controlled_payoff": float(payoff),
        },
        "protocol": {"passed": bool(mechanics), "violations": []},
    }


def _passing_primary_inputs():
    fixed = []
    for value in release.FIXED_VALUES:
        if value <= 0.5:
            fixed.append(_fixed(
                value, purchases=5.0, shots=5.0,
                payoff=max(0.1, 5.0 - 5.0 * value),
            ))
        else:
            demand = (1.0 - value) * 10.0
            fixed.append(_fixed(value, purchases=demand, shots=demand, payoff=0.0))
    random = {
        "summary": {"mean_controlled_payoff": 1.0},
        "protocol": {"passed": True, "violations": []},
    }
    return random, fixed


def _timing_results():
    rows = []
    for timing in ("early", "late"):
        for price in (0.5, 0.75, 0.9):
            rows.append({
                "timing": timing,
                "fifth_price": price,
                "policy_mode": "actual",
                "summary": {
                    "mean_controlled_payoff": 1.95 if timing == "early" else 1.0,
                    "event_5_acceptance_rate": 1.0 if timing == "early" else 0.0,
                },
                "protocol": {"passed": True, "violations": []},
            })
        rows.extend((
            {
                "timing": timing,
                "fifth_price": 0.75,
                "policy_mode": "forced_buy",
                "summary": {
                    "mean_controlled_payoff": 2.0 if timing == "early" else 0.0,
                    "event_5_acceptance_rate": 1.0,
                },
                "protocol": {"passed": True, "violations": []},
            },
            {
                "timing": timing,
                "fifth_price": 0.75,
                "policy_mode": "forced_reject",
                "summary": {
                    "mean_controlled_payoff": 1.0,
                    "event_5_acceptance_rate": 0.0,
                },
                "protocol": {"passed": True, "violations": []},
            },
        ))
    return rows


def test_primary_gate_is_exact_legacy_prefix_of_sixteen_checks():
    random, fixed = _passing_primary_inputs()
    primary = evaluator.buyer_primary_economic_gate(
        random_result=random, fixed_results=fixed
    )
    independent = release.primary_economic_gate(
        random_result=random, fixed_results=fixed
    )
    legacy = evaluator.behavioral_gate(
        role=BUYER,
        random_result=random,
        fixed_results=fixed,
        timing_results=_timing_results(),
    )
    assert primary == independent
    assert len(primary["checks"]) == 16
    assert [row["name"] for row in primary["checks"]] == list(
        release.EXPECTED_CHECK_NAMES
    )
    assert primary["checks"] == legacy["checks"][:16]
    assert primary["passed"]


def test_primary_gate_requires_mechanics_and_exact_unique_grid():
    random, fixed = _passing_primary_inputs()
    fixed[3]["protocol"]["passed"] = False
    assert not release.primary_economic_gate(
        random_result=random, fixed_results=fixed
    )["passed"]
    malformed = deepcopy(fixed)
    malformed[-1]["opponent_value"] = 0.9
    with pytest.raises(ValueError, match="duplicate fixed-grid"):
        release.primary_economic_gate(
            random_result=random, fixed_results=malformed
        )


def _episode(*, seed, context, event_steps, phase, checkpoint_path="/tmp/source.zip"):
    events = []
    payments = 0.0
    for index, price in enumerate(context):
        threshold = 1.0
        accepted = bool(price <= threshold)
        payments += float(price) * accepted
        events.append({
            "event_index": index,
            "game_step": event_steps[index],
            "price": float(price),
            "threshold": threshold,
            "accepted": accepted,
            "seller_ammo_before": 1,
            "seller_ammo_after": 0,
            "buyer_ammo_before": 0,
            "buyer_ammo_after": 1,
        })
    purchases = 5
    buyer_game = 5.0
    row = {
        "phase": phase,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": release.EXPECTED_SHA256["source_checkpoint"],
        "training_timesteps": 2_400_960,
        "evaluation_seed": seed,
        "evaluation_steps": 205,
        "evaluation_return": buyer_game - payments,
        "controlled_role": BUYER,
        "opponent_commitment": [float(value) for value in context],
        "event_steps": list(event_steps),
        "events": events,
        "outer_transition_count": 205,
        "reward_transition_count": 205,
        "gameplay_transitions": 200,
        "trade_transitions": 5,
        "seller_emulator_step_calls": 200,
        "buyer_emulator_step_calls": 200,
        "bullets_arrived": 5,
        "purchases": purchases,
        "payments": payments,
        "seller_game_reward": 0.0,
        "buyer_game_reward": buyer_game,
        "seller_reward": payments,
        "buyer_reward": buyer_game - payments,
        "seller_shots_fired": 0,
        "buyer_shots_fired": 5,
        "seller_final_ammo": 0,
        "buyer_final_ammo": 0,
        "seller_bullet_error": 0,
        "buyer_bullet_error": 0,
        "seller_payoff_error": 0.0,
        "buyer_payoff_error": 0.0,
        "fifth_economic_override": None,
        "fifth_economic_override_applied": 0,
        "e1_sampler_mode": "uniform",
        "e1_context_stratum": "external",
        "e1_schedule_stratum": "fixed",
    }
    return row


def _serialized_result(row):
    rows = [row]
    return {
        "summary": evaluator._summary(rows, role=BUYER),
        "protocol": {"passed": True, "violations": []},
        "episode_rows": rows,
    }


def test_raw_revalidation_rejects_valid_looking_schedule_summary_and_policy_tampering():
    context = np.full(5, 0.2, dtype=np.float32)
    row = _episode(
        seed=release.FIXED_SEED_START,
        context=context,
        event_steps=release.FIXED_EVENT_STEPS,
        phase="fixed_0.20",
    )
    result = _serialized_result(row)
    release._recomputed_result(
        result,
        expected_seeds=[release.FIXED_SEED_START],
        expected_contexts=[context],
        expected_steps=release.FIXED_EVENT_STEPS,
        phase="fixed_0.20",
        schedule_stratum="fixed",
        checkpoint_path=Path("/tmp/source.zip"),
    )
    bad_schedule = deepcopy(result)
    bad_schedule["episode_rows"][0]["event_steps"] = [21, 50, 80, 110, 140]
    with pytest.raises(ValueError, match="another event schedule"):
        release._recomputed_result(
            bad_schedule,
            expected_seeds=[release.FIXED_SEED_START],
            expected_contexts=[context],
            expected_steps=release.FIXED_EVENT_STEPS,
            phase="fixed_0.20",
            schedule_stratum="fixed",
            checkpoint_path=Path("/tmp/source.zip"),
        )
    bad_summary = deepcopy(result)
    bad_summary["summary"]["mean_purchases"] = 4.9
    with pytest.raises(ValueError, match="serialized summary"):
        release._recomputed_result(
            bad_summary,
            expected_seeds=[release.FIXED_SEED_START],
            expected_contexts=[context],
            expected_steps=release.FIXED_EVENT_STEPS,
            phase="fixed_0.20",
            schedule_stratum="fixed",
            checkpoint_path=Path("/tmp/source.zip"),
        )
    bad_policy = deepcopy(result)
    bad_policy["episode_rows"][0]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="eligible policy"):
        release._recomputed_result(
            bad_policy,
            expected_seeds=[release.FIXED_SEED_START],
            expected_contexts=[context],
            expected_steps=release.FIXED_EVENT_STEPS,
            phase="fixed_0.20",
            schedule_stratum="fixed",
            checkpoint_path=Path("/tmp/source.zip"),
        )


def test_random_schedule_reconstruction_uses_full_seed_chain():
    seed = release.RANDOM_SEED_START
    wrapper = np.random.default_rng(seed + 74_711)
    inner = int(wrapper.integers(0, 2 ** 31 - 1))
    expected = tuple(sorted(
        np.random.default_rng(inner).choice(np.arange(200), 5, replace=False)
    ))
    assert release.expected_random_schedule(seed) == expected
    assert release.expected_random_schedule(seed + 1) != expected


def test_atomic_publication_refuses_overwrite_and_symlink(tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"checkpoint")
    target = tmp_path / "selected.zip"
    release.atomic_copy_new(source, target)
    assert target.read_bytes() == b"checkpoint"
    with pytest.raises(FileExistsError):
        release.atomic_copy_new(source, target)
    symlink = tmp_path / "link.zip"
    symlink.symlink_to(source)
    with pytest.raises(FileNotFoundError, match="nonsymlink"):
        release.sha256_file(symlink)


def _stub_release_validation(monkeypatch, source):
    digest = release.sha256_file(source)
    protocol = {
        "candidate_policy": {
            "checkpoint": {"path": str(source), "sha256": digest},
        },
    }

    def validate_report(**kwargs):
        return {"passed": True}

    def build_gate(**kwargs):
        return {"kind": "test-gate", "passed": True, "sha256": digest}

    def validate_gate(**kwargs):
        value = release.load_json(kwargs["gate_path"])
        assert value == build_gate()
        assert release.sha256_file(kwargs["selected_checkpoint"]) == digest
        return value

    monkeypatch.setattr(release, "validate_report", validate_report)
    monkeypatch.setattr(release, "validate_protocol", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(release, "build_gate", build_gate)
    monkeypatch.setattr(release, "validate_gate", validate_gate)
    return digest


def test_finalize_release_recovers_report_only_without_rerunning(monkeypatch, tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"eligible checkpoint")
    digest = _stub_release_validation(monkeypatch, source)
    report = tmp_path / "confirmation.json"
    report.write_text("{}")
    selected = tmp_path / "selected.zip"
    gate = tmp_path / "confirmation.gate.json"

    first = release.finalize_release(
        protocol_path=tmp_path / "protocol.json",
        report_path=report,
        selected_checkpoint=selected,
        gate_path=gate,
        code_root=tmp_path,
    )
    assert first["gate"]["passed"]
    assert release.sha256_file(selected) == digest
    first_gate_bytes = gate.read_bytes()

    second = release.finalize_release(
        protocol_path=tmp_path / "protocol.json",
        report_path=report,
        selected_checkpoint=selected,
        gate_path=gate,
        code_root=tmp_path,
    )
    assert second["gate"] == first["gate"]
    assert gate.read_bytes() == first_gate_bytes


def test_finalize_release_recovers_alias_without_gate(monkeypatch, tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"eligible checkpoint")
    _stub_release_validation(monkeypatch, source)
    report = tmp_path / "confirmation.json"
    report.write_text("{}")
    selected = tmp_path / "selected.zip"
    release.atomic_copy_new(source, selected)
    gate = tmp_path / "confirmation.gate.json"

    result = release.finalize_release(
        protocol_path=tmp_path / "protocol.json",
        report_path=report,
        selected_checkpoint=selected,
        gate_path=gate,
        code_root=tmp_path,
    )
    assert result["gate"]["passed"]
    assert gate.is_file()


def test_finalize_release_rejects_gate_without_alias(monkeypatch, tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"eligible checkpoint")
    _stub_release_validation(monkeypatch, source)
    report = tmp_path / "confirmation.json"
    report.write_text("{}")
    gate = tmp_path / "confirmation.gate.json"
    gate.write_text('{"kind":"test-gate","passed":true}')

    with pytest.raises(ValueError, match="gate exists without"):
        release.finalize_release(
            protocol_path=tmp_path / "protocol.json",
            report_path=report,
            selected_checkpoint=tmp_path / "selected.zip",
            gate_path=gate,
            code_root=tmp_path,
        )


def test_launcher_uses_clean_code_root_and_separate_artifact_root():
    script = Path(
        "replication/atari/automation/"
        "run_atari_clean_e1_buyer_primary_economic_release.sh"
    ).read_text()
    assert 'CODE_ROOT="${AUTOMATION_DIR:h:h:h}"' in script
    assert "STACKPOMDP_ARTIFACT_ROOT" in script
    assert "write-protocol" in script
    assert "finalize-release" in script
    assert "--device cpu" in script
    assert "wandb" not in script.lower()
