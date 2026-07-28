import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replication.atari.automation import (
    preflight_atari_e1_temporal_contingency as preflight,
)
from replication.atari.automation import (
    validate_atari_e1_temporal_contingency as validator,
)


def _write_failed_report(
        tmp_path, *, name, actor_loss_mode, rom_sha256="r" * 64,
        passed=False,
):
    root = tmp_path / name
    root.mkdir()
    e0b = tmp_path / "e0b.zip"
    e0b.write_bytes(b"e0b")
    candidates = []
    results = []
    ranking = []
    for index in range(6):
        candidate = root / f"buyer_step{validator.CHECKPOINT_INTERVAL * (index + 1)}.zip"
        candidate.write_bytes(f"{name}-{index}".encode())
        digest = validator.sha256_file(candidate)
        candidates.append(digest)
        metadata = {
            "path": str(candidate),
            "sha256": digest,
            "training_timesteps": validator.CHECKPOINT_INTERVAL * (index + 1),
        }
        results.append({
            "metadata": metadata,
            "protocol": {"passed": True},
            "summary": {"mean_controlled_payoff": 6 - index},
            "episode_rows": [{"evaluation_seed": seed} for seed in range(20)],
            "event_rows": [],
        })
        ranking.append({
            "rank": index + 1,
            "checkpoint_path": str(candidate),
            "checkpoint_sha256": digest,
            "training_timesteps": metadata["training_timesteps"],
            "mechanically_valid": True,
        })
    report_path = root / "report.json"
    report = {
        "evaluator": validator.evaluator.EVALUATOR_NAME,
        "role": "buyer",
        "passed": bool(passed),
        "selected_alias": ({"path": "selected.zip"} if passed else None),
        "protocol": {
            "screen_episodes": 20,
            "confirmation_episodes": 100,
            "outer_transitions": 205,
        },
        "environment": {
            "gameplay_horizon": 200,
            "event_tail_steps": 0,
            "rom_sha256": rom_sha256,
        },
        "training_family": {
            "common_training_config": {
                "actor_loss_mode": actor_loss_mode,
                "n_steps": 205,
                "batch_size": 820,
                "gamma": 1.0,
                "gae_lambda": 1.0,
            },
            "common_sampler_provenance": None,
            "common_sampler_history": None,
        },
        "screen": {
            "common_pairing": {
                "passed": True,
                "candidates_checked": 6,
                "seed_context_pairs": [{} for _ in range(20)],
            },
            "results": results,
        },
        "ranking": ranking,
        "confirmation_attempts": ([{
            "behavioral_gate": {"passed": False}
        }] if not passed else [{"behavioral_gate": {"passed": True}}]),
        "immutable_evaluation": {"candidate_sha256": candidates},
        "e0b_source": {
            "path": str(e0b),
            "sha256": validator.sha256_file(e0b),
        },
        "artifacts": {"json": str(report_path)},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, report


def test_activation_is_inert_until_both_uniform_all_six_reports_fail(
        monkeypatch, tmp_path,
):
    rom = tmp_path / "space_invaders.bin"
    rom.write_bytes(b"rom")
    rom_sha256 = validator.sha256_file(rom)
    monkeypatch.setattr(validator, "CANONICAL_ROM_SHA256", rom_sha256)
    standard_path, _ = _write_failed_report(
        tmp_path, name="standard", actor_loss_mode="standard",
        rom_sha256=rom_sha256,
    )
    balanced_path, _ = _write_failed_report(
        tmp_path, name="balanced", actor_loss_mode="balanced",
        rom_sha256=rom_sha256, passed=True,
    )
    output = tmp_path / "activation.json"
    args = SimpleNamespace(
        standard_report=str(standard_path),
        balanced_report=str(balanced_path),
        rom=str(rom),
        base=str(tmp_path / "temporal.zip"),
        code_root=str(Path(__file__).resolve().parents[1]),
        output=str(output),
    )
    with pytest.raises(validator.ContingencyInactive):
        validator.write_or_validate_activation(args)
    assert not output.exists()

    balanced = json.loads(balanced_path.read_text())
    balanced["passed"] = False
    balanced["selected_alias"] = None
    balanced["confirmation_attempts"] = [{
        "behavioral_gate": {"passed": False}
    }]
    balanced_path.write_text(json.dumps(balanced), encoding="utf-8")
    activation = validator.write_or_validate_activation(args)
    assert output.is_file()
    assert activation["resume_source"]["sha256"] == balanced["ranking"][0]["checkpoint_sha256"]
    assert activation["resume_source"]["rank"] == 1
    assert activation["protocol"]["additional_timesteps"] == 2_000_800
    assert len(activation["protocol"]["candidate_paths"]) == 6
    assert activation["protocol"]["confirmation"]["fallback_allowed"] is False
    with pytest.raises(FileExistsError):
        validator.atomic_write_json(output, {"replacement": True})


def test_sampler_preflight_checks_frequencies_support_and_independence():
    result = preflight.audit_sampler(seed=97_531, draws=10_000)
    assert result["passed"]
    assert result["low_prefix_draws_checked"] > 2_000
    assert result["maximum_independence_error"] < 0.02
    assert result["schedule_frequencies"]["early_fifth"] == pytest.approx(
        0.25, abs=0.02
    )
    assert result["context_frequencies"]["low_prefix"] == pytest.approx(
        0.25, abs=0.02
    )


class _FakeModel:
    def predict(self, observation, deterministic):
        assert deterministic
        return np.zeros((1, 2), dtype=np.float32), None


class _FakeVecEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return {"observation": np.zeros((1, 1), dtype=np.float32)}

    def step(self, action):
        self.steps += 1
        done = self.steps == 205
        info = {}
        if done:
            info = {
                "gameplay_transitions": 200,
                "trade_transitions": 5,
                "reward_transition_count": 205,
                "outer_transition_count": 205,
                "e1_sampler_mode": "temporal-marginal-v1",
                "e1_schedule_stratum": "late_fifth",
                "e1_context_stratum": "low_prefix",
                "events": [{} for _ in range(5)],
                "event_steps": (20, 50, 80, 110, 190),
                "opponent_commitment": (0.1, 0.1, 0.1, 0.1, 0.8),
                "purchases": 3,
                "bullets_arrived": 3,
                "payments": 1.1,
                "buyer_game_reward": 4.0,
                "buyer_shots_fired": 3,
                "buyer_final_ammo": 0,
            }
        return self.reset(), np.array([0.0]), np.array([done]), [info]


def test_real_ale_preflight_contract_stops_after_one_complete_episode():
    env = _FakeVecEnv()
    result = preflight._run_one_ale_episode(_FakeModel(), env)
    assert env.steps == 205
    assert result["trade_transitions"] == 5
    assert result["purchases"] == result["bullets_arrived"] == 3


def test_launcher_orders_activation_and_preflight_before_wandb_or_training():
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "replication/atari/automation"
        / "run_atari_clean_e1_buyer_temporal_contingency_2m.sh"
    ).read_text(encoding="utf-8")
    activation = launcher.index("e1_activate")
    preflight_index = launcher.index("e1_run_preflight")
    wandb = launcher.index("WANDB_MODE=online")
    training = launcher.index("train_atari_meta_response_sb3")
    assert activation < preflight_index < wandb < training


def test_temporal_selector_publishes_a_separate_immutable_gate():
    root = Path(__file__).resolve().parents[1]
    selector = (
        root / "replication/atari/automation"
        / "run_e1_buyer_temporal_contingency_final_selector.sh"
    ).read_text(encoding="utf-8")
    assert "--gate-output \"$E1_SELECTOR_GATE\"" in selector
    assert "validate-selection-gate" in (
        root / "replication/atari/automation"
        / "validate_atari_e1_temporal_contingency.py"
    ).read_text(encoding="utf-8")


def test_temporal_selector_validator_forbids_confirmation_fallback(
        monkeypatch, tmp_path,
):
    hashes = [f"{index + 1:064x}" for index in range(6)]
    sampler = {"mode": "temporal-marginal-v1"}
    history = [{"sampler": {"mode": "uniform"}}, {"sampler": sampler}]
    family = {
        "candidate_sha256": hashes,
        "candidate_metadata": [{
            "atari_e1_sampler_provenance": sampler,
            "atari_e1_sampler_history": history,
        } for _ in hashes],
    }
    monkeypatch.setattr(
        validator, "validate_training_family", lambda path: family
    )
    report_path = tmp_path / "selection.json"
    selected = tmp_path / "selected.zip"
    attempt = {
        "metadata": {"sha256": hashes[0]},
        "random": {"episode_rows": [
            {"evaluation_seed": seed}
            for seed in range(6_100_001, 6_100_101)
        ]},
        "behavioral_gate": {"passed": False},
    }
    report = {
        "evaluator": validator.evaluator.EVALUATOR_NAME,
        "role": "buyer",
        "passed": False,
        "selected_alias": None,
        "protocol": {
            "confirmation_policy": "screen_winner_only_no_fallback",
            "screen_episodes": 20,
            "screen_seed_start": 6_000_001,
            "confirmation_episodes": 100,
            "confirmation_seed_start": 6_100_001,
            "fixed_context_seed_start": 6_200_001,
            "paired_timing_seed_start": 6_300_001,
        },
        "selection": {
            "fallback_allowed": False,
            "screen_selected_checkpoint_sha256": hashes[0],
            "selected_checkpoint_sha256": None,
        },
        "screen": {
            "common_pairing": {
                "passed": True,
                "candidates_checked": 6,
                "seed_context_pairs": [
                    {"evaluation_seed": seed}
                    for seed in range(6_000_001, 6_000_021)
                ],
            },
            "results": [{"metadata": {"sha256": digest}} for digest in hashes],
        },
        "immutable_evaluation": {"candidate_sha256": hashes},
        "ranking": [{
            "checkpoint_sha256": hashes[0], "rank": 1
        }],
        "confirmation_attempts": [attempt],
        "training_family": {
            "common_sampler_provenance": sampler,
            "common_sampler_history": history,
        },
        "artifacts": {"json": str(report_path)},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    args = SimpleNamespace(
        family=str(tmp_path / "family.json"),
        report=str(report_path),
        selected=str(selected),
    )
    assert validator.validate_selection(args)["passed"] is False
    report["confirmation_attempts"].append(dict(attempt))
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="confirm only the screen winner"):
        validator.validate_selection(args)
