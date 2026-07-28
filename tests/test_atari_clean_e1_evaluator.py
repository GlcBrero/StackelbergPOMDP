from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from stackelberg_pomdp.atari.stackpomdp_env import BUYER, SELLER


def _episode(*, role=BUYER, seed=10, context=None, event_steps=None):
    context = [0.2] * 5 if context is None else list(context)
    event_steps = [20, 50, 80, 110, 140] if event_steps is None else list(event_steps)
    events = []
    payments = 0.0
    for index, opponent in enumerate(context):
        price = float(opponent if role == BUYER else 0.2)
        threshold = float(0.8 if role == BUYER else opponent)
        accepted = price <= threshold
        payments += price * accepted
        events.append({
            "event_index": index,
            "game_step": event_steps[index],
            "price": price,
            "threshold": threshold,
            "accepted": accepted,
            "seller_ammo_before": 1,
            "seller_ammo_after": 1 - int(accepted),
            "buyer_ammo_before": 0,
            "buyer_ammo_after": int(accepted),
        })
    purchases = sum(event["accepted"] for event in events)
    seller_game = float(5 - purchases)
    buyer_game = float(purchases)
    return {
        "evaluation_seed": seed,
        "evaluation_steps": 205,
        "evaluation_return": (
            buyer_game - payments if role == BUYER
            else 0.1 * seller_game + payments
        ),
        "controlled_role": role,
        "opponent_commitment": context,
        "event_steps": event_steps,
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
        "seller_game_reward": seller_game,
        "buyer_game_reward": buyer_game,
        "seller_reward": 0.1 * seller_game + payments,
        "buyer_reward": buyer_game - payments,
        "seller_shots_fired": 5 - purchases,
        "buyer_shots_fired": purchases,
        "seller_final_ammo": 0,
        "buyer_final_ammo": 0,
        "seller_bullet_error": 0,
        "buyer_bullet_error": 0,
        "seller_payoff_error": 0.0,
        "buyer_payoff_error": 0.0,
    }


@pytest.mark.parametrize("role", (BUYER, SELLER))
def test_e1_episode_audit_accepts_exact_bilateral_accounting(role):
    assert evaluator.audit_episode(_episode(role=role), role=role) == []


def test_e1_episode_audit_rejects_protocol_payment_and_inventory_errors():
    row = _episode()
    row["evaluation_steps"] = 204
    row["payments"] += 0.1
    row["events"][0]["buyer_ammo_after"] = 0
    fields = {item["field"] for item in evaluator.audit_episode(row, role=BUYER)}
    assert "evaluation_steps" in fields
    assert "payments" in fields
    assert "event_0_buyer_inventory" in fields


def test_e1_episode_audit_rejects_nonfinite_and_recomputes_bullets():
    row = _episode()
    row["seller_bullet_error"] = float("nan")
    row["buyer_shots_fired"] -= 1
    fields = {item["field"] for item in evaluator.audit_episode(row, role=BUYER)}
    assert "seller_bullet_error" in fields
    assert "independent_buyer_bullet_conservation" in fields


def test_pin_file_detects_source_mutation(monkeypatch, tmp_path):
    source = tmp_path / "source.zip"
    destination = tmp_path / "pinned.zip"
    source.write_bytes(b"source")
    values = iter(("a" * 64, "b" * 64, "a" * 64))
    monkeypatch.setattr(evaluator, "checkpoint_sha256", lambda path: next(values))
    with pytest.raises(RuntimeError, match="changed while being pinned"):
        evaluator.pin_file(source, destination)
    assert not destination.exists()


def _screen_result(path, digest, payoff, *, valid=True, timestep=100):
    row = _episode(seed=1)
    return {
        "metadata": {
            "path": str(path),
            "sha256": digest,
            "training_timesteps": timestep,
            "training_config": {"algorithm": "PPO", "seed": 1},
        },
        "summary": {
            "mean_controlled_payoff": payoff,
            "median_controlled_payoff": payoff,
            "minimum_controlled_payoff": payoff,
            "std_controlled_payoff": 0.0,
        },
        "protocol": {"passed": valid, "violations": [] if valid else [{}]},
        "episode_rows": [row],
        "event_rows": [],
    }


def test_common_screen_checks_contexts_and_schedules():
    seeds = [1]
    contexts = [np.full(5, 0.2, dtype=np.float32)]
    results = [
        _screen_result("a.zip", "a", 1.0),
        _screen_result("b.zip", "b", 2.0),
    ]
    expected_context = [float(value) for value in contexts[0]]
    for result in results:
        result["episode_rows"][0]["opponent_commitment"] = expected_context
    result = evaluator.validate_common_screen(results, seeds=seeds, contexts=contexts)
    assert result["passed"]
    results[1]["episode_rows"][0]["event_steps"][-1] = 190
    with pytest.raises(RuntimeError, match="event schedules"):
        evaluator.validate_common_screen(results, seeds=seeds, contexts=contexts)


def test_rank_excludes_mechanical_failure_and_uses_stable_payoff_order():
    first = _screen_result("a.zip", "a", 2.0, timestep=200)
    second = _screen_result("b.zip", "b", 2.0, timestep=100)
    invalid = _screen_result("c.zip", "c", 100.0, valid=False)
    ranked, rows = evaluator.rank_candidates([first, second, invalid])
    assert [row["metadata"]["sha256"] for row in ranked] == ["b", "a"]
    assert [row["rank"] for row in rows] == [1, 2, None]


def test_checkpoint_family_accepts_steps_and_final_but_distinguishes_runs(
        tmp_path,
):
    final = tmp_path / "response.zip"
    step = tmp_path / "response_step400160.zip"
    other = tmp_path / "other_step400160.zip"
    for path in (final, step, other):
        path.write_bytes(path.name.encode("utf-8"))
    assert evaluator.checkpoint_family(final) == evaluator.checkpoint_family(step)
    assert evaluator.checkpoint_family(other) != evaluator.checkpoint_family(step)


def _fixed(value, **updates):
    summary = {
        "mean_controlled_payoff": 2.0,
        "mean_purchases": 5.0 - 5.0 * value,
        "mean_payments": 1.0,
        "mean_seller_game_reward": 5.0 * (1.0 - value),
        "mean_buyer_game_reward": 5.0,
        "mean_seller_shots_fired": 5.0 * (1.0 - value),
        "mean_buyer_shots_fired": 5.0 - 0.5 * value,
        "mean_seller_final_ammo": 0.0,
        "mean_buyer_final_ammo": 0.0,
        "mean_price": value,
        "mean_threshold": 0.8,
        "acceptance_rate": 1.0 - value,
    }
    summary.update(updates)
    return {
        "opponent_value": value,
        "summary": summary,
        "protocol": {"passed": True, "violations": []},
        "episode_rows": [],
        "event_rows": [],
    }


def test_buyer_behavior_gate_requires_low_price_use_and_high_price_rejection():
    fixed = [
        _fixed(0.0, mean_purchases=5.0, mean_buyer_shots_fired=5.0),
        _fixed(0.5, mean_purchases=5.0, mean_buyer_shots_fired=5.0, mean_controlled_payoff=2.5),
        _fixed(1.0, mean_purchases=0.0, mean_buyer_shots_fired=0.0, mean_controlled_payoff=0.0),
    ]
    random = {
        "summary": {
            "mean_controlled_payoff": 1.0,
            "early_mean_threshold": 0.8,
            "late_mean_threshold": 0.6,
        },
        "protocol": {"passed": True},
    }
    assert evaluator.behavioral_gate(role=BUYER, random_result=random, fixed_results=fixed)["passed"]
    random["summary"]["late_mean_threshold"] = 0.8
    assert not evaluator.behavioral_gate(
        role=BUYER, random_result=random, fixed_results=fixed
    )["passed"]
    random["summary"]["late_mean_threshold"] = 0.6
    fixed[-1]["summary"]["mean_purchases"] = 5.0
    assert not evaluator.behavioral_gate(role=BUYER, random_result=random, fixed_results=fixed)["passed"]


def test_seller_behavior_gate_requires_retention_then_high_value_sales():
    fixed = [
        _fixed(0.0, mean_purchases=0.0, mean_seller_shots_fired=5.0, mean_price=0.2),
        _fixed(0.5, mean_purchases=5.0, mean_seller_shots_fired=0.0, mean_price=0.5),
        _fixed(1.0, mean_purchases=5.0, mean_seller_shots_fired=0.0, mean_price=0.9, mean_controlled_payoff=4.5),
    ]
    random = {"summary": {"mean_controlled_payoff": 2.0}, "protocol": {"passed": True}}
    assert evaluator.behavioral_gate(role=SELLER, random_result=random, fixed_results=fixed)["passed"]
    fixed[-1]["summary"]["mean_price"] = 0.4
    assert not evaluator.behavioral_gate(role=SELLER, random_result=random, fixed_results=fixed)["passed"]


def test_selection_tries_next_ranked_candidate_after_failed_confirmation(monkeypatch, tmp_path):
    first = tmp_path / "response_step100.zip"
    second = tmp_path / "response_step200.zip"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    e0b = tmp_path / "e0b.zip"
    e0b.write_bytes(b"e0b")
    args = SimpleNamespace(
        role=BUYER,
        e0b_checkpoint=str(e0b),
        device="cpu",
        checkpoint=[str(first), str(second)],
        screen_seed_start=100,
        confirmation_seed_start=200,
        fixed_seed_start=400,
        selected_checkpoint=str(tmp_path / "selected.zip"),
        fixed_eval_values=(0.0, 0.5, 1.0),
        grid_event_steps=(20, 50, 80, 110, 140),
        rom_path=None,
    )
    monkeypatch.setattr(evaluator, "validate_e0b", lambda *a, **k: {"sha256": "e0b"})
    monkeypatch.setattr(evaluator, "environment_config", lambda args: {"test": True})
    screen_results = {
        str(first.resolve()): _screen_result(first, evaluator.checkpoint_sha256(first), 2.0),
        str(second.resolve()): _screen_result(second, evaluator.checkpoint_sha256(second), 1.0),
    }
    monkeypatch.setattr(
        evaluator,
        "screen_candidate",
        lambda path, *a, **k: screen_results[str(Path(k["display_path"]).resolve())],
    )
    monkeypatch.setattr(evaluator, "validate_common_screen", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(
        evaluator,
        "load_candidate",
        lambda path, **k: (
            object(),
            screen_results[str(Path(k["display_path"]).resolve())]["metadata"],
        ),
    )
    fake_random = {"summary": {"mean_controlled_payoff": 1.0}, "protocol": {"passed": True}, "episode_rows": [], "event_rows": []}
    monkeypatch.setattr(evaluator, "evaluate_rows", lambda *a, **k: fake_random)
    monkeypatch.setattr(evaluator, "fixed_grid", lambda *a, **k: [])
    calls = iter((False, True))
    monkeypatch.setattr(evaluator, "behavioral_gate", lambda **k: {"passed": next(calls)})
    copied = []
    monkeypatch.setattr(evaluator, "atomic_copy_no_overwrite", lambda source, destination: copied.append(str(source)) or {"path": str(destination), "sha256": "selected"})
    report = evaluator.run_selection(args)
    assert report["passed"]
    assert len(report["confirmation_attempts"]) == 2
    assert copied
    assert Path(report["selected_alias"]["source_path"]).resolve() == second.resolve()


def test_parser_enforces_exact_disjoint_screen_and_confirmation(tmp_path):
    base = [
        "--role", BUYER,
        "--checkpoint", str(tmp_path / "candidate.zip"),
        "--e0b-checkpoint", str(tmp_path / "e0b.zip"),
        "--selected-checkpoint", str(tmp_path / "selected.zip"),
    ]
    args = evaluator.parse_args(base)
    assert args.screen_episodes == 20
    assert args.confirmation_episodes == 100
    with pytest.raises(SystemExit):
        evaluator.parse_args(base + ["--screen-episodes", "19"])
    with pytest.raises(SystemExit):
        evaluator.parse_args(base + [
            "--screen-seed-start", "100",
            "--confirmation-seed-start", "110",
        ])
    with pytest.raises(SystemExit):
        evaluator.parse_args(base + ["--fixed-eval-values", "0,0.5,1"])
    with pytest.raises(SystemExit):
        evaluator.parse_args(base + [
            "--grid-event-steps", "10,40,70,100,130",
        ])
