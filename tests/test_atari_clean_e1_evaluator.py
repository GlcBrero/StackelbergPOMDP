from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari.sb3_common import (
    ECONOMIC_INIT_ATTRIBUTE,
    PHASE_BALANCED_ACTOR_LOSS_MODE,
)
from stackelberg_pomdp.atari.protocol import ACTOR_STATE, actor_state
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


def test_fifth_economic_override_preserves_game_action_and_only_hits_trade_five():
    action = np.array([3.0, 0.4], dtype=np.float32)
    fifth_trade = {
        ACTOR_STATE: actor_state(
            ammo_fraction=0.0,
            projectile_active=0.0,
            normalized_time=0.975,
            trade_mode=1.0,
            event_index=4,
            opponent_commitment=np.array([0, 0, 0, 0, 0.75]),
        )
    }
    overridden, applied = evaluator.apply_fifth_economic_override(
        fifth_trade, action, 1.0
    )
    assert applied
    assert overridden[0] == action[0]
    assert overridden[1] == 1.0
    assert np.allclose(action, np.array([3.0, 0.4], dtype=np.float32))

    fourth_trade = {
        ACTOR_STATE: actor_state(
            ammo_fraction=0.0,
            projectile_active=0.0,
            normalized_time=0.55,
            trade_mode=1.0,
            event_index=3,
            opponent_commitment=np.array([0, 0, 0, 0, 0.75]),
        )
    }
    unchanged, applied = evaluator.apply_fifth_economic_override(
        fourth_trade, action, 0.0
    )
    assert not applied
    assert np.array_equal(unchanged, action)


def test_episode_audit_verifies_fifth_override_application():
    row = _episode(context=[0, 0, 0, 0, 0.75])
    row["fifth_economic_override"] = 1.0
    row["fifth_economic_override_applied"] = 1
    row["events"][-1]["threshold"] = 1.0
    assert evaluator.audit_episode(row, role=BUYER) == []
    row["fifth_economic_override_applied"] = 0
    fields = {item["field"] for item in evaluator.audit_episode(row, role=BUYER)}
    assert "fifth_economic_override_applied" in fields


def test_paired_timing_confirmation_uses_six_actual_and_four_calibration_runs(
        monkeypatch,
):
    args = SimpleNamespace(fixed_event_steps=None)
    seeds = [101, 102]
    calls = []

    def fake_evaluate_rows(
            model,
            received_args,
            metadata,
            *,
            seeds,
            contexts,
            phase,
            fifth_economic_override=None,
    ):
        del model, metadata
        calls.append({
            "event_steps": tuple(received_args.fixed_event_steps),
            "seeds": tuple(seeds),
            "contexts": tuple(tuple(value) for value in contexts),
            "phase": phase,
            "override": fifth_economic_override,
        })
        return {
            "summary": {
                "mean_controlled_payoff": 1.0,
                "event_5_acceptance_rate": 0.5,
            },
            "protocol": {"passed": True, "violations": []},
            "episode_rows": [],
            "event_rows": [],
        }

    monkeypatch.setattr(evaluator, "evaluate_rows", fake_evaluate_rows)
    results = evaluator.paired_timing_confirmation(
        object(), args, {"sha256": "a"}, seeds=seeds
    )
    assert args.fixed_event_steps is None
    assert len(results) == 10
    assert sum(row["policy_mode"] == "actual" for row in results) == 6
    assert sum(row["policy_mode"] != "actual" for row in results) == 4
    assert {call["event_steps"][-1] for call in calls} == {140, 195}
    assert all(call["event_steps"][:4] == (20, 50, 80, 110) for call in calls)
    assert all(call["seeds"] == tuple(seeds) for call in calls)
    forced = [call for call in calls if call["override"] is not None]
    assert {call["override"] for call in forced} == {0.0, 1.0}
    assert all(np.isclose(call["contexts"][0][-1], 0.75) for call in forced)


def test_paired_timing_csv_artifacts_are_buyer_only(tmp_path):
    buyer = SimpleNamespace(
        output_dir=str(tmp_path), run_name="buyer", role=BUYER
    )
    seller = SimpleNamespace(
        output_dir=str(tmp_path), run_name="seller", role=SELLER
    )
    buyer_paths = evaluator.artifact_paths(buyer)
    seller_paths = evaluator.artifact_paths(seller)
    assert {
        "paired_timing_conditions",
        "paired_timing_episodes",
        "paired_timing_events",
    } <= set(buyer_paths)
    assert not any(name.startswith("paired_timing") for name in seller_paths)


def test_pin_file_detects_source_mutation(monkeypatch, tmp_path):
    source = tmp_path / "source.zip"
    destination = tmp_path / "pinned.zip"
    source.write_bytes(b"source")
    values = iter(("a" * 64, "b" * 64, "a" * 64))
    monkeypatch.setattr(evaluator, "checkpoint_sha256", lambda path: next(values))
    with pytest.raises(RuntimeError, match="changed while being pinned"):
        evaluator.pin_file(source, destination)
    assert not destination.exists()


def test_candidate_metadata_records_loss_and_initialization_with_legacy_defaults(
        monkeypatch, tmp_path
):
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.write_bytes(b"candidate")

    class FakePolicy:
        economic_role = BUYER
        economic_input_mode = "full"
        pretrained_lr_scale = 0.1
        visual_features = 512
        state_features = 64
        economic_hidden = 64
        critic_hidden = 256

        def set_training_mode(self, mode):
            assert mode is False

    model = SimpleNamespace(
        policy=FakePolicy(),
        atari_e1_source_provenance={
            "sha256": "e" * 64,
            "source_economic_role": "gameplay",
            "source_economic_input_mode": "full",
            "source_pretrained_lr_scale": 0.1,
            "zero_initialized_actor_state_indices": [9, 10, 11, 12, 13],
        },
        n_steps=205,
        gamma=1.0,
        gae_lambda=1.0,
        num_timesteps=205,
        seed=1,
        learning_rate=1.0e-4,
        batch_size=205,
        n_epochs=4,
        clip_range=lambda _: 0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    monkeypatch.setattr(evaluator, "StackPOMDPAtariPolicy", FakePolicy)
    monkeypatch.setattr(evaluator, "_load_model", lambda *a, **k: model)

    _, legacy = evaluator.load_candidate(
        checkpoint,
        role=BUYER,
        e0b_sha256="e" * 64,
    )
    assert legacy["training_config"]["actor_loss_mode"] == "standard"
    assert legacy["training_config"]["economic_head_initialization"] == {
        "mean": 0.95,
        "concentration": 10.0,
    }
    assert legacy["atari_e1_sampler_provenance"]["mode"] == "uniform"
    assert legacy["atari_e1_sampler_history"] == [{
        "start_total_timesteps": 0,
        "sampler": legacy["atari_e1_sampler_provenance"],
        "inferred_for_legacy_checkpoint": True,
        "resume_sources": [],
    }]
    assert legacy["sampler_contract_inferred_for_legacy_checkpoint"]

    model.atari_actor_loss_mode = PHASE_BALANCED_ACTOR_LOSS_MODE
    setattr(model, ECONOMIC_INIT_ATTRIBUTE, {
        "mean": 0.9,
        "concentration": 8.0,
    })
    _, explicit = evaluator.load_candidate(
        checkpoint,
        role=BUYER,
        e0b_sha256="e" * 64,
    )
    assert explicit["training_config"]["actor_loss_mode"] == "balanced"
    assert explicit["training_config"]["economic_head_initialization"] == {
        "mean": 0.9,
        "concentration": 8.0,
    }

    del model.atari_actor_loss_mode
    delattr(model, ECONOMIC_INIT_ATTRIBUTE)
    model.policy.economic_role = SELLER
    _, legacy_seller = evaluator.load_candidate(
        checkpoint,
        role=SELLER,
        e0b_sha256="e" * 64,
    )
    assert legacy_seller["training_config"][
        "economic_head_initialization"
    ] == {"mean": 0.5, "concentration": 2.0}

    model.policy.economic_role = BUYER
    model.atari_actor_loss_mode = "unknown"
    with pytest.raises(ValueError, match="unknown actor-loss mode"):
        evaluator.load_candidate(
            checkpoint,
            role=BUYER,
            e0b_sha256="e" * 64,
        )

    model.atari_actor_loss_mode = PHASE_BALANCED_ACTOR_LOSS_MODE
    setattr(model, ECONOMIC_INIT_ATTRIBUTE, {
        "mean": 1.0,
        "concentration": 8.0,
    })
    with pytest.raises(ValueError, match="metadata is invalid"):
        evaluator.load_candidate(
            checkpoint,
            role=BUYER,
            e0b_sha256="e" * 64,
        )


def _temporal_sampler_contract(*, source_digest="a" * 64):
    uniform = evaluator.trainer.e1_sampler_provenance(
        evaluator.trainer.UNIFORM_E1_SAMPLER,
        gameplay_horizon=200,
        event_tail_steps=0,
    )
    temporal = evaluator.trainer.e1_sampler_provenance(
        evaluator.trainer.TEMPORAL_MIX_E1_SAMPLER,
        gameplay_horizon=200,
        event_tail_steps=0,
    )
    history = [
        {
            "start_total_timesteps": 0,
            "sampler": uniform,
            "inferred_for_legacy_checkpoint": True,
            "resume_sources": [],
        },
        {
            "start_total_timesteps": 205,
            "sampler": temporal,
            "inferred_for_legacy_checkpoint": False,
            "resume_sources": [{
                "path": "/immutable/uniform_parent.zip",
                "sha256": source_digest,
                "training_total_timesteps": 205,
                "resume_total_timesteps": 205,
            }],
        },
    ]
    return temporal, history


def test_candidate_sampler_contract_validates_and_records_full_history():
    temporal, history = _temporal_sampler_contract()
    model = SimpleNamespace(
        num_timesteps=820,
        atari_e1_sampler_provenance=temporal,
        atari_e1_sampler_history=history,
    )
    result = evaluator.candidate_sampler_contract(model)
    assert result["atari_e1_sampler_provenance"] == temporal
    assert result["atari_e1_sampler_history"] == history
    assert not result["sampler_contract_inferred_for_legacy_checkpoint"]

    model.atari_e1_sampler_provenance = None
    with pytest.raises(ValueError, match="provenance and history together"):
        evaluator.candidate_sampler_contract(model)

    model.atari_e1_sampler_provenance = temporal
    model.atari_e1_sampler_history[-1]["sampler"] = (
        evaluator.trainer.e1_sampler_provenance(
            evaluator.trainer.UNIFORM_E1_SAMPLER,
            gameplay_horizon=200,
            event_tail_steps=0,
        )
    )
    with pytest.raises(ValueError, match="final history stage"):
        evaluator.candidate_sampler_contract(model)


def _family_result(contract):
    return {
        "metadata": {
            "training_config": {"algorithm": "PPO", "seed": 1},
            **contract,
        },
        "episode_rows": [{}],
    }


def test_common_training_family_rejects_sampler_mode_or_history_mixing():
    legacy_model = SimpleNamespace(num_timesteps=820)
    uniform = evaluator.candidate_sampler_contract(legacy_model)
    temporal_provenance, temporal_history = _temporal_sampler_contract()
    temporal = {
        "atari_e1_sampler_provenance": temporal_provenance,
        "atari_e1_sampler_history": temporal_history,
        "sampler_contract_inferred_for_legacy_checkpoint": False,
    }

    family = evaluator.common_training_family([
        _family_result(temporal), _family_result(temporal)
    ])
    assert family["common_sampler_provenance"] == temporal_provenance
    assert family["common_sampler_history"] == temporal_history

    with pytest.raises(ValueError, match="sampler provenance/history"):
        evaluator.common_training_family([
            _family_result(uniform), _family_result(temporal)
        ])

    _, different_history = _temporal_sampler_contract(source_digest="b" * 64)
    with pytest.raises(ValueError, match="sampler provenance/history"):
        evaluator.common_training_family([
            _family_result(temporal),
            _family_result({
                **temporal,
                "atari_e1_sampler_history": different_history,
            }),
        ])


def _screen_result(path, digest, payoff, *, valid=True, timestep=100):
    row = _episode(seed=1)
    sampler_contract = evaluator.candidate_sampler_contract(
        SimpleNamespace(num_timesteps=timestep)
    )
    return {
        "metadata": {
            "path": str(path),
            "sha256": digest,
            "training_timesteps": timestep,
            "training_config": {"algorithm": "PPO", "seed": 1},
            **sampler_contract,
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


def _timing_results():
    rows = []
    for timing in ("early", "late"):
        for price in evaluator.CANONICAL_TIMING_PRICES:
            rows.append({
                "timing": timing,
                "fifth_price": price,
                "policy_mode": "actual",
                "summary": {
                    "mean_controlled_payoff": (
                        1.95 if timing == "early" else 0.95
                    ),
                    "event_5_acceptance_rate": (
                        1.0 if timing == "early" else 0.0
                    ),
                },
                "protocol": {"passed": True, "violations": []},
            })
        rows.extend([
            {
                "timing": timing,
                "fifth_price": 0.75,
                "policy_mode": "forced_buy",
                "summary": {
                    "mean_controlled_payoff": (
                        2.0 if timing == "early" else 0.0
                    ),
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
        ])
    return rows


def test_buyer_behavior_gate_requires_low_price_use_and_high_price_rejection():
    fixed = [
        _fixed(0.0, mean_purchases=5.0, mean_buyer_shots_fired=5.0),
        _fixed(0.5, mean_purchases=5.0, mean_buyer_shots_fired=5.0, mean_controlled_payoff=2.5),
        _fixed(1.0, mean_purchases=0.0, mean_buyer_shots_fired=0.0, mean_controlled_payoff=0.0),
    ]
    random = {
        "summary": {"mean_controlled_payoff": 1.0},
        "protocol": {"passed": True},
    }
    timing = _timing_results()
    gate = evaluator.behavioral_gate(
        role=BUYER,
        random_result=random,
        fixed_results=fixed,
        timing_results=timing,
    )
    assert gate["passed"]
    assert gate["data_calibration_passed"]
    assert gate["timing_behavior_passed"]
    next(
        row for row in timing
        if row["timing"] == "late"
        and row["fifth_price"] == 0.75
        and row["policy_mode"] == "actual"
    )["summary"]["event_5_acceptance_rate"] = 1.0
    assert not evaluator.behavioral_gate(
        role=BUYER,
        random_result=random,
        fixed_results=fixed,
        timing_results=timing,
    )["passed"]
    timing = _timing_results()
    fixed[-1]["summary"]["mean_purchases"] = 5.0
    assert not evaluator.behavioral_gate(
        role=BUYER,
        random_result=random,
        fixed_results=fixed,
        timing_results=timing,
    )["passed"]


def test_buyer_behavior_gate_requires_paired_timing_calibration_and_regret():
    fixed = [
        _fixed(0.0, mean_purchases=5.0, mean_buyer_shots_fired=5.0),
        _fixed(
            0.5,
            mean_purchases=5.0,
            mean_buyer_shots_fired=5.0,
            mean_controlled_payoff=2.5,
        ),
        _fixed(
            1.0,
            mean_purchases=0.0,
            mean_buyer_shots_fired=0.0,
            mean_controlled_payoff=0.0,
        ),
    ]
    random = {
        "summary": {"mean_controlled_payoff": 1.0},
        "protocol": {"passed": True},
    }
    timing = _timing_results()
    forced_early_buy = next(
        row for row in timing
        if row["timing"] == "early" and row["policy_mode"] == "forced_buy"
    )
    forced_early_buy["summary"]["mean_controlled_payoff"] = 1.1
    gate = evaluator.behavioral_gate(
        role=BUYER,
        random_result=random,
        fixed_results=fixed,
        timing_results=timing,
    )
    assert not gate["passed"]
    assert not gate["data_calibration_passed"]

    timing = _timing_results()
    early_actual = next(
        row for row in timing
        if row["timing"] == "early"
        and row["fifth_price"] == 0.75
        and row["policy_mode"] == "actual"
    )
    early_actual["summary"]["mean_controlled_payoff"] = 1.0
    gate = evaluator.behavioral_gate(
        role=BUYER,
        random_result=random,
        fixed_results=fixed,
        timing_results=timing,
    )
    assert not gate["passed"]
    assert not gate["timing_behavior_passed"]


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


def test_selection_never_falls_back_after_screen_winner_fails(monkeypatch, tmp_path):
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
        timing_seed_start=500,
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
    monkeypatch.setattr(
        evaluator, "paired_timing_confirmation", lambda *a, **k: []
    )
    calls = []
    monkeypatch.setattr(
        evaluator,
        "behavioral_gate",
        lambda **k: calls.append(k) or {"passed": False},
    )
    copied = []
    monkeypatch.setattr(evaluator, "atomic_copy_no_overwrite", lambda source, destination: copied.append(str(source)) or {"path": str(destination), "sha256": "selected"})
    report = evaluator.run_selection(args)
    assert not report["passed"]
    assert len(report["confirmation_attempts"]) == 1
    assert len(calls) == 1
    assert not copied
    assert report["selected_alias"] is None
    assert report["selection"] == {
        "fallback_allowed": False,
        "screen_selected_checkpoint_sha256": evaluator.checkpoint_sha256(first),
        "selected_checkpoint_sha256": None,
    }


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
    assert args.timing_episodes == 20
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
    with pytest.raises(SystemExit):
        evaluator.parse_args(base + [
            "--fixed-seed-start", "100",
            "--timing-seed-start", "110",
        ])
