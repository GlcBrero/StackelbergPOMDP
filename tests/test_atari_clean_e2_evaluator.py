import csv
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replication.atari import evaluate_atari_stackpomdp_leader_sb3 as evaluator
from replication.atari import train_atari_stackpomdp_leader_sb3 as trainer
from replication.atari.sb3_common import (
    PHASE_BALANCED_ACTOR_LOSS_MODE,
    STANDARD_ACTOR_LOSS_MODE,
)
from stackelberg_pomdp.atari.protocol import (
    CACHED_TRADE_REPLAY,
    GAMEPLAY,
    LEADER_QUERY,
)


def _episode_row(
        episode=0,
        *,
        checkpoint_hash="a" * 64,
        trace="b" * 64,
        intervention_id="factual",
        commitment_override=None,
):
    events = []
    for event, game_step in enumerate((0, 1, 2, 3, 6)):
        events.append({
            "event_index": event,
            "game_step": game_step,
            "price": 0.2,
            "threshold": 0.5,
            "accepted": True,
            "seller_ammo_before": 1,
            "buyer_ammo_before": 0,
            "seller_ammo_after": 0,
            "buyer_ammo_after": 1,
        })
    return {
        "phase": "screen",
        "checkpoint_path": "/tmp/candidate.zip",
        "checkpoint_sha256": checkpoint_hash,
        "response_sha256": "c" * 64,
        "environment_config_sha256": "d" * 64,
        "evaluation_episode": episode,
        "evaluation_seed": 100 + episode,
        "evaluation_return": 1.0,
        "evaluation_steps": 17,
        "economic_intervention_id": intervention_id,
        "economic_intervention_sha256": "9" * 64,
        "economic_commitment_override": commitment_override,
        "economic_override_query_applications": (
            0 if commitment_override is None else 5
        ),
        "economic_override_replay_applications": (
            0 if commitment_override is None else 5
        ),
        "economic_override_query_event_counts": (
            [0] * 5 if commitment_override is None else [1] * 5
        ),
        "economic_override_replay_event_counts": (
            [0] * 5 if commitment_override is None else [1] * 5
        ),
        "terminal_episode_summary": {"r": 1.0, "l": 17},
        "leader_role": "seller",
        "follower_role": "buyer",
        "event_steps": [0, 1, 2, 3, 6],
        "events": events,
        "gameplay_transitions": 7,
        "trade_transitions": 5,
        "reward_transition_count": 12,
        "query_transitions": 5,
        "outer_transition_count": 17,
        "cache_hits": 5,
        "bullets_arrived": 5,
        "purchases": 5,
        "payments": 1.0,
        "seller_game_reward": 0.0,
        "buyer_game_reward": 5.0,
        "seller_reward": 1.0,
        "buyer_reward": 4.0,
        "leader_reward": 1.0,
        "seller_shots_fired": 0,
        "buyer_shots_fired": 5,
        "seller_final_ammo": 0,
        "buyer_final_ammo": 0,
        "seller_emulator_step_calls": 7,
        "buyer_emulator_step_calls": 7,
        "seller_bullet_error": 0,
        "buyer_bullet_error": 0,
        "seller_payoff_error": 0.0,
        "buyer_payoff_error": 0.0,
        "query_actions": [[0.0, 0.2]] * 5,
        "query_trace_sha256": trace,
        "leader_commitment": [0.2] * 5,
        "follower_actions": [[1.0, 0.5]] * 5,
        "response_algorithm": "frozen_meta_policy",
    }


def _transition(
        episode,
        index,
        substep,
        *,
        reward=0.0,
        query_index=None,
        event_index=None,
        actor_hash="game",
        requested_action=(0.0, 0.2),
        game_step=None,
        done=False,
):
    canonical_event = query_index if substep == LEADER_QUERY else event_index
    actor_state = (
        evaluator.canonical_leader_state(canonical_event).tolist()
        if substep in (LEADER_QUERY, CACHED_TRADE_REPLAY)
        else [0.0] * 14
    )
    return {
        "phase": "screen",
        "checkpoint_path": "/tmp/candidate.zip",
        "checkpoint_sha256": "a" * 64,
        "response_sha256": "c" * 64,
        "environment_config_sha256": "d" * 64,
        "evaluation_episode": episode,
        "evaluation_seed": 100 + episode,
        "transition_index": index,
        "substep_type": substep,
        "reward": reward,
        "cumulative_return": 0.0,
        "done": done,
        "is_reward_phase": substep != LEADER_QUERY,
        "reward_generated": True,
        "emulator_advanced": substep == GAMEPLAY,
        "action_credit": {
            LEADER_QUERY: [0.0, 1.0],
            GAMEPLAY: [1.0, 0.0],
            CACHED_TRADE_REPLAY: [0.0, 0.0],
        }[substep],
        "actor_state": actor_state,
        "action_mask": (
            list(evaluator.CANONICAL_EVENT_ACTION_MASK)
            if substep in (LEADER_QUERY, CACHED_TRADE_REPLAY)
            else [1.0] * 6
        ),
        "actor_image_nonzero": 0,
        "actor_observation_sha256": actor_hash,
        "economic_intervention_id": "factual",
        "economic_intervention_sha256": "9" * 64,
        "economic_commitment_override": None,
        "economic_override_applied": False,
        "economic_override_kind": None,
        "economic_override_event": None,
        "policy_action_before_intervention": list(requested_action),
        "requested_action": list(requested_action),
        "query_index": query_index,
        "event_index": event_index,
        "cache_hit": substep == CACHED_TRADE_REPLAY,
        "leader_executed_action": (
            list(requested_action) if substep == CACHED_TRADE_REPLAY else None
        ),
        "follower_action": (
            [1.0, 0.5] if substep == CACHED_TRADE_REPLAY else None
        ),
        "game_step": game_step,
        "next_event": None,
    }


def _valid_evaluation(episodes=1):
    episode_rows = []
    transitions = []
    decisions = []
    schedule = (0, 1, 2, 3, 6)
    for episode in range(episodes):
        episode_rows.append(_episode_row(episode))
        index = 0
        for event in range(5):
            row = _transition(
                episode,
                index,
                LEADER_QUERY,
                query_index=event,
                actor_hash=f"query-{event}",
            )
            transitions.append(row)
            decisions.append(dict(row))
            index += 1
        next_event = 0
        for game_step in range(7):
            if next_event < 5 and schedule[next_event] == game_step:
                row = _transition(
                    episode,
                    index,
                    CACHED_TRADE_REPLAY,
                    reward=0.2,
                    event_index=next_event,
                    actor_hash=f"query-{next_event}",
                    game_step=game_step,
                )
                transitions.append(row)
                decisions.append(dict(row))
                index += 1
                next_event += 1
            transitions.append(_transition(
                episode,
                index,
                GAMEPLAY,
                requested_action=(0.0, 0.5),
                done=(game_step == 6),
            ))
            index += 1
    return {
        "episode_rows": episode_rows,
        "transition_rows": transitions,
        "decision_rows": decisions,
    }


def _refresh_decision_rows(evaluation):
    evaluation["decision_rows"] = [
        dict(row) for row in evaluation["transition_rows"]
        if row["substep_type"] in (LEADER_QUERY, CACHED_TRADE_REPLAY)
    ]


def _buyer_evaluation():
    evaluation = _valid_evaluation()
    episode = evaluation["episode_rows"][0]
    episode.update({
        "leader_role": "buyer",
        "follower_role": "seller",
        "leader_reward": 4.0,
        "evaluation_return": 4.0,
        "terminal_episode_summary": {"r": 4.0, "l": 17},
        "query_actions": [[0.0, 0.5]] * 5,
        "leader_commitment": [0.5] * 5,
        "follower_actions": [[1.0, 0.2]] * 5,
    })
    gameplay = [
        row for row in evaluation["transition_rows"]
        if row["substep_type"] == GAMEPLAY
    ]
    gameplay[-1]["reward"] = 5.0
    for row in evaluation["transition_rows"]:
        if row["substep_type"] in (LEADER_QUERY, CACHED_TRADE_REPLAY):
            row["policy_action_before_intervention"] = [0.0, 0.5]
            row["requested_action"] = [0.0, 0.5]
        if row["substep_type"] == CACHED_TRADE_REPLAY:
            row["leader_executed_action"] = [0.0, 0.5]
            row["reward"] = -0.2
    _refresh_decision_rows(evaluation)
    return evaluation


def _candidate(
        name,
        *,
        mean,
        median=None,
        minimum=None,
        std=0.0,
        timesteps=100,
        passed=True,
        digest=None,
):
    digest = digest or (name[0] * 64)
    return {
        "checkpoint_id": name,
        "checkpoint_path": f"/tmp/{name}.zip",
        "checkpoint_sha256": digest,
        "training_total_timesteps": timesteps,
        "summary": {
            "episodes": 20,
            "mean_leader_payoff": mean,
            "median_leader_payoff": mean if median is None else median,
            "min_leader_payoff": mean if minimum is None else minimum,
            "max_leader_payoff": mean,
            "std_leader_payoff": std,
        },
        "protocol": {"passed": passed, "violations": [] if passed else [{}]},
        "economic_gate": {
            "passed": passed,
            "checks": [],
        },
    }


def _confirmation_control(intervention_id, value, *, trace):
    return {
        "economic_intervention": {
            "intervention_id": intervention_id,
            "economic_commitment": [value] * 5,
            "manifest_sha256": ("0" if value == 0.0 else "1") * 64,
        },
        "protocol": {
            "passed": True,
            "query_trace_sha256": trace,
            "leader_commitment": [value] * 5,
            "full_query_actions": [[0.0, value]] * 5,
        },
    }


def _gate_result(intervention_id, payoffs, *, value=None, shots=4.0):
    commitment = None if value is None else [float(value)] * 5
    manifest_sha256 = intervention_id[0] * 64
    rows = []
    for episode, payoff in enumerate(payoffs):
        rows.append({
            "evaluation_seed": 1_000 + episode,
            "event_steps": [0, 20, 40, 60, 80],
            "leader_reward": float(payoff),
            "purchases": 4.0,
            "payments": 2.0,
            "seller_shots_fired": float(shots) / 2.0,
            "buyer_shots_fired": float(shots) / 2.0,
            "economic_intervention_id": intervention_id,
            "economic_intervention_sha256": manifest_sha256,
            "economic_commitment_override": commitment,
        })
    return {
        "checkpoint_path": "/tmp/candidate.zip",
        "checkpoint_sha256": "a" * 64,
        "response_checkpoint_sha256": "b" * 64,
        "environment_config_sha256": "c" * 64,
        "e2_provenance_fingerprint": "d" * 64,
        "seed_start": 1_000,
        "seed_end": 1_000 + len(rows) - 1,
        "phase": "screen",
        "protocol": {"passed": True, "violations": []},
        "summary": {
            "episodes": len(rows),
            "mean_leader_payoff": float(np.mean(payoffs)),
        },
        "economic_intervention": {
            "intervention_id": intervention_id,
            "economic_commitment": commitment,
            "manifest_sha256": manifest_sha256,
        },
        "episode_rows": rows,
    }


def test_exact_210_analogue_protocol_audits_query_cache_trade_and_payoffs():
    evaluation = _valid_evaluation(episodes=2)
    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=2,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert protocol["passed"]
    assert protocol["violations"] == []
    assert protocol["single_query_trace"]
    assert protocol["single_leader_commitment"]
    assert protocol["leader_commitment"] == [0.2] * 5


def test_protocol_rejects_cache_mismatch_and_payoff_accounting_error():
    evaluation = _valid_evaluation()
    replay = next(
        row for row in evaluation["transition_rows"]
        if row["substep_type"] == CACHED_TRADE_REPLAY
    )
    replay["requested_action"] = [0.0, 0.7]
    _refresh_decision_rows(evaluation)
    evaluation["episode_rows"][0]["seller_reward"] = 1.1

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=1,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert not protocol["passed"]
    fields = {row["field"] for row in protocol["violations"]}
    assert "seller payoff identity" in fields
    assert "event 0 requested action cache identity" in fields


def test_protocol_rejects_decision_row_not_copied_from_transition_subset():
    evaluation = _valid_evaluation()
    evaluation["decision_rows"][0]["requested_action"] = [0.0, 0.7]

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=1,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert not protocol["passed"]
    assert any(
        row["field"] == "decision rows equal economic transition subset"
        for row in protocol["violations"]
    )


def test_protocol_binds_buyer_threshold_and_seller_response_price():
    evaluation = _buyer_evaluation()

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=1,
        leader_role="buyer",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert protocol["passed"]


@pytest.mark.parametrize("leader_role", ["seller", "buyer"])
def test_protocol_rejects_delayed_instead_of_immediate_trade_payment(
        leader_role):
    evaluation = (
        _valid_evaluation()
        if leader_role == "seller"
        else _buyer_evaluation()
    )
    replay = next(
        row for row in evaluation["transition_rows"]
        if row["substep_type"] == CACHED_TRADE_REPLAY
        and row["event_index"] == 0
    )
    gameplay = [
        row for row in evaluation["transition_rows"]
        if row["substep_type"] == GAMEPLAY
    ]
    displaced = float(replay["reward"])
    replay["reward"] = 0.0
    gameplay[-1]["reward"] += displaced
    _refresh_decision_rows(evaluation)

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=1,
        leader_role=leader_role,
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert not protocol["passed"]
    assert any(
        row["field"] == f"event 0 immediate {leader_role} trade reward"
        for row in protocol["violations"]
    )


def test_protocol_rejects_unreported_factual_economic_override():
    evaluation = _valid_evaluation()
    for row in evaluation["transition_rows"]:
        if row["substep_type"] in (LEADER_QUERY, CACHED_TRADE_REPLAY):
            row["requested_action"] = [0.0, 0.7]
            if row["substep_type"] == CACHED_TRADE_REPLAY:
                row["leader_executed_action"] = [0.0, 0.7]
    _refresh_decision_rows(evaluation)

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=1,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert not protocol["passed"]
    fields = {row["field"] for row in protocol["violations"]}
    assert "transition 0 no economic override action identity" in fields


def test_protocol_rejects_noncanonical_event_mask_across_episodes():
    evaluation = _valid_evaluation(episodes=2)
    for row in evaluation["transition_rows"]:
        if (
                row["evaluation_episode"] == 1
                and row["substep_type"] in (
                    LEADER_QUERY, CACHED_TRADE_REPLAY
                )
        ):
            # Enables RIGHTFIRE and therefore differs from the event-only mask.
            row["action_mask"] = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    _refresh_decision_rows(evaluation)

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=2,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert not protocol["passed"]
    assert protocol["canonical_event_action_mask"] == list(
        evaluator.CANONICAL_EVENT_ACTION_MASK
    )
    assert any(
        "canonical event-only action mask" in row["field"]
        and row["evaluation_episode"] == 1
        for row in protocol["violations"]
    )


def test_protocol_rejects_multiple_deterministic_commitments_or_traces():
    evaluation = _valid_evaluation(episodes=2)
    evaluation["episode_rows"][1]["leader_commitment"][4] = 0.3
    evaluation["episode_rows"][1]["query_actions"][4][1] = 0.3
    evaluation["episode_rows"][1]["query_trace_sha256"] = "e" * 64

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=2,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert not protocol["passed"]
    assert not protocol["single_query_trace"]
    assert not protocol["single_leader_commitment"]
    assert not protocol["single_full_query_action_trace"]


def test_commitment_override_changes_only_economics_at_query_and_replay():
    commitment = [0.1, 0.2, 0.3, 0.4, 0.5]
    query = {
        evaluator.ACTOR_STATE: evaluator.canonical_leader_state(2),
        evaluator.ACTION_CREDIT: np.asarray([0.0, 1.0], dtype=np.float32),
    }
    replay = {
        evaluator.ACTOR_STATE: evaluator.canonical_leader_state(2),
        evaluator.ACTION_CREDIT: np.asarray([0.0, 0.0], dtype=np.float32),
    }
    gameplay = {
        evaluator.ACTOR_STATE: np.zeros(14, dtype=np.float32),
        evaluator.ACTION_CREDIT: np.asarray([1.0, 0.0], dtype=np.float32),
    }

    query_action, query_tag = evaluator.apply_economic_commitment_override(
        query, [4.0, 0.9], commitment
    )
    replay_action, replay_tag = evaluator.apply_economic_commitment_override(
        replay, [4.0, 0.9], commitment
    )
    gameplay_action, gameplay_tag = (
        evaluator.apply_economic_commitment_override(
            gameplay, [3.0, 0.9], commitment
        )
    )

    assert query_action == replay_action == [4.0, 0.3]
    assert query_tag == (LEADER_QUERY, 2)
    assert replay_tag == (CACHED_TRADE_REPLAY, 2)
    assert gameplay_action == [3.0, 0.9]
    assert gameplay_tag is None
    manifest = evaluator.economic_intervention_manifest(
        intervention_id="all_zero",
        commitment=[0.0] * 5,
        checkpoint_hash="a" * 64,
        response_hash="b" * 64,
        config_hash="c" * 64,
        provenance_fingerprint="d" * 64,
    )
    assert manifest["no_direct_gameplay_action_override"] is True
    assert "preserve_candidate_gameplay_actions" not in manifest


def test_protocol_audits_effective_override_and_preintervention_cache():
    evaluation = _valid_evaluation()
    episode = evaluation["episode_rows"][0]
    episode.update({
        "evaluation_return": 0.0,
        "terminal_episode_summary": {"r": 0.0, "l": 17},
        "payments": 0.0,
        "seller_reward": 0.0,
        "buyer_reward": 5.0,
        "leader_reward": 0.0,
        "query_actions": [[0.0, 0.0]] * 5,
        "leader_commitment": [0.0] * 5,
        "economic_intervention_id": "all_zero",
        "economic_intervention_sha256": "0" * 64,
        "economic_commitment_override": [0.0] * 5,
        "economic_override_query_applications": 5,
        "economic_override_replay_applications": 5,
        "economic_override_query_event_counts": [1] * 5,
        "economic_override_replay_event_counts": [1] * 5,
    })
    for event in episode["events"]:
        event["price"] = 0.0
    for row in evaluation["transition_rows"]:
        row.update({
            "economic_intervention_id": "all_zero",
            "economic_intervention_sha256": "0" * 64,
            "economic_commitment_override": [0.0] * 5,
        })
        if row["substep_type"] in (LEADER_QUERY, CACHED_TRADE_REPLAY):
            event_index = (
                row["query_index"]
                if row["substep_type"] == LEADER_QUERY
                else row["event_index"]
            )
            row.update({
                "economic_override_applied": True,
                "economic_override_kind": row["substep_type"],
                "economic_override_event": event_index,
                "requested_action": [0.0, 0.0],
            })
            if row["substep_type"] == CACHED_TRADE_REPLAY:
                row["leader_executed_action"] = [0.0, 0.0]
                row["reward"] = 0.0
    _refresh_decision_rows(evaluation)

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=1,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert protocol["passed"]


def test_paired_economic_gate_requires_endpoint_dominance_and_four_shots():
    factual = _gate_result("factual", [1.0] * 5, shots=4.0)
    controls = [
        _gate_result("all_zero", [0.0] * 5, value=0.0),
        _gate_result("all_one", [0.5] * 5, value=1.0),
    ]

    passed = evaluator.paired_economic_gate(
        factual, controls, required_episodes=5
    )
    assert passed["passed"]
    assert "not a universal equilibrium condition" in passed["hypothesis"]
    assert len(passed["paired_rows"]) == 5

    weak = deepcopy(controls)
    for row in weak[1]["episode_rows"]:
        row["leader_reward"] = 0.8
    failed_advantage = evaluator.paired_economic_gate(
        factual, weak, required_episodes=5
    )
    assert not failed_advantage["passed"]
    assert any(
        row["name"] == "mean paired advantage over all_one"
        and not row["passed"]
        for row in failed_advantage["checks"]
    )

    sparse = deepcopy(controls)
    sparse_payoffs = [-0.25, 1.0, 1.0, 1.0, 1.0]
    for row, payoff in zip(sparse[1]["episode_rows"], sparse_payoffs):
        row["leader_reward"] = payoff
    failed_win_rate = evaluator.paired_economic_gate(
        factual, sparse, required_episodes=5
    )
    assert not failed_win_rate["passed"]
    assert any(
        row["name"] == "paired win rate over all_one"
        and not row["passed"]
        for row in failed_win_rate["checks"]
    )

    low_shots = _gate_result("factual", [1.0] * 5, shots=3.9)
    failed_shots = evaluator.paired_economic_gate(
        low_shots, controls, required_episodes=5
    )
    assert not failed_shots["passed"]
    assert any(
        row["name"] == "factual mean total bullets fired"
        and not row["passed"]
        for row in failed_shots["checks"]
    )


def test_paired_economic_gate_rejects_unmatched_schedule_or_provenance():
    factual = _gate_result("factual", [1.0] * 5)
    controls = [
        _gate_result("all_zero", [0.0] * 5, value=0.0),
        _gate_result("all_one", [0.5] * 5, value=1.0),
    ]
    controls[0]["episode_rows"][0]["event_steps"][-1] = 81
    controls[1]["response_checkpoint_sha256"] = "e" * 64

    gate = evaluator.paired_economic_gate(
        factual, controls, required_episodes=5
    )

    assert not gate["passed"]
    fields = {row["field"] for row in gate["errors"]}
    assert "all_zero.event_steps" in fields
    assert "all_one.response_checkpoint_sha256" in fields


def test_outcome_summary_reports_shots_ammo_and_safe_accepted_price():
    row = _episode_row()
    summary = evaluator.leader_outcome_summary([row])
    assert summary["mean_seller_shots_fired"] == 0.0
    assert summary["mean_buyer_shots_fired"] == 5.0
    assert summary["mean_total_shots_fired"] == 5.0
    assert summary["mean_seller_final_ammo"] == 0.0
    assert summary["mean_buyer_final_ammo"] == 0.0
    assert summary["mean_accepted_price"] == pytest.approx(0.2)
    assert summary["mean_purchase_rate"] == 1.0
    assert summary["buyer_purchased_bullet_utilization"] == 1.0
    assert summary["seller_retained_bullet_utilization"] == 0.0

    row["purchases"] = 0
    row["payments"] = 0.0
    row["seller_shots_fired"] = 4
    no_trade = evaluator.leader_outcome_summary([row])
    assert no_trade["mean_accepted_price"] == 0.0
    assert no_trade["buyer_purchased_bullet_utilization"] == 0.0
    assert no_trade["seller_retained_bullet_utilization"] == pytest.approx(0.8)


def test_selection_excludes_invalid_then_applies_documented_tiebreaks():
    invalid_high = _candidate("invalid", mean=100.0, passed=False, digest="f" * 64)
    economic_failure = _candidate(
        "economic_failure", mean=200.0, passed=True, digest="9" * 64
    )
    economic_failure["economic_gate"]["passed"] = False
    unstable = _candidate(
        "unstable", mean=4.0, median=4.0, minimum=2.0, std=1.0,
        digest="e" * 64,
    )
    robust_late = _candidate(
        "robust_late", mean=4.0, median=4.0, minimum=3.0, std=0.5,
        timesteps=200, digest="d" * 64,
    )
    robust_early = _candidate(
        "robust_early", mean=4.0, median=4.0, minimum=3.0, std=0.5,
        timesteps=100, digest="c" * 64,
    )

    ranked = evaluator.rank_candidates([
        invalid_high, economic_failure, unstable, robust_late, robust_early
    ])

    assert ranked["selected_checkpoint_sha256"] == "c" * 64
    assert ranked["eligible_checkpoints"] == 3
    assert ranked["ranking_rows"][0]["checkpoint_id"] == "robust_early"
    assert sum(not row["eligible"] for row in ranked["ranking_rows"]) == 2
    assert ranked["selection_rule"] == list(evaluator.SELECTION_RULE)


def test_common_screen_requires_identical_seed_schedule_pairs():
    left = {
        "response_checkpoint_sha256": "c" * 64,
        "environment_config_sha256": "d" * 64,
        "e2_provenance_fingerprint": "e" * 64,
        "seed_start": 100,
        "seed_end": 101,
        "episode_rows": [
            _episode_row(0),
            _episode_row(1),
        ]
    }
    right = deepcopy(left)
    assert evaluator.validate_common_screen([left, right])["passed"]

    right["episode_rows"][1]["event_steps"][-1] = 5
    with pytest.raises(RuntimeError, match="different event schedules"):
        evaluator.validate_common_screen([left, right])

    duplicate = deepcopy(left)
    duplicate["episode_rows"][1]["evaluation_seed"] = (
        duplicate["episode_rows"][0]["evaluation_seed"]
    )
    with pytest.raises(RuntimeError, match="duplicate evaluation seeds"):
        evaluator.validate_common_screen([duplicate])

    other_provenance = deepcopy(left)
    other_provenance["e2_provenance_fingerprint"] = "f" * 64
    with pytest.raises(RuntimeError, match="different E2 scientific provenance"):
        evaluator.validate_common_screen([left, other_provenance])


def test_selected_alias_is_exact_and_never_overwrites(tmp_path):
    source = tmp_path / "step200.zip"
    source.write_bytes(b"checkpoint bytes")
    target = tmp_path / "selected.zip"

    copied = evaluator.atomic_copy_no_overwrite(source, target)

    assert target.read_bytes() == source.read_bytes()
    assert copied["copy_verified"]
    assert copied["checkpoint_sha256"] == evaluator.checkpoint_sha256(source)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.atomic_copy_no_overwrite(source, target)


def test_selected_alias_verification_failure_removes_only_its_new_copy(
        tmp_path, monkeypatch):
    source = tmp_path / "step200.zip"
    source.write_bytes(b"checkpoint bytes")
    target = tmp_path / "selected.zip"
    real_sha256 = evaluator.checkpoint_sha256
    target_hash_calls = 0

    def one_false_target_hash(path):
        nonlocal target_hash_calls
        if Path(path).resolve() == target.resolve():
            target_hash_calls += 1
            if target_hash_calls == 1:
                return "0" * 64
        return real_sha256(path)

    monkeypatch.setattr(
        evaluator, "checkpoint_sha256", one_false_target_hash
    )
    with pytest.raises(RuntimeError, match="failed its SHA-256 check"):
        evaluator.atomic_copy_no_overwrite(source, target)
    assert target_hash_calls >= 2
    assert not target.exists()


def test_artifact_failure_rolls_back_only_unchanged_new_alias(
        tmp_path, monkeypatch):
    source = tmp_path / "source.zip"
    source.write_bytes(b"selected checkpoint")
    target = tmp_path / "selected.zip"
    output = tmp_path / "results"
    args = SimpleNamespace(
        selected_checkpoint=str(target),
        run_name="rollback",
        output_dir=str(output),
        leader_role="seller",
        screen_seed_start=100,
        screen_episodes=20,
    )

    monkeypatch.setattr(evaluator, "parse_args", lambda argv=None: args)

    def fake_selection(values):
        alias = evaluator.atomic_copy_no_overwrite(
            source, values.selected_checkpoint
        )
        return {"selected_alias": alias, "passed": True}

    monkeypatch.setattr(evaluator, "run_selection", fake_selection)
    monkeypatch.setattr(
        evaluator,
        "write_selection_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("artifact failure")
        ),
    )

    with pytest.raises(RuntimeError, match="artifact failure"):
        evaluator.main([])
    assert not target.exists()

    alias = evaluator.atomic_copy_no_overwrite(source, target)
    target.write_bytes(b"externally replaced bytes")
    assert not evaluator.rollback_new_selected_alias(
        {"selected_alias": alias}, expected_path=target
    )
    assert target.read_bytes() == b"externally replaced bytes"


def test_model_loader_rejects_checkpoint_mutation_during_load(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.write_bytes(b"validated bytes")

    def mutate(path, *, device):
        del path, device
        checkpoint.write_bytes(b"changed during load")
        return object()

    monkeypatch.setattr(evaluator.ScaledLearningRatePPO, "load", mutate)
    with pytest.raises(RuntimeError, match="changed while"):
        evaluator._load_model(checkpoint, device="cpu")


def test_checkpoint_evaluation_is_bound_to_loaded_bytes_and_rechecks_afterward(
        tmp_path, monkeypatch):
    checkpoint = (tmp_path / "candidate.zip").resolve()
    checkpoint.write_bytes(b"evaluated bytes")
    digest = evaluator.checkpoint_sha256(checkpoint)
    model = SimpleNamespace(
        policy=SimpleNamespace(
            economic_role="seller", economic_input_mode="event_only"
        ),
        num_timesteps=10,
        e2_evaluation_loaded_checkpoint={
            "path": str(checkpoint), "sha256": digest,
        },
    )
    args = SimpleNamespace(leader_role="seller")
    config = {
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "fixed_event_steps": None,
        "seller_game_reward_scale": 0.1,
        "buyer_game_reward_scale": 1.0,
    }
    monkeypatch.setattr(
        evaluator,
        "validate_candidate_provenance",
        lambda *args, **kwargs: {"fingerprint_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        evaluator, "model_actor_loss_mode", lambda model: "standard"
    )
    monkeypatch.setattr(
        evaluator,
        "model_economic_initialization",
        lambda *args, **kwargs: {"mean": 0.5, "concentration": 2.0},
    )
    monkeypatch.setattr(
        evaluator,
        "audit_e2_protocol",
        lambda *args, **kwargs: {"passed": True, "violations": []},
    )

    wrong_model = deepcopy(model)
    wrong_model.e2_evaluation_loaded_checkpoint["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="in-memory model is not bound"):
        evaluator.evaluate_checkpoint(
            wrong_model,
            checkpoint,
            args=args,
            response_model=object(),
            response_hash="b" * 64,
            config=config,
            config_hash="c" * 64,
            episodes=1,
            seed_start=1,
            phase="screen",
            expected_checkpoint_sha256=digest,
        )

    def mutate_after_rollout(*args, **kwargs):
        checkpoint.write_bytes(b"changed after load")
        return {"episode_rows": [], "transition_rows": [], "decision_rows": []}

    monkeypatch.setattr(evaluator, "evaluate_e2_model", mutate_after_rollout)
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        evaluator.evaluate_checkpoint(
            model,
            checkpoint,
            args=args,
            response_model=object(),
            response_hash="b" * 64,
            config=config,
            config_hash="c" * 64,
            episodes=1,
            seed_start=1,
            phase="screen",
            expected_checkpoint_sha256=digest,
        )


def test_confirmation_requires_same_hash_trace_and_commitment():
    screen = {
        "checkpoint_sha256": "a" * 64,
        "e2_provenance_fingerprint": "c" * 64,
        "economic_gate": {"passed": True},
        "counterfactual_controls": [
            _confirmation_control("all_zero", 0.0, trace="0" * 64),
            _confirmation_control("all_one", 1.0, trace="1" * 64),
        ],
        "protocol": {
            "passed": True,
            "query_trace_sha256": "b" * 64,
            "leader_commitment": [0.2] * 5,
            "full_query_actions": [[0.0, 0.2]] * 5,
        },
    }
    confirmation = deepcopy(screen)
    assert evaluator.confirmation_matches_screen(screen, confirmation)["passed"]

    confirmation["protocol"]["leader_commitment"][0] = 0.3
    check = evaluator.confirmation_matches_screen(screen, confirmation)
    assert not check["passed"]
    assert not check["leader_commitment_matches_screen"]


def _artifact_report_with_controls():
    evaluation = _valid_evaluation()
    result = {
        "checkpoint_id": "candidate",
        "checkpoint_path": "/tmp/candidate.zip",
        "checkpoint_sha256": "a" * 64,
        "response_checkpoint_sha256": "c" * 64,
        "environment_config_sha256": "d" * 64,
        "e2_provenance_fingerprint": "e" * 64,
        "training_total_timesteps": 100,
        "economic_role": "seller",
        "economic_input_mode": "event_only",
        "phase": "screen",
        "seed_start": 100,
        "seed_end": 100,
        "summary": evaluator.leader_outcome_summary(evaluation["episode_rows"]),
        "protocol": {"passed": True, "violations": []},
        "economic_intervention": {
            "intervention_id": "factual",
            "economic_commitment": None,
            "manifest_sha256": "f" * 64,
        },
        **evaluation,
    }
    for rows in (
            result["episode_rows"],
            result["transition_rows"],
            result["decision_rows"],
    ):
        for row in rows:
            row["e2_provenance_fingerprint"] = "e" * 64
    controls = []
    for intervention_id, value in (("all_zero", 0.0), ("all_one", 1.0)):
        control = deepcopy(result)
        control["economic_intervention"] = {
            "intervention_id": intervention_id,
            "economic_commitment": [value] * 5,
            "manifest_sha256": str(int(value)) * 64,
        }
        for rows in (
                control["episode_rows"],
                control["transition_rows"],
                control["decision_rows"],
        ):
            for row in rows:
                row.update({
                    "economic_intervention_id": intervention_id,
                    "economic_intervention_sha256": str(int(value)) * 64,
                    "economic_commitment_override": [value] * 5,
                    "e2_provenance_fingerprint": "e" * 64,
                })
        controls.append(control)
    result["counterfactual_controls"] = controls
    result["economic_gate"] = {
        "passed": True,
        "checks": [],
        "paired_rows": [{
            "checkpoint_path": "/tmp/candidate.zip",
            "checkpoint_sha256": "a" * 64,
            "response_sha256": "c" * 64,
            "environment_config_sha256": "d" * 64,
            "e2_provenance_fingerprint": "e" * 64,
            "phase": "screen",
            "evaluation_seed": 100,
            "event_steps": [0, 1, 2, 3, 6],
            "factual_minus_all_zero": 1.0,
            "factual_minus_all_one": 0.5,
        }],
    }
    ranking = evaluator.rank_candidates([result])
    report = {
        "environment_config": {"leader_role": "seller"},
        "screen": {
            "seed_start": 100,
            "seed_end": 100,
            "checkpoint_results": [result],
        },
        "selection": ranking,
        "confirmation": None,
        "confirmation_attempts": [],
        "passed": False,
    }
    return report


def test_artifacts_retain_all_rows_and_refuse_collisions(tmp_path):
    report = _artifact_report_with_controls()

    written = evaluator.write_selection_artifacts(
        report, output_dir=tmp_path, run_name="audit"
    )

    artifacts = written["artifacts"]
    assert Path(artifacts["report_json"]).is_file()
    assert Path(artifacts["screen_episodes_csv"]).is_file()
    assert Path(artifacts["screen_transitions_csv"]).is_file()
    assert Path(artifacts["screen_decisions_csv"]).is_file()
    assert Path(artifacts["screen_events_csv"]).is_file()
    assert Path(artifacts["screen_counterfactual_conditions_csv"]).is_file()
    assert Path(artifacts["screen_counterfactual_paired_csv"]).is_file()
    assert Path(artifacts["screen_counterfactual_transitions_csv"]).is_file()
    assert len(written["screen"]["checkpoint_results"][0]["transition_rows"]) == 17
    assert all(Path(path).is_file() for path in artifacts.values())
    with Path(artifacts["report_json"]).open(encoding="utf-8") as handle:
        assert json.load(handle)["artifacts"] == artifacts

    with Path(artifacts["screen_counterfactual_conditions_csv"]).open(
            newline="", encoding="utf-8"
    ) as handle:
        conditions = list(csv.DictReader(handle))
    all_zero = next(
        row for row in conditions if row["intervention_id"] == "all_zero"
    )
    assert all_zero["intervention_sha256"] == "0" * 64
    assert all_zero["response_sha256"] == "c" * 64
    assert all_zero["environment_config_sha256"] == "d" * 64
    assert all_zero["e2_provenance_fingerprint"] == "e" * 64
    assert json.loads(all_zero["economic_commitment"]) == [0.0] * 5

    with Path(artifacts["screen_counterfactual_transitions_csv"]).open(
            newline="", encoding="utf-8"
    ) as handle:
        transition_rows = list(csv.DictReader(handle))
    assert {row["economic_intervention_id"] for row in transition_rows} == {
        "all_zero", "all_one",
    }
    assert all(
        row["e2_provenance_fingerprint"] == "e" * 64
        for row in transition_rows
    )
    assert all(
        row["economic_intervention_sha256"]
        == ("0" if row["economic_intervention_id"] == "all_zero" else "1")
        * 64
        for row in transition_rows
    )
    for artifact_key in (
            "screen_counterfactual_episodes_csv",
            "screen_counterfactual_decisions_csv",
            "screen_counterfactual_events_csv",
    ):
        with Path(artifacts[artifact_key]).open(
                newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert {row["economic_intervention_id"] for row in rows} == {
            "all_zero", "all_one",
        }
        assert all(
            row["e2_provenance_fingerprint"] == "e" * 64 for row in rows
        )
    with Path(artifacts["screen_counterfactual_paired_csv"]).open(
            newline="", encoding="utf-8"
    ) as handle:
        paired_rows = list(csv.DictReader(handle))
    assert len(paired_rows) == 1
    assert paired_rows[0]["checkpoint_sha256"] == "a" * 64
    assert paired_rows[0]["response_sha256"] == "c" * 64
    assert paired_rows[0]["environment_config_sha256"] == "d" * 64
    assert paired_rows[0]["e2_provenance_fingerprint"] == "e" * 64
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.write_selection_artifacts(
            report, output_dir=tmp_path, run_name="audit"
        )


def test_artifacts_omit_absent_counterfactual_files_and_paths(tmp_path):
    report = _artifact_report_with_controls()
    result = report["screen"]["checkpoint_results"][0]
    result.pop("counterfactual_controls")
    result.pop("economic_gate")

    written = evaluator.write_selection_artifacts(
        report, output_dir=tmp_path, run_name="no_controls"
    )

    assert not any(
        "counterfactual" in key for key in written["artifacts"]
    )
    assert all(
        Path(path).is_file() for path in written["artifacts"].values()
    )
    assert not list(tmp_path.glob("no_controls.*counterfactual*.csv"))


def test_artifact_mid_stage_failure_leaves_no_final_or_staging_paths(
        tmp_path, monkeypatch):
    report = _artifact_report_with_controls()
    real_write_csv = evaluator.write_csv
    calls = 0

    def fail_third_csv(path, rows):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected staged CSV failure")
        return real_write_csv(path, rows)

    monkeypatch.setattr(evaluator, "write_csv", fail_third_csv)
    with pytest.raises(RuntimeError, match="injected staged CSV failure"):
        evaluator.write_selection_artifacts(
            report, output_dir=tmp_path, run_name="midwrite"
        )

    assert calls == 3
    assert not any("midwrite" in path.name for path in tmp_path.iterdir())


def test_artifact_publication_race_never_overwrites_and_rolls_back(
        tmp_path, monkeypatch):
    report = _artifact_report_with_controls()
    real_link = evaluator.os.link
    calls = 0
    raced_path = None

    def inject_racing_writer(source, destination, **kwargs):
        nonlocal calls, raced_path
        calls += 1
        if calls == 2:
            raced_path = Path(destination)
            raced_path.write_text("concurrent writer", encoding="utf-8")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(evaluator.os, "link", inject_racing_writer)
    with pytest.raises(FileExistsError):
        evaluator.write_selection_artifacts(
            report, output_dir=tmp_path, run_name="race"
        )

    assert calls == 2
    assert raced_path is not None
    assert raced_path.read_text(encoding="utf-8") == "concurrent writer"
    remaining = [path for path in tmp_path.iterdir() if "race" in path.name]
    assert remaining == [raced_path]


def test_artifact_publication_lock_rejects_concurrent_writer(tmp_path):
    report = _artifact_report_with_controls()
    lock = tmp_path / ".locked.publish.lock"
    lock.mkdir()

    with pytest.raises(FileExistsError, match="already in progress"):
        evaluator.write_selection_artifacts(
            report, output_dir=tmp_path, run_name="locked"
        )

    assert lock.is_dir()
    assert set(tmp_path.iterdir()) == {lock}


def test_parser_defaults_to_common_20_screen_and_disjoint_100_confirmation(
        tmp_path):
    args = evaluator.parse_args([
        "--leader-role", "seller",
        "--response-checkpoint", str(tmp_path / "buyer.zip"),
        "--checkpoint", str(tmp_path / "step100.zip"),
        "--selected-checkpoint", str(tmp_path / "selected.zip"),
    ])
    assert args.screen_episodes == 20
    assert args.confirmation_episodes == 100
    assert not evaluator._ranges_overlap(
        args.screen_seed_start,
        args.screen_episodes,
        args.confirmation_seed_start,
        args.confirmation_episodes,
    )

    with pytest.raises(SystemExit):
        evaluator.parse_args([
            "--leader-role", "seller",
            "--response-checkpoint", "buyer.zip",
            "--checkpoint", "step.zip",
            "--selected-checkpoint", "selected.zip",
            "--screen-seed-start", "100",
            "--screen-episodes", "20",
            "--confirmation-seed-start", "110",
            "--confirmation-episodes", "100",
        ])

    with pytest.raises(SystemExit):
        evaluator.parse_args([
            "--leader-role", "seller",
            "--response-checkpoint", "buyer.zip",
            "--checkpoint", "step.zip",
            "--selected-checkpoint", "selected.zip",
            "--gameplay-horizon", "199",
        ])


def test_run_selection_rechecks_canonical_counts_and_disjoint_seeds():
    base = SimpleNamespace(
        screen_episodes=20,
        screen_seed_start=100,
        confirmation_episodes=100,
        confirmation_seed_start=1_000,
        gameplay_horizon=200,
    )
    with pytest.raises(ValueError, match="exactly 20"):
        evaluator.run_selection(SimpleNamespace(**{
            **vars(base), "screen_episodes": 19,
        }))
    with pytest.raises(ValueError, match="exactly 100"):
        evaluator.run_selection(SimpleNamespace(**{
            **vars(base), "confirmation_episodes": 99,
        }))
    with pytest.raises(ValueError, match="must be disjoint"):
        evaluator.run_selection(SimpleNamespace(**{
            **vars(base), "confirmation_seed_start": 110,
        }))
    with pytest.raises(ValueError, match="canonical 200-step"):
        evaluator.run_selection(SimpleNamespace(**{
            **vars(base), "gameplay_horizon": 199,
        }))


def test_environment_hash_binds_role_response_contract_and_schedule(tmp_path):
    args = SimpleNamespace(
        leader_role="buyer",
        gameplay_horizon=200,
        event_tail_steps=0,
        fixed_event_steps=None,
        noop_max=30,
        max_frames=100_000,
        rom_path=None,
    )
    config = evaluator.environment_config(args)
    digest = evaluator.environment_config_sha256(config)
    changed = dict(config)
    changed["noop_max"] = 0

    assert config["follower_role"] == "seller"
    assert config["leader_action_cache"] is True
    assert Path(config["rom_path"]).is_file()
    assert len(config["rom_sha256"]) == 64
    assert len(digest) == 64
    assert digest != evaluator.environment_config_sha256(changed)


def test_candidate_provenance_requires_exact_response_and_evaluation_config():
    config = {
        "leader_role": "seller",
        "follower_role": "buyer",
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "fixed_event_steps": None,
        "seller_game_reward_scale": 0.1,
        "buyer_game_reward_scale": 1.0,
        "noop_max": 30,
        "frame_skip": 4,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
        "max_frames": 100_000,
        "rom_sha256": "1" * 64,
    }
    response_hash = "a" * 64
    policy = SimpleNamespace(
        economic_role="seller",
        economic_input_mode="event_only",
        visual_features=512,
        state_features=64,
        economic_hidden=64,
        critic_hidden=256,
        pretrained_lr_scale=0.1,
        game_action_count=6,
    )
    leader_policy = {
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "economic_role": "seller",
        "economic_input_mode": "event_only",
        "visual_features": 512,
        "state_features": 64,
        "economic_hidden": 64,
        "critic_hidden": 256,
        "pretrained_lr_scale": 0.1,
        "game_action_count": 6,
        "actor_loss_mode": STANDARD_ACTOR_LOSS_MODE,
        "economic_head_initialization": {
            "mean": 0.5,
            "concentration": 2.0,
        },
    }
    scientific_config = {
        "leader_role": "seller",
        "follower_role": "buyer",
        "environment": {"seed": 1, **{
            key: config[key]
            for key in (
                "gameplay_horizon",
                "event_tail_steps",
                "fixed_event_steps",
                "seller_game_reward_scale",
                "buyer_game_reward_scale",
                "noop_max",
                "frame_skip",
                "frame_stack",
                "episodic_life",
                "clip_game_rewards",
                "max_frames",
                "rom_sha256",
            )
        }},
        "leader_policy": leader_policy,
        "implementation": trainer.e2_implementation_provenance(),
        "protocol": {
            "trade_events": 5,
            "query_transitions": 5,
            "cached_trade_replays": 5,
            "outer_episode_transitions": 210,
            "policy_action_cache": True,
            "leader_economic_input": "event_only",
            "response_economic_input": "full",
            "response_algorithm": "frozen_meta_policy",
        },
    }
    artifacts = {
        "frozen_response": {
            "sha256": response_hash,
            "policy": {
                "economic_role": "buyer",
                "economic_input_mode": "full",
            },
        },
    }
    identity = {
        "scientific_config": scientific_config,
        "artifacts": artifacts,
    }
    unsigned = {
        "schema": trainer.E2_PROVENANCE_SCHEMA,
        "version": trainer.E2_PROVENANCE_VERSION,
        "run_lineage_id": "0" * 32,
        "scientific_identity_sha256": trainer._canonical_sha256(identity),
        **identity,
    }
    manifest = {
        **unsigned,
        "fingerprint_sha256": trainer._canonical_sha256(unsigned),
    }
    model = SimpleNamespace(e2_provenance_manifest=manifest, policy=policy)

    validated = evaluator.validate_candidate_provenance(
        model, response_hash=response_hash, config=config
    )
    assert validated["fingerprint_sha256"] == manifest["fingerprint_sha256"]

    model.atari_actor_loss_mode = PHASE_BALANCED_ACTOR_LOSS_MODE
    with pytest.raises(ValueError, match="actor loss mode"):
        evaluator.validate_candidate_provenance(
            model, response_hash=response_hash, config=config
        )
    del model.atari_actor_loss_mode

    model.atari_economic_head_initialization = {
        "mean": 0.75,
        "concentration": 2.0,
    }
    with pytest.raises(ValueError, match="economic initialization"):
        evaluator.validate_candidate_provenance(
            model, response_hash=response_hash, config=config
        )
    del model.atari_economic_head_initialization

    model.atari_actor_loss_mode = "unknown"
    with pytest.raises(ValueError, match="unknown actor loss mode"):
        evaluator.validate_candidate_provenance(
            model, response_hash=response_hash, config=config
        )
    del model.atari_actor_loss_mode

    model.target_kl = 0.01
    with pytest.raises(ValueError, match="target KL does not match"):
        evaluator.validate_candidate_provenance(
            model, response_hash=response_hash, config=config
        )
    del model.target_kl

    with pytest.raises(ValueError, match="different frozen E1 response"):
        evaluator.validate_candidate_provenance(
            model, response_hash="b" * 64, config=config
        )
    changed = {**config, "noop_max": 0}
    with pytest.raises(ValueError, match="environment provenance"):
        evaluator.validate_candidate_provenance(
            model, response_hash=response_hash, config=changed
        )

    policy.critic_hidden = 128
    with pytest.raises(ValueError, match="policy architecture"):
        evaluator.validate_candidate_provenance(
            model, response_hash=response_hash, config=config
        )


def test_run_selection_confirms_only_screen_winner_and_never_falls_back(
        tmp_path, monkeypatch):
    first = tmp_path / "step100.zip"
    second = tmp_path / "step200.zip"
    invalid = tmp_path / "step300.zip"
    response = tmp_path / "buyer_e1.zip"
    selected = tmp_path / "selected.zip"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    invalid.write_bytes(b"invalid")
    response.write_bytes(b"response")
    args = SimpleNamespace(
        leader_role="seller",
        response_checkpoint=str(response),
        checkpoint=[str(first), str(second), str(invalid)],
        selected_checkpoint=str(selected),
        screen_episodes=20,
        screen_seed_start=4_000_001,
        confirmation_episodes=100,
        confirmation_seed_start=5_000_001,
        gameplay_horizon=200,
        event_tail_steps=0,
        fixed_event_steps=None,
        noop_max=30,
        max_frames=100_000,
        rom_path=None,
        device="cpu",
    )

    class _Policy:
        economic_role = "buyer"
        economic_input_mode = "full"

    response_model = SimpleNamespace(
        policy=_Policy(), num_timesteps=500
    )
    monkeypatch.setattr(
        evaluator, "load_e1_response", lambda *a, **k: response_model
    )

    def fake_load(path, *, leader_role, device, expected_sha256=None):
        del leader_role, device, expected_sha256
        return SimpleNamespace(
            policy=SimpleNamespace(
                economic_role="seller", economic_input_mode="event_only"
            ),
            num_timesteps=100 if "100" in str(path) else 200,
        )

    monkeypatch.setattr(evaluator, "load_e2_checkpoint", fake_load)
    calls = []

    def fake_evaluate(
            model,
            checkpoint,
            *,
            args,
            response_model,
            response_hash,
            config,
            config_hash,
            episodes,
            seed_start,
            phase,
            intervention_id="factual",
            economic_commitment_override=None,
            expected_checkpoint_sha256=None,
    ):
        del args, response_model, config, expected_checkpoint_sha256
        path = Path(checkpoint)
        digest = evaluator.checkpoint_sha256(path)
        calls.append((
            phase, path.name, episodes, seed_start, selected.exists(),
            intervention_id,
        ))
        if phase == "screen":
            assert not selected.exists()
        else:
            assert path != selected
            assert not selected.exists()
        factual_score = {
            b"first": 1.0,
            b"second": 2.0,
            b"invalid": 100.0,
        }[path.read_bytes()]
        if phase == "confirmation" and path.read_bytes() == b"second":
            factual_score = 0.0
        score = factual_score if intervention_id == "factual" else 0.0
        protocol_passed = not (
            phase == "screen"
            and path.read_bytes() == b"invalid"
            and intervention_id == "factual"
        )
        commitment = (
            [0.5] * 5
            if economic_commitment_override is None
            else list(economic_commitment_override)
        )
        intervention_sha256 = (
            digest[:62]
            + {"factual": "fa", "all_zero": "00", "all_one": "11"}[
                intervention_id
            ]
        )
        rows = [
            {
                "evaluation_seed": seed_start + episode,
                "event_steps": [0, 20, 40, 60, 80],
                "leader_reward": score,
                "purchases": 5,
                "payments": 2.5,
                "seller_shots_fired": 2,
                "buyer_shots_fired": 2,
                "economic_intervention_id": intervention_id,
                "economic_intervention_sha256": intervention_sha256,
                "economic_commitment_override": (
                    None
                    if economic_commitment_override is None
                    else list(economic_commitment_override)
                ),
            }
            for episode in range(episodes)
        ]
        return {
            "checkpoint_id": path.stem,
            "checkpoint_path": str(path),
            "checkpoint_sha256": digest,
            "response_checkpoint_sha256": response_hash,
            "environment_config_sha256": config_hash,
            "e2_provenance_fingerprint": "e" * 64,
            "training_total_timesteps": model.num_timesteps,
            "economic_role": "seller",
            "economic_input_mode": "event_only",
            "economic_intervention": {
                "intervention_id": intervention_id,
                "economic_commitment": (
                    None
                    if economic_commitment_override is None
                    else list(economic_commitment_override)
                ),
                "manifest_sha256": intervention_sha256,
            },
            "phase": phase,
            "seed_start": seed_start,
            "seed_end": seed_start + episodes - 1,
            "summary": {
                "episodes": episodes,
                "mean_leader_payoff": score,
                "median_leader_payoff": score,
                "std_leader_payoff": 0.0,
                "min_leader_payoff": score,
                "max_leader_payoff": score,
            },
            "protocol": {
                "passed": protocol_passed,
                "violations": [] if protocol_passed else [{"field": "forced"}],
                "query_trace_sha256": "f" * 64,
                "leader_commitment": commitment,
                "full_query_actions": [
                    [0.0, value] for value in commitment
                ],
            },
            "episode_rows": rows,
            "transition_rows": [],
            "decision_rows": [],
        }

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", fake_evaluate)
    report = evaluator.run_selection(args)

    assert [(call[0], call[1], call[5]) for call in calls] == [
        ("screen", "step100.zip", "factual"),
        ("screen", "step100.zip", "all_zero"),
        ("screen", "step100.zip", "all_one"),
        ("screen", "step200.zip", "factual"),
        ("screen", "step200.zip", "all_zero"),
        ("screen", "step200.zip", "all_one"),
        ("screen", "step300.zip", "factual"),
        ("confirmation", "step200.zip", "factual"),
        ("confirmation", "step200.zip", "all_zero"),
        ("confirmation", "step200.zip", "all_one"),
    ]
    assert all(not call[4] for call in calls)
    assert not selected.exists()
    assert not report["passed"]
    assert report["confirmation"]["disjoint_from_screen"]
    assert len(report["confirmation_attempts"]) == 1
    assert report["selection"]["screen_selected_checkpoint_sha256"] == (
        evaluator.checkpoint_sha256(second)
    )
    assert report["selection"]["selected_checkpoint_sha256"] is None
