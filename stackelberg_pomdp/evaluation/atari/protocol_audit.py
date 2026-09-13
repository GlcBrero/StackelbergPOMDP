"""Validate leader-query, trade, inventory, and payoff invariants in retained trajectories."""

from collections import defaultdict
import json
import re

import numpy as np

from stackelberg_pomdp.atari.protocol import (
    CACHED_TRADE_REPLAY,
    GAMEPLAY,
    LEADER_QUERY,
    NUM_TRADE_EVENTS,
    canonical_leader_state,
)
from stackelberg_pomdp.envs.atari.bilateral import (
    BUYER,
    SELLER,
)
from stackelberg_pomdp.evaluation.atari.contracts import (
    CANONICAL_EVENT_ACTION_MASK,
    PROTOCOL_ATOL,
    _action_list,
    _canonical_json_bytes,
    _close,
    _number,
    opposite_role,
)
from stackelberg_pomdp.evaluation.atari.interventions import _normalized_commitment


def _violation(row, field, expected, actual):
    return {
        "evaluation_episode": int(row.get("evaluation_episode", -1)),
        "evaluation_seed": int(row.get("evaluation_seed", -1)),
        "field": str(field),
        "expected": expected,
        "actual": actual,
    }


def _audit_episode_summary(
        row, *, leader_role, gameplay_horizon,
):
    expected_reward = gameplay_horizon + NUM_TRADE_EVENTS
    expected_outer = expected_reward + NUM_TRADE_EVENTS
    exact = {
        "evaluation_steps": expected_outer,
        "outer_transition_count": expected_outer,
        "query_transitions": NUM_TRADE_EVENTS,
        "gameplay_transitions": gameplay_horizon,
        "trade_transitions": NUM_TRADE_EVENTS,
        "reward_transition_count": expected_reward,
        "cache_hits": NUM_TRADE_EVENTS,
        "bullets_arrived": NUM_TRADE_EVENTS,
        "seller_emulator_step_calls": gameplay_horizon,
        "buyer_emulator_step_calls": gameplay_horizon,
        "response_algorithm": "frozen_meta_policy",
        "leader_role": leader_role,
        "follower_role": opposite_role(leader_role),
    }
    violations = []
    for field, expected in exact.items():
        if row.get(field) != expected:
            violations.append(_violation(row, field, expected, row.get(field)))

    for field in (
            "seller_bullet_error",
            "buyer_bullet_error",
            "seller_payoff_error",
            "buyer_payoff_error",
    ):
        actual = _number(row.get(field))
        if actual is None or abs(actual) > PROTOCOL_ATOL:
            violations.append(_violation(row, field, 0.0, row.get(field)))

    return expected_outer, violations


def _audit_episode_payoffs(
        row, *, leader_role, seller_game_reward_scale,
        buyer_game_reward_scale, violations,
):
    numeric = {}
    for field in (
            "evaluation_return",
            "leader_reward",
            "seller_reward",
            "buyer_reward",
            "seller_game_reward",
            "buyer_game_reward",
            "payments",
            "purchases",
            "seller_shots_fired",
            "buyer_shots_fired",
            "seller_final_ammo",
            "buyer_final_ammo",
    ):
        numeric[field] = _number(row.get(field))
        if numeric[field] is None:
            violations.append(_violation(row, field, "numeric", row.get(field)))
    if all(numeric[field] is not None for field in (
            "evaluation_return", "leader_reward"
    )) and not _close(numeric["evaluation_return"], numeric["leader_reward"]):
        violations.append(_violation(
            row, "evaluation_return == leader_reward",
            numeric["leader_reward"], numeric["evaluation_return"],
        ))
    role_reward = numeric.get(f"{leader_role}_reward")
    if (
            numeric["leader_reward"] is not None
            and role_reward is not None
            and not _close(numeric["leader_reward"], role_reward)
    ):
        violations.append(_violation(
            row, "leader_reward == role reward", role_reward,
            numeric["leader_reward"],
        ))
    if all(numeric[field] is not None for field in (
            "seller_reward", "seller_game_reward", "payments"
    )):
        expected = (
            seller_game_reward_scale * numeric["seller_game_reward"]
            + numeric["payments"]
        )
        if not _close(numeric["seller_reward"], expected):
            violations.append(_violation(
                row, "seller payoff identity", expected,
                numeric["seller_reward"],
            ))
    if all(numeric[field] is not None for field in (
            "buyer_reward", "buyer_game_reward", "payments"
    )):
        expected = (
            buyer_game_reward_scale * numeric["buyer_game_reward"]
            - numeric["payments"]
        )
        if not _close(numeric["buyer_reward"], expected):
            violations.append(_violation(
                row, "buyer payoff identity", expected,
                numeric["buyer_reward"],
            ))
    if all(numeric[field] is not None for field in (
            "seller_shots_fired", "purchases", "seller_final_ammo"
    )):
        accounted = (
            numeric["seller_shots_fired"]
            + numeric["purchases"]
            + numeric["seller_final_ammo"]
        )
        if not _close(accounted, NUM_TRADE_EVENTS):
            violations.append(_violation(
                row, "seller bullet identity", NUM_TRADE_EVENTS, accounted
            ))
    if all(numeric[field] is not None for field in (
            "buyer_shots_fired", "buyer_final_ammo", "purchases"
    )):
        accounted = numeric["buyer_shots_fired"] + numeric["buyer_final_ammo"]
        if not _close(accounted, numeric["purchases"]):
            violations.append(_violation(
                row, "buyer bullet identity", numeric["purchases"], accounted
            ))

    return numeric


def _audit_episode_trades(
        row, *, gameplay_horizon, event_tail_steps, fixed_event_steps,
        numeric, violations,
):
    event_stop = gameplay_horizon - event_tail_steps
    try:
        event_steps = tuple(int(value) for value in row.get("event_steps", ()))
    except (TypeError, ValueError):
        event_steps = ()
    valid_schedule = bool(
        len(event_steps) == NUM_TRADE_EVENTS
        and tuple(sorted(set(event_steps))) == event_steps
        and event_steps[0] >= 0
        and event_steps[-1] < event_stop
    )
    if not valid_schedule:
        violations.append(_violation(
            row, "event_steps",
            f"five sorted distinct steps in [0, {event_stop})",
            row.get("event_steps"),
        ))
    elif fixed_event_steps is not None and event_steps != tuple(fixed_event_steps):
        violations.append(_violation(
            row, "event_steps", list(fixed_event_steps), list(event_steps)
        ))

    events = row.get("events")
    if not isinstance(events, list) or len(events) != NUM_TRADE_EVENTS:
        violations.append(_violation(
            row, "events", f"{NUM_TRADE_EVENTS} trade rows", events
        ))
        events = []
    accepted_count = 0
    payment_sum = 0.0
    for index, event in enumerate(events):
        expected_event = event_steps[index] if valid_schedule else None
        if int(event.get("event_index", -1)) != index:
            violations.append(_violation(
                row, f"events[{index}].event_index", index,
                event.get("event_index"),
            ))
        if expected_event is not None and int(event.get("game_step", -1)) != expected_event:
            violations.append(_violation(
                row, f"events[{index}].game_step", expected_event,
                event.get("game_step"),
            ))
        price = _number(event.get("price"))
        threshold = _number(event.get("threshold"))
        accepted = event.get("accepted")
        if (
                price is None or threshold is None
                or not 0.0 <= price <= 1.0
                or not 0.0 <= threshold <= 1.0
                or not isinstance(accepted, bool)
        ):
            violations.append(_violation(
                row, f"events[{index}].trade", "valid [0,1] trade", event
            ))
            continue
        expected_acceptance = bool(price <= threshold)
        if accepted != expected_acceptance:
            violations.append(_violation(
                row, f"events[{index}].accepted", expected_acceptance, accepted
            ))
        accepted_count += int(accepted)
        payment_sum += price if accepted else 0.0
        seller_delta = int(event["seller_ammo_after"]) - int(event["seller_ammo_before"])
        buyer_delta = int(event["buyer_ammo_after"]) - int(event["buyer_ammo_before"])
        expected_seller_delta = -1 if accepted else 0
        expected_buyer_delta = 1 if accepted else 0
        if seller_delta != expected_seller_delta:
            violations.append(_violation(
                row, f"events[{index}].seller_ammo_delta",
                expected_seller_delta, seller_delta,
            ))
        if buyer_delta != expected_buyer_delta:
            violations.append(_violation(
                row, f"events[{index}].buyer_ammo_delta",
                expected_buyer_delta, buyer_delta,
            ))
    if numeric["purchases"] is not None and not _close(
            accepted_count, numeric["purchases"]
    ):
        violations.append(_violation(
            row, "purchases == accepted trades", accepted_count,
            numeric["purchases"],
        ))
    if numeric["payments"] is not None and not _close(
            payment_sum, numeric["payments"]
    ):
        violations.append(_violation(
            row, "payments == accepted prices", payment_sum,
            numeric["payments"],
        ))

    return events


def _audit_episode_economic_actions(
        row, *, leader_role, expected_outer, numeric, events, violations,
):
    query_actions = row.get("query_actions")
    commitment = row.get("leader_commitment")
    follower_actions = row.get("follower_actions")
    intervention_id = row.get("economic_intervention_id")
    override = row.get("economic_commitment_override")
    if not isinstance(intervention_id, str) or not intervention_id:
        violations.append(_violation(
            row, "economic_intervention_id", "nonempty string", intervention_id
        ))
    intervention_sha = row.get("economic_intervention_sha256")
    if (
            not isinstance(intervention_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", intervention_sha) is None
    ):
        violations.append(_violation(
            row,
            "economic_intervention_sha256",
            "64 lowercase hex characters",
            intervention_sha,
        ))
    normalized_override = None
    try:
        normalized_override = _normalized_commitment(override)
    except ValueError:
        violations.append(_violation(
            row,
            "economic_commitment_override",
            "null or five values in [0,1]",
            override,
        ))
    expected_applications = 0 if normalized_override is None else NUM_TRADE_EVENTS
    for field in (
            "economic_override_query_applications",
            "economic_override_replay_applications",
    ):
        if row.get(field) != expected_applications:
            violations.append(_violation(
                row, field, expected_applications, row.get(field)
            ))
    expected_event_counts = [
        0 if normalized_override is None else 1
    ] * NUM_TRADE_EVENTS
    for field in (
            "economic_override_query_event_counts",
            "economic_override_replay_event_counts",
    ):
        if row.get(field) != expected_event_counts:
            violations.append(_violation(
                row, field, expected_event_counts, row.get(field)
            ))
    valid_query_actions = bool(
        isinstance(query_actions, list)
        and len(query_actions) == NUM_TRADE_EVENTS
        and all(isinstance(action, list) and len(action) == 2 for action in query_actions)
    )
    if not valid_query_actions:
        violations.append(_violation(
            row, "query_actions", "five full actions", query_actions
        ))
    if not isinstance(commitment, list) or len(commitment) != NUM_TRADE_EVENTS:
        violations.append(_violation(
            row, "leader_commitment", "five economic actions", commitment
        ))
    elif valid_query_actions and not np.allclose(
            np.asarray(commitment, dtype=float),
            np.asarray(query_actions, dtype=float)[:, 1],
            rtol=0.0,
            atol=PROTOCOL_ATOL,
    ):
        violations.append(_violation(
            row, "leader_commitment == query economic actions",
            np.asarray(query_actions, dtype=float)[:, 1].tolist(), commitment,
        ))
    if normalized_override is not None:
        expected_commitment = list(normalized_override)
        if (
                not isinstance(commitment, list)
                or len(commitment) != NUM_TRADE_EVENTS
                or not np.allclose(
                np.asarray(commitment, dtype=float),
                np.asarray(expected_commitment, dtype=float),
                rtol=0.0,
                atol=PROTOCOL_ATOL,
                )
        ):
            violations.append(_violation(
                row,
                "leader commitment == economic override",
                expected_commitment,
                commitment,
            ))
    valid_follower_actions = bool(
        isinstance(follower_actions, list)
        and len(follower_actions) == NUM_TRADE_EVENTS
        and all(
            isinstance(action, list) and len(action) == 2
            for action in follower_actions
        )
    )
    if not valid_follower_actions:
        violations.append(_violation(
            row, "follower_actions", "five full trade actions", follower_actions
        ))
    if valid_query_actions and valid_follower_actions and len(events) == 5:
        for index, event in enumerate(events):
            leader_economic = float(query_actions[index][1])
            follower_economic = float(follower_actions[index][1])
            if leader_role == SELLER:
                leader_field, follower_field = "price", "threshold"
            else:
                leader_field, follower_field = "threshold", "price"
            if not _close(leader_economic, event[leader_field]):
                violations.append(_violation(
                    row, f"events[{index}].{leader_field} == leader query",
                    leader_economic, event[leader_field],
                ))
            if not _close(follower_economic, event[follower_field]):
                violations.append(_violation(
                    row, f"events[{index}].{follower_field} == response action",
                    follower_economic, event[follower_field],
                ))
    trace = row.get("query_trace_sha256")
    if not isinstance(trace, str) or re.fullmatch(r"[0-9a-f]{64}", trace) is None:
        violations.append(_violation(
            row, "query_trace_sha256", "64 lowercase hex characters", trace
        ))
    nested = row.get("terminal_episode_summary")
    if not isinstance(nested, dict):
        violations.append(_violation(
            row, "terminal_episode_summary", "terminal episode mapping", nested
        ))
    else:
        if nested.get("l") != expected_outer:
            violations.append(_violation(
                row, "terminal episode length", expected_outer, nested.get("l")
            ))
        if (
                numeric["leader_reward"] is not None
                and _number(nested.get("r")) is not None
                and not _close(nested["r"], numeric["leader_reward"])
        ):
            violations.append(_violation(
                row, "terminal episode return", numeric["leader_reward"],
                nested.get("r"),
            ))


def _audit_episode(
        row,
        *,
        leader_role,
        gameplay_horizon,
        event_tail_steps,
        fixed_event_steps,
        seller_game_reward_scale,
        buyer_game_reward_scale,
):
    expected_outer, violations = _audit_episode_summary(
        row, leader_role=leader_role, gameplay_horizon=gameplay_horizon,
    )
    numeric = _audit_episode_payoffs(
        row,
        leader_role=leader_role,
        seller_game_reward_scale=seller_game_reward_scale,
        buyer_game_reward_scale=buyer_game_reward_scale,
        violations=violations,
    )
    events = _audit_episode_trades(
        row,
        gameplay_horizon=gameplay_horizon,
        event_tail_steps=event_tail_steps,
        fixed_event_steps=fixed_event_steps,
        numeric=numeric,
        violations=violations,
    )
    _audit_episode_economic_actions(
        row,
        leader_role=leader_role,
        expected_outer=expected_outer,
        numeric=numeric,
        events=events,
        violations=violations,
    )
    return violations


def _audit_transition_layout(
        episode_row, selected, *, gameplay_horizon, violations,
):
    expected_outer = gameplay_horizon + 2 * NUM_TRADE_EVENTS
    indices = [int(row["transition_index"]) for row in selected]
    if indices != list(range(expected_outer)):
        violations.append(_violation(
            episode_row, "transition indices", list(range(expected_outer)), indices
        ))
    counts = defaultdict(int)
    for row in selected:
        counts[row["substep_type"]] += 1
    expected_counts = {
        LEADER_QUERY: NUM_TRADE_EVENTS,
        GAMEPLAY: gameplay_horizon,
        CACHED_TRADE_REPLAY: NUM_TRADE_EVENTS,
    }
    for substep, expected in expected_counts.items():
        if counts[substep] != expected:
            violations.append(_violation(
                episode_row, f"transition count: {substep}", expected,
                counts[substep],
            ))
    if [row["substep_type"] for row in selected[:NUM_TRADE_EVENTS]] != [
            LEADER_QUERY
    ] * NUM_TRADE_EVENTS:
        violations.append(_violation(
            episode_row, "first five transitions", [LEADER_QUERY] * 5,
            [row["substep_type"] for row in selected[:5]],
        ))
    for index, row in enumerate(selected[:NUM_TRADE_EVENTS]):
        if row.get("query_index") != index:
            violations.append(_violation(
                episode_row, f"query transition {index} index", index,
                row.get("query_index"),
            ))
        expected_state = canonical_leader_state(index)
        if not np.array_equal(
                np.asarray(row.get("actor_state"), dtype=np.float32),
                expected_state,
        ):
            violations.append(_violation(
                episode_row, f"query transition {index} canonical actor state",
                expected_state.tolist(), row.get("actor_state"),
            ))
        if row.get("actor_image_nonzero") != 0:
            violations.append(_violation(
                episode_row, f"query transition {index} dummy image", 0,
                row.get("actor_image_nonzero"),
            ))


def _audit_transition_reward_schedule(
        episode_row, selected, *, gameplay_horizon, violations,
):
    reward_rows = selected[NUM_TRADE_EVENTS:]
    gameplay_seen = 0
    replay_seen = 0
    event_steps = tuple(int(value) for value in episode_row.get("event_steps", ()))
    for row in reward_rows:
        if row["substep_type"] == CACHED_TRADE_REPLAY:
            if row.get("event_index") != replay_seen:
                violations.append(_violation(
                    episode_row, "cached replay event order", replay_seen,
                    row.get("event_index"),
                ))
            if (
                    len(event_steps) == NUM_TRADE_EVENTS
                    and replay_seen < NUM_TRADE_EVENTS
                    and gameplay_seen != event_steps[replay_seen]
            ):
                violations.append(_violation(
                    episode_row, f"cached replay {replay_seen} gameplay position",
                    event_steps[replay_seen], gameplay_seen,
                ))
            replay_seen += 1
        elif row["substep_type"] == GAMEPLAY:
            gameplay_seen += 1
    if gameplay_seen != gameplay_horizon or replay_seen != NUM_TRADE_EVENTS:
        violations.append(_violation(
            episode_row, "reward-phase gameplay/trade progression",
            [gameplay_horizon, NUM_TRADE_EVENTS],
            [gameplay_seen, replay_seen],
        ))
    for index, row in enumerate(selected):
        expected_done = index == len(selected) - 1
        if bool(row["done"]) != expected_done:
            violations.append(_violation(
                episode_row, f"transition {index} done", expected_done,
                row["done"],
            ))
    transition_return = float(sum(float(row["reward"]) for row in selected))
    if not _close(transition_return, episode_row["evaluation_return"]):
        violations.append(_violation(
            episode_row, "sum transition rewards", episode_row["evaluation_return"],
            transition_return,
        ))


def _audit_transition_decision_replays(
        episode_row, selected_decisions, *, leader_role, violations,
):
    if len(selected_decisions) != 2 * NUM_TRADE_EVENTS:
        violations.append(_violation(
            episode_row, "economic decision rows", 2 * NUM_TRADE_EVENTS,
            len(selected_decisions),
        ))
        return
    queries = {
        int(row["query_index"]): row
        for row in selected_decisions
        if row["substep_type"] == LEADER_QUERY
    }
    replays = {
        int(row["event_index"]): row
        for row in selected_decisions
        if row["substep_type"] == CACHED_TRADE_REPLAY
    }
    if set(queries) != set(range(NUM_TRADE_EVENTS)):
        violations.append(_violation(
            episode_row, "query indices", list(range(NUM_TRADE_EVENTS)),
            sorted(queries),
        ))
    if set(replays) != set(range(NUM_TRADE_EVENTS)):
        violations.append(_violation(
            episode_row, "cached replay indices", list(range(NUM_TRADE_EVENTS)),
            sorted(replays),
        ))
    for event in sorted(set(queries) & set(replays)):
        query = queries[event]
        replay = replays[event]
        expected_state = canonical_leader_state(event)
        if not np.array_equal(
                np.asarray(replay.get("actor_state"), dtype=np.float32),
                expected_state,
        ):
            violations.append(_violation(
                episode_row, f"event {event} cached canonical actor state",
                expected_state.tolist(), replay.get("actor_state"),
            ))
        if replay.get("actor_image_nonzero") != 0:
            violations.append(_violation(
                episode_row, f"event {event} cached dummy image", 0,
                replay.get("actor_image_nonzero"),
            ))
        if query["actor_observation_sha256"] != replay["actor_observation_sha256"]:
            violations.append(_violation(
                episode_row, f"event {event} actor observation cache identity",
                query["actor_observation_sha256"],
                replay["actor_observation_sha256"],
            ))
        if not np.allclose(
                query["requested_action"], replay["requested_action"],
                rtol=0.0, atol=PROTOCOL_ATOL,
        ):
            violations.append(_violation(
                episode_row, f"event {event} requested action cache identity",
                query["requested_action"], replay["requested_action"],
            ))
        if not np.allclose(
                query.get("policy_action_before_intervention"),
                replay.get("policy_action_before_intervention"),
                rtol=0.0,
                atol=PROTOCOL_ATOL,
        ):
            violations.append(_violation(
                episode_row,
                f"event {event} pre-intervention policy cache identity",
                query.get("policy_action_before_intervention"),
                replay.get("policy_action_before_intervention"),
            ))
        executed = replay.get("leader_executed_action")
        if executed is None or not np.allclose(
                query["requested_action"], executed,
                rtol=0.0, atol=PROTOCOL_ATOL,
        ):
            violations.append(_violation(
                episode_row, f"event {event} executed action cache identity",
                query["requested_action"], executed,
            ))
        events = episode_row.get("events")
        event_row = (
            events[event]
            if isinstance(events, list) and event < len(events)
            else None
        )
        if isinstance(event_row, dict):
            if replay.get("game_step") != event_row.get("game_step"):
                violations.append(_violation(
                    episode_row,
                    f"event {event} replay game step",
                    event_row.get("game_step"),
                    replay.get("game_step"),
                ))
            price = _number(event_row.get("price"))
            accepted = event_row.get("accepted")
            replay_reward = _number(replay.get("reward"))
            if price is not None and isinstance(accepted, bool):
                signed_payment = price if accepted else 0.0
                if leader_role == BUYER:
                    signed_payment = -signed_payment
                if (
                        replay_reward is None
                        or not _close(replay_reward, signed_payment)
                ):
                    violations.append(_violation(
                        episode_row,
                        f"event {event} immediate {leader_role} trade reward",
                        signed_payment,
                        replay.get("reward"),
                    ))


def _audit_transition_phase_credit(episode_row, row, *, substep, violations):
    if not row["reward_generated"]:
        violations.append(_violation(
            episode_row, f"transition {row['transition_index']} reward_generated",
            True, False,
        ))
    expected_reward_phase = substep != LEADER_QUERY
    if row["is_reward_phase"] != expected_reward_phase:
        violations.append(_violation(
            episode_row, f"transition {row['transition_index']} is_reward_phase",
            expected_reward_phase, row["is_reward_phase"],
        ))
    expected_credit = {
        LEADER_QUERY: [0.0, 1.0],
        GAMEPLAY: [1.0, 0.0],
        CACHED_TRADE_REPLAY: [0.0, 0.0],
    }.get(substep)
    if expected_credit is not None and not np.array_equal(
            np.asarray(row["action_credit"], dtype=float), expected_credit
    ):
        violations.append(_violation(
            episode_row, f"transition {row['transition_index']} action_credit",
            expected_credit, row["action_credit"],
        ))
    if substep == LEADER_QUERY:
        if float(row["reward"]) != 0.0 or row["emulator_advanced"]:
            violations.append(_violation(
                episode_row, f"query transition {row['transition_index']}",
                "zero reward and no emulator advance", row,
            ))
    if substep == CACHED_TRADE_REPLAY and (
            not row["cache_hit"] or row["emulator_advanced"]
    ):
        violations.append(_violation(
            episode_row, f"cached trade transition {row['transition_index']}",
            "cache hit and no emulator advance", row,
        ))


def _audit_transition_intervention(
        episode_row, selected, *, violations,
):
    try:
        override = _normalized_commitment(
            episode_row.get("economic_commitment_override")
        )
    except ValueError:
        override = None
    for row in selected:
        substep = row["substep_type"]
        for field in (
                "economic_intervention_id",
                "economic_intervention_sha256",
                "economic_commitment_override",
        ):
            if row.get(field) != episode_row.get(field):
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} {field}",
                    episode_row.get(field),
                    row.get(field),
                ))
        try:
            policy_action = _action_list(
                row.get("policy_action_before_intervention")
            )
            requested_action = _action_list(row.get("requested_action"))
        except RuntimeError:
            violations.append(_violation(
                episode_row,
                f"transition {row['transition_index']} intervention action",
                "two finite full actions",
                {
                    "policy": row.get("policy_action_before_intervention"),
                    "requested": row.get("requested_action"),
                },
            ))
            policy_action = requested_action = None
        if (
                policy_action is not None
                and not _close(policy_action[0], requested_action[0])
        ):
            violations.append(_violation(
                episode_row,
                f"transition {row['transition_index']} preserved game action",
                policy_action[0],
                requested_action[0],
            ))
        expected_override = bool(
            override is not None
            and substep in (LEADER_QUERY, CACHED_TRADE_REPLAY)
        )
        if bool(row.get("economic_override_applied")) != expected_override:
            violations.append(_violation(
                episode_row,
                f"transition {row['transition_index']} override application",
                expected_override,
                row.get("economic_override_applied"),
            ))
        if not expected_override:
            if (
                    policy_action is not None
                    and not np.array_equal(
                        np.asarray(requested_action, dtype=np.float64),
                        np.asarray(policy_action, dtype=np.float64),
                    )
            ):
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} "
                    "no economic override action identity",
                    policy_action,
                    requested_action,
                ))
            for field in (
                    "economic_override_kind", "economic_override_event"
            ):
                if row.get(field) is not None:
                    violations.append(_violation(
                        episode_row,
                        f"transition {row['transition_index']} {field}",
                        None,
                        row.get(field),
                    ))
        if expected_override and requested_action is not None:
            event_index = (
                row.get("query_index")
                if substep == LEADER_QUERY
                else row.get("event_index")
            )
            if row.get("economic_override_kind") != substep:
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} override kind",
                    substep,
                    row.get("economic_override_kind"),
                ))
            if row.get("economic_override_event") != event_index:
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} override event",
                    event_index,
                    row.get("economic_override_event"),
                ))
            if (
                    not isinstance(event_index, (int, np.integer))
                    or not 0 <= int(event_index) < NUM_TRADE_EVENTS
                    or not _close(requested_action[1], override[int(event_index)])
            ):
                expected_economic = (
                    override[int(event_index)]
                    if isinstance(event_index, (int, np.integer))
                    and 0 <= int(event_index) < NUM_TRADE_EVENTS
                    else "valid event-indexed override"
                )
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} overridden economics",
                    expected_economic,
                    requested_action[1],
                ))
        if substep in (LEADER_QUERY, CACHED_TRADE_REPLAY):
            try:
                action_mask = np.asarray(
                    row.get("action_mask"), dtype=np.float64
                ).reshape(-1)
            except (TypeError, ValueError):
                action_mask = np.asarray([], dtype=np.float64)
            required_mask = np.asarray(
                CANONICAL_EVENT_ACTION_MASK, dtype=np.float64
            )
            if not np.array_equal(action_mask, required_mask):
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} "
                    "canonical event-only action mask",
                    required_mask.tolist(),
                    row.get("action_mask"),
                ))
        _audit_transition_phase_credit(
            episode_row, row, substep=substep, violations=violations,
        )


def _audit_transitions(
        rows,
        decisions,
        episode_row,
        *,
        gameplay_horizon,
        leader_role,
):
    episode = int(episode_row["evaluation_episode"])
    selected = [row for row in rows if int(row["evaluation_episode"]) == episode]
    selected_decisions = [
        row for row in decisions if int(row["evaluation_episode"]) == episode
    ]
    expected_outer = gameplay_horizon + 2 * NUM_TRADE_EVENTS
    violations = []
    if len(selected) != expected_outer:
        violations.append(_violation(
            episode_row, "retained transition rows", expected_outer, len(selected)
        ))
        return violations
    _audit_transition_layout(
        episode_row, selected, gameplay_horizon=gameplay_horizon,
        violations=violations,
    )
    _audit_transition_reward_schedule(
        episode_row, selected, gameplay_horizon=gameplay_horizon,
        violations=violations,
    )
    _audit_transition_intervention(
        episode_row, selected, violations=violations,
    )
    _audit_transition_decision_replays(
        episode_row,
        selected_decisions,
        leader_role=leader_role,
        violations=violations,
    )
    return violations


def audit_e2_protocol(
        evaluation,
        *,
        required_episodes,
        leader_role,
        gameplay_horizon,
        event_tail_steps=0,
        fixed_event_steps=None,
        seller_game_reward_scale=0.1,
        buyer_game_reward_scale=1.0,
):
    """Enforce every transition, trace, trade, bullet, and payoff identity."""

    episode_rows = list(evaluation.get("episode_rows", ()))
    transition_rows = list(evaluation.get("transition_rows", ()))
    decision_rows = list(evaluation.get("decision_rows", ()))
    violations = []
    expected_decision_rows = [
        row for row in transition_rows
        if row.get("substep_type") in (LEADER_QUERY, CACHED_TRADE_REPLAY)
    ]
    try:
        decisions_match_transitions = (
            _canonical_json_bytes(decision_rows)
            == _canonical_json_bytes(expected_decision_rows)
        )
    except (TypeError, ValueError):
        decisions_match_transitions = False
    if not decisions_match_transitions:
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "decision rows equal economic transition subset",
            "expected": expected_decision_rows,
            "actual": decision_rows,
        })

    first_query = next((
        row for row in transition_rows
        if row.get("substep_type") == LEADER_QUERY
    ), None)
    observed_event_mask = (
        None if first_query is None else first_query.get("action_mask")
    )
    try:
        normalized_event_mask = np.asarray(
            observed_event_mask, dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError):
        normalized_event_mask = np.asarray([], dtype=np.float64)
    if not np.array_equal(
            normalized_event_mask,
            np.asarray(CANONICAL_EVENT_ACTION_MASK, dtype=np.float64),
    ):
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "derived canonical event-only action mask",
            "expected": list(CANONICAL_EVENT_ACTION_MASK),
            "actual": observed_event_mask,
        })
    if len(episode_rows) != int(required_episodes):
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "episode_count",
            "expected": int(required_episodes),
            "actual": len(episode_rows),
        })
    for row in episode_rows:
        violations.extend(_audit_episode(
            row,
            leader_role=leader_role,
            gameplay_horizon=int(gameplay_horizon),
            event_tail_steps=int(event_tail_steps),
            fixed_event_steps=fixed_event_steps,
            seller_game_reward_scale=float(seller_game_reward_scale),
            buyer_game_reward_scale=float(buyer_game_reward_scale),
        ))
        violations.extend(_audit_transitions(
            transition_rows, decision_rows, row,
            gameplay_horizon=int(gameplay_horizon),
            leader_role=leader_role,
        ))

    traces = sorted({str(row.get("query_trace_sha256")) for row in episode_rows})
    commitments = sorted({
        _canonical_json_bytes(row.get("leader_commitment")).decode("ascii")
        for row in episode_rows
    })
    query_actions = sorted({
        _canonical_json_bytes(row.get("query_actions")).decode("ascii")
        for row in episode_rows
    })
    if len(traces) != 1:
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "deterministic query trace",
            "expected": "one trace SHA-256 across all episodes",
            "actual": traces,
        })
    if len(commitments) != 1:
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "deterministic leader commitment",
            "expected": "one five-action commitment across all episodes",
            "actual": commitments,
        })
    if len(query_actions) != 1:
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "deterministic full query actions",
            "expected": "one full five-action trace across all episodes",
            "actual": query_actions,
        })
    return {
        "passed": not violations,
        "episodes": len(episode_rows),
        "required_episodes": int(required_episodes),
        "expected_outer_transitions": int(gameplay_horizon) + 10,
        "expected_query_transitions": NUM_TRADE_EVENTS,
        "expected_gameplay_transitions": int(gameplay_horizon),
        "expected_cached_trade_replays": NUM_TRADE_EVENTS,
        "single_query_trace": len(traces) == 1,
        "single_leader_commitment": len(commitments) == 1,
        "single_full_query_action_trace": len(query_actions) == 1,
        "query_trace_sha256": traces[0] if len(traces) == 1 else None,
        "leader_commitment": (
            json.loads(commitments[0]) if len(commitments) == 1 else None
        ),
        "full_query_actions": (
            json.loads(query_actions[0]) if len(query_actions) == 1 else None
        ),
        "canonical_event_action_mask": (
            normalized_event_mask.tolist()
            if np.array_equal(
                normalized_event_mask,
                np.asarray(CANONICAL_EVENT_ACTION_MASK, dtype=np.float64),
            )
            else None
        ),
        "violations": violations,
    }
