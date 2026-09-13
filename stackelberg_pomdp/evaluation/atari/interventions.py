"""Apply fixed economic commitments while preserving the gameplay action."""

import hashlib

import numpy as np

from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTOR_STATE,
    CACHED_TRADE_REPLAY,
    EVENT_SLICE,
    LEADER_QUERY,
    NUM_TRADE_EVENTS,
    TRADE_MODE_INDEX,
    canonical_leader_state,
)
from stackelberg_pomdp.evaluation.atari.contracts import (
    ECONOMIC_INTERVENTION_SCHEMA,
    _action_list,
    _canonical_json_bytes,
)


def _normalized_commitment(values):
    if values is None:
        return None
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
            result.shape != (NUM_TRADE_EVENTS,)
            or not np.all(np.isfinite(result))
            or not np.all((0.0 <= result) & (result <= 1.0))
    ):
        raise ValueError(
            "economic commitment override must contain five finite values "
            "in [0, 1]"
        )
    return tuple(float(value) for value in result)


def _canonical_economic_transition(observation):
    """Identify a canonical E2 query/replay without using hidden env state."""

    credit = np.asarray(observation[ACTION_CREDIT], dtype=np.float32).reshape(-1)
    state = np.asarray(observation[ACTOR_STATE], dtype=np.float32).reshape(-1)
    if state.shape[0] <= max(TRADE_MODE_INDEX, EVENT_SLICE.stop - 1):
        raise RuntimeError("E2 actor state is too short for an event identity")
    trade_mode = bool(np.isclose(
        state[TRADE_MODE_INDEX], 1.0, atol=0.0, rtol=0.0
    ))
    query_credit = np.array_equal(credit, np.asarray([0.0, 1.0]))
    replay_credit = np.array_equal(credit, np.asarray([0.0, 0.0]))
    if not trade_mode:
        if query_credit:
            raise RuntimeError("economic actor credit appeared outside trade mode")
        return None
    event_values = state[EVENT_SLICE]
    active = np.flatnonzero(np.isclose(event_values, 1.0, atol=0.0, rtol=0.0))
    if (
            active.shape != (1,)
            or not np.array_equal(
                event_values,
                np.eye(NUM_TRADE_EVENTS, dtype=np.float32)[active[0]],
            )
    ):
        raise RuntimeError("canonical E2 economic state has an invalid event one-hot")
    event_index = int(active[0])
    expected_state = canonical_leader_state(event_index)
    if not np.array_equal(state, expected_state):
        raise RuntimeError("canonical E2 economic state is not event-only")
    if query_credit:
        return LEADER_QUERY, event_index
    if replay_credit:
        return CACHED_TRADE_REPLAY, event_index
    raise RuntimeError("trade-mode E2 state has invalid action credit")


def apply_economic_commitment_override(observation, action, commitment):
    """Apply one deterministic commitment to both queries and cache replays.

    The policy's Atari coordinate is never changed.  The returned decision tag
    lets the evaluator prove that exactly five queries and five corresponding
    reward-trade replays were intervened on, while gameplay was untouched.
    """

    policy_action = _action_list(action)
    normalized = _normalized_commitment(commitment)
    if normalized is None:
        return policy_action, None
    decision = _canonical_economic_transition(observation)
    if decision is None:
        return policy_action, None
    kind, event_index = decision
    effective = list(policy_action)
    effective[1] = float(normalized[event_index])
    return effective, (kind, event_index)


def economic_intervention_manifest(
        *,
        intervention_id,
        commitment,
        checkpoint_hash,
        response_hash,
        config_hash,
        provenance_fingerprint,
):
    normalized = _normalized_commitment(commitment)
    body = {
        "schema": ECONOMIC_INTERVENTION_SCHEMA,
        "intervention_id": str(intervention_id),
        "target": "leader_economic_commitment",
        "economic_commitment": (
            None if normalized is None else list(normalized)
        ),
        "apply_to": [LEADER_QUERY, CACHED_TRADE_REPLAY],
        "preserve_candidate_game_coordinate": True,
        "no_direct_gameplay_action_override": True,
        "fresh_follower_response_episode": True,
        "checkpoint_sha256": str(checkpoint_hash),
        "response_sha256": str(response_hash),
        "environment_config_sha256": str(config_hash),
        "e2_provenance_fingerprint": str(provenance_fingerprint),
    }
    return {
        **body,
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
    }
