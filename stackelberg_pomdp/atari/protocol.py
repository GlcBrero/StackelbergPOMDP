"""Canonical observation, action, and policy-credit protocol for Atari.

The clean Atari curriculum uses one actor interface from E0a through E2.  The
only stage-specific inputs are critic-prefixed training fields; those fields
never enter an actor network or an observation--action cache key.
"""

from collections import OrderedDict

import gym
import numpy as np
from gym import spaces


NUM_TRADE_EVENTS = 5
ACTOR_STATE_DIM = 14
CRITIC_STATE_DIM = 32
FULL_ACTION_DIM = 2

IMAGE = "image"
ACTOR_STATE = "actor_state"
ACTION_MASK = "action_mask"
ACTOR_OBSERVATION_FIELDS = (IMAGE, ACTOR_STATE, ACTION_MASK)

CRITIC_PREFIX = "critic:"
CRITIC_STATE = f"{CRITIC_PREFIX}state"
ACTION_CREDIT = f"{CRITIC_PREFIX}action_credit"

AMMO_INDEX = 0
PROJECTILE_INDEX = 1
TIME_INDEX = 2
TRADE_MODE_INDEX = 3
EVENT_SLICE = slice(4, 9)
OPPONENT_COMMITMENT_SLICE = slice(9, 14)

GAMEPLAY = "gameplay"
FOLLOWER_TRADE = "follower_trade"
LEADER_QUERY = "leader_query"
CACHED_TRADE_REPLAY = "cached_trade_replay"
AUTOMATIC_TRANSFER = "automatic_transfer"
TERMINAL = "terminal"

_ACTION_CREDIT = {
    GAMEPLAY: (1.0, 0.0),
    FOLLOWER_TRADE: (0.0, 1.0),
    LEADER_QUERY: (0.0, 1.0),
    CACHED_TRADE_REPLAY: (0.0, 0.0),
    AUTOMATIC_TRANSFER: (0.0, 0.0),
    TERMINAL: (0.0, 0.0),
}


def event_one_hot(event_index):
    """Return the canonical five-entry event identity.

    ``None`` denotes that no trade event remains (or that the field is unused
    in E0a).  Otherwise the index must identify one of the five events.
    """

    result = np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
    if event_index is None:
        return result
    index = int(event_index)
    if not 0 <= index < NUM_TRADE_EVENTS:
        raise ValueError(f"event_index must lie in [0, 4], got {index}")
    result[index] = 1.0
    return result


def actor_state(
        *,
        ammo_fraction,
        projectile_active,
        normalized_time,
        trade_mode,
        event_index,
        opponent_commitment=None,
):
    """Build ``xi=(ammo, projectile, time, mode, event, commitment)``."""

    context = (
        np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        if opponent_commitment is None
        else np.asarray(opponent_commitment, dtype=np.float32).reshape(-1)
    )
    if context.shape != (NUM_TRADE_EVENTS,):
        raise ValueError("opponent_commitment must contain exactly five scalars")
    scalars = np.asarray(
        [ammo_fraction, projectile_active, normalized_time, trade_mode],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(scalars)) or not np.all(
            (0.0 <= scalars) & (scalars <= 1.0)
    ):
        raise ValueError("actor-state scalars must be finite values in [0, 1]")
    if not np.all(np.isfinite(context)) or not np.all(
            (0.0 <= context) & (context <= 1.0)
    ):
        raise ValueError("opponent commitment must lie in [0, 1]^5")

    result = np.zeros(ACTOR_STATE_DIM, dtype=np.float32)
    result[:4] = scalars
    result[EVENT_SLICE] = event_one_hot(event_index)
    result[OPPONENT_COMMITMENT_SLICE] = context
    return result


def canonical_leader_state(event_index):
    """Event-only state used at both a leader query and its cached replay."""

    return actor_state(
        ammo_fraction=0.0,
        projectile_active=0.0,
        normalized_time=0.0,
        trade_mode=1.0,
        event_index=event_index,
        opponent_commitment=np.zeros(NUM_TRADE_EVENTS, dtype=np.float32),
    )


def action_credit(decision_kind):
    """Return ``[game, economic]`` PPO log-probability gates.

    The gate is critic-prefixed because it is training bookkeeping, not an
    actor input.  In particular, a cached reward-trade replay has no actor
    credit even though its actor-visible observation matches the query.
    """

    try:
        values = _ACTION_CREDIT[str(decision_kind)]
    except KeyError as error:
        raise ValueError(f"unknown Atari decision kind: {decision_kind!r}") from error
    return np.asarray(values, dtype=np.float32)


def observation_space(image_space, game_action_count):
    """Return the single padded Dict space used by every clean stage."""

    count = int(game_action_count)
    if count <= 0:
        raise ValueError("game_action_count must be positive")
    if not isinstance(image_space, gym.spaces.Box):
        raise TypeError("image_space must be a Gym Box")
    return spaces.Dict(OrderedDict([
        (IMAGE, image_space),
        (
            ACTOR_STATE,
            spaces.Box(0.0, 1.0, shape=(ACTOR_STATE_DIM,), dtype=np.float32),
        ),
        (
            ACTION_MASK,
            spaces.Box(0.0, 1.0, shape=(count,), dtype=np.float32),
        ),
        (
            CRITIC_STATE,
            spaces.Box(
                -np.inf,
                np.inf,
                shape=(CRITIC_STATE_DIM,),
                dtype=np.float32,
            ),
        ),
        (
            ACTION_CREDIT,
            spaces.Box(0.0, 1.0, shape=(FULL_ACTION_DIM,), dtype=np.float32),
        ),
    ]))


def action_space(game_action_count):
    """Return ``[Atari action, normalized economic scalar]``."""

    count = int(game_action_count)
    if count <= 0:
        raise ValueError("game_action_count must be positive")
    return spaces.Box(
        low=np.array([0.0, 0.0], dtype=np.float32),
        high=np.array([float(count - 1), 1.0], dtype=np.float32),
        dtype=np.float32,
    )


def validate_action(action, game_action_count):
    """Return one clipped ``[game action, economic action]`` vector."""

    count = int(game_action_count)
    if count <= 0:
        raise ValueError("game_action_count must be positive")
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape != (FULL_ACTION_DIM,):
        raise ValueError("Atari action must be [game_action, economic]")
    return np.array([
        np.clip(values[0], 0.0, count - 1),
        np.clip(values[1], 0.0, 1.0),
    ], dtype=np.float32)


def observation(
        *,
        image,
        state,
        action_mask,
        decision_kind,
        critic_state=None,
):
    """Build one canonical observation without sharing mutable arrays."""

    state_values = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_values.shape != (ACTOR_STATE_DIM,):
        raise ValueError(f"actor state must have shape ({ACTOR_STATE_DIM},)")
    mask_values = np.asarray(action_mask, dtype=np.float32).reshape(-1)
    if mask_values.size <= 0:
        raise ValueError("action mask cannot be empty")
    critic_values = (
        np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
        if critic_state is None
        else np.asarray(critic_state, dtype=np.float32).reshape(-1)
    )
    if critic_values.shape != (CRITIC_STATE_DIM,):
        raise ValueError(f"critic state must have shape ({CRITIC_STATE_DIM},)")
    return OrderedDict([
        (IMAGE, np.array(image, copy=True)),
        (ACTOR_STATE, np.array(state_values, copy=True)),
        (ACTION_MASK, np.array(mask_values, copy=True)),
        (CRITIC_STATE, np.array(critic_values, copy=True)),
        (ACTION_CREDIT, action_credit(decision_kind)),
    ])


def actor_observation(values):
    """Copy exactly the fields that may influence the deployed actor."""

    return OrderedDict(
        (key, np.array(values[key], copy=True))
        for key in ACTOR_OBSERVATION_FIELDS
    )


def assert_actor_observations_equal(left, right):
    """Fail if two observations differ in any actor-visible coordinate."""

    for key in ACTOR_OBSERVATION_FIELDS:
        if not np.array_equal(np.asarray(left[key]), np.asarray(right[key])):
            raise RuntimeError(
                f"actor observations differ in {key!r}; cache hit is invalid"
            )


__all__ = [
    "ACTION_CREDIT",
    "ACTION_MASK",
    "ACTOR_OBSERVATION_FIELDS",
    "ACTOR_STATE",
    "ACTOR_STATE_DIM",
    "AMMO_INDEX",
    "AUTOMATIC_TRANSFER",
    "CACHED_TRADE_REPLAY",
    "CRITIC_STATE",
    "CRITIC_STATE_DIM",
    "CRITIC_PREFIX",
    "EVENT_SLICE",
    "FOLLOWER_TRADE",
    "FULL_ACTION_DIM",
    "GAMEPLAY",
    "IMAGE",
    "LEADER_QUERY",
    "NUM_TRADE_EVENTS",
    "OPPONENT_COMMITMENT_SLICE",
    "PROJECTILE_INDEX",
    "TERMINAL",
    "TIME_INDEX",
    "TRADE_MODE_INDEX",
    "action_credit",
    "action_space",
    "actor_observation",
    "actor_state",
    "assert_actor_observations_equal",
    "canonical_leader_state",
    "event_one_hot",
    "observation",
    "observation_space",
    "validate_action",
]
