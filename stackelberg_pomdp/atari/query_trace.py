"""Exact, serializable leader-query traces for the Atari PI response.

The exact ordered trace ``Q`` is authoritative.  The five economic actions are
also exposed as the declared opponent commitment ``omega``; this is a chosen
response statistic, not a claim that five numbers losslessly encode arbitrary
observations and full actions.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from stackelberg_pomdp.atari.protocol import (
    ACTOR_OBSERVATION_FIELDS,
    ACTOR_STATE,
    ACTOR_STATE_DIM,
    CRITIC_PREFIX,
    EVENT_SLICE,
    FULL_ACTION_DIM,
    NUM_TRADE_EVENTS,
    TRADE_MODE_INDEX,
)


TRACE_FORMAT = "stackpomdp.atari.leader_query_trace"
TRACE_FORMAT_VERSION = 2


class QueryTraceError(ValueError):
    """Raised when an exact leader trace is incomplete or malformed."""


@dataclass(frozen=True)
class ArrayPayload:
    dtype: str
    shape: Tuple[int, ...]
    data: bytes

    @classmethod
    def capture(cls, value: Any) -> "ArrayPayload":
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise QueryTraceError("query arrays must be numeric and unstructured")
        copied = np.array(array, copy=True, order="C")
        return cls(
            dtype=copied.dtype.str,
            shape=tuple(int(size) for size in copied.shape),
            data=copied.tobytes(order="C"),
        )

    def __post_init__(self):
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise QueryTraceError(f"invalid array dtype {self.dtype!r}") from error
        if dtype.hasobject or dtype.fields is not None:
            raise QueryTraceError("query arrays must be numeric and unstructured")
        if any(int(size) < 0 for size in self.shape):
            raise QueryTraceError("array shape cannot contain negative entries")
        entries = int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1
        if len(self.data) != entries * dtype.itemsize:
            raise QueryTraceError("array payload byte count is inconsistent")

    def reconstruct(self):
        return np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(
            self.shape
        ).copy()

    def to_wire(self):
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_base64": b64encode(self.data).decode("ascii"),
        }

    @classmethod
    def from_wire(cls, payload):
        if set(payload) != {"dtype", "shape", "data_base64"}:
            raise QueryTraceError("invalid array wire payload")
        try:
            data = b64decode(payload["data_base64"], validate=True)
        except Exception as error:
            raise QueryTraceError("invalid base64 array payload") from error
        return cls(
            dtype=str(payload["dtype"]),
            shape=tuple(int(size) for size in payload["shape"]),
            data=data,
        )


@dataclass(frozen=True)
class ObservationField:
    name: str
    value: ArrayPayload

    def __post_init__(self):
        if self.name not in ACTOR_OBSERVATION_FIELDS:
            raise QueryTraceError(
                f"invalid actor-visible observation field {self.name!r}"
            )

    def to_wire(self):
        return {"name": self.name, "value": self.value.to_wire()}

    @classmethod
    def from_wire(cls, payload):
        if set(payload) != {"name", "value"}:
            raise QueryTraceError("invalid observation-field payload")
        return cls(
            name=str(payload["name"]),
            value=ArrayPayload.from_wire(payload["value"]),
        )


@dataclass(frozen=True)
class LeaderQuery:
    event_index: int
    observation_fields: Tuple[ObservationField, ...]
    full_action: ArrayPayload

    @classmethod
    def capture(cls, event_index, observation, full_action):
        if not isinstance(observation, Mapping):
            raise QueryTraceError("leader observation must be a mapping")
        actor_names = tuple(
            str(name)
            for name in observation
            if not str(name).startswith(CRITIC_PREFIX)
        )
        if set(actor_names) != set(ACTOR_OBSERVATION_FIELDS):
            raise QueryTraceError(
                "leader observation must contain exactly the canonical "
                f"actor fields {ACTOR_OBSERVATION_FIELDS!r}"
            )
        return cls(
            event_index=int(event_index),
            observation_fields=tuple(
                ObservationField(name, ArrayPayload.capture(observation[name]))
                for name in ACTOR_OBSERVATION_FIELDS
            ),
            full_action=ArrayPayload.capture(full_action),
        )

    def __post_init__(self):
        names = tuple(field.name for field in self.observation_fields)
        if not names or len(names) != len(set(names)):
            raise QueryTraceError("leader observation fields must be unique")

    def observation(self):
        return OrderedDict(
            (field.name, field.value.reconstruct())
            for field in self.observation_fields
        )

    def action(self):
        return self.full_action.reconstruct()

    def to_wire(self):
        return {
            "event_index": self.event_index,
            "observation_fields": [
                field.to_wire() for field in self.observation_fields
            ],
            "full_action": self.full_action.to_wire(),
        }

    @classmethod
    def from_wire(cls, payload):
        if set(payload) != {"event_index", "observation_fields", "full_action"}:
            raise QueryTraceError("invalid leader-query payload")
        return cls(
            event_index=int(payload["event_index"]),
            observation_fields=tuple(
                ObservationField.from_wire(value)
                for value in payload["observation_fields"]
            ),
            full_action=ArrayPayload.from_wire(payload["full_action"]),
        )


@dataclass(frozen=True)
class LeaderQueryTrace:
    """The exact five actor-observation/full-action query pairs."""

    queries: Tuple[LeaderQuery, ...]

    @classmethod
    def capture(cls, observations: Sequence, full_actions: Sequence):
        observations = tuple(observations)
        full_actions = tuple(full_actions)
        if len(observations) != len(full_actions):
            raise QueryTraceError("observation/action sequence lengths differ")
        result = cls(tuple(
            LeaderQuery.capture(index, obs, action)
            for index, (obs, action) in enumerate(
                zip(observations, full_actions)
            )
        ))
        result.validate_canonical()
        return result

    def validate_canonical(self):
        if len(self.queries) != NUM_TRADE_EVENTS:
            raise QueryTraceError("canonical Atari trace requires five queries")
        first_names = tuple(
            field.name for field in self.queries[0].observation_fields
        )
        if first_names != ACTOR_OBSERVATION_FIELDS:
            raise QueryTraceError(
                "canonical trace requires the ordered actor fields "
                f"{ACTOR_OBSERVATION_FIELDS!r}"
            )
        schema = tuple(
            (field.name, field.value.dtype, field.value.shape)
            for field in self.queries[0].observation_fields
        )
        action_dtype = self.queries[0].full_action.dtype
        for expected_index, query in enumerate(self.queries):
            if query.event_index != expected_index:
                raise QueryTraceError("query events must be ordered zero through four")
            current_schema = tuple(
                (field.name, field.value.dtype, field.value.shape)
                for field in query.observation_fields
            )
            if current_schema != schema:
                raise QueryTraceError("query observations must share one schema")
            if query.full_action.shape != (FULL_ACTION_DIM,):
                raise QueryTraceError("full query action must have shape (2,)")
            if query.full_action.dtype != action_dtype:
                raise QueryTraceError("query actions must share one dtype")
            action = query.action()
            if action.dtype.kind not in "fiu" or not np.all(np.isfinite(action)):
                raise QueryTraceError("query actions must be finite and numeric")

            state = np.asarray(query.observation()[ACTOR_STATE]).reshape(-1)
            if state.shape != (ACTOR_STATE_DIM,):
                raise QueryTraceError(
                    "canonical actor state must have "
                    f"{ACTOR_STATE_DIM} entries"
                )
            if float(state[TRADE_MODE_INDEX]) != 1.0:
                raise QueryTraceError("canonical query must be a trade-mode state")
            expected_event = np.zeros(NUM_TRADE_EVENTS, dtype=state.dtype)
            expected_event[expected_index] = 1
            if not np.array_equal(state[EVENT_SLICE], expected_event):
                raise QueryTraceError("canonical state has the wrong event identity")

    def reconstruct(self):
        return tuple(
            (query.observation(), query.action()) for query in self.queries
        )

    def full_action_matrix(self):
        return np.stack([query.action() for query in self.queries], axis=0)

    @property
    def economic_commitment(self):
        """The declared five-action response statistic ``omega``."""

        return self.full_action_matrix()[:, 1].copy()

    def to_wire(self):
        return {
            "format": TRACE_FORMAT,
            "version": TRACE_FORMAT_VERSION,
            "queries": [query.to_wire() for query in self.queries],
        }

    @classmethod
    def from_wire(cls, payload):
        if set(payload) != {"format", "version", "queries"}:
            raise QueryTraceError("invalid query-trace payload")
        if payload["format"] != TRACE_FORMAT:
            raise QueryTraceError(f"unknown trace format {payload['format']!r}")
        if int(payload["version"]) != TRACE_FORMAT_VERSION:
            raise QueryTraceError(
                f"unsupported trace version {payload['version']!r}"
            )
        result = cls(tuple(
            LeaderQuery.from_wire(query) for query in payload["queries"]
        ))
        result.validate_canonical()
        return result

    def to_json_bytes(self):
        return json.dumps(
            self.to_wire(), ensure_ascii=True, separators=(",", ":")
        ).encode("ascii")

    @classmethod
    def from_json_bytes(cls, payload):
        try:
            decoded = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QueryTraceError("invalid query-trace JSON") from error
        if not isinstance(decoded, Mapping):
            raise QueryTraceError("query-trace JSON must contain an object")
        return cls.from_wire(decoded)

    @property
    def sha256(self):
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


__all__ = [
    "ArrayPayload",
    "LeaderQuery",
    "LeaderQueryTrace",
    "ObservationField",
    "QueryTraceError",
]
