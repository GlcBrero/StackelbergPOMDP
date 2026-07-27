"""Exact leader-query traces for the Atari Stackelberg-POMDP.

The PI response algorithm conditions on what the leader did during the
response phase.  Its source-of-truth context is therefore a *structured
trace*, not just the five prices (or thresholds) extracted from those actions.
For the canonical Atari protocol the trace contains five ordered pairs

``(actor-visible leader observation, full leader action)``.

The full action is ``[game_action, economic_action]`` even though the game
component is ignored while Atari is paused at a trade-only query.  Critic-only
observation entries (the ``"critic:"`` namespace) are deliberately omitted:
they are not behavior observed by the follower and may reveal the hidden
StackPOMDP phase.

This module keeps the exact trace and any compact neural-network encoding as
two distinct objects.  :class:`FixedCanonicalQueryFeatureEncoder` is lossless
only after it has verified that all five observations equal its bound
canonical templates.  Under that explicit condition it can omit the fixed
observations and encode the complete five-by-two action matrix in ten values.
The older five-economic-action context remains available only through the
explicitly named :func:`legacy_economic_action_projection`.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


CANONICAL_QUERY_COUNT = 5
FULL_ACTION_DIM = 2
CRITIC_PREFIX = "critic:"
TRACE_FORMAT = "stackpomdp.atari.leader_query_trace"
TRACE_FORMAT_VERSION = 1

REQUIRED_CANONICAL_EVENT_FIELDS = ("event_active", "event_one_hot")


class QueryTraceError(ValueError):
    """Base exception for invalid or incompatible leader-query traces."""


class ObservationTemplateMismatch(QueryTraceError):
    """Raised when a trace is not compatible with a fixed-observation encoder."""


@dataclass(frozen=True)
class ArrayPayload:
    """Immutable, exact representation of one numeric NumPy value.

    Shape, dtype (including byte order), and C-order value bytes are retained.
    Original strides are intentionally not retained because they are not part
    of the observation value supplied to a policy.
    """

    dtype: str
    shape: Tuple[int, ...]
    data: bytes

    @classmethod
    def capture(cls, value: Any) -> "ArrayPayload":
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise QueryTraceError(
                "query traces support only unstructured, non-object arrays"
            )
        contiguous = np.array(array, copy=True, order="C")
        return cls(
            dtype=contiguous.dtype.str,
            shape=tuple(int(size) for size in contiguous.shape),
            data=contiguous.tobytes(order="C"),
        )

    def __post_init__(self) -> None:
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise QueryTraceError(f"invalid array dtype {self.dtype!r}") from error
        if dtype.hasobject or dtype.fields is not None:
            raise QueryTraceError(
                "query traces support only unstructured, non-object arrays"
            )
        if any(int(size) < 0 for size in self.shape):
            raise QueryTraceError("array shapes cannot contain negative entries")
        element_count = int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1
        expected_bytes = element_count * dtype.itemsize
        if len(self.data) != expected_bytes:
            raise QueryTraceError(
                "array payload byte count does not match its dtype and shape: "
                f"{len(self.data)} != {expected_bytes}"
            )

    def reconstruct(self) -> np.ndarray:
        """Return a writable array independent of the immutable payload."""

        array = np.frombuffer(self.data, dtype=np.dtype(self.dtype))
        return array.reshape(self.shape).copy()

    def to_wire(self) -> Mapping[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_base64": b64encode(self.data).decode("ascii"),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ArrayPayload":
        required = {"dtype", "shape", "data_base64"}
        if set(payload) != required:
            raise QueryTraceError(
                "array wire payload must contain exactly "
                f"{sorted(required)}, got {sorted(payload)}"
            )
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
    """One named actor-visible observation value, preserving field order."""

    name: str
    value: ArrayPayload

    def __post_init__(self) -> None:
        if not self.name:
            raise QueryTraceError("observation field names cannot be empty")
        if self.name.startswith(CRITIC_PREFIX):
            raise QueryTraceError(
                f"critic-only field {self.name!r} cannot enter a follower trace"
            )

    def to_wire(self) -> Mapping[str, Any]:
        return {"name": self.name, "value": self.value.to_wire()}

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ObservationField":
        if set(payload) != {"name", "value"}:
            raise QueryTraceError(
                "observation field payload must contain exactly name and value"
            )
        return cls(
            name=str(payload["name"]),
            value=ArrayPayload.from_wire(payload["value"]),
        )


@dataclass(frozen=True)
class LeaderQuery:
    """One canonical leader observation and the full action taken there."""

    event_index: int
    observation_fields: Tuple[ObservationField, ...]
    full_action: ArrayPayload

    @classmethod
    def capture(
        cls,
        event_index: int,
        observation: Mapping[str, Any],
        full_action: Any,
    ) -> "LeaderQuery":
        if not isinstance(observation, Mapping):
            raise QueryTraceError("leader observations must be mappings")
        fields = tuple(
            ObservationField(str(name), ArrayPayload.capture(value))
            for name, value in observation.items()
            if not str(name).startswith(CRITIC_PREFIX)
        )
        return cls(
            event_index=int(event_index),
            observation_fields=fields,
            full_action=ArrayPayload.capture(full_action),
        )

    def __post_init__(self) -> None:
        names = tuple(field.name for field in self.observation_fields)
        if len(names) != len(set(names)):
            raise QueryTraceError("leader observations cannot repeat field names")
        if not names:
            raise QueryTraceError("leader observations cannot be empty")

    def observation(self) -> OrderedDict:
        return OrderedDict(
            (field.name, field.value.reconstruct())
            for field in self.observation_fields
        )

    def action(self) -> np.ndarray:
        return self.full_action.reconstruct()

    def to_wire(self) -> Mapping[str, Any]:
        return {
            "event_index": self.event_index,
            "observation_fields": [
                field.to_wire() for field in self.observation_fields
            ],
            "full_action": self.full_action.to_wire(),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "LeaderQuery":
        required = {"event_index", "observation_fields", "full_action"}
        if set(payload) != required:
            raise QueryTraceError(
                "query payload must contain exactly "
                f"{sorted(required)}, got {sorted(payload)}"
            )
        return cls(
            event_index=int(payload["event_index"]),
            observation_fields=tuple(
                ObservationField.from_wire(field)
                for field in payload["observation_fields"]
            ),
            full_action=ArrayPayload.from_wire(payload["full_action"]),
        )


@dataclass(frozen=True)
class LeaderQueryTrace:
    """The exact five-query context supplied to a PI meta-response."""

    queries: Tuple[LeaderQuery, ...]

    @classmethod
    def capture(
        cls,
        observations: Sequence[Mapping[str, Any]],
        full_actions: Sequence[Any],
    ) -> "LeaderQueryTrace":
        observations = tuple(observations)
        full_actions = tuple(full_actions)
        if len(observations) != len(full_actions):
            raise QueryTraceError(
                "leader observation and action sequences must have equal length"
            )
        trace = cls(tuple(
            LeaderQuery.capture(index, observation, action)
            for index, (observation, action) in enumerate(
                zip(observations, full_actions)
            )
        ))
        trace.validate_canonical()
        return trace

    def validate_canonical(self) -> None:
        if len(self.queries) != CANONICAL_QUERY_COUNT:
            raise QueryTraceError(
                "the canonical Atari response phase requires exactly "
                f"{CANONICAL_QUERY_COUNT} queries, got {len(self.queries)}"
            )
        expected_names = tuple(
            field.name for field in self.queries[0].observation_fields
        )
        missing = set(REQUIRED_CANONICAL_EVENT_FIELDS) - set(expected_names)
        if missing:
            raise QueryTraceError(
                "canonical leader observation is missing event fields: "
                f"{sorted(missing)}"
            )

        reference_schema = tuple(
            (field.name, field.value.dtype, field.value.shape)
            for field in self.queries[0].observation_fields
        )
        expected_action_dtype = self.queries[0].full_action.dtype
        for expected_index, query in enumerate(self.queries):
            if query.event_index != expected_index:
                raise QueryTraceError(
                    "canonical query indices must be ordered 0 through 4"
                )
            schema = tuple(
                (field.name, field.value.dtype, field.value.shape)
                for field in query.observation_fields
            )
            if schema != reference_schema:
                raise QueryTraceError(
                    "all canonical query observations must share one schema"
                )
            if query.full_action.shape != (FULL_ACTION_DIM,):
                raise QueryTraceError(
                    "each leader action must have shape (2,): "
                    "[game_action, economic_action]"
                )
            if query.full_action.dtype != expected_action_dtype:
                raise QueryTraceError(
                    "all five full actions must use the same dtype"
                )
            action = query.action()
            if action.dtype.kind not in "fiu" or not np.all(np.isfinite(action)):
                raise QueryTraceError("leader actions must be finite numeric values")

            observation = query.observation()
            event_active = np.asarray(observation["event_active"]).reshape(-1)
            if event_active.shape != (1,) or float(event_active[0]) != 1.0:
                raise QueryTraceError(
                    "canonical query observations require event_active=1"
                )
            one_hot = np.asarray(observation["event_one_hot"]).reshape(-1)
            expected_one_hot = np.zeros(CANONICAL_QUERY_COUNT, dtype=one_hot.dtype)
            expected_one_hot[expected_index] = 1
            if one_hot.shape != (CANONICAL_QUERY_COUNT,) or not np.array_equal(
                one_hot, expected_one_hot
            ):
                raise QueryTraceError(
                    "event_one_hot must identify the corresponding canonical query"
                )

    def reconstruct(self) -> Tuple[Tuple[OrderedDict, np.ndarray], ...]:
        """Return independent observation/action pairs in their original order."""

        return tuple(
            (query.observation(), query.action()) for query in self.queries
        )

    def full_action_matrix(self) -> np.ndarray:
        return np.stack([query.action() for query in self.queries], axis=0)

    def to_wire(self) -> Mapping[str, Any]:
        return {
            "format": TRACE_FORMAT,
            "version": TRACE_FORMAT_VERSION,
            "queries": [query.to_wire() for query in self.queries],
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "LeaderQueryTrace":
        required = {"format", "version", "queries"}
        if set(payload) != required:
            raise QueryTraceError(
                "trace payload must contain exactly "
                f"{sorted(required)}, got {sorted(payload)}"
            )
        if payload["format"] != TRACE_FORMAT:
            raise QueryTraceError(f"unknown query trace format {payload['format']!r}")
        if int(payload["version"]) != TRACE_FORMAT_VERSION:
            raise QueryTraceError(
                f"unsupported query trace version {payload['version']!r}"
            )
        trace = cls(tuple(
            LeaderQuery.from_wire(query) for query in payload["queries"]
        ))
        trace.validate_canonical()
        return trace

    def to_json_bytes(self) -> bytes:
        """Return a stable, safe serialization suitable for artifacts/hashing."""

        return json.dumps(
            self.to_wire(),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "LeaderQueryTrace":
        try:
            decoded = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QueryTraceError("invalid query trace JSON") from error
        if not isinstance(decoded, Mapping):
            raise QueryTraceError("query trace JSON must contain an object")
        return cls.from_wire(decoded)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True)
class FixedCanonicalQueryFeatureEncoder:
    """Lossless ten-feature encoding for fixed canonical observations.

    Row ``j`` contains ``[game_action_j, economic_action_j]``.  Query position
    supplies event identity.  The five observation templates are bound into
    the encoder and checked byte-for-byte on every call, so omitting them from
    the feature tensor does not silently discard varying follower context.
    """

    observation_templates: Tuple[Tuple[ObservationField, ...], ...]
    action_dtype: str

    @classmethod
    def bind(
        cls,
        reference_trace: LeaderQueryTrace,
    ) -> "FixedCanonicalQueryFeatureEncoder":
        reference_trace.validate_canonical()
        return cls(
            observation_templates=tuple(
                query.observation_fields for query in reference_trace.queries
            ),
            action_dtype=reference_trace.queries[0].full_action.dtype,
        )

    def __post_init__(self) -> None:
        if len(self.observation_templates) != CANONICAL_QUERY_COUNT:
            raise QueryTraceError("encoder requires five observation templates")
        dtype = np.dtype(self.action_dtype)
        if dtype.kind not in "fiu":
            raise QueryTraceError("encoder action dtype must be numeric")

    @property
    def feature_shape(self) -> Tuple[int, int]:
        return (CANONICAL_QUERY_COUNT, FULL_ACTION_DIM)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(
            f"query_{query_index + 1}.{component}"
            for query_index in range(CANONICAL_QUERY_COUNT)
            for component in ("game_action", "economic_action")
        )

    def encode(self, trace: LeaderQueryTrace) -> np.ndarray:
        trace.validate_canonical()
        for query, template in zip(trace.queries, self.observation_templates):
            if query.observation_fields != template:
                raise ObservationTemplateMismatch(
                    "leader query observations differ from the encoder's fixed "
                    f"canonical template at event {query.event_index + 1}"
                )
            if query.full_action.dtype != self.action_dtype:
                raise ObservationTemplateMismatch(
                    "leader action dtype differs from the bound trace schema"
                )
        return trace.full_action_matrix()

    def encode_flat(self, trace: LeaderQueryTrace) -> np.ndarray:
        return self.encode(trace).reshape(-1)

    def decode(self, features: Any) -> LeaderQueryTrace:
        values = np.asarray(features)
        if values.shape != self.feature_shape:
            raise QueryTraceError(
                f"query features must have shape {self.feature_shape}, "
                f"got {values.shape}"
            )
        if values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
            raise QueryTraceError("query features must be finite numeric values")
        cast_values = values.astype(np.dtype(self.action_dtype), copy=False)
        trace = LeaderQueryTrace(tuple(
            LeaderQuery(
                event_index=event_index,
                observation_fields=self.observation_templates[event_index],
                full_action=ArrayPayload.capture(cast_values[event_index]),
            )
            for event_index in range(CANONICAL_QUERY_COUNT)
        ))
        trace.validate_canonical()
        return trace

    def decode_flat(self, features: Any) -> LeaderQueryTrace:
        values = np.asarray(features)
        if values.shape != (CANONICAL_QUERY_COUNT * FULL_ACTION_DIM,):
            raise QueryTraceError(
                "flat query features must contain ten entries"
            )
        return self.decode(values.reshape(self.feature_shape))


def legacy_economic_action_projection(trace: LeaderQueryTrace) -> np.ndarray:
    """Project a full trace to the old five-price/threshold context.

    This projection is intentionally explicit because it is lossy in general:
    it discards all queried observations and all five queried game actions.  It
    is a sufficient statistic only when the canonical observations are fixed
    and the trade protocol ignores the game-action component.
    """

    trace.validate_canonical()
    return trace.full_action_matrix()[:, 1].copy()


def expand_legacy_economic_context_linear_weight(
    old_weight: Any,
    *,
    unchanged_prefix_width: int,
    unchanged_suffix_width: int = 0,
    query_count: int = CANONICAL_QUERY_COUNT,
) -> np.ndarray:
    """Expand a first-layer weight from five scalars to five full actions.

    The old input layout is

    ``[unchanged_prefix, economic_1, ..., economic_5, unchanged_suffix]``

    and the new layout is

    ``[unchanged_prefix, game_1, economic_1, ..., game_5, economic_5,
    unchanged_suffix]``.

    New game-action columns are initialized to zero and every other column is
    copied exactly.  With fixed canonical query observations and ignored query
    game actions this preserves the old network function exactly.
    """

    weight = np.asarray(old_weight)
    if weight.ndim != 2:
        raise QueryTraceError("linear-layer weights must be a two-dimensional array")
    prefix = int(unchanged_prefix_width)
    suffix = int(unchanged_suffix_width)
    query_count = int(query_count)
    if prefix < 0 or suffix < 0 or query_count <= 0:
        raise QueryTraceError(
            "prefix/suffix widths must be nonnegative and query count positive"
        )
    expected_input_width = prefix + query_count + suffix
    if weight.shape[1] != expected_input_width:
        raise QueryTraceError(
            "legacy weight has the wrong input width: "
            f"{weight.shape[1]} != {expected_input_width}"
        )

    expanded = np.zeros(
        (
            weight.shape[0],
            prefix + FULL_ACTION_DIM * query_count + suffix,
        ),
        dtype=weight.dtype,
    )
    expanded[:, :prefix] = weight[:, :prefix]
    for event_index in range(query_count):
        old_column = prefix + event_index
        new_economic_column = prefix + FULL_ACTION_DIM * event_index + 1
        expanded[:, new_economic_column] = weight[:, old_column]
    if suffix:
        old_suffix_start = prefix + query_count
        new_suffix_start = prefix + FULL_ACTION_DIM * query_count
        expanded[:, new_suffix_start:] = weight[:, old_suffix_start:]
    return expanded
