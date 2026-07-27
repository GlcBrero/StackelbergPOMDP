from collections import OrderedDict

import numpy as np
import pytest

from stackelberg_pomdp.atari.query_trace import (
    CANONICAL_QUERY_COUNT,
    FixedCanonicalQueryFeatureEncoder,
    LeaderQueryTrace,
    ObservationTemplateMismatch,
    QueryTraceError,
    expand_legacy_economic_context_linear_weight,
    legacy_economic_action_projection,
)


def canonical_observations(*, critic_offset=0.0):
    observations = []
    for event_index in range(CANONICAL_QUERY_COUNT):
        one_hot = np.zeros(CANONICAL_QUERY_COUNT, dtype=np.float32)
        one_hot[event_index] = 1.0
        observations.append(OrderedDict([
            ("image", np.zeros((4, 6, 6), dtype=np.uint8)),
            ("ammo_fraction", np.array([0.0], dtype=np.float32)),
            ("projectile_active", np.array([0.0], dtype=np.float32)),
            ("action_mask", np.ones(6, dtype=np.float32)),
            ("offer_active", np.array([0.0], dtype=np.float32)),
            ("opportunities_remaining", np.array([1.0], dtype=np.float32)),
            ("critic:price", np.array([critic_offset + event_index], dtype=np.float32)),
            ("event_active", np.array([1.0], dtype=np.float32)),
            ("event_one_hot", one_hot),
            ("opponent_context", np.zeros(5, dtype=np.float32)),
            (
                "critic:state",
                np.full(12, critic_offset + event_index, dtype=np.float32),
            ),
        ]))
    return observations


def canonical_actions():
    return [
        np.array([event_index % 3, 0.1 + 0.15 * event_index], dtype=np.float32)
        for event_index in range(CANONICAL_QUERY_COUNT)
    ]


def assert_array_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def test_full_actor_visible_trace_round_trips_exactly_and_filters_critic():
    observations = canonical_observations(critic_offset=4.0)
    actions = canonical_actions()
    trace = LeaderQueryTrace.capture(observations, actions)

    reconstructed = trace.reconstruct()
    expected_actor_keys = tuple(
        key for key in observations[0] if not key.startswith("critic:")
    )
    for event_index, (observation, action) in enumerate(reconstructed):
        assert tuple(observation) == expected_actor_keys
        assert not any(key.startswith("critic:") for key in observation)
        for key in expected_actor_keys:
            assert_array_exact(observation[key], observations[event_index][key])
        assert_array_exact(action, actions[event_index])

    serialized = trace.to_json_bytes()
    restored = LeaderQueryTrace.from_json_bytes(serialized)
    assert restored == trace
    assert restored.sha256 == trace.sha256

    # Capturing owns the values; subsequent environment mutation cannot alter
    # the PI response context.
    observations[0]["image"][0, 0, 0] = 255
    actions[0][1] = 0.99
    assert restored == trace


def test_critic_only_state_never_changes_the_follower_context():
    actions = canonical_actions()
    trace_a = LeaderQueryTrace.capture(
        canonical_observations(critic_offset=0.0), actions
    )
    trace_b = LeaderQueryTrace.capture(
        canonical_observations(critic_offset=100.0), actions
    )
    assert trace_a == trace_b
    assert trace_a.sha256 == trace_b.sha256


def test_fixed_template_encoder_is_lossless_and_rejects_actor_drift():
    trace = LeaderQueryTrace.capture(canonical_observations(), canonical_actions())
    encoder = FixedCanonicalQueryFeatureEncoder.bind(trace)

    encoded = encoder.encode(trace)
    assert encoded.shape == (5, 2)
    assert encoder.feature_names == (
        "query_1.game_action",
        "query_1.economic_action",
        "query_2.game_action",
        "query_2.economic_action",
        "query_3.game_action",
        "query_3.economic_action",
        "query_4.game_action",
        "query_4.economic_action",
        "query_5.game_action",
        "query_5.economic_action",
    )
    assert encoder.decode(encoded) == trace
    assert encoder.decode_flat(encoder.encode_flat(trace)) == trace

    changed_observations = canonical_observations()
    changed_observations[2]["ammo_fraction"][0] = 0.2
    changed_trace = LeaderQueryTrace.capture(
        changed_observations, canonical_actions()
    )
    with pytest.raises(ObservationTemplateMismatch, match="event 3"):
        encoder.encode(changed_trace)


def test_legacy_projection_is_named_and_weight_expansion_preserves_network():
    trace = LeaderQueryTrace.capture(canonical_observations(), canonical_actions())
    full_actions = trace.full_action_matrix()
    legacy_context = legacy_economic_action_projection(trace)
    np.testing.assert_array_equal(legacy_context, full_actions[:, 1])

    # Current meta-response economic input is ammo (1), event one-hot (5),
    # then five economic actions: unchanged prefix width = 6.
    rng = np.random.default_rng(123)
    prefix = rng.normal(size=6).astype(np.float32)
    old_input = np.concatenate([prefix, legacy_context])
    old_weight = rng.normal(size=(7, old_input.size)).astype(np.float32)
    bias = rng.normal(size=7).astype(np.float32)

    new_weight = expand_legacy_economic_context_linear_weight(
        old_weight, unchanged_prefix_width=prefix.size
    )
    new_input = np.concatenate([prefix, full_actions.reshape(-1)])
    np.testing.assert_allclose(
        new_weight @ new_input + bias,
        old_weight @ old_input + bias,
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(new_weight[:, prefix.size::2], 0.0)

    # The value network has critic-only features after the context.  Those
    # suffix columns are copied exactly as well.
    suffix = rng.normal(size=3).astype(np.float32)
    old_value_input = np.concatenate([prefix, legacy_context, suffix])
    old_value_weight = rng.normal(
        size=(4, old_value_input.size)
    ).astype(np.float32)
    new_value_weight = expand_legacy_economic_context_linear_weight(
        old_value_weight,
        unchanged_prefix_width=prefix.size,
        unchanged_suffix_width=suffix.size,
    )
    new_value_input = np.concatenate([
        prefix,
        full_actions.reshape(-1),
        suffix,
    ])
    np.testing.assert_allclose(
        new_value_weight @ new_value_input,
        old_value_weight @ old_value_input,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_trace_rejects_noncanonical_event_identity_and_partial_actions():
    observations = canonical_observations()
    observations[3]["event_one_hot"].fill(0.0)
    with pytest.raises(QueryTraceError, match="event_one_hot"):
        LeaderQueryTrace.capture(observations, canonical_actions())

    partial_actions = canonical_actions()
    partial_actions[1] = np.array([0.5], dtype=np.float32)
    with pytest.raises(QueryTraceError, match=r"shape \(2,\)"):
        LeaderQueryTrace.capture(canonical_observations(), partial_actions)
