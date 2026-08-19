import pickle

import numpy as np
import pytest

from stackelberg_pomdp.atari.sampling import (
    ALL_EQUAL_E1_SAMPLER,
    CONTEXT_STRATUM_WEIGHTS,
    EARLY_FIFTH_INTERVAL,
    LATE_FIFTH_INTERVAL,
    LOW_PREFIX_PRICE_HIGH,
    SCHEDULE_STRATUM_WEIGHTS,
    TEMPORAL_MIX_E1_SAMPLER,
    TemporalMarginalE1Sampler,
    e1_sampler_provenance,
    sample_all_equal_e1_context,
)


def _draw_signature(draw):
    return (
        draw.event_steps,
        tuple(float(value) for value in draw.opponent_commitment),
        draw.schedule_stratum,
        draw.context_stratum,
    )


def test_all_equal_sampler_is_exact_uniform_and_reproducible():
    first_rng = np.random.default_rng(17)
    second_rng = np.random.default_rng(17)
    other_rng = np.random.default_rng(18)

    first = np.stack([
        sample_all_equal_e1_context(first_rng) for _ in range(20_000)
    ])
    second = np.stack([
        sample_all_equal_e1_context(second_rng) for _ in range(20_000)
    ])
    other = np.stack([
        sample_all_equal_e1_context(other_rng) for _ in range(20_000)
    ])

    assert first.shape == (20_000, 5)
    assert first.dtype == np.float32
    assert np.all(first == first[:, :1])
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all(first >= 0.0)
    assert np.all(first <= 1.0)

    shared = first[:, 0]
    assert float(np.mean(shared)) == pytest.approx(0.5, abs=0.01)
    assert float(np.var(shared)) == pytest.approx(1.0 / 12.0, abs=0.004)
    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert float(np.mean(shared <= quantile)) == pytest.approx(
            quantile, abs=0.012
        )


def test_all_equal_sampler_provenance_is_explicit():
    provenance = e1_sampler_provenance(
        ALL_EQUAL_E1_SAMPLER,
        gameplay_horizon=200,
        event_tail_steps=0,
    )

    assert provenance == {
        "mode": ALL_EQUAL_E1_SAMPLER,
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "schedule": "ExactFiveEventSchedule.sample",
        "context": "one Uniform(0,1) scalar replicated across five events",
        "context_scalar_distribution": "Uniform(0,1)",
        "context_replication_count": 5,
        "commitment_entries_all_equal": True,
        "per_event_marginal": "Uniform(0,1)",
        "schedule_context_rngs_independent": False,
        "legacy_sampling_path": False,
    }


def test_temporal_sampler_matches_marginals_bounds_and_independence():
    assert SCHEDULE_STRATUM_WEIGHTS == {
        "unconditional": 0.50,
        "early_fifth": 0.25,
        "late_fifth": 0.25,
    }
    assert CONTEXT_STRATUM_WEIGHTS == {
        "uniform": 0.75,
        "low_prefix": 0.25,
    }
    assert EARLY_FIFTH_INTERVAL == (120, 160)
    assert LATE_FIFTH_INTERVAL == (180, 200)
    assert LOW_PREFIX_PRICE_HIGH == 0.25
    sampler = TemporalMarginalE1Sampler(
        seed=41,
        gameplay_horizon=200,
        event_tail_steps=0,
    )
    draws = [sampler.sample() for _ in range(20_000)]

    schedule_counts = {
        name: sum(draw.schedule_stratum == name for draw in draws)
        for name in SCHEDULE_STRATUM_WEIGHTS
    }
    context_counts = {
        name: sum(draw.context_stratum == name for draw in draws)
        for name in CONTEXT_STRATUM_WEIGHTS
    }
    for name, probability in SCHEDULE_STRATUM_WEIGHTS.items():
        assert schedule_counts[name] / len(draws) == pytest.approx(
            probability, abs=0.015
        )
    for name, probability in CONTEXT_STRATUM_WEIGHTS.items():
        assert context_counts[name] / len(draws) == pytest.approx(
            probability, abs=0.015
        )

    for draw in draws:
        assert len(draw.event_steps) == 5
        assert draw.event_steps == tuple(sorted(set(draw.event_steps)))
        assert 0 <= draw.event_steps[0] <= draw.event_steps[-1] < 200
        if draw.schedule_stratum == "early_fifth":
            assert EARLY_FIFTH_INTERVAL[0] <= draw.event_steps[-1]
            assert draw.event_steps[-1] < EARLY_FIFTH_INTERVAL[1]
        elif draw.schedule_stratum == "late_fifth":
            assert LATE_FIFTH_INTERVAL[0] <= draw.event_steps[-1]
            assert draw.event_steps[-1] < LATE_FIFTH_INTERVAL[1]

        context = draw.opponent_commitment
        assert context.shape == (5,)
        assert context.dtype == np.float32
        assert np.all(context >= 0.0)
        assert np.all(context <= 1.0)
        if draw.context_stratum == "low_prefix":
            assert np.all(context[:4] <= LOW_PREFIX_PRICE_HIGH)

    # Schedule and price strata use independent RNG streams.  Consequently,
    # every empirical joint mass should be close to the product of its
    # empirical marginals, rather than leaking one curriculum selector into
    # the other.
    for schedule_name in SCHEDULE_STRATUM_WEIGHTS:
        for context_name in CONTEXT_STRATUM_WEIGHTS:
            observed = sum(
                draw.schedule_stratum == schedule_name
                and draw.context_stratum == context_name
                for draw in draws
            ) / len(draws)
            expected = (
                schedule_counts[schedule_name]
                * context_counts[context_name]
                / len(draws) ** 2
            )
            assert observed == pytest.approx(expected, abs=0.012)


def test_temporal_sampler_is_reproducible_and_pickle_safe():
    first = TemporalMarginalE1Sampler(seed=7, gameplay_horizon=200)
    second = TemporalMarginalE1Sampler(seed=7, gameplay_horizon=200)
    other = TemporalMarginalE1Sampler(seed=8, gameplay_horizon=200)

    first_draws = [_draw_signature(first.sample()) for _ in range(30)]
    assert first_draws == [_draw_signature(second.sample()) for _ in range(30)]
    assert first_draws != [_draw_signature(other.sample()) for _ in range(30)]

    restored = pickle.loads(pickle.dumps(first))
    assert [
        _draw_signature(first.sample()) for _ in range(30)
    ] == [
        _draw_signature(restored.sample()) for _ in range(30)
    ]


def test_temporal_sampler_provenance_and_horizon_contract():
    provenance = e1_sampler_provenance(
        TEMPORAL_MIX_E1_SAMPLER,
        gameplay_horizon=200,
        event_tail_steps=0,
    )
    assert provenance["gameplay_horizon"] == 200
    assert provenance["event_tail_steps"] == 0
    assert provenance["schedule_stratum_weights"] == (
        SCHEDULE_STRATUM_WEIGHTS
    )
    assert provenance["context_stratum_weights"] == CONTEXT_STRATUM_WEIGHTS
    assert provenance["schedule_context_rngs_independent"]

    with pytest.raises(ValueError, match="gameplay_horizon=200"):
        TemporalMarginalE1Sampler(seed=1, gameplay_horizon=199)
    with pytest.raises(ValueError, match="event_tail_steps=0"):
        TemporalMarginalE1Sampler(
            seed=1, gameplay_horizon=200, event_tail_steps=1
        )
    with pytest.raises(ValueError, match="fixed event steps"):
        TemporalMarginalE1Sampler(
            seed=1,
            gameplay_horizon=200,
            fixed_event_steps=(0, 1, 2, 3, 4),
        )
    with pytest.raises(ValueError, match="event_tail_steps=0"):
        e1_sampler_provenance(
            TEMPORAL_MIX_E1_SAMPLER,
            gameplay_horizon=200,
            event_tail_steps=1,
        )
