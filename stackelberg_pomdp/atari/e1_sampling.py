"""Canonical episode samplers for E1 opponent commitments and schedules."""

from dataclasses import dataclass

import numpy as np

from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS
from stackelberg_pomdp.atari.schedule import ExactFiveEventSchedule


UNIFORM_E1_SAMPLER = "uniform"
ALL_EQUAL_E1_SAMPLER = "all-equal-v1"
TEMPORAL_MIX_E1_SAMPLER = "temporal-marginal-v1"
E1_SAMPLER_MODES = (
    UNIFORM_E1_SAMPLER,
    ALL_EQUAL_E1_SAMPLER,
    TEMPORAL_MIX_E1_SAMPLER,
)

TEMPORAL_MIX_GAMEPLAY_HORIZON = 200
SCHEDULE_STRATA = (
    "unconditional",
    "early_fifth",
    "late_fifth",
    "fixed",
)
CONTEXT_STRATA = ("uniform", "all_equal", "low_prefix", "external")
SCHEDULE_STRATUM_WEIGHTS = {
    "unconditional": 0.50,
    "early_fifth": 0.25,
    "late_fifth": 0.25,
}
CONTEXT_STRATUM_WEIGHTS = {"uniform": 0.75, "low_prefix": 0.25}
EARLY_FIFTH_INTERVAL = (120, 160)
LATE_FIFTH_INTERVAL = (180, 200)
LOW_PREFIX_PRICE_HIGH = 0.25


def e1_sampler_provenance(mode, *, gameplay_horizon, event_tail_steps=0):
    """Return the immutable scientific contract for one sampler mode."""

    mode = str(mode)
    horizon = int(gameplay_horizon)
    tail_steps = int(event_tail_steps)
    if mode not in E1_SAMPLER_MODES:
        raise ValueError(f"unknown E1 sampler mode: {mode!r}")
    if mode == UNIFORM_E1_SAMPLER:
        return {
            "mode": mode,
            "gameplay_horizon": horizon,
            "event_tail_steps": tail_steps,
            "schedule": "ExactFiveEventSchedule.sample",
            "context": "five independent Uniform(0,1) prices",
            "schedule_context_rngs_independent": False,
            "legacy_sampling_path": True,
        }
    if mode == ALL_EQUAL_E1_SAMPLER:
        return {
            "mode": mode,
            "gameplay_horizon": horizon,
            "event_tail_steps": tail_steps,
            "schedule": "ExactFiveEventSchedule.sample",
            "context": (
                "one Uniform(0,1) scalar replicated across five events"
            ),
            "context_scalar_distribution": "Uniform(0,1)",
            "context_replication_count": NUM_TRADE_EVENTS,
            "commitment_entries_all_equal": True,
            "per_event_marginal": "Uniform(0,1)",
            "schedule_context_rngs_independent": False,
            "legacy_sampling_path": False,
        }
    if horizon != TEMPORAL_MIX_GAMEPLAY_HORIZON:
        raise ValueError(
            f"{TEMPORAL_MIX_E1_SAMPLER} requires gameplay_horizon="
            f"{TEMPORAL_MIX_GAMEPLAY_HORIZON}"
        )
    if tail_steps != 0:
        raise ValueError(
            f"{TEMPORAL_MIX_E1_SAMPLER} requires event_tail_steps=0"
        )
    return {
        "mode": mode,
        "gameplay_horizon": horizon,
        "event_tail_steps": 0,
        "schedule_stratum_weights": dict(SCHEDULE_STRATUM_WEIGHTS),
        "early_fifth_interval_half_open": list(EARLY_FIFTH_INTERVAL),
        "late_fifth_interval_half_open": list(LATE_FIFTH_INTERVAL),
        "conditional_schedule_method": (
            "rejection sample from ExactFiveEventSchedule"
        ),
        "context_stratum_weights": dict(CONTEXT_STRATUM_WEIGHTS),
        "uniform_context": "five independent Uniform(0,1) prices",
        "low_prefix_context": {
            "first_four": (
                f"four independent Uniform(0,{LOW_PREFIX_PRICE_HIGH}) prices"
            ),
            "fifth": "Uniform(0,1) price",
        },
        "schedule_context_rngs_independent": True,
        "legacy_sampling_path": False,
    }


@dataclass(frozen=True)
class E1SamplerDraw:
    event_steps: tuple
    opponent_commitment: np.ndarray
    schedule_stratum: str
    context_stratum: str


def sample_all_equal_e1_context(rng):
    """Draw one uniform scalar and repeat it at all five trade events."""

    shared_value = np.float32(rng.uniform(0.0, 1.0))
    return np.full(NUM_TRADE_EVENTS, shared_value, dtype=np.float32)


class TemporalMarginalE1Sampler:
    """Draw broad timing/value contrasts without using selector conditions."""

    _SCHEDULE_SEED_OFFSET = 1_904_711
    _CONTEXT_SEED_OFFSET = 7_313_993

    def __init__(
            self,
            *,
            seed,
            gameplay_horizon,
            event_tail_steps=0,
            fixed_event_steps=None,
    ):
        self.gameplay_horizon = int(gameplay_horizon)
        if self.gameplay_horizon != TEMPORAL_MIX_GAMEPLAY_HORIZON:
            raise ValueError(
                f"{TEMPORAL_MIX_E1_SAMPLER} requires gameplay_horizon="
                f"{TEMPORAL_MIX_GAMEPLAY_HORIZON}"
            )
        if int(event_tail_steps) != 0:
            raise ValueError(
                f"{TEMPORAL_MIX_E1_SAMPLER} requires event_tail_steps=0"
            )
        if fixed_event_steps is not None:
            raise ValueError(
                f"{TEMPORAL_MIX_E1_SAMPLER} is incompatible with fixed event "
                "steps"
            )
        self.schedule = ExactFiveEventSchedule(
            gameplay_horizon=self.gameplay_horizon,
            tail_steps=0,
        )
        self.seed(seed)

    def seed(self, seed):
        seed = int(seed)
        self.schedule_rng = np.random.default_rng(
            seed + self._SCHEDULE_SEED_OFFSET
        )
        self.context_rng = np.random.default_rng(
            seed + self._CONTEXT_SEED_OFFSET
        )

    @staticmethod
    def _stratum(rng, weights):
        draw = float(rng.random())
        cumulative = 0.0
        for name, probability in weights.items():
            cumulative += float(probability)
            if draw < cumulative:
                return name
        return next(reversed(weights))

    def _conditioned_schedule(self, interval):
        low, high = (int(value) for value in interval)
        while True:
            values = self.schedule.sample(self.schedule_rng)
            if low <= values[-1] < high:
                return values

    def sample_schedule(self):
        stratum = self._stratum(
            self.schedule_rng, SCHEDULE_STRATUM_WEIGHTS
        )
        if stratum == "unconditional":
            values = self.schedule.sample(self.schedule_rng)
        elif stratum == "early_fifth":
            values = self._conditioned_schedule(EARLY_FIFTH_INTERVAL)
        elif stratum == "late_fifth":
            values = self._conditioned_schedule(LATE_FIFTH_INTERVAL)
        else:  # pragma: no cover - constants above define all cases
            raise RuntimeError(f"unknown schedule stratum: {stratum!r}")
        return tuple(int(value) for value in values), stratum

    def sample_context(self):
        stratum = self._stratum(
            self.context_rng, CONTEXT_STRATUM_WEIGHTS
        )
        if stratum == "uniform":
            values = self.context_rng.uniform(
                0.0, 1.0, size=NUM_TRADE_EVENTS
            )
        elif stratum == "low_prefix":
            values = np.empty(NUM_TRADE_EVENTS, dtype=np.float64)
            values[:-1] = self.context_rng.uniform(
                0.0, LOW_PREFIX_PRICE_HIGH, size=NUM_TRADE_EVENTS - 1
            )
            values[-1] = self.context_rng.uniform(0.0, 1.0)
        else:  # pragma: no cover - constants above define all cases
            raise RuntimeError(f"unknown context stratum: {stratum!r}")
        return np.asarray(values, dtype=np.float32), stratum

    def sample(self):
        event_steps, schedule_stratum = self.sample_schedule()
        context, context_stratum = self.sample_context()
        return E1SamplerDraw(
            event_steps=event_steps,
            opponent_commitment=context,
            schedule_stratum=schedule_stratum,
            context_stratum=context_stratum,
        )


__all__ = [
    "ALL_EQUAL_E1_SAMPLER",
    "CONTEXT_STRATA",
    "CONTEXT_STRATUM_WEIGHTS",
    "E1_SAMPLER_MODES",
    "E1SamplerDraw",
    "EARLY_FIFTH_INTERVAL",
    "LATE_FIFTH_INTERVAL",
    "LOW_PREFIX_PRICE_HIGH",
    "SCHEDULE_STRATA",
    "SCHEDULE_STRATUM_WEIGHTS",
    "TEMPORAL_MIX_E1_SAMPLER",
    "TEMPORAL_MIX_GAMEPLAY_HORIZON",
    "TemporalMarginalE1Sampler",
    "UNIFORM_E1_SAMPLER",
    "e1_sampler_provenance",
    "sample_all_equal_e1_context",
]
