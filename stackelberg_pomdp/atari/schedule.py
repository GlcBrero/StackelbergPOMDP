"""Exogenous five-event timing shared by E0b, E1, and E2."""

import numpy as np

from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS


class ExactFiveEventSchedule:
    """Sample five distinct event times before a reserved gameplay tail."""

    def __init__(
            self,
            *,
            gameplay_horizon,
            tail_steps,
            fixed_event_steps=None,
    ):
        self.gameplay_horizon = int(gameplay_horizon)
        self.tail_steps = int(tail_steps)
        if self.gameplay_horizon <= NUM_TRADE_EVENTS:
            raise ValueError("gameplay_horizon is too short")
        if not 1 <= self.tail_steps < self.gameplay_horizon:
            raise ValueError("tail_steps must leave positive gameplay")
        self.event_stop = self.gameplay_horizon - self.tail_steps
        if self.event_stop < NUM_TRADE_EVENTS:
            raise ValueError("event window must contain at least five steps")
        self.fixed_event_steps = self._validate_fixed(fixed_event_steps)

    def _validate_fixed(self, values):
        if values is None:
            return None
        fixed = tuple(int(value) for value in values)
        if len(fixed) != NUM_TRADE_EVENTS:
            raise ValueError("fixed_event_steps must contain five entries")
        if tuple(sorted(set(fixed))) != fixed:
            raise ValueError("fixed_event_steps must be strictly increasing")
        if fixed[0] < 0 or fixed[-1] >= self.event_stop:
            raise ValueError("fixed_event_steps must fit before the tail")
        return fixed

    def sample(self, rng):
        if self.fixed_event_steps is not None:
            return self.fixed_event_steps
        candidates = np.arange(self.event_stop, dtype=np.int64)
        values = rng.choice(
            candidates,
            size=NUM_TRADE_EVENTS,
            replace=False,
        )
        return tuple(int(value) for value in np.sort(values))


__all__ = ["ExactFiveEventSchedule"]
