"""Shared helpers for Calvano-style Bertrand pricing experiments.

This module owns the economic constants and summary metrics used by the
Calvano replication, deviation diagnostic, and platform-intervention
evaluation scripts. Environment dynamics still live in
`BertrandCompetitionEnv`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalvanoConfig:
    n_agents: int = 2
    marginal_cost: float = 1.0
    quality: float = 2.0
    outside_option: float = 0.0
    mu: float = 0.25
    discount: float = 0.95
    alpha: float = 0.15
    beta: float = 4e-6
    convergence_window: int = 100_000
    max_steps: int = 10_000_000


DEFAULT_CALVANO = CalvanoConfig()
PAPER_INTERVENTION_CALVANO = CalvanoConfig(alpha=0.25, beta=1e-4)


def symmetric_demand(price, config=DEFAULT_CALVANO):
    """Demand for one firm when all firms set the same price."""
    x = np.exp((config.quality - price) / config.mu)
    y = np.exp(config.outside_option / config.mu)
    return x / (config.n_agents * x + y)


def symmetric_profit(price, config=DEFAULT_CALVANO):
    return (price - config.marginal_cost) * symmetric_demand(price, config)


def gain_index(avg_profit, nash_profit, monopoly_profit):
    """Calvano gain index, normalized between Nash and monopoly profits."""
    return (avg_profit - nash_profit) / (monopoly_profit - nash_profit)


def average_symmetric_gain_index(prices, nash_price, monopoly_price, config=DEFAULT_CALVANO):
    profits = [symmetric_profit(price, config) for price in prices]
    return gain_index(
        np.mean(profits),
        symmetric_profit(nash_price, config),
        symmetric_profit(monopoly_price, config),
    )
