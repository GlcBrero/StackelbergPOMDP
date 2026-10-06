"""Draw the frozen Atari responses and the 25-seed leader curves."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    DEEP_PALETTE,
    ATARI_LINE_WIDTH as LINE_WIDTH,
    ATARI_MARKER_SIZE as MARKER_SIZE,
    ATARI_STANDARD_ERROR_ALPHA as STANDARD_ERROR_ALPHA,
    ATARI_THREE_PANEL_FIGSIZE as THREE_PANEL_FIGSIZE,
    configure_atari_style,
    despine,
)

EXPECTED_GRID = np.linspace(0.0, 1.0, 11)

BLUE = DEEP_PALETTE[0]

ORANGE = DEEP_PALETTE[1]

GRAY = "#666666"

def role_data(data: pd.DataFrame, role: str) -> pd.DataFrame:
    result = data[data["controlled_role"] == role].sort_values("opponent_value")
    if len(result) != 11:
        raise ValueError(f"Missing response data for {role}")
    return result

def errorbar(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    mean_column: str,
    sem_column: str,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
    zorder: float = 2.0,
) -> None:
    ax.errorbar(
        data["opponent_value"].to_numpy(dtype=float),
        data[mean_column].to_numpy(dtype=float),
        yerr=data[sem_column].to_numpy(dtype=float),
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        capsize=2.0,
        elinewidth=0.9,
        zorder=zorder,
    )

def make_response_figure(data: pd.DataFrame) -> plt.Figure:
    configure_atari_style()
    buyer = role_data(data, "buyer")
    seller = role_data(data, "seller")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=THREE_PANEL_FIGSIZE,
        constrained_layout=True,
    )

    ax = axes[0]
    errorbar(
        ax,
        buyer,
        mean_column="transfers_mean",
        sem_column="transfers_sem",
        label="Purchased",
        color=BLUE,
        marker="o",
    )
    errorbar(
        ax,
        buyer,
        mean_column="controlled_shots_mean",
        sem_column="controlled_shots_sem",
        label="Fired",
        color=ORANGE,
        marker="x",
        linestyle=":",
        zorder=3.0,
    )
    ax.set_title("(a) Buyer demand and use", loc="left")
    ax.set_xlabel("Seller price")
    ax.set_ylabel("Bullets")
    ax.set_ylim(-0.15, 5.3)
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    errorbar(
        ax,
        seller,
        mean_column="quoted_price_mean",
        sem_column="quoted_price_sem",
        label="Quoted price",
        color=BLUE,
        marker="o",
        zorder=3.0,
    )
    ax.plot(
        EXPECTED_GRID,
        EXPECTED_GRID,
        color=GRAY,
        linestyle=":",
        linewidth=1.4,
        label=r"Buyer threshold ($p=t$)",
        zorder=2.0,
    )
    ax.set_title("(b) Seller price response", loc="left")
    ax.set_xlabel("Buyer threshold")
    ax.set_ylabel("Quoted price")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[2]
    errorbar(
        ax,
        buyer,
        mean_column="controlled_payoff_mean",
        sem_column="controlled_payoff_sem",
        label="Buyer (price)",
        color=ORANGE,
        marker="s",
        zorder=3.0,
    )
    errorbar(
        ax,
        seller,
        mean_column="controlled_payoff_mean",
        sem_column="controlled_payoff_sem",
        label="Seller (threshold)",
        color=BLUE,
        marker="o",
        zorder=3.0,
    )
    ax.set_title("(c) Follower payoffs", loc="left")
    ax.set_xlabel("Opponent commitment")
    ax.set_ylabel("Episode payoff")
    ax.set_ylim(-0.15, 5.3)
    ax.legend(loc="upper center", frameon=False)

    for ax in axes:
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        despine(ax)
    return figure


EXPECTED_METRICS = (
    "mean_seller_payoff",
    "mean_buyer_payoff",
    "mean_accepted_price",
)

EXPECTED_TIMESTEPS = (0, 400680, 800520, 1200360, 1600200, 2000040)

METRIC_SPECS = (
    ("mean_seller_payoff", "(a) Seller payoff", "Episode payoff"),
    ("mean_buyer_payoff", "(b) Buyer payoff", "Episode payoff"),
    ("mean_accepted_price", "(c) Mean accepted price", "Price"),
)

ROLE_ORDER = ("seller", "buyer")

ROLE_LABELS = {"seller": "Seller leader", "buyer": "Buyer leader"}

ROLE_STYLES = {
    "seller": {"color": DEEP_PALETTE[0], "marker": "o"},
    "buyer": {"color": DEEP_PALETTE[1], "marker": "s"},
}

def make_leader_figure(data: pd.DataFrame) -> plt.Figure:
    configure_atari_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=THREE_PANEL_FIGSIZE,
        constrained_layout=True,
    )
    legend_handles = []
    for ax, (metric, title, ylabel) in zip(axes, METRIC_SPECS):
        for role in ROLE_ORDER:
            series = data[
                (data["metric"] == metric) & (data["leader_role"] == role)
            ].sort_values("training_total_timesteps")
            x = series["training_total_timesteps"].to_numpy(dtype=float) / 1e6
            mean = series["mean_across_policy_means"].to_numpy(dtype=float)
            sem = series["sem_across_policy_means"].to_numpy(dtype=float)
            style = ROLE_STYLES[role]
            line, = ax.plot(
                x,
                mean,
                color=style["color"],
                marker=style["marker"],
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=ROLE_LABELS[role],
                zorder=3,
            )
            ax.fill_between(
                x,
                mean - sem,
                mean + sem,
                color=style["color"],
                alpha=STANDARD_ERROR_ALPHA,
                linewidth=0,
                zorder=2,
            )
            if ax is axes[2]:
                legend_handles.append(line)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.05, 2.06)
        ax.set_xticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
        ax.margins(y=0.10)
        despine(ax)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(
        handles=legend_handles,
        title="Leader",
        loc="upper right",
        frameon=False,
    )
    figure.supxlabel(
        "Training transitions (millions)",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    return figure
