"""Deterministic evaluation for every stage of the native SB3 Atari buyer.

The evaluator operates on an in-memory SB3 model so training callbacks and the
standalone command use exactly the same episode accounting and pass criteria.
"""

from dataclasses import replace
import statistics

import numpy as np

from stackelberg_pomdp.atari.factory import make_atari_buyer_env


DEFAULT_FIXED_PRICES = tuple(round(index / 10.0, 1) for index in range(11))
TIMING_BINS = ("early", "middle", "late")


def _timing_bin(normalized_timestep):
    """Assign an offer to equal thirds of the episode horizon."""

    normalized_timestep = float(np.clip(normalized_timestep, 0.0, 1.0))
    if normalized_timestep < 1.0 / 3.0:
        return "early"
    if normalized_timestep < 2.0 / 3.0:
        return "middle"
    return "late"


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def _trade_event(info, *, event_index, decision_step, max_steps):
    """Return one JSON-safe audit record for an offered trade.

    New stochastic-timing environments provide the timing fields directly.
    The fallbacks keep checkpoints evaluated against the original immediate
    market wrapper readable rather than silently dropping their decisions.
    """

    offer_step = int(info.get("offer_step", decision_step))
    if "normalized_timestep" in info:
        normalized_timestep = float(info["normalized_timestep"])
    elif max_steps:
        normalized_timestep = float(
            np.clip(offer_step / max(int(max_steps), 1), 0.0, 1.0)
        )
    else:
        # Legacy immediate-offer environments had no finite time coordinate.
        normalized_timestep = 0.0
    time_remaining = float(
        info.get("time_remaining", 1.0 - normalized_timestep)
    )
    purchased = bool(info.get("trade_this_step", False))
    timing_bin = _timing_bin(normalized_timestep)
    price_offered = float(info.get("price_offered", 0.0))
    return {
        "opportunity_index": int(event_index),
        "offer_step": offer_step,
        "normalized_timestep": normalized_timestep,
        "time_remaining": time_remaining,
        "timing_bin": timing_bin,
        "time_bin": timing_bin,
        "price_offered": price_offered,
        "price": price_offered,
        "threshold": float(info.get("threshold", 0.0)),
        "purchased": purchased,
        "accepted": purchased,
        "trade_this_step": purchased,
        "ammo_before_trade": int(info.get("ammo_before_trade", 0)),
        "ammo_after_trade": int(
            info.get("ammo_after_trade", int(purchased))
        ),
        "ammo_remaining": int(
            info.get("ammo_remaining", info.get("final_ammo", 0))
        ),
        "shots_fired_this_step": int(
            info.get("shots_fired_this_step", 0)
        ),
        "opportunities_remaining_before": int(
            info.get("opportunities_remaining_before", 0)
        ),
        "opportunities_remaining_after": int(
            info.get("opportunities_remaining_after", 0)
        ),
    }


def _event_was_purchased(event):
    return bool(event.get(
        "purchased",
        event.get("accepted", event.get("trade_this_step", False)),
    ))


def _event_timing_bin(event):
    return str(event.get(
        "timing_bin",
        event.get("time_bin", _timing_bin(event["normalized_timestep"])),
    ))


def run_deterministic_episode(model, config, *, seed):
    """Run one deterministic episode and retain its economic decisions."""

    env = make_atari_buyer_env(replace(config, seed=int(seed)))
    observation = env.reset()
    done = False
    net_reward = 0.0
    last_info = {}
    offered_prices = []
    thresholds = []
    accepted_prices = []
    trade_events = []
    decision_step = 0
    try:
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, last_info = env.step(action)
            decision_step += 1
            net_reward += float(reward)
            if last_info.get("trade_event", False):
                event = _trade_event(
                    last_info,
                    event_index=len(trade_events),
                    decision_step=decision_step,
                    max_steps=config.max_steps,
                )
                trade_events.append(event)
                offered_price = event["price_offered"]
                offered_prices.append(offered_price)
                thresholds.append(event["threshold"])
                if event["purchased"]:
                    accepted_prices.append(offered_price)
    finally:
        env.close()

    game_reward = float(last_info.get("episode_game_reward", net_reward))
    payments = float(last_info.get("episode_payments", 0.0))
    purchases = int(last_info.get("purchases", 0))
    shots = int(last_info.get("shots_fired", 0))
    opportunities = int(last_info.get("trade_opportunities", 0))
    return {
        "seed": int(seed),
        "net_reward": float(net_reward),
        "game_reward": game_reward,
        "payments": payments,
        "length": int(last_info.get("episode_length", 0)),
        "trade_opportunities": opportunities,
        "purchases": purchases,
        "acceptance_rate": (
            float(purchases) / float(opportunities) if opportunities else 0.0
        ),
        "shots_fired": shots,
        "final_ammo": int(last_info.get("final_ammo", 0)),
        "fired_fraction_of_purchases": (
            float(shots) / float(purchases) if purchases else 0.0
        ),
        "reward_per_bullet": float(last_info.get("reward_per_bullet", 0.0)),
        "threshold": (
            float(np.mean(thresholds)) if thresholds else 0.0
        ),
        "offered_price": (
            float(np.mean(offered_prices)) if offered_prices else 0.0
        ),
        "offered_prices": offered_prices,
        "thresholds": thresholds,
        "accepted_prices": accepted_prices,
        "trade_events": trade_events,
        "accounting_error": float(net_reward - (game_reward - payments)),
    }


def summarize_episodes(episodes):
    if not episodes:
        raise ValueError("cannot summarize zero Atari episodes")

    mean_fields = (
        "net_reward",
        "game_reward",
        "payments",
        "length",
        "trade_opportunities",
        "purchases",
        "acceptance_rate",
        "shots_fired",
        "final_ammo",
        "fired_fraction_of_purchases",
        "reward_per_bullet",
        "threshold",
        "offered_price",
    )
    total_purchases = sum(row["purchases"] for row in episodes)
    total_opportunities = sum(row["trade_opportunities"] for row in episodes)
    total_shots = sum(row["shots_fired"] for row in episodes)
    game_rewards = [row["game_reward"] for row in episodes]
    all_events = [
        event
        for episode in episodes
        for event in episode.get("trade_events", [])
    ]
    purchased_events = [
        event for event in all_events if _event_was_purchased(event)
    ]
    result = {
        "episodes": len(episodes),
        **{
            f"mean_{field}": float(np.mean([row[field] for row in episodes]))
            for field in mean_fields
        },
        "median_game_reward": float(statistics.median(game_rewards)),
        "score_five_rate": float(np.mean([value >= 5.0 for value in game_rewards])),
        "positive_net_reward_rate": float(
            np.mean([row["net_reward"] > 0.0 for row in episodes])
        ),
        "bought_all_five_rate": float(
            np.mean([row["purchases"] == 5 for row in episodes])
        ),
        "fired_all_five_rate": float(
            np.mean([row["shots_fired"] == 5 for row in episodes])
        ),
        "used_all_ammo_rate": float(
            np.mean([row["final_ammo"] == 0 for row in episodes])
        ),
        "fired_all_purchased_rate": float(
            np.mean([
                row["shots_fired"] == row["purchases"]
                for row in episodes
            ])
        ),
        "aggregate_fired_fraction_of_purchases": (
            float(total_shots) / float(total_purchases)
            if total_purchases
            else 1.0
        ),
        "aggregate_acceptance_rate": (
            float(total_purchases) / float(total_opportunities)
            if total_opportunities
            else 0.0
        ),
        "mean_offer_normalized_timestep": _mean_or_none([
            event["normalized_timestep"] for event in all_events
        ]),
        "mean_purchase_normalized_timestep": _mean_or_none([
            event["normalized_timestep"] for event in purchased_events
        ]),
        "mean_offer_step": _mean_or_none([
            event["offer_step"] for event in all_events
        ]),
        "mean_purchase_step": _mean_or_none([
            event["offer_step"] for event in purchased_events
        ]),
        "max_absolute_accounting_error": float(
            max(abs(row["accounting_error"]) for row in episodes)
        ),
    }

    for label in TIMING_BINS:
        bin_events = [
            event for event in all_events if _event_timing_bin(event) == label
        ]
        bin_purchases = [
            event for event in bin_events if _event_was_purchased(event)
        ]
        offer_count = len(bin_events)
        purchase_count = len(bin_purchases)
        result[f"offer_count_{label}"] = offer_count
        result[f"purchase_count_{label}"] = purchase_count
        result[f"rejection_count_{label}"] = offer_count - purchase_count
        result[f"acceptance_rate_{label}"] = (
            float(purchase_count) / float(offer_count)
            if offer_count
            else 0.0
        )
        result[f"mean_threshold_{label}"] = _mean_or_none([
            event["threshold"] for event in bin_events
        ])
    late_offers = result["offer_count_late"]
    result["late_rejection_rate"] = (
        float(result["rejection_count_late"]) / float(late_offers)
        if late_offers
        else 0.0
    )

    opportunity_indices = sorted({
        int(event["opportunity_index"]) for event in all_events
    })
    result["opportunity_histogram"] = {}
    for opportunity_index in opportunity_indices:
        opportunity_events = [
            event
            for event in all_events
            if int(event["opportunity_index"]) == opportunity_index
        ]
        purchases = sum(
            _event_was_purchased(event) for event in opportunity_events
        )
        offers = len(opportunity_events)
        result["opportunity_histogram"][str(opportunity_index)] = {
            "offers": offers,
            "purchases": int(purchases),
            "acceptance_rate": float(purchases) / float(offers),
        }
    result["episode_opportunity_count_histogram"] = {
        str(count): int(sum(
            int(row["trade_opportunities"]) == count for row in episodes
        ))
        for count in range(6)
    }
    return result


def _seeded_episodes(model, config, *, episodes, eval_seed):
    return [
        run_deterministic_episode(
            model,
            config,
            seed=int(eval_seed) + 1009 * episode_index,
        )
        for episode_index in range(int(episodes))
    ]


def evaluate_gameplay(model, config, *, episodes=20, eval_seed=100_001):
    """Evaluate E0 gameplay or the zero-price free-trade bridge."""

    rows = _seeded_episodes(
        model,
        config,
        episodes=episodes,
        eval_seed=eval_seed,
    )
    summary = summarize_episodes(rows)
    summary["five_point_gate_passed"] = bool(
        summary["median_game_reward"] >= 5.0
        and summary["mean_game_reward"] >= 4.8
        and summary["fired_all_five_rate"] >= 0.95
    )
    summary["free_trade_gate_passed"] = bool(
        summary["five_point_gate_passed"]
        and summary["mean_purchases"] >= 4.9
        and summary["aggregate_fired_fraction_of_purchases"] >= 0.95
        and summary["mean_payments"] == 0.0
    )
    return {"summary": summary, "episodes": rows}


def evaluate_economics(
        model,
        config,
        *,
        fixed_prices=DEFAULT_FIXED_PRICES,
        episodes_per_price=20,
        random_episodes=100,
        eval_seed=100_001,
):
    """Evaluate E1/joint policies at paired fixed prices and Uniform prices."""

    contextual_timing = bool(
        getattr(config, "actor_economic_context", False)
        and getattr(config, "offer_timing", "immediate") == "bernoulli"
    )

    fixed_prices = tuple(float(price) for price in fixed_prices)
    if not fixed_prices:
        raise ValueError("fixed-price evaluation needs at least one price")
    if not any(np.isclose(price, 0.0) for price in fixed_prices):
        raise ValueError("fixed-price evaluation must include price zero")

    fixed_rows = []
    fixed_episodes = {}
    for price in fixed_prices:
        fixed_config = replace(
            config,
            stage="fixed_price",
            fixed_price=price,
        )
        episodes = _seeded_episodes(
            model,
            fixed_config,
            episodes=episodes_per_price,
            eval_seed=eval_seed,
        )
        row = {"price": price, **summarize_episodes(episodes)}
        fixed_rows.append(row)
        fixed_episodes[f"{price:.6g}"] = episodes

    zero_price_row = next(
        row for row in fixed_rows if np.isclose(row["price"], 0.0)
    )
    zero_price_game_value = float(zero_price_row["mean_game_reward"])
    profitable_rows = []
    for row in fixed_rows:
        # Under stochastic timing an episode may expose fewer than five offers.
        # Charge the all-available-offers benchmark for offers that actually
        # occurred, and judge buying with its summed-opportunity acceptance rate.
        opportunity_benchmark = (
            zero_price_game_value
            - float(row["mean_trade_opportunities"]) * float(row["price"])
        )
        profitable = opportunity_benchmark > 1.0e-9
        # Retain the old field for downstream readers.  It is numerically equal
        # to the new benchmark in the original five-immediate-offer protocol.
        row["five_bullet_net_benchmark"] = (
            zero_price_game_value - 5.0 * float(row["price"])
        )
        row["opportunity_normalized_net_benchmark"] = opportunity_benchmark
        row["opportunity_normalized_acceptance_rate"] = row[
            "aggregate_acceptance_rate"
        ]
        row["buying_profitable"] = profitable
        if contextual_timing:
            # A late bullet can rationally be rejected even when accepting all
            # opportunities would be profitable on average.  Do not impose the
            # immediate-offer control's 90% acceptance gate on a timing-aware
            # policy; realized net value and use of purchased bullets are the
            # relevant checks.
            row["profitable_price_passed"] = bool(
                not profitable
                or (
                    row["aggregate_fired_fraction_of_purchases"] >= 0.9
                    and row["mean_net_reward"] > 0.0
                )
            )
        else:
            row["profitable_price_passed"] = bool(
                not profitable
                or (
                    row["opportunity_normalized_acceptance_rate"] >= 0.9
                    and row["aggregate_fired_fraction_of_purchases"] >= 0.9
                    and row["mean_net_reward"] > 0.0
                )
            )
        if profitable:
            profitable_rows.append(row)

    random_config = replace(config, stage="priced")
    random_rows = _seeded_episodes(
        model,
        random_config,
        episodes=random_episodes,
        eval_seed=int(eval_seed) + 1_000_003,
    )
    random_summary = summarize_episodes(random_rows)
    random_passed = bool(
        random_summary["mean_net_reward"] > 0.0
        and random_summary["aggregate_fired_fraction_of_purchases"] >= 0.9
        and random_summary["max_absolute_accounting_error"] <= 1.0e-6
    )
    fixed_passed = bool(
        profitable_rows
        and all(row["profitable_price_passed"] for row in profitable_rows)
    )
    if contextual_timing:
        # Free bullets should still be accepted; this retains a concrete
        # transfer sanity check without ruling out costly late-offer rejection.
        fixed_passed = bool(
            fixed_passed
            and zero_price_row["aggregate_acceptance_rate"] >= 0.9
        )
    max_profitable_price = max(
        (row["price"] for row in profitable_rows),
        default=None,
    )
    summary = {
        "zero_price_game_value": zero_price_game_value,
        "max_evaluated_profitable_price": max_profitable_price,
        "profitable_fixed_prices": len(profitable_rows),
        "fixed_price_passed": fixed_passed,
        "random_price_passed": random_passed,
        "pass_condition": bool(fixed_passed and random_passed),
        "contextual_timing": contextual_timing,
        "zero_price_acceptance_rate": zero_price_row[
            "aggregate_acceptance_rate"
        ],
        "random_mean_net_reward": random_summary["mean_net_reward"],
        "random_mean_game_reward": random_summary["mean_game_reward"],
        "random_mean_payments": random_summary["mean_payments"],
        "random_mean_purchases": random_summary["mean_purchases"],
        "random_mean_shots_fired": random_summary["mean_shots_fired"],
        "random_positive_net_reward_rate": random_summary[
            "positive_net_reward_rate"
        ],
        "random_fired_fraction_of_purchases": random_summary[
            "aggregate_fired_fraction_of_purchases"
        ],
        "random_aggregate_acceptance_rate": random_summary[
            "aggregate_acceptance_rate"
        ],
        "random_mean_offer_normalized_timestep": random_summary[
            "mean_offer_normalized_timestep"
        ],
        "random_mean_purchase_normalized_timestep": random_summary[
            "mean_purchase_normalized_timestep"
        ],
        "random_mean_offer_step": random_summary["mean_offer_step"],
        "random_mean_purchase_step": random_summary["mean_purchase_step"],
    }
    for label in TIMING_BINS:
        for field in (
                "offer_count",
                "purchase_count",
                "rejection_count",
                "acceptance_rate",
                "mean_threshold",
        ):
            summary[f"random_{field}_{label}"] = random_summary[
                f"{field}_{label}"
            ]
    summary["random_late_rejection_rate"] = random_summary[
        "late_rejection_rate"
    ]
    summary["random_opportunity_histogram"] = random_summary[
        "opportunity_histogram"
    ]
    return {
        "summary": summary,
        "fixed_price_table": fixed_rows,
        "fixed_price_episodes": fixed_episodes,
        "random_price_summary": random_summary,
        "random_price_episodes": random_rows,
    }
