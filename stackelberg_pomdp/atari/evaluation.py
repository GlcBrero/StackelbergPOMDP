"""Deterministic evaluation for every stage of the native SB3 Atari buyer.

The evaluator operates on an in-memory SB3 model so training callbacks and the
standalone command use exactly the same episode accounting and pass criteria.
"""

from dataclasses import replace
import statistics

import numpy as np

from stackelberg_pomdp.atari.factory import make_atari_buyer_env


DEFAULT_FIXED_PRICES = tuple(round(index / 10.0, 1) for index in range(11))


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
    try:
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, last_info = env.step(action)
            net_reward += float(reward)
            if last_info.get("trade_event", False):
                offered_price = float(last_info["price_offered"])
                offered_prices.append(offered_price)
                thresholds.append(float(last_info["threshold"]))
                if last_info.get("trade_this_step", False):
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
    total_shots = sum(row["shots_fired"] for row in episodes)
    game_rewards = [row["game_reward"] for row in episodes]
    return {
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
        "max_absolute_accounting_error": float(
            max(abs(row["accounting_error"]) for row in episodes)
        ),
    }


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
        benchmark = zero_price_game_value - 5.0 * float(row["price"])
        profitable = benchmark > 1.0e-9
        row["five_bullet_net_benchmark"] = benchmark
        row["buying_profitable"] = profitable
        row["profitable_price_passed"] = bool(
            not profitable
            or (
                row["mean_purchases"] >= 4.5
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
    }
    return {
        "summary": summary,
        "fixed_price_table": fixed_rows,
        "fixed_price_episodes": fixed_episodes,
        "random_price_summary": random_summary,
        "random_price_episodes": random_rows,
    }
