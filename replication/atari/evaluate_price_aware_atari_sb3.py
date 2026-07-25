"""Evaluate a native SB3 Atari buyer checkpoint with deterministic actions."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile


os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stackpomdp-matplotlib")
)

from stable_baselines3 import PPO

from stackelberg_pomdp.atari import AtariBuyerEnvConfig
from stackelberg_pomdp.atari.evaluation import (
    DEFAULT_FIXED_PRICES,
    evaluate_economics,
    evaluate_gameplay,
)


def parse_prices(raw):
    prices = tuple(
        float(value.strip())
        for value in str(raw).split(",")
        if value.strip()
    )
    if not prices:
        raise ValueError("at least one fixed price is required")
    if any(price < 0.0 or price > 1.0 for price in prices):
        raise ValueError("fixed prices must lie in [0, 1]")
    return prices


def default_output_path(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    return checkpoint.with_suffix(".evaluation.json")


def write_fixed_price_csv(path, rows):
    scalar_rows = [
        {
            key: value
            for key, value in row.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        for row in rows
    ]
    if not scalar_rows:
        return
    fieldnames = list(scalar_rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def iter_trade_event_rows(result):
    """Flatten nested episode audit trails into one row per offer."""

    episode_groups = []
    for fixed_price, episodes in result.get("fixed_price_episodes", {}).items():
        episode_groups.append(("fixed_price", float(fixed_price), episodes))
    if "random_price_episodes" in result:
        episode_groups.append(
            ("random_price", None, result["random_price_episodes"])
        )
    if not episode_groups and "episodes" in result:
        episode_groups.append(("gameplay", None, result["episodes"]))

    for split, fixed_price, episodes in episode_groups:
        for episode_index, episode in enumerate(episodes):
            for event in episode.get("trade_events", []):
                yield {
                    "evaluation_split": split,
                    "fixed_price": fixed_price,
                    "episode_index": int(episode_index),
                    "seed": int(episode["seed"]),
                    "episode_net_reward": float(episode["net_reward"]),
                    "episode_game_reward": float(episode["game_reward"]),
                    "episode_trade_opportunities": int(
                        episode["trade_opportunities"]
                    ),
                    "episode_purchases": int(episode["purchases"]),
                    **event,
                }


def write_trade_events_csv(path, result):
    """Write a stable, flat audit table even when an evaluation has no offers."""

    rows = list(iter_trade_event_rows(result))
    fieldnames = [
        "evaluation_split",
        "fixed_price",
        "episode_index",
        "seed",
        "episode_net_reward",
        "episode_game_reward",
        "episode_trade_opportunities",
        "episode_purchases",
        "opportunity_index",
        "offer_step",
        "normalized_timestep",
        "time_remaining",
        "timing_bin",
        "price_offered",
        "threshold",
        "purchased",
        "ammo_before_trade",
        "ammo_after_trade",
        "ammo_remaining",
        "shots_fired_this_step",
        "opportunities_remaining_before",
        "opportunities_remaining_after",
    ]
    extra_fields = sorted({
        key for row in rows for key in row if key not in fieldnames
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(args):
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model = PPO.load(checkpoint, device=args.device)
    saved_stage = getattr(model.policy, "stage", "gameplay")
    actor_economic_context = bool(
        getattr(model.policy, "actor_economic_context", False)
    )
    requested_offer_timing = getattr(args, "offer_timing", "auto")
    offer_timing = (
        "bernoulli" if actor_economic_context else "immediate"
    ) if requested_offer_timing == "auto" else requested_offer_timing
    requested_max_steps = getattr(args, "max_steps", None)
    max_steps = (
        125
        if requested_max_steps is None and actor_economic_context
        else requested_max_steps
    )
    stage = args.stage or saved_stage
    if stage == "joint":
        environment_stage = "priced"
    else:
        environment_stage = stage
    if environment_stage not in {"gameplay", "free_trade", "priced"}:
        raise ValueError(f"cannot evaluate stage {stage!r}")

    config = AtariBuyerEnvConfig(
        stage=environment_stage,
        seed=args.eval_seed,
        initial_bullets=5 if environment_stage == "gameplay" else 0,
        bullet_capacity=5,
        offer_chances=5,
        max_purchases=5,
        price_min=args.price_min,
        price_max=args.price_max,
        noop_max=30,
        frame_skip=4,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        offer_timing=offer_timing,
        offer_probability=getattr(args, "offer_probability", 0.04),
        actor_economic_context=actor_economic_context,
        max_steps=max_steps,
        rom_path=args.rom_path,
    )
    if environment_stage in {"gameplay", "free_trade"}:
        result = evaluate_gameplay(
            model,
            config,
            episodes=args.episodes,
            eval_seed=args.eval_seed,
        )
    else:
        result = evaluate_economics(
            model,
            config,
            fixed_prices=parse_prices(args.fixed_prices),
            episodes_per_price=args.episodes_per_price,
            random_episodes=args.random_episodes,
            eval_seed=args.eval_seed,
        )
    result["evaluation_config"] = {
        "checkpoint": str(checkpoint),
        "saved_stage": saved_stage,
        "evaluated_stage": stage,
        "actor_economic_context": actor_economic_context,
        "offer_timing": offer_timing,
        "offer_probability": float(
            getattr(args, "offer_probability", 0.04)
        ),
        "max_steps": max_steps,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--stage",
        choices=["gameplay", "free_trade", "priced", "joint"],
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--fixed-prices",
        default=",".join(str(price) for price in DEFAULT_FIXED_PRICES),
    )
    parser.add_argument("--episodes-per-price", type=int, default=20)
    parser.add_argument("--random-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=100_001)
    parser.add_argument("--price-min", type=float, default=0.0)
    parser.add_argument("--price-max", type=float, default=1.0)
    parser.add_argument(
        "--offer-timing",
        choices=["auto", "immediate", "bernoulli"],
        default="auto",
        help=(
            "auto uses Bernoulli timing for an economic-context policy and "
            "legacy immediate offers otherwise"
        ),
    )
    parser.add_argument("--offer-probability", type=float, default=0.04)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="defaults to 125 for economic-context policies and unlimited otherwise",
    )
    parser.add_argument("--rom-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--fixed-price-csv")
    parser.add_argument("--trade-events-csv")
    return parser.parse_args()


def validate_args(args):
    for name in ("episodes", "episodes_per_price", "random_episodes"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= args.price_min <= args.price_max <= 1.0:
        raise ValueError("price range must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= args.offer_probability <= 1.0:
        raise ValueError("offer_probability must lie in [0, 1]")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")


def main():
    args = parse_args()
    validate_args(args)
    result = evaluate_checkpoint(args)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(args.checkpoint)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if "fixed_price_table" in result:
        csv_path = (
            Path(args.fixed_price_csv).expanduser().resolve()
            if args.fixed_price_csv
            else output_path.with_suffix(".fixed_prices.csv")
        )
        write_fixed_price_csv(csv_path, result["fixed_price_table"])
        print(f"fixed_price_csv={csv_path}")
    trade_events_path = (
        Path(args.trade_events_csv).expanduser().resolve()
        if args.trade_events_csv
        else output_path.with_suffix(".trade_events.csv")
    )
    write_trade_events_csv(trade_events_path, result)
    print(f"trade_events_csv={trade_events_path}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"evaluation={output_path}")


if __name__ == "__main__":
    main()
