"""Deterministically evaluate the E1 buyer economic threshold head.

The threshold actor is evaluated at its Gaussian mean.  The protected E0 Atari
policy remains frozen and uses argmax game actions inside the environment.
Outputs include a paired fixed-price table and a Uniform(0, 1) summary.
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import gym
import numpy as np
import torch

from stackelberg_pomdp.atari_models import BuyerThresholdHeadTorch
from stackelberg_pomdp.gym_envs.envs.atari_envs import (
    FrozenGameplayBuyerThresholdEnv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAME_CHECKPOINT = (
    PROJECT_ROOT
    / "replication/atari/checkpoints/space_invaders_5bullets_a3c_best.json"
)
DEFAULT_FIXED_PRICES = tuple(round(value / 10.0, 1) for value in range(11))
MAX_PURCHASES = 5


def parse_prices(raw):
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("at least one fixed evaluation price is required")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("fixed evaluation prices must lie in [0, 1]")
    return values


def _resolve_existing_path(reference, *, anchor=None):
    path = Path(reference).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
        if anchor is not None:
            candidates.append(Path(anchor).resolve().parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not resolve {reference}; tried {rendered}")


def resolve_game_checkpoint(reference, *, anchor=None):
    """Resolve either an E0 pickle or its protected JSON manifest."""

    path = _resolve_existing_path(reference, anchor=anchor)
    if path.suffix.lower() != ".json":
        return path
    with path.open() as handle:
        manifest = json.load(handle)
    checkpoint = manifest.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"{path} has no checkpoint field")
    return _resolve_existing_path(checkpoint, anchor=path)


def resolve_threshold_checkpoint(reference):
    """Resolve either an E1 threshold pickle or its selected JSON manifest."""

    path = _resolve_existing_path(reference)
    if path.suffix.lower() != ".json":
        return path
    with path.open() as handle:
        manifest = json.load(handle)
    checkpoint = manifest.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"{path} has no checkpoint field")
    return _resolve_existing_path(checkpoint, anchor=path)


def _load_threshold_model(checkpoint):
    checkpoint = resolve_threshold_checkpoint(checkpoint)
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    metadata = dict(payload.get("metadata", {}))
    source = payload.get("default_policy") or payload.get("agent_1")
    if source is None:
        raise ValueError(
            f"{checkpoint} has no threshold policy weights: {list(payload)}"
        )

    price_max = float(metadata.get("price_max", 1.0))
    observation_space = gym.spaces.Box(
        low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([price_max, 1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        0.0, price_max, shape=(1,), dtype=np.float32
    )
    model = BuyerThresholdHeadTorch(
        observation_space,
        action_space,
        2,
        {
            "vf_share_layers": True,
            "custom_model_config": {
                "hidden_size": int(metadata.get("hidden_size", 64)),
                "price_max": price_max,
                "initial_threshold": float(
                    metadata.get("initial_threshold", 0.5)
                ),
                "initial_log_std": float(metadata.get("initial_log_std", -0.7)),
                "hide_price_from_actor": bool(
                    metadata.get("hide_price_from_actor", False)
                ),
            },
        },
        "e1_buyer_threshold_evaluation",
    )
    current = model.state_dict()
    patched = {}
    copied = []
    for key, value in current.items():
        if key in source and tuple(source[key].shape) == tuple(value.shape):
            patched[key] = torch.as_tensor(source[key])
            copied.append(key)
        else:
            patched[key] = value
    if not any(key.startswith("threshold_raw_mean") for key in copied):
        raise ValueError(f"no compatible threshold actor tensors found in {checkpoint}")
    model.load_state_dict(patched)
    model.eval()
    return model, metadata, copied


def deterministic_threshold(model, observation, price_max=1.0):
    with torch.no_grad():
        output, _ = model(
            {"obs": torch.as_tensor(observation[None, ...])}, [], None
        )
    return float(np.clip(output[0, 0].item(), 0.0, price_max))


def _episode(
    model,
    *,
    game_checkpoint,
    seed,
    price_mode,
    fixed_price,
    price_min,
    price_max,
    max_steps,
):
    env = FrozenGameplayBuyerThresholdEnv(
        {
            "seed": int(seed),
            "game_checkpoint": str(game_checkpoint),
            "price_mode": price_mode,
            "fixed_price": float(fixed_price),
            "price_min": float(price_min),
            "price_max": float(price_max),
            "offer_chances": 5,
            "max_replenish": 5,
            "buyer_initial_bullets": 0,
            "buyer_game_reward_scale": 1.0,
            "payment_penalty_lambda": 1.0,
            "max_steps": max_steps,
            "noop_max": 30,
            "frame_skip": 4,
            "frame_stack": 4,
            "episodic_life": True,
            "clip_game_rewards": True,
            "rollout_after_last_offer": True,
        }
    )
    observation = env.reset()
    done = False
    total_reward = 0.0
    decisions = 0
    last_info = {}
    while not done:
        threshold = deterministic_threshold(model, observation, price_max)
        observation, reward, done, last_info = env.step(
            np.array([threshold], dtype=np.float32)
        )
        total_reward += float(reward)
        decisions += 1
    env.close()

    game_reward = float(last_info.get("episode_game_reward", 0.0))
    payments = float(last_info.get("episode_payments", 0.0))
    purchases = int(last_info.get("purchases", 0))
    shots = int(last_info.get("shots_fired", 0))
    opportunities = int(last_info.get("trade_opportunities", 0))
    return {
        "seed": int(seed),
        "net_reward": total_reward,
        "game_reward": game_reward,
        "payments": payments,
        "economic_decisions": decisions,
        "game_steps": int(last_info.get("game_steps", 0)),
        "trade_opportunities": opportunities,
        "purchases": purchases,
        "acceptance_rate": purchases / opportunities if opportunities else 0.0,
        "shots_fired": shots,
        "final_ammo": float(last_info.get("final_ammo", 0.0)),
        "fired_fraction_of_purchases": shots / purchases if purchases else 0.0,
        "reward_per_purchased_bullet": (
            game_reward / purchases if purchases else 0.0
        ),
        "threshold": float(np.mean(last_info.get("thresholds", (0.0,)))),
        "offered_price": float(
            np.mean(last_info.get("offered_prices", (0.0,)))
        ),
        "offered_prices": list(last_info.get("offered_prices", ())),
        "thresholds": list(last_info.get("thresholds", ())),
        "accepted_prices": list(last_info.get("accepted_prices", ())),
        "accounting_error": total_reward - (game_reward - payments),
    }


def _summary(episodes):
    mean_keys = (
        "net_reward",
        "game_reward",
        "payments",
        "economic_decisions",
        "game_steps",
        "trade_opportunities",
        "purchases",
        "acceptance_rate",
        "shots_fired",
        "final_ammo",
        "fired_fraction_of_purchases",
        "reward_per_purchased_bullet",
        "threshold",
        "offered_price",
    )
    total_purchases = sum(episode["purchases"] for episode in episodes)
    total_shots = sum(episode["shots_fired"] for episode in episodes)
    return {
        "episodes": len(episodes),
        **{
            f"mean_{key}": float(np.mean([episode[key] for episode in episodes]))
            for key in mean_keys
        },
        "positive_net_reward_episodes": sum(
            episode["net_reward"] > 0.0 for episode in episodes
        ),
        "bought_all_five_episodes": sum(
            episode["purchases"] == 5 for episode in episodes
        ),
        "fired_all_purchased_episodes": sum(
            episode["shots_fired"] == episode["purchases"]
            for episode in episodes
        ),
        "aggregate_fired_fraction_of_purchases": (
            float(total_shots / total_purchases) if total_purchases else 1.0
        ),
        "max_absolute_accounting_error": float(
            max(abs(episode["accounting_error"]) for episode in episodes)
        ),
    }


def evaluate_checkpoint(
    checkpoint,
    *,
    game_checkpoint=None,
    fixed_prices=DEFAULT_FIXED_PRICES,
    episodes_per_price=20,
    random_episodes=100,
    eval_seed=100_001,
    max_steps=300,
):
    checkpoint = resolve_threshold_checkpoint(checkpoint)
    model, metadata, copied = _load_threshold_model(checkpoint)
    game_reference = game_checkpoint or metadata.get("game_checkpoint")
    if game_reference is None:
        raise ValueError(
            "game checkpoint is absent from E1 metadata; pass --game-checkpoint"
        )
    game_checkpoint = resolve_game_checkpoint(game_reference, anchor=checkpoint)
    price_min = float(metadata.get("price_min", 0.0))
    price_max = float(metadata.get("price_max", 1.0))

    fixed_rows = []
    fixed_episode_results = {}
    for price in fixed_prices:
        episodes = [
            _episode(
                model,
                game_checkpoint=game_checkpoint,
                seed=eval_seed + episode_idx * 1009,
                price_mode="fixed",
                fixed_price=float(price),
                price_min=price_min,
                price_max=price_max,
                max_steps=max_steps,
            )
            for episode_idx in range(episodes_per_price)
        ]
        summary = _summary(episodes)
        row = {"price": float(price), **summary}
        fixed_rows.append(row)
        fixed_episode_results[f"{float(price):.6g}"] = episodes

    random_results = [
        _episode(
            model,
            game_checkpoint=game_checkpoint,
            seed=eval_seed + 1_000_003 + episode_idx * 1009,
            price_mode="uniform",
            fixed_price=0.5,
            price_min=price_min,
            price_max=price_max,
            max_steps=max_steps,
        )
        for episode_idx in range(random_episodes)
    ]
    random_summary = _summary(random_results)

    zero_price_rows = [
        row for row in fixed_rows if abs(float(row["price"])) <= 1.0e-8
    ]
    zero_price_game_reward = (
        float(zero_price_rows[0]["mean_game_reward"])
        if zero_price_rows
        else None
    )
    for row in fixed_rows:
        row["all_five_counterfactual_net_reward"] = (
            zero_price_game_reward - MAX_PURCHASES * float(row["price"])
            if zero_price_game_reward is not None
            else float("nan")
        )
    profitable_rows = (
        [
            row
            for row in fixed_rows
            if row["all_five_counterfactual_net_reward"] > 1.0e-8
        ]
        if zero_price_game_reward is not None
        else []
    )
    pass_condition = {
        "criterion": (
            "all fixed prices where five bullets have positive paired "
            "zero-price counterfactual net value"
        ),
        "zero_price_reference_available": zero_price_game_reward is not None,
        "zero_price_game_reward": zero_price_game_reward,
        "profitable_fixed_prices": [row["price"] for row in profitable_rows],
        "buy_close_to_all_five": bool(profitable_rows)
        and all(row["mean_purchases"] >= 4.5 for row in profitable_rows),
        "fire_close_to_all_purchased": bool(profitable_rows)
        and all(
            row["aggregate_fired_fraction_of_purchases"] >= 0.9
            for row in profitable_rows
        ),
        "positive_net_reward": bool(profitable_rows)
        and all(row["mean_net_reward"] > 0.0 for row in profitable_rows),
        "random_price_fire_close_to_all_purchased": (
            random_summary["aggregate_fired_fraction_of_purchases"] >= 0.9
        ),
        "random_price_positive_net_reward": (
            random_summary["mean_net_reward"] > 0.0
        ),
    }
    pass_condition["passed"] = all(
        pass_condition[key]
        for key in (
            "zero_price_reference_available",
            "buy_close_to_all_five",
            "fire_close_to_all_purchased",
            "positive_net_reward",
            "random_price_fire_close_to_all_purchased",
            "random_price_positive_net_reward",
        )
    )

    return {
        "checkpoint": str(checkpoint),
        "game_checkpoint": str(game_checkpoint),
        "checkpoint_metadata": metadata,
        "copied_threshold_tensors": copied,
        "evaluation_policy": "deterministic_threshold_mean_and_gameplay_argmax",
        "gamma": 1.0,
        "fixed_price_table": fixed_rows,
        "random_price_summary": random_summary,
        "pass_condition": pass_condition,
        "fixed_price_episode_results": fixed_episode_results,
        "random_price_episode_results": random_results,
    }


def write_evaluation(result, output, fixed_csv):
    output = Path(output).expanduser().resolve()
    fixed_csv = Path(fixed_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_csv.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)

    rows = result["fixed_price_table"]
    with fixed_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return output, fixed_csv


def _print_table(result):
    print(
        "price  net_reward  game_reward  payments  purchases  shots  final_ammo  threshold",
        flush=True,
    )
    for row in result["fixed_price_table"]:
        print(
            f"{row['price']:>5.2f}  {row['mean_net_reward']:>10.3f}  "
            f"{row['mean_game_reward']:>11.3f}  {row['mean_payments']:>8.3f}  "
            f"{row['mean_purchases']:>9.3f}  {row['mean_shots_fired']:>5.3f}  "
            f"{row['mean_final_ammo']:>10.3f}  {row['mean_threshold']:>9.3f}",
            flush=True,
        )
    print(
        "random_price_summary="
        + json.dumps(result["random_price_summary"], sort_keys=True),
        flush=True,
    )
    print(
        "pass_condition=" + json.dumps(result["pass_condition"], sort_keys=True),
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--game-checkpoint")
    parser.add_argument(
        "--fixed-prices",
        default=",".join(str(value) for value in DEFAULT_FIXED_PRICES),
    )
    parser.add_argument("--episodes-per-price", type=int, default=20)
    parser.add_argument("--random-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=100_001)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output")
    parser.add_argument("--fixed-csv")
    args = parser.parse_args()
    if args.episodes_per_price <= 0 or args.random_episodes <= 0:
        parser.error("evaluation episode counts must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    return args


def main():
    args = parse_args()
    checkpoint = resolve_threshold_checkpoint(args.checkpoint)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint.with_suffix(".evaluation.json")
    )
    fixed_csv = (
        Path(args.fixed_csv).expanduser().resolve()
        if args.fixed_csv
        else checkpoint.with_suffix(".fixed_prices.csv")
    )
    result = evaluate_checkpoint(
        checkpoint,
        game_checkpoint=args.game_checkpoint,
        fixed_prices=parse_prices(args.fixed_prices),
        episodes_per_price=args.episodes_per_price,
        random_episodes=args.random_episodes,
        eval_seed=args.eval_seed,
        max_steps=args.max_steps,
    )
    output, fixed_csv = write_evaluation(result, output, fixed_csv)
    _print_table(result)
    print(f"evaluation={output}", flush=True)
    print(f"fixed_price_csv={fixed_csv}", flush=True)


if __name__ == "__main__":
    main()
