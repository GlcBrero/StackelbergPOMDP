"""Train the E1 buyer economic threshold head with frozen Atari gameplay.

The buyer starts with no bullets.  Five exogenous Uniform(0, 1) offers arrive,
each accepted offer transfers exactly one bullet, and payment is subtracted on
the purchase transition.  PPO trains only a four-scalar economic head; the
protected E0 gameplay policy executes deterministic argmax actions inside the
environment.  Rewards and GAE are undiscounted (gamma=lambda=1).
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from replication.atari.evaluate_buyer_threshold_head import (
    DEFAULT_GAME_CHECKPOINT,
    evaluate_checkpoint,
    parse_prices,
    resolve_game_checkpoint,
    write_evaluation,
)
from stackelberg_pomdp.atari_models import BuyerThresholdHeadTorch
from stackelberg_pomdp.gym_envs.envs.atari_envs import (
    FrozenGameplayBuyerThresholdEnv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_NAME = "stackpomdp_e1_frozen_gameplay_buyer_threshold"
MODEL_NAME = "stackpomdp_e1_buyer_threshold_head"
WANDB_PROJECT = "StackPOMDP"
ALGORITHM = "PPO"
OFFER_CHANCES = 5
MAX_PURCHASES = 5
GAMMA = 1.0
GAE_LAMBDA = 1.0


class BuyerThresholdMetricsCallbacks(DefaultCallbacks):
    """Expose complete E1 episode economics through RLlib results."""

    def on_episode_end(self, *, episode, **kwargs):
        info = episode.last_info_for()
        if not info:
            return
        scalar_keys = (
            "episode_net_reward",
            "episode_game_reward",
            "episode_payments",
            "economic_decisions",
            "game_steps",
            "trade_opportunities",
            "purchases",
            "shots_fired",
            "final_ammo",
            "acceptance_rate",
            "fired_fraction_of_purchases",
            "reward_per_purchased_bullet",
        )
        for key in scalar_keys:
            episode.custom_metrics[key] = float(info.get(key, 0.0))
        episode.custom_metrics["mean_threshold"] = float(
            np.mean(info.get("thresholds", (0.0,)))
        )
        episode.custom_metrics["mean_offered_price"] = float(
            np.mean(info.get("offered_prices", (0.0,)))
        )


def _portable_path(path):
    path = Path(path).expanduser().resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_env_config(args, game_checkpoint, *, seed=None, price_mode=None, fixed_price=None):
    return {
        "seed": int(args.seed if seed is None else seed),
        "game_checkpoint": str(game_checkpoint),
        "price_mode": price_mode or args.price_mode,
        "fixed_price": float(args.fixed_price if fixed_price is None else fixed_price),
        "price_min": float(args.price_min),
        "price_max": float(args.price_max),
        "offer_chances": OFFER_CHANCES,
        "max_replenish": MAX_PURCHASES,
        "buyer_initial_bullets": 0,
        "buyer_game_reward_scale": 1.0,
        "payment_penalty_lambda": 1.0,
        "max_steps": args.max_steps,
        "noop_max": 30,
        "frame_skip": 4,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
        "rollout_after_last_offer": True,
    }


def build_config(args, game_checkpoint):
    return {
        "env": ENV_NAME,
        "env_config": make_env_config(args, game_checkpoint),
        "framework": "torch",
        "preprocessor_pref": "rllib",
        "_disable_preprocessor_api": True,
        "callbacks": BuyerThresholdMetricsCallbacks,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_gpus": 0,
        "num_gpus_per_worker": 0,
        "rollout_fragment_length": args.rollout_fragment_length,
        "train_batch_size": args.train_batch_size,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        "num_sgd_iter": args.num_sgd_iter,
        "lr": args.learning_rate,
        "entropy_coeff": args.entropy_coeff,
        "gamma": GAMMA,
        "lambda": GAE_LAMBDA,
        "clip_param": args.clip_param,
        "vf_clip_param": args.vf_clip_param,
        "vf_loss_coeff": args.vf_loss_coeff,
        "clip_rewards": False,
        "clip_actions": True,
        "normalize_actions": False,
        "batch_mode": "complete_episodes",
        "model": {
            "custom_model": MODEL_NAME,
            "vf_share_layers": True,
            "custom_model_config": {
                "hidden_size": args.hidden_size,
                "price_max": args.price_max,
                "initial_threshold": args.initial_threshold,
                "initial_log_std": args.initial_log_std,
                "hide_price_from_actor": args.hide_price_from_actor,
            },
        },
        "seed": args.seed,
    }


def _custom_metric(result, key):
    value = result.get("custom_metrics", {}).get(f"{key}_mean", float("nan"))
    return float("nan") if value is None else float(value)


def _result_metric(result, key):
    value = result.get(key, float("nan"))
    return float("nan") if value is None else float(value)


def _checkpoint_metadata(args, game_checkpoint, iteration, timesteps):
    return {
        "format_version": 1,
        "stage": "E1_buyer_threshold_head",
        "algorithm": ALGORITHM,
        "seed": args.seed,
        "threshold_head_only": True,
        "gameplay_frozen": True,
        "gameplay_policy": "deterministic_argmax",
        "game_checkpoint": _portable_path(game_checkpoint),
        "game_checkpoint_sha256": _sha256(game_checkpoint),
        "buyer_initial_bullets": 0,
        "offer_chances": OFFER_CHANCES,
        "max_purchases": MAX_PURCHASES,
        "bullets_per_purchase": 1,
        "price_mode": args.price_mode,
        "price_min": args.price_min,
        "price_max": args.price_max,
        "fixed_price": args.fixed_price,
        "payment_timing": "immediate",
        "payment_penalty_lambda": 1.0,
        "clip_game_rewards": True,
        "gamma": GAMMA,
        "lambda": GAE_LAMBDA,
        "event_driven_trade_decisions": True,
        "rollout_game_after_last_offer": True,
        "observation_fields": list(
            FrozenGameplayBuyerThresholdEnv.OBSERVATION_FIELDS
        ),
        "normalized_remaining_opportunities": True,
        "normalized_buyer_ammo": True,
        "hidden_size": args.hidden_size,
        "initial_threshold": args.initial_threshold,
        "initial_log_std": args.initial_log_std,
        "reset_log_std_on_warmstart": args.reset_log_std_on_warmstart,
        "hide_price_from_actor": args.hide_price_from_actor,
        "initial_checkpoint": (
            _portable_path(args.initial_checkpoint)
            if args.initial_checkpoint is not None
            else None
        ),
        "initial_checkpoint_total_decision_timesteps": getattr(
            args, "initial_checkpoint_timesteps", 0
        ),
        "iteration": iteration,
        "total_decision_timesteps": timesteps,
        "cumulative_total_decision_timesteps": (
            getattr(args, "initial_checkpoint_timesteps", 0) + timesteps
        ),
        "max_game_steps_per_episode": args.max_steps,
        "noop_reset": True,
        "episodic_life": True,
        "frame_skip": 4,
        "frame_stack": 4,
    }


def save_checkpoint(trainer, checkpoint, args, game_checkpoint, iteration, timesteps):
    checkpoint = Path(checkpoint).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    weights = trainer.get_weights(["default_policy"])["default_policy"]
    payload = {
        "default_policy": weights,
        "agent_1": weights,
        "metadata": _checkpoint_metadata(
            args, game_checkpoint, iteration, timesteps
        ),
    }
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(checkpoint)
    return checkpoint


def load_initial_checkpoint(trainer, checkpoint, *, reset_log_std=False):
    """Warm-start only the E1 economic policy; gameplay is environment-owned."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    source = payload.get("default_policy") or payload.get("agent_1")
    if source is None:
        raise ValueError(
            f"{checkpoint} has no threshold policy weights: {list(payload)}"
        )
    current = trainer.get_weights(["default_policy"])["default_policy"]
    patched = {}
    copied = []
    actor_critic_split = any(key.startswith("critic_hidden") for key in current)
    for key, value in current.items():
        compatible = key in source and tuple(source[key].shape) == tuple(value.shape)
        if actor_critic_split and key.startswith("value"):
            compatible = False
        if reset_log_std and key == "threshold_log_std":
            compatible = False
        if compatible:
            patched[key] = source[key]
            copied.append(key)
        else:
            patched[key] = value
    if not any(key.startswith("threshold_raw_mean") for key in copied):
        raise ValueError(f"no compatible E1 actor tensors found in {checkpoint}")
    trainer.set_weights({"default_policy": patched})
    trainer.workers.sync_weights()
    metadata = dict(payload.get("metadata", {}))
    print(
        f"initial_checkpoint={checkpoint} copied_tensors={len(copied)} "
        f"parent_decisions={metadata.get('total_decision_timesteps', 0)} "
        f"reset_log_std={reset_log_std}",
        flush=True,
    )
    return metadata, copied


def _scripted_episode(args, game_checkpoint, *, threshold, fixed_price, seed):
    env = FrozenGameplayBuyerThresholdEnv(
        make_env_config(
            args,
            game_checkpoint,
            seed=seed,
            price_mode="fixed",
            fixed_price=fixed_price,
        )
    )
    observation = env.reset()
    done = False
    total_reward = 0.0
    last_info = {}
    while not done:
        assert observation[1] == 1.0
        assert abs(float(observation[0]) - fixed_price) < 1.0e-6
        observation, reward, done, last_info = env.step(
            np.array([threshold], dtype=np.float32)
        )
        total_reward += float(reward)
    env.close()
    return {"total_reward": total_reward, **last_info}


def validate_mechanics(args, game_checkpoint):
    """Run accepted/rejected paths before allocating a PPO trainer."""

    price = 0.2
    accepted = _scripted_episode(
        args,
        game_checkpoint,
        threshold=1.0,
        fixed_price=price,
        seed=args.mechanics_seed,
    )
    rejected = _scripted_episode(
        args,
        game_checkpoint,
        threshold=0.0,
        fixed_price=price,
        seed=args.mechanics_seed,
    )

    expected_payment = OFFER_CHANCES * price
    if accepted["trade_opportunities"] != OFFER_CHANCES:
        raise AssertionError("accepted-path smoke did not expose five offers")
    if accepted["purchases"] != MAX_PURCHASES:
        raise AssertionError("threshold=1 failed to buy all five bullets")
    if not math.isclose(
        accepted["episode_payments"], expected_payment, abs_tol=1.0e-6
    ):
        raise AssertionError("accepted-path payments are not immediate price sums")
    if not math.isclose(
        accepted["total_reward"],
        accepted["episode_game_reward"] - accepted["episode_payments"],
        abs_tol=1.0e-6,
    ):
        raise AssertionError("buyer net reward does not equal game reward minus payments")
    if accepted["shots_fired"] <= 0:
        raise AssertionError("frozen gameplay did not fire any purchased bullet")
    if not math.isclose(
        accepted["final_ammo"],
        accepted["purchases"] - accepted["shots_fired"],
        abs_tol=1.0e-6,
    ):
        raise AssertionError("bullet transfers and shot consumption do not reconcile")

    if rejected["trade_opportunities"] != OFFER_CHANCES:
        raise AssertionError("rejected-path smoke did not expose five offers")
    if rejected["purchases"] != 0 or rejected["episode_payments"] != 0.0:
        raise AssertionError("threshold=0 accepted a strictly positive price")
    if rejected["shots_fired"] != 0 or rejected["final_ammo"] != 0.0:
        raise AssertionError("rejected-path buyer acquired or fired a bullet")

    summary = {
        "passed": True,
        "price": price,
        "accepted": {
            key: accepted[key]
            for key in (
                "episode_game_reward",
                "episode_payments",
                "total_reward",
                "trade_opportunities",
                "purchases",
                "shots_fired",
                "final_ammo",
                "game_steps",
            )
        },
        "rejected": {
            key: rejected[key]
            for key in (
                "episode_game_reward",
                "episode_payments",
                "total_reward",
                "trade_opportunities",
                "purchases",
                "shots_fired",
                "final_ammo",
                "game_steps",
            )
        },
    }
    print("mechanics_smoke=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _init_wandb(args, checkpoint, game_checkpoint):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type=args.wandb_job_type,
        name=args.wandb_name
        or f"e1_buyer_threshold_ppo_{args.price_mode}_seed{args.seed}",
        config={
            "stage": "E1_buyer_threshold_head",
            "algorithm": ALGORITHM,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "checkpoint_path": str(checkpoint),
            "game_checkpoint": _portable_path(game_checkpoint),
            "gameplay_frozen": True,
            "gameplay_policy": "deterministic_argmax",
            "buyer_initial_bullets": 0,
            "offer_chances": OFFER_CHANCES,
            "max_purchases": MAX_PURCHASES,
            "bullets_per_purchase": 1,
            "price_process": (
                "Uniform(0, 1)" if args.price_mode == "uniform" else "fixed"
            ),
            "continuous_price_observation": True,
            "payment_timing": "immediate",
            "gamma": GAMMA,
            "lambda": GAE_LAMBDA,
            "clip_game_rewards": True,
            "event_driven_trade_decisions": True,
            "observation_fields": list(
                FrozenGameplayBuyerThresholdEnv.OBSERVATION_FIELDS
            ),
            "hidden_size": args.hidden_size,
            "initial_threshold": args.initial_threshold,
            "initial_log_std": args.initial_log_std,
            "reset_log_std_on_warmstart": args.reset_log_std_on_warmstart,
            "hide_price_from_actor": args.hide_price_from_actor,
            "target_decision_timesteps": args.timesteps,
            "iterations": args.iterations,
            "initial_checkpoint": args.initial_checkpoint,
            "initial_checkpoint_total_decision_timesteps": getattr(
                args, "initial_checkpoint_timesteps", 0
            ),
            "wandb_job_type": args.wandb_job_type,
        },
    )


def _write_jsonl(handle, record):
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument(
        "--iterations",
        type=int,
        help="Stop after this many PPO iterations instead of --timesteps.",
    )
    parser.add_argument("--game-checkpoint", default=str(DEFAULT_GAME_CHECKPOINT))
    parser.add_argument(
        "--initial-checkpoint",
        help="Optional E1 threshold checkpoint used to warm-start PPO.",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--archive-checkpoints", action="store_true")
    parser.add_argument("--training-log")
    parser.add_argument("--price-mode", choices=("uniform", "fixed"), default="uniform")
    parser.add_argument("--price-min", type=float, default=0.0)
    parser.add_argument("--price-max", type=float, default=1.0)
    parser.add_argument("--fixed-price", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--rollout-fragment-length", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=500)
    parser.add_argument("--sgd-minibatch-size", type=int, default=100)
    parser.add_argument("--num-sgd-iter", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--vf-clip-param", type=float, default=10.0)
    parser.add_argument("--vf-loss-coeff", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--initial-threshold", type=float, default=0.5)
    parser.add_argument("--initial-log-std", type=float, default=-0.7)
    parser.add_argument(
        "--reset-log-std-on-warmstart",
        action="store_true",
        help=(
            "Keep --initial-log-std instead of importing the parent's "
            "exploration scale when warm-starting an E1 head."
        ),
    )
    parser.add_argument(
        "--hide-price-from-actor",
        action="store_true",
        help=(
            "Keep price in the observation/value function but exclude it from "
            "the willingness-to-pay actor, making the threshold price-invariant."
        ),
    )
    parser.add_argument(
        "--mechanics-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mechanics-seed", type=int, default=100_001)
    parser.add_argument(
        "--fixed-eval-prices", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
    )
    parser.add_argument("--eval-episodes-per-price", type=int, default=20)
    parser.add_argument("--random-eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=100_001)
    parser.add_argument("--evaluation-output")
    parser.add_argument("--fixed-price-csv")
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default="glcbrero")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-job-type", default="atari_e1_buyer_threshold")
    parser.add_argument("--ray-local-mode", action="store_true")
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.iterations is not None and args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.train_batch_size <= 0 or args.sgd_minibatch_size <= 0:
        parser.error("PPO batch sizes must be positive")
    if args.sgd_minibatch_size > args.train_batch_size:
        parser.error("--sgd-minibatch-size cannot exceed --train-batch-size")
    if not 0.0 <= args.price_min <= args.price_max:
        parser.error("prices must satisfy 0 <= price_min <= price_max")
    if not args.price_min <= args.fixed_price <= args.price_max:
        parser.error("--fixed-price must lie inside the configured price range")
    if not 0.0 <= args.initial_threshold <= args.price_max:
        parser.error("--initial-threshold must lie in [0, price_max]")
    if args.eval_episodes_per_price <= 0 or args.random_eval_episodes <= 0:
        parser.error("evaluation episode counts must be positive")
    return args


def main():
    args = parse_args()
    game_checkpoint = resolve_game_checkpoint(args.game_checkpoint)
    checkpoint = Path(
        args.checkpoint
        or (
            "replication/atari/checkpoints/e1/"
            f"buyer_threshold_ppo_uniform_seed{args.seed}.pkl"
        )
    ).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.initial_checkpoint_timesteps = 0
    if args.initial_checkpoint is not None:
        args.initial_checkpoint = str(
            Path(args.initial_checkpoint).expanduser().resolve()
        )
        if not Path(args.initial_checkpoint).exists():
            raise FileNotFoundError(args.initial_checkpoint)
        with Path(args.initial_checkpoint).open("rb") as handle:
            initial_payload = pickle.load(handle)
        args.initial_checkpoint_timesteps = int(
            initial_payload.get("metadata", {}).get(
                "total_decision_timesteps", 0
            )
        )
    training_log = Path(
        args.training_log or checkpoint.with_suffix(".training.jsonl")
    ).expanduser().resolve()
    training_log.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output = Path(
        args.evaluation_output or checkpoint.with_suffix(".evaluation.json")
    ).expanduser().resolve()
    fixed_price_csv = Path(
        args.fixed_price_csv or checkpoint.with_suffix(".fixed_prices.csv")
    ).expanduser().resolve()

    if args.mechanics_check:
        mechanics = validate_mechanics(args, game_checkpoint)
    else:
        mechanics = {"passed": None, "skipped": True}

    os.environ.setdefault("WANDB_START_METHOD", "thread")
    register_env(
        ENV_NAME,
        lambda config: FrozenGameplayBuyerThresholdEnv(config),
    )
    ModelCatalog.register_custom_model(MODEL_NAME, BuyerThresholdHeadTorch)
    wandb_run = _init_wandb(args, checkpoint, game_checkpoint)
    if wandb_run is not None:
        print(f"wandb_run={wandb_run.url}", flush=True)
    ray.init(
        local_mode=args.ray_local_mode,
        ignore_reinit_error=True,
        include_dashboard=False,
    )
    trainer = None
    completed_training = False
    iteration = 0
    timesteps = 0
    estimated_game_steps = 0
    started = time.time()
    try:
        trainer = PPO(config=build_config(args, game_checkpoint))
        if args.initial_checkpoint is not None:
            parent_metadata, _ = load_initial_checkpoint(
                trainer,
                args.initial_checkpoint,
                reset_log_std=args.reset_log_std_on_warmstart,
            )
            args.initial_checkpoint_timesteps = int(
                parent_metadata.get("total_decision_timesteps", 0)
            )
        with training_log.open("w") as log_handle:
            _write_jsonl(
                log_handle,
                {
                    "record": "configuration",
                    "algorithm": ALGORITHM,
                    "seed": args.seed,
                    "game_checkpoint": _portable_path(game_checkpoint),
                    "checkpoint": str(checkpoint),
                    "learning_rate": args.learning_rate,
                    "entropy_coeff": args.entropy_coeff,
                    "gamma": GAMMA,
                    "lambda": GAE_LAMBDA,
                    "price_mode": args.price_mode,
                    "offer_chances": OFFER_CHANCES,
                    "max_purchases": MAX_PURCHASES,
                    "hidden_size": args.hidden_size,
                    "hide_price_from_actor": args.hide_price_from_actor,
                    "initial_log_std": args.initial_log_std,
                    "num_workers": args.num_workers,
                    "train_batch_size": args.train_batch_size,
                    "sgd_minibatch_size": args.sgd_minibatch_size,
                    "num_sgd_iter": args.num_sgd_iter,
                    "mechanics_smoke": mechanics,
                    "initial_checkpoint": args.initial_checkpoint,
                    "initial_checkpoint_total_decision_timesteps": (
                        args.initial_checkpoint_timesteps
                    ),
                    "reset_log_std_on_warmstart": (
                        args.reset_log_std_on_warmstart
                    ),
                    "wandb_run_url": (
                        wandb_run.url if wandb_run is not None else None
                    ),
                },
            )
            while True:
                result = trainer.train()
                iteration += 1
                timesteps = int(
                    result.get("num_env_steps_sampled")
                    or result.get("timesteps_total")
                    or 0
                )
                episodes_this_iter = int(result.get("episodes_this_iter") or 0)
                mean_game_steps = _custom_metric(result, "game_steps")
                if episodes_this_iter and math.isfinite(mean_game_steps):
                    estimated_game_steps += int(
                        round(episodes_this_iter * mean_game_steps)
                    )
                metrics = {
                    "episode_reward": _result_metric(result, "episode_reward_mean"),
                    "episode_length": _result_metric(result, "episode_len_mean"),
                    "game_reward": _custom_metric(result, "episode_game_reward"),
                    "payments": _custom_metric(result, "episode_payments"),
                    "purchases": _custom_metric(result, "purchases"),
                    "trade_opportunities": _custom_metric(
                        result, "trade_opportunities"
                    ),
                    "acceptance_rate": _custom_metric(result, "acceptance_rate"),
                    "shots_fired": _custom_metric(result, "shots_fired"),
                    "final_ammo": _custom_metric(result, "final_ammo"),
                    "fired_fraction_of_purchases": _custom_metric(
                        result, "fired_fraction_of_purchases"
                    ),
                    "reward_per_purchased_bullet": _custom_metric(
                        result, "reward_per_purchased_bullet"
                    ),
                    "threshold": _custom_metric(result, "mean_threshold"),
                    "offered_price": _custom_metric(result, "mean_offered_price"),
                    "game_steps_per_episode": mean_game_steps,
                    "total_decision_timesteps": timesteps,
                    "cumulative_total_decision_timesteps": (
                        args.initial_checkpoint_timesteps + timesteps
                    ),
                    "estimated_total_game_steps": estimated_game_steps,
                }
                elapsed = time.time() - started
                print(
                    f"iter={iteration:>4} decisions={timesteps:>7} "
                    f"game_steps~={estimated_game_steps:>9} "
                    f"net={metrics['episode_reward']:>7.3f} "
                    f"game={metrics['game_reward']:>6.3f} "
                    f"paid={metrics['payments']:>6.3f} "
                    f"buys={metrics['purchases']:>5.2f} "
                    f"shots={metrics['shots_fired']:>5.2f} "
                    f"thr={metrics['threshold']:>5.3f} "
                    f"elapsed={elapsed:>7.0f}s",
                    flush=True,
                )
                _write_jsonl(
                    log_handle,
                    {"record": "training", "iteration": iteration, **metrics},
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=timesteps)

                stop = (
                    iteration >= args.iterations
                    if args.iterations is not None
                    else timesteps >= args.timesteps
                )
                if iteration % args.checkpoint_every == 0 or stop:
                    saved = save_checkpoint(
                        trainer,
                        checkpoint,
                        args,
                        game_checkpoint,
                        iteration,
                        timesteps,
                    )
                    print(f"checkpoint={saved}", flush=True)
                    if args.archive_checkpoints:
                        archive = saved.with_name(
                            f"{saved.stem}_step{timesteps}{saved.suffix}"
                        )
                        shutil.copy2(saved, archive)
                        print(f"checkpoint_archive={archive}", flush=True)
                if stop:
                    completed_training = True
                    break
    finally:
        if trainer is not None:
            trainer.stop()
        ray.shutdown()

    if not completed_training:
        raise RuntimeError("training exited before reaching its stopping condition")

    evaluation = evaluate_checkpoint(
        checkpoint,
        game_checkpoint=game_checkpoint,
        fixed_prices=parse_prices(args.fixed_eval_prices),
        episodes_per_price=args.eval_episodes_per_price,
        random_episodes=args.random_eval_episodes,
        eval_seed=args.eval_seed,
        max_steps=args.max_steps,
    )
    evaluation_output, fixed_price_csv = write_evaluation(
        evaluation, evaluation_output, fixed_price_csv
    )
    print(
        "random_price_summary="
        + json.dumps(evaluation["random_price_summary"], sort_keys=True),
        flush=True,
    )
    print(
        "pass_condition="
        + json.dumps(evaluation["pass_condition"], sort_keys=True),
        flush=True,
    )
    print(f"evaluation={evaluation_output}", flush=True)
    print(f"fixed_price_csv={fixed_price_csv}", flush=True)

    if wandb_run is not None:
        import wandb

        table_rows = evaluation["fixed_price_table"]
        columns = list(table_rows[0])
        wandb_run.log(
            {
                "evaluation/fixed_price_table": wandb.Table(
                    columns=columns,
                    data=[[row[column] for column in columns] for row in table_rows],
                )
            },
            step=timesteps,
        )
        for key, value in evaluation["random_price_summary"].items():
            if isinstance(value, (int, float)):
                wandb_run.summary[f"random_eval/{key}"] = value
        for key, value in evaluation["pass_condition"].items():
            wandb_run.summary[f"pass_condition/{key}"] = value
        wandb_run.summary["checkpoint_path"] = str(checkpoint)
        wandb_run.summary["training_log"] = str(training_log)
        wandb_run.summary["evaluation_path"] = str(evaluation_output)
        wandb_run.summary["fixed_price_csv"] = str(fixed_price_csv)
        artifact = wandb.Artifact(
            f"e1-buyer-threshold-seed{args.seed}", type="model"
        )
        artifact.add_file(str(checkpoint))
        artifact.add_file(str(training_log))
        artifact.add_file(str(evaluation_output))
        artifact.add_file(str(fixed_price_csv))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()
