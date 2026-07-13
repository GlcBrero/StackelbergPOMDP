"""Train and evaluate a scarce-bullet Space Invaders gameplay policy.

The policy sees only four stacked Atari frames and selects only a gameplay
action.  Every episode starts with exactly five bullets and has no mechanism,
seller, pricing action, threshold action, or bullet replenishment.
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import ray
from ray.rllib.algorithms.a3c import A3C
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from stackelberg_pomdp.gym_envs.envs.atari_envs import SinglePlayerAtariBulletEnv
from stackerlberg.train.atari_models import NatureCNNTorch


ENV_NAME = "stackpomdp_space_invaders_five_bullets"
MODEL_NAME = "stackpomdp_nature_cnn"
WANDB_PROJECT = "StackPOMDP"
INITIAL_BULLETS = 5


class ScarceBulletMetricsCallbacks(DefaultCallbacks):
    """Publish per-episode bullet-use metrics through RLlib results."""

    def on_episode_end(self, *, episode, **kwargs):
        info = episode.last_info_for()
        if not info:
            return
        episode.custom_metrics["shots_fired"] = float(info.get("shots_fired", 0.0))
        episode.custom_metrics["final_ammo"] = float(info.get("final_ammo", 0.0))
        episode.custom_metrics["reward_per_bullet"] = float(
            info.get("reward_per_bullet", 0.0)
        )


def make_env_config(seed, max_steps=None):
    return {
        "seed": int(seed),
        "initial_bullets": INITIAL_BULLETS,
        "max_steps": max_steps,
        "noop_max": 30,
        "frame_skip": 4,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
    }


def build_config(args):
    config = {
        "env": ENV_NAME,
        "env_config": make_env_config(args.seed, args.max_steps),
        "framework": "torch",
        "preprocessor_pref": "rllib",
        "_disable_preprocessor_api": True,
        "callbacks": ScarceBulletMetricsCallbacks,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_gpus": 0,
        "num_gpus_per_worker": 0,
        "rollout_fragment_length": args.rollout_fragment_length,
        "lr": args.learning_rate,
        "entropy_coeff": args.entropy_coeff,
        "gamma": args.gamma,
        # Reward clipping occurs inside the shared Atari wrapper before frame
        # skipping.  RLlib must not clip the accumulated skip-window reward again.
        "clip_rewards": False,
        "model": {
            "custom_model": MODEL_NAME,
            "vf_share_layers": True,
        },
        "seed": args.seed,
    }
    if args.algorithm == "PPO":
        config.update({
            "train_batch_size": args.train_batch_size,
            "sgd_minibatch_size": args.sgd_minibatch_size,
            "num_sgd_iter": args.num_sgd_iter,
            "clip_param": args.clip_param,
            "batch_mode": "truncate_episodes",
        })
    else:
        config.update({
            "min_sample_timesteps_per_iteration": args.timesteps_per_iteration,
            "grad_clip": args.grad_clip,
            "vf_loss_coeff": args.vf_loss_coeff,
        })
    return config


def _metric(result, name, default=float("nan")):
    value = result.get(name, default)
    return default if value is None else value


def _custom_metric(result, name):
    return result.get("custom_metrics", {}).get(f"{name}_mean", float("nan"))


def save_checkpoint(trainer, path, args, iteration, timesteps):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = trainer.get_weights(["default_policy"])["default_policy"]
    payload = {
        # agent_1 is the key consumed by the frozen buyer gameplay environment.
        "agent_1": weights,
        "default_policy": weights,
        "metadata": {
            "algorithm": args.algorithm,
            "seed": args.seed,
            "initial_bullets": INITIAL_BULLETS,
            "no_replenishment": True,
            "clip_game_rewards": True,
            "iteration": iteration,
            "total_timesteps": timesteps,
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def evaluate_argmax(trainer, args):
    episodes = []
    for episode_idx in range(args.eval_episodes):
        env = SinglePlayerAtariBulletEnv(
            make_env_config(args.eval_seed + episode_idx * 1009, args.max_steps)
        )
        observation = env.reset()
        done = False
        reward = 0.0
        length = 0
        last_info = {}
        while not done:
            action = trainer.compute_single_action(observation, explore=False)
            observation, step_reward, done, last_info = env.step(action)
            reward += float(step_reward)
            length += 1
        episodes.append(
            {
                "episode": episode_idx,
                "reward": reward,
                "length": length,
                "shots_fired": int(last_info.get("shots_fired", 0)),
                "final_ammo": float(last_info.get("final_ammo", 0.0)),
                "reward_per_bullet": float(last_info.get("reward_per_bullet", 0.0)),
            }
        )
        env.close()

    keys = ("reward", "length", "shots_fired", "final_ammo", "reward_per_bullet")
    summary = {
        "episodes": args.eval_episodes,
        "evaluator": "deterministic_argmax",
        "initial_bullets": INITIAL_BULLETS,
        "no_replenishment": True,
        **{
            f"mean_{key}": float(np.mean([episode[key] for episode in episodes]))
            for key in keys
        },
        "positive_reward_episodes": sum(episode["reward"] > 0 for episode in episodes),
        "used_all_bullets_episodes": sum(
            episode["final_ammo"] == 0 for episode in episodes
        ),
    }
    return {"summary": summary, "episode_results": episodes}


def _init_wandb(args, checkpoint_path):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project=WANDB_PROJECT,
        entity=args.wandb_entity,
        name=args.wandb_name or f"five_bullet_{args.algorithm.lower()}_seed{args.seed}",
        config={
            "seed": args.seed,
            "algorithm": args.algorithm,
            "checkpoint_path": str(checkpoint_path),
            "initial_bullets": INITIAL_BULLETS,
            "no_replenishment": True,
            "timesteps": args.timesteps,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "clip_game_rewards": True,
            "noop_reset": True,
            "episodic_life": True,
            "frame_skip": 4,
            "frame_stack": 4,
            "evaluation_policy": "deterministic_argmax",
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["PPO", "A3C"], default="PPO")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument(
        "--iterations",
        type=int,
        help="Stop after this many training iterations instead of using --timesteps.",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=100_001)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--rollout-fragment-length", type=int)
    parser.add_argument("--train-batch-size", type=int, default=5_000)
    parser.add_argument("--sgd-minibatch-size", type=int, default=100)
    parser.add_argument("--num-sgd-iter", type=int, default=10)
    parser.add_argument("--timesteps-per-iteration", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--clip-param", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--vf-loss-coeff", type=float, default=0.25)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-entity", default="glcbrero")
    parser.add_argument("--wandb-name")
    parser.add_argument("--ray-local-mode", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.learning_rate is None:
        # RLlib's asynchronous A3C is substantially less tolerant of the
        # historical synchronous-A2C Atari rate (7e-4).  Use the native A3C
        # default here; the higher rate produced a flat/declining five-bullet
        # reward curve through 1.39M sampled steps.
        args.learning_rate = 2.5e-4 if args.algorithm == "PPO" else 1e-4
    if args.rollout_fragment_length is None:
        args.rollout_fragment_length = 100 if args.algorithm == "PPO" else 20
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive")

    checkpoint_path = Path(
        args.checkpoint
        or (
            "replication/atari/checkpoints/"
            f"space_invaders_5bullets_{args.algorithm.lower()}_seed{args.seed}.pkl"
        )
    ).expanduser().resolve()
    os.environ.setdefault("WANDB_START_METHOD", "thread")

    register_env(ENV_NAME, lambda config: SinglePlayerAtariBulletEnv(config))
    ModelCatalog.register_custom_model(MODEL_NAME, NatureCNNTorch)
    ray.init(
        local_mode=args.ray_local_mode,
        ignore_reinit_error=True,
        include_dashboard=False,
    )
    trainer_class = PPO if args.algorithm == "PPO" else A3C
    trainer = trainer_class(config=build_config(args))
    wandb_run = _init_wandb(args, checkpoint_path)

    started = time.time()
    iteration = 0
    timesteps = 0
    try:
        while True:
            result = trainer.train()
            iteration += 1
            timesteps = int(
                result.get("num_env_steps_sampled")
                or result.get("timesteps_total")
                or 0
            )
            metrics = {
                "episode_reward": float(_metric(result, "episode_reward_mean")),
                "episode_length": float(_metric(result, "episode_len_mean")),
                "shots_fired": float(_custom_metric(result, "shots_fired")),
                "final_ammo": float(_custom_metric(result, "final_ammo")),
                "reward_per_bullet": float(_custom_metric(result, "reward_per_bullet")),
                "total_timesteps": timesteps,
            }
            elapsed = time.time() - started
            print(
                f"iter={iteration:>4} steps={timesteps:>9} "
                f"reward={metrics['episode_reward']:>7.3f} "
                f"length={metrics['episode_length']:>7.1f} "
                f"shots={metrics['shots_fired']:>5.2f} "
                f"ammo={metrics['final_ammo']:>5.2f} "
                f"reward_per_bullet={metrics['reward_per_bullet']:>7.3f} "
                f"sps={timesteps / elapsed if elapsed else 0:>6.0f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(metrics, step=timesteps)

            stop = (
                iteration >= args.iterations
                if args.iterations is not None
                else timesteps >= args.timesteps
            )
            if iteration % args.checkpoint_every == 0 or stop:
                saved = save_checkpoint(trainer, checkpoint_path, args, iteration, timesteps)
                print(f"checkpoint={saved}", flush=True)
            if stop:
                break

        evaluation = evaluate_argmax(trainer, args)
        evaluation_path = checkpoint_path.with_suffix(".evaluation.json")
        with evaluation_path.open("w") as handle:
            json.dump(evaluation, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(evaluation["summary"], sort_keys=True), flush=True)
        print(f"evaluation={evaluation_path}", flush=True)

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"evaluation/{key}": value
                    for key, value in evaluation["summary"].items()
                    if isinstance(value, (int, float))
                },
                step=timesteps,
            )
            wandb_run.summary.update(evaluation["summary"])
            wandb_run.summary["checkpoint_path"] = str(checkpoint_path)
            wandb_run.summary["evaluation_path"] = str(evaluation_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        trainer.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
