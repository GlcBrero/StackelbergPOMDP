"""Train a clean full-trajectory E1 Atari buyer or seller meta-response."""

import argparse
import math
import os
from pathlib import Path
import tempfile

import numpy as np

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stackpomdp-matplotlib")
)
os.environ.setdefault("WANDB_START_METHOD", "thread")

from stable_baselines3.common.callbacks import CallbackList

from replication.atari.sb3_common import (
    EpisodeCheckpointCallback,
    ScaledLearningRatePPO,
    WANDB_GROUP,
    WANDB_PROJECT,
    checkpoint_path,
    e1_episode_transitions,
    evaluate_model,
    fixed_context_csv_path,
    finish_run,
    init_wandb,
    make_vec_env,
    write_csv,
)
from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
    make_atari_meta_response_env,
)
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _validate_e0b_source(provenance):
    expected_role = "gameplay"
    actual_role = provenance["source_economic_role"]
    if actual_role != expected_role:
        raise ValueError(
            "E0b checkpoint must have economic_role="
            f"{expected_role!r}, got {actual_role!r}"
        )
    if provenance["source_economic_input_mode"] != "full":
        raise ValueError("E1 actor sources must use economic_input_mode='full'")


def bilateral_config(args, *, seed):
    return BilateralAtariConfig(
        seed=int(seed),
        gameplay_horizon=args.gameplay_horizon,
        event_tail_steps=args.event_tail_steps,
        seller_game_reward_scale=0.1,
        buyer_game_reward_scale=1.0,
        noop_max=args.noop_max,
        frame_skip=4,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        max_frames=args.max_frames,
        rom_path=args.rom_path,
        fixed_event_steps=args.fixed_event_steps,
    )


def make_env(args, *, seed, context_sampler=None):
    return make_atari_meta_response_env(
        controlled_role=args.role,
        e0b_checkpoint=args.e0b_checkpoint,
        config=bilateral_config(args, seed=seed),
        context_sampler=context_sampler,
        device=args.device,
    )


def _new_model(args, vec_env):
    model = ScaledLearningRatePPO(
        StackPOMDPAtariPolicy,
        vec_env,
        policy_kwargs={
            "economic_role": args.role,
            "economic_input_mode": "full",
            "visual_features": 512,
            "state_features": 64,
            "economic_hidden": 64,
            "critic_hidden": 256,
            "pretrained_lr_scale": args.pretrained_lr_scale,
        },
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=1.0,
        gae_lambda=1.0,
        clip_range=args.clip_range,
        ent_coef=args.entropy_coeff,
        vf_coef=args.value_coefficient,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    provenance = model.policy.load_actor_checkpoint(
        args.e0b_checkpoint,
        include_economic=False,
        device=args.device,
    )
    _validate_e0b_source(provenance)
    if args.role == BUYER:
        model.policy.reset_economic_head(mean=0.95, concentration=10.0)
    else:
        model.policy.reset_economic_head(mean=0.5, concentration=2.0)
    print({"actor_transfer": provenance}, flush=True)
    return model


def _resumed_model(args, vec_env):
    model = ScaledLearningRatePPO.load(
        args.resume,
        env=vec_env,
        device=args.device,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=1.0,
        gae_lambda=1.0,
        clip_range=args.clip_range,
        ent_coef=args.entropy_coeff,
        vf_coef=args.value_coefficient,
        max_grad_norm=args.max_grad_norm,
    )
    policy = model.policy
    if not isinstance(policy, StackPOMDPAtariPolicy):
        raise TypeError("--resume must contain the clean Atari composite policy")
    if policy.economic_role != args.role:
        raise ValueError("--resume role does not match --role")
    if policy.economic_input_mode != "full":
        raise ValueError("--resume is not a full-state E1 response")
    if not math.isclose(
            policy.pretrained_lr_scale,
            args.pretrained_lr_scale,
            rel_tol=0.0,
            abs_tol=1.0e-12,
    ):
        raise ValueError(
            "--pretrained-lr-scale must match the saved E1 checkpoint "
            f"({policy.pretrained_lr_scale})"
        )
    return model


def build_model(args, vec_env):
    """Start E1 from E0b actors, or resume a complete E1 optimizer state."""

    return _resumed_model(args, vec_env) if args.resume else _new_model(args, vec_env)


def _trade_diagnostics(episode_rows, *, gameplay_horizon):
    events = [
        event
        for row in episode_rows
        for event in row.get("events", ())
    ]
    result = {"trade_events": len(events)}
    for event_index in range(5):
        selected = [
            event for event in events
            if int(event["event_index"]) == event_index
        ]
        if selected:
            result[f"event_{event_index + 1}_acceptance_rate"] = float(
                np.mean([event["accepted"] for event in selected])
            )
            result[f"event_{event_index + 1}_mean_game_step"] = float(
                np.mean([event["game_step"] for event in selected])
            )
    time_bins = {
        "early": (0.0, 1.0 / 3.0),
        "middle": (1.0 / 3.0, 2.0 / 3.0),
        "late": (2.0 / 3.0, 1.0 + 1.0e-12),
    }
    for name, (low, high) in time_bins.items():
        selected = [
            event for event in events
            if low
            <= float(event["game_step"]) / float(gameplay_horizon)
            < high
        ]
        result[f"{name}_trade_events"] = len(selected)
        if selected:
            result[f"{name}_acceptance_rate"] = float(
                np.mean([event["accepted"] for event in selected])
            )
            result[f"{name}_mean_price"] = float(
                np.mean([event["price"] for event in selected])
            )
            result[f"{name}_mean_threshold"] = float(
                np.mean([event["threshold"] for event in selected])
            )
    return result


def evaluate_response(model, args):
    """Evaluate random commitments and a paired fixed-context grid."""

    random_evaluation = evaluate_model(
        model,
        lambda episode: make_env(
            args, seed=args.seed + 300_000 + episode
        ),
        episodes=args.eval_episodes,
    )
    random_evaluation["summary"].update(_trade_diagnostics(
        random_evaluation["episode_rows"],
        gameplay_horizon=args.gameplay_horizon,
    ))
    fixed_rows = []
    for value_index, value in enumerate(args.fixed_eval_values):
        fixed = float(value)
        context = np.full(5, fixed, dtype=np.float32)
        result = evaluate_model(
            model,
            lambda episode, context=context, value_index=value_index: make_env(
                args,
                seed=args.seed + 400_000 + 10_000 * value_index + episode,
                context_sampler=lambda rng, context=context: context,
            ),
            episodes=args.fixed_eval_episodes,
        )
        fixed_rows.append({
            "opponent_value": fixed,
            **result["summary"],
        })
    return {
        "summary": random_evaluation["summary"],
        "random": random_evaluation,
        "fixed_contexts": fixed_rows,
    }


def _parse_event_steps(raw):
    if raw is None:
        return None
    values = tuple(int(value.strip()) for value in raw.split(","))
    if len(values) != 5:
        raise ValueError("--fixed-event-steps requires five comma-separated steps")
    return values


def _parse_float_list(raw):
    values = tuple(float(value.strip()) for value in raw.split(","))
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("fixed evaluation values must lie in [0, 1]")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=(BUYER, SELLER), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--gameplay-horizon", type=int, default=200)
    parser.add_argument("--event-tail-steps", type=int, default=50)
    parser.add_argument("--fixed-event-steps", type=str)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--pretrained-lr-scale", type=float, default=0.1)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=100_000)
    parser.add_argument("--rom-path")
    parser.add_argument("--e0b-checkpoint", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--fixed-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--fixed-eval-values",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    parser.add_argument("--wandb-name")
    args = parser.parse_args(argv)
    try:
        args.fixed_event_steps = _parse_event_steps(args.fixed_event_steps)
        args.fixed_eval_values = _parse_float_list(args.fixed_eval_values)
    except ValueError as error:
        parser.error(str(error))
    transitions = e1_episode_transitions(args.gameplay_horizon)
    args.n_steps = transitions if args.n_steps is None else args.n_steps
    buffer_size = args.n_steps * args.num_envs
    args.batch_size = buffer_size if args.batch_size is None else args.batch_size
    if args.gameplay_horizon <= 0:
        parser.error("--gameplay-horizon must be positive")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.n_steps != transitions:
        parser.error(f"--n-steps must equal one full E1 episode ({transitions})")
    if args.batch_size <= 0 or args.batch_size > buffer_size:
        parser.error("--batch-size must lie in [1, n_steps * num_envs]")
    if buffer_size % args.batch_size:
        parser.error("--batch-size must divide n_steps * num_envs exactly")
    if args.timesteps <= 0 and not args.eval_only:
        parser.error("--timesteps must be positive during training")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.fixed_eval_episodes <= 0:
        parser.error("--fixed-eval-episodes must be positive")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    default = (
        REPOSITORY_ROOT
        / "replication/atari/checkpoints/clean"
        / f"meta_{args.role}_e1_ppo_seed{args.seed}.zip"
    )
    args.checkpoint = str(checkpoint_path(args.checkpoint or default))
    return args


def main(argv=None):
    args = parse_args(argv)
    vec_env = make_vec_env(
        lambda rank: make_env(args, seed=args.seed + 10_000 * rank),
        num_envs=args.num_envs,
        start_method=args.start_method,
    )
    run = init_wandb(args, stage=f"e1_{args.role}", checkpoint=args.checkpoint)
    try:
        model = build_model(args, vec_env)
        if not args.eval_only:
            callback = EpisodeCheckpointCallback(
                checkpoint=args.checkpoint,
                checkpoint_every=args.checkpoint_every,
                seed=args.seed,
                wandb_run=run,
                resume=bool(args.resume),
            )
            model.learn(
                total_timesteps=args.timesteps,
                callback=CallbackList([callback]),
                reset_num_timesteps=not bool(args.resume),
            )
            model.save(args.checkpoint)
        evaluation = evaluate_response(model, args)
        write_csv(
            fixed_context_csv_path(args.checkpoint),
            evaluation["fixed_contexts"],
        )
        finish_run(
            run,
            checkpoint=args.checkpoint,
            evaluation=evaluation,
            total_timesteps=model.num_timesteps,
        )
        run = None
        print(evaluation["summary"], flush=True)
    finally:
        vec_env.close()
        if run is not None:
            run.finish(exit_code=1)


if __name__ == "__main__":
    main()
