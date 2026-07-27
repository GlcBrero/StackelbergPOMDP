"""Train the clean E0a/E0b composite Atari policy with Stable-Baselines3."""

import argparse
import math
import os
from pathlib import Path
import tempfile


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
    e0_episode_transitions,
    evaluate_model,
    finish_run,
    init_wandb,
    make_vec_env,
)
from stackelberg_pomdp.atari.curriculum_env import (
    AtariCurriculumConfig,
    AtariCurriculumEnv,
)
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def env_config(args, *, seed):
    return AtariCurriculumConfig(
        stage=args.stage,
        seed=int(seed),
        gameplay_horizon=args.gameplay_horizon,
        event_tail_steps=args.event_tail_steps,
        noop_max=args.noop_max,
        frame_skip=4,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        max_frames=args.max_frames,
        rom_path=args.rom_path,
        fixed_event_steps=args.fixed_event_steps,
    )


def make_env(args, *, seed):
    return AtariCurriculumEnv(env_config(args, seed=seed))


def _stage_lr_scale(args):
    return args.pretrained_lr_scale if args.stage == "e0b" else 1.0


def _new_model(args, vec_env):
    model = ScaledLearningRatePPO(
        StackPOMDPAtariPolicy,
        vec_env,
        policy_kwargs={
            "economic_role": "gameplay",
            "economic_input_mode": "full",
            "visual_features": 512,
            "state_features": 64,
            "economic_hidden": 64,
            "critic_hidden": 256,
            "pretrained_lr_scale": _stage_lr_scale(args),
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
    if args.init_checkpoint is not None:
        provenance = model.policy.load_actor_checkpoint(
            args.init_checkpoint, include_economic=False, device=args.device
        )
        if provenance["source_economic_role"] != "gameplay":
            raise ValueError("E0b must initialize from a gameplay checkpoint")
        if provenance["source_economic_input_mode"] != "full":
            raise ValueError("E0b initialization requires the full actor state")
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
    if policy.economic_role != "gameplay":
        raise ValueError("--resume must contain an E0 gameplay policy")
    if policy.economic_input_mode != "full":
        raise ValueError("--resume is not a full-state E0 policy")
    expected_scale = _stage_lr_scale(args)
    if not math.isclose(
            policy.pretrained_lr_scale,
            expected_scale,
            rel_tol=0.0,
            abs_tol=1.0e-12,
    ):
        raise ValueError(
            "the saved E0 learning-rate scale does not match this stage "
            f"({policy.pretrained_lr_scale} != {expected_scale})"
        )
    return model


def build_model(args, vec_env):
    """Create a curriculum stage or resume its complete optimizer state."""

    return _resumed_model(args, vec_env) if args.resume else _new_model(args, vec_env)


def _parse_event_steps(raw):
    if raw is None:
        return None
    values = tuple(int(value.strip()) for value in raw.split(","))
    if len(values) != 5:
        raise ValueError("--fixed-event-steps requires five comma-separated steps")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("e0a", "e0b"), default="e0a")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--gameplay-horizon", type=int, default=200)
    parser.add_argument("--event-tail-steps", type=int, default=50)
    parser.add_argument("--fixed-event-steps", type=str)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--pretrained-lr-scale", type=float, default=0.1)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=100_000)
    parser.add_argument("--rom-path")
    parser.add_argument(
        "--init-checkpoint",
        help="clean E0a actor checkpoint used only to initialize a new E0b run",
    )
    parser.add_argument(
        "--resume",
        help="same-stage checkpoint whose model, critic, optimizer, and clock resume",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
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
    except ValueError as error:
        parser.error(str(error))
    transitions = e0_episode_transitions(args.stage, args.gameplay_horizon)
    args.n_steps = transitions if args.n_steps is None else args.n_steps
    expected_batch = args.n_steps * args.num_envs
    args.batch_size = expected_batch if args.batch_size is None else args.batch_size
    if args.gameplay_horizon <= 0:
        parser.error("--gameplay-horizon must be positive")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.n_steps != transitions:
        parser.error(
            f"--n-steps must equal one complete {args.stage} episode ({transitions})"
        )
    if args.batch_size <= 0 or args.batch_size > expected_batch:
        parser.error("--batch-size must lie in [1, n_steps * num_envs]")
    if expected_batch % args.batch_size:
        parser.error("--batch-size must divide n_steps * num_envs exactly")
    if args.timesteps <= 0 and not args.eval_only:
        parser.error("--timesteps must be positive during training")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.init_checkpoint and args.resume:
        parser.error("--init-checkpoint and --resume are mutually exclusive")
    if args.stage == "e0a" and args.init_checkpoint:
        parser.error("E0a does not accept --init-checkpoint")
    if (
            args.stage == "e0b"
            and args.init_checkpoint is None
            and args.resume is None
    ):
        parser.error("a new E0b run requires --init-checkpoint from clean E0a")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    default = (
        REPOSITORY_ROOT
        / "replication/atari/checkpoints/clean"
        / f"space_invaders_{args.stage}_ppo_seed{args.seed}.zip"
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
    run = init_wandb(args, stage=args.stage, checkpoint=args.checkpoint)
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
        evaluation = evaluate_model(
            model,
            lambda episode: make_env(
                args, seed=args.seed + 200_000 + episode
            ),
            episodes=args.eval_episodes,
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
