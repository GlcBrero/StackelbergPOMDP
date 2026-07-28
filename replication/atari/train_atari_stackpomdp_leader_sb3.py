"""Train a clean full-trajectory E2 Atari StackPOMDP leader with PPO.

The leader first answers five canonical event-only queries.  A fresh bilateral
game then executes 200 gameplay decisions and five actor-identical cached
trade replays against a frozen opposite-role E1 composite response.  The new
leader inherits the same-role E1 visual, state, and Atari-game actor modules;
its economic actor and stage-private critic are initialized from scratch.
"""

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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from replication.atari.sb3_common import (
    EpisodeCheckpointCallback,
    ScaledLearningRatePPO,
    WANDB_GROUP,
    WANDB_PROJECT,
    checkpoint_path,
    e2_episode_transitions,
    evaluate_model,
    finish_run,
    init_wandb,
    make_vec_env,
)
from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS
from stackelberg_pomdp.atari.meta_response import (
    make_stackpomdp_atari_leader_env,
)
from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
)
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy
from stackelberg_pomdp.callbacks import FixPolicyActionsCallback


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def follower_role(leader_role):
    """Return the role whose frozen E1 response the leader faces."""

    if leader_role == SELLER:
        return BUYER
    if leader_role == BUYER:
        return SELLER
    raise ValueError(f"unknown leader role: {leader_role!r}")


def _existing_checkpoint(path, label):
    candidate = Path(path).expanduser()
    alternatives = (candidate, Path(f"{candidate}.zip"))
    for alternative in alternatives:
        if alternative.is_file():
            return alternative.resolve()
    raise FileNotFoundError(f"{label} checkpoint does not exist: {candidate}")


def checkpoint_policy_metadata(path, *, device="cpu", label="Atari"):
    """Read and validate the curriculum identity stored in one checkpoint."""

    resolved = _existing_checkpoint(path, label)
    model = PPO.load(str(resolved), device=device)
    try:
        policy = model.policy
        if not isinstance(policy, StackPOMDPAtariPolicy):
            raise TypeError(
                f"{label} checkpoint is not a clean StackPOMDPAtariPolicy: "
                f"{resolved}"
            )
        return {
            "path": str(resolved),
            "economic_role": policy.economic_role,
            "economic_input_mode": policy.economic_input_mode,
        }
    finally:
        del model


def validate_stage_checkpoints(args):
    """Enforce the E1-to-E2 role and actor-input contracts before rollout."""

    expected_follower = follower_role(args.leader_role)
    response = checkpoint_policy_metadata(
        args.response_checkpoint,
        device=args.device,
        label="frozen E1 response",
    )
    if response["economic_role"] != expected_follower:
        raise ValueError(
            f"{args.leader_role} leader requires a {expected_follower} E1 "
            f"response, got {response['economic_role']!r}"
        )
    if response["economic_input_mode"] != "full":
        raise ValueError("the frozen E1 response must use the full actor state")

    if args.resume:
        leader = checkpoint_policy_metadata(
            args.resume,
            device=args.device,
            label="E2 resume",
        )
        required_mode = "event_only"
    else:
        if args.leader_e1_checkpoint is None:
            raise ValueError(
                "a new E2 run requires --leader-e1-checkpoint from the "
                "same role"
            )
        leader = checkpoint_policy_metadata(
            args.leader_e1_checkpoint,
            device=args.device,
            label="same-role E1 initialization",
        )
        required_mode = "full"
    if leader["economic_role"] != args.leader_role:
        raise ValueError(
            f"{args.leader_role} leader requires a same-role checkpoint, got "
            f"{leader['economic_role']!r}"
        )
    if leader["economic_input_mode"] != required_mode:
        raise ValueError(
            f"leader checkpoint must use {required_mode!r} economic input, "
            f"got {leader['economic_input_mode']!r}"
        )
    return {"response": response, "leader": leader}


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


def make_env(args, *, seed):
    return make_stackpomdp_atari_leader_env(
        leader_role=args.leader_role,
        response_checkpoint=args.response_checkpoint,
        config=bilateral_config(args, seed=seed),
        device=args.device,
    )


def _new_model(args, vec_env):
    model = ScaledLearningRatePPO(
        StackPOMDPAtariPolicy,
        vec_env,
        policy_kwargs={
            "economic_role": args.leader_role,
            "economic_input_mode": "event_only",
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
        args.leader_e1_checkpoint,
        include_economic=False,
        device=args.device,
    )
    if provenance.get("critic_transferred") is not False:
        raise RuntimeError("E2 initialization must not transfer the E1 critic")
    model.policy.reset_economic_head(mean=0.5, concentration=2.0)
    model.policy.clear_obs_action_map()
    print({"actor_transfer": provenance, "fresh_economic_head": True}, flush=True)
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
    if not isinstance(model.policy, StackPOMDPAtariPolicy):
        raise TypeError("--resume must contain the clean Atari composite policy")
    if model.policy.economic_role != args.leader_role:
        raise ValueError("--resume role does not match --leader-role")
    if model.policy.economic_input_mode != "event_only":
        raise ValueError("--resume is not an event-only E2 leader")
    if not math.isclose(
            model.policy.pretrained_lr_scale,
            args.pretrained_lr_scale,
            rel_tol=0.0,
            abs_tol=1.0e-12,
    ):
        raise ValueError(
            "--pretrained-lr-scale must match the saved E2 checkpoint "
            f"({model.policy.pretrained_lr_scale})"
        )
    model.policy.clear_obs_action_map()
    return model


def build_model(args, vec_env):
    """Build E2 from E1 actors, or restore a complete in-progress E2 run."""

    return _resumed_model(args, vec_env) if args.resume else _new_model(args, vec_env)


def make_training_callback(args, *, wandb_run=None):
    """Return callbacks with the theoretical action cache always enabled."""

    return CallbackList([
        FixPolicyActionsCallback(),
        EpisodeCheckpointCallback(
            checkpoint=args.checkpoint,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            wandb_run=wandb_run,
            resume=bool(args.resume),
        ),
    ])


def evaluate_leader(model, args):
    """Evaluate deterministic event commitments with cached trade execution."""

    return evaluate_model(
        model,
        lambda episode: make_env(
            args, seed=args.seed + 300_000 + episode
        ),
        episodes=args.eval_episodes,
        use_action_cache=True,
    )


def _parse_event_steps(raw):
    if raw is None:
        return None
    values = tuple(int(value.strip()) for value in raw.split(","))
    if len(values) != NUM_TRADE_EVENTS:
        raise ValueError(
            "--fixed-event-steps requires five comma-separated steps"
        )
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-role", choices=(BUYER, SELLER), required=True)
    parser.add_argument("--response-checkpoint", required=True)
    parser.add_argument(
        "--leader-e1-checkpoint",
        "--leader-init-checkpoint",
        dest="leader_e1_checkpoint",
    )
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--gameplay-horizon", type=int, default=200)
    parser.add_argument("--event-tail-steps", type=int, default=0)
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
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
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
    transitions = e2_episode_transitions(args.gameplay_horizon)
    args.n_steps = transitions if args.n_steps is None else args.n_steps
    buffer_size = args.n_steps * args.num_envs
    args.batch_size = buffer_size if args.batch_size is None else args.batch_size
    if args.gameplay_horizon < NUM_TRADE_EVENTS:
        parser.error("--gameplay-horizon must be at least five")
    if args.event_tail_steps < 0:
        parser.error("--event-tail-steps must be nonnegative")
    if (
            args.gameplay_horizon - args.event_tail_steps
            < NUM_TRADE_EVENTS
    ):
        parser.error("E2 event window must contain at least five steps")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.n_steps != transitions:
        parser.error(
            "--n-steps must equal one complete E2 episode: five queries + "
            f"{args.gameplay_horizon} gameplay + five cached trades = "
            f"{transitions}"
        )
    if args.batch_size <= 0 or args.batch_size > buffer_size:
        parser.error("--batch-size must lie in [1, n_steps * num_envs]")
    if buffer_size % args.batch_size:
        parser.error("--batch-size must divide n_steps * num_envs exactly")
    if args.timesteps <= 0 and not args.eval_only:
        parser.error("--timesteps must be positive during training")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.entropy_coeff < 0.01:
        parser.error("E2 requires --entropy-coeff >= 0.01")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    if not args.resume and args.leader_e1_checkpoint is None:
        parser.error("a new E2 run requires --leader-e1-checkpoint")

    default = (
        REPOSITORY_ROOT
        / "replication/atari/checkpoints/clean"
        / f"leader_{args.leader_role}_e2_ppo_seed{args.seed}.zip"
    )
    args.checkpoint = str(checkpoint_path(args.checkpoint or default))
    return args


def main(argv=None):
    args = parse_args(argv)
    metadata = validate_stage_checkpoints(args)
    vec_env = make_vec_env(
        lambda rank: make_env(args, seed=args.seed + 10_000 * rank),
        num_envs=args.num_envs,
        start_method=args.start_method,
    )
    run = init_wandb(
        args, stage=f"e2_{args.leader_role}", checkpoint=args.checkpoint
    )
    if run is not None:
        run.config.update({
            "stage_checkpoints": metadata,
            "query_transitions_per_episode": NUM_TRADE_EVENTS,
            "gameplay_transitions_per_episode": args.gameplay_horizon,
            "cached_trade_replays_per_episode": NUM_TRADE_EVENTS,
            "policy_action_cache": True,
            "phase_wrapper": "StackPOMDPWrapper",
            "response_algorithm": "frozen_meta_policy",
            "leader_economic_input": "event_only",
            "response_economic_input": "full",
        }, allow_val_change=True)
    try:
        model = build_model(args, vec_env)
        if not args.eval_only:
            model.learn(
                total_timesteps=args.timesteps,
                callback=make_training_callback(args, wandb_run=run),
                reset_num_timesteps=not bool(args.resume),
            )
            model.save(args.checkpoint)
        evaluation = evaluate_leader(model, args)
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
