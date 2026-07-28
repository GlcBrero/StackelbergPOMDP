"""Train a clean full-trajectory E1 Atari buyer or seller meta-response."""

import argparse
from copy import copy, deepcopy
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch as th

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stackpomdp-matplotlib")
)
os.environ.setdefault("WANDB_START_METHOD", "thread")

from stable_baselines3.common.callbacks import CallbackList

from replication.atari.sb3_common import (
    ACTOR_LOSS_MODES,
    EpisodeCheckpointCallback,
    PHASE_BALANCED_ACTOR_LOSS_MODE,
    STANDARD_ACTOR_LOSS_MODE,
    WANDB_GROUP,
    WANDB_PROJECT,
    attach_atari_training_contract,
    checkpoint_path,
    e1_episode_transitions,
    evaluate_model,
    fixed_context_csv_path,
    finish_run,
    init_wandb,
    make_vec_env,
    model_actor_loss_mode,
    model_economic_initialization,
    ppo_class_for_actor_loss_mode,
    write_csv,
)
from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
    make_atari_meta_response_env,
)
from stackelberg_pomdp.atari.e1_sampling import (
    E1_SAMPLER_MODES,
    TEMPORAL_MIX_E1_SAMPLER,
    TEMPORAL_MIX_GAMEPLAY_HORIZON,
    UNIFORM_E1_SAMPLER,
    e1_sampler_provenance,
)
from stackelberg_pomdp.atari.protocol import (
    NUM_TRADE_EVENTS,
    OPPONENT_COMMITMENT_SLICE,
)
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUYER_INIT_MEAN = 0.95
BUYER_INIT_CONCENTRATION = 10.0
SELLER_INIT_MEAN = 0.5
SELLER_INIT_CONCENTRATION = 2.0


def _actor_loss_mode(args):
    return str(getattr(args, "actor_loss_mode", STANDARD_ACTOR_LOSS_MODE))


def _e1_sampler_mode(args):
    return str(getattr(args, "e1_sampler_mode", UNIFORM_E1_SAMPLER))


def _economic_initialization(args):
    if args.role == BUYER:
        return {
            "mean": float(getattr(args, "buyer_init_mean", BUYER_INIT_MEAN)),
            "concentration": float(getattr(
                args, "buyer_init_concentration", BUYER_INIT_CONCENTRATION
            )),
        }
    return {
        "mean": SELLER_INIT_MEAN,
        "concentration": SELLER_INIT_CONCENTRATION,
    }


def _value_slug(value):
    return format(float(value), ".6g").replace("-", "m").replace(".", "p")


def _run_variant_suffix(args):
    parts = []
    if _actor_loss_mode(args) != STANDARD_ACTOR_LOSS_MODE:
        parts.append(_actor_loss_mode(args))
    if args.role == BUYER and (
            not math.isclose(args.buyer_init_mean, BUYER_INIT_MEAN)
            or not math.isclose(
                args.buyer_init_concentration, BUYER_INIT_CONCENTRATION
            )
    ):
        parts.append(
            f"initm{_value_slug(args.buyer_init_mean)}"
            f"c{_value_slug(args.buyer_init_concentration)}"
        )
    if args.target_kl is not None:
        parts.append(f"kl{_value_slug(args.target_kl)}")
    if _e1_sampler_mode(args) != UNIFORM_E1_SAMPLER:
        parts.append("temporal_mix_v1")
    return "" if not parts else "_" + "_".join(parts)


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
    if not math.isclose(
            float(provenance["source_pretrained_lr_scale"]),
            0.1,
            rel_tol=0.0,
            abs_tol=1.0e-12,
    ):
        raise ValueError(
            "E1 requires an E0b checkpoint with pretrained_lr_scale=0.1; "
            f"got {provenance['source_pretrained_lr_scale']!r}"
        )


def _checkpoint_sha256(raw):
    path = Path(raw).expanduser()
    if not path.is_file() and Path(f"{path}.zip").is_file():
        path = Path(f"{path}.zip")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(raw, *, label):
    path = Path(raw).expanduser()
    if not path.is_file() and Path(f"{path}.zip").is_file():
        path = Path(f"{path}.zip")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return {"path": str(path), "sha256": _checkpoint_sha256(path)}


def _sampler_provenance(args):
    return e1_sampler_provenance(
        _e1_sampler_mode(args),
        gameplay_horizon=int(getattr(args, "gameplay_horizon", 200)),
        event_tail_steps=int(getattr(args, "event_tail_steps", 0)),
    )


def _attach_sampler_contract(
        model,
        args,
        *,
        resume_source=None,
        preserve_existing_sampler=False,
):
    """Persist the current sampler stage and its byte-bound parent."""

    saved_current = getattr(model, "atari_e1_sampler_provenance", None)
    current = (
        deepcopy(saved_current)
        if preserve_existing_sampler and saved_current is not None
        else _sampler_provenance(args)
    )
    existing = getattr(model, "atari_e1_sampler_history", None)
    if existing is None:
        parent = e1_sampler_provenance(
            UNIFORM_E1_SAMPLER,
            gameplay_horizon=int(getattr(args, "gameplay_horizon", 200)),
            event_tail_steps=int(getattr(args, "event_tail_steps", 0)),
        )
        history = [{
            "start_total_timesteps": 0,
            "sampler": parent,
            "inferred_for_legacy_checkpoint": resume_source is not None,
            "resume_sources": [],
        }]
    elif not isinstance(existing, list):
        raise ValueError("E1 sampler history must be a list")
    else:
        history = deepcopy(existing)

    if resume_source is not None:
        source_record = {
            **resume_source,
            "resume_total_timesteps": int(model.num_timesteps),
        }
        if history[-1].get("sampler") == current:
            history[-1].setdefault("resume_sources", []).append(source_record)
        else:
            history.append({
                "start_total_timesteps": int(model.num_timesteps),
                "sampler": current,
                "inferred_for_legacy_checkpoint": False,
                "resume_sources": [source_record],
            })
    else:
        history[0]["sampler"] = current
    model.atari_e1_sampler_provenance = dict(current)
    model.atari_e1_sampler_history = history
    model.atari_e1_resume_source_provenance = (
        None if resume_source is None else dict(resume_source)
    )


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
        e1_sampler_mode=_e1_sampler_mode(args),
        device=args.device,
    )


def _new_model(args, vec_env):
    actor_loss_mode = _actor_loss_mode(args)
    algorithm_class = ppo_class_for_actor_loss_mode(actor_loss_mode)
    model = algorithm_class(
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
        target_kl=getattr(args, "target_kl", None),
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
    # The opponent commitment is identically zero throughout E0a/E0b, so its
    # five input columns retain arbitrary initialization values.  Reset only
    # those previously unseen columns before E1: gameplay therefore starts
    # exactly invariant to price/threshold context, while ordinary PPO
    # gradients remain free to learn economic effects during fine-tuning.
    state_input = model.policy.features_extractor.state_encoder[0]
    with th.no_grad():
        state_input.weight[:, OPPONENT_COMMITMENT_SLICE].zero_()
    provenance = {
        **provenance,
        "modules": list(provenance["modules"]),
        "zero_initialized_actor_state_inputs": ["opponent_commitment"],
        "zero_initialized_actor_state_indices": list(range(
            OPPONENT_COMMITMENT_SLICE.start,
            OPPONENT_COMMITMENT_SLICE.stop,
        )),
    }
    model.atari_e1_source_provenance = dict(provenance)
    initialization = _economic_initialization(args)
    model.policy.reset_economic_head(**initialization)
    attach_atari_training_contract(
        model,
        actor_loss_mode=actor_loss_mode,
        economic_init_mean=initialization["mean"],
        economic_init_concentration=initialization["concentration"],
    )
    _attach_sampler_contract(model, args)
    print({"actor_transfer": provenance}, flush=True)
    return model


def _resumed_model(args, vec_env):
    actor_loss_mode = _actor_loss_mode(args)
    algorithm_class = ppo_class_for_actor_loss_mode(actor_loss_mode)
    resume_source = _checkpoint_identity(
        args.resume, label="E1 resume checkpoint"
    )
    model = algorithm_class.load(
        resume_source["path"],
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
    expected_target_kl = getattr(args, "target_kl", None)
    saved_target_kl = getattr(model, "target_kl", None)
    if (
            (saved_target_kl is None) != (expected_target_kl is None)
            or (
                saved_target_kl is not None
                and not math.isclose(
                    float(saved_target_kl),
                    float(expected_target_kl),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
    ):
        raise ValueError(
            "--target-kl must match the saved E1 checkpoint "
            f"({saved_target_kl})"
        )
    saved_actor_loss_mode = model_actor_loss_mode(model)
    if saved_actor_loss_mode not in ACTOR_LOSS_MODES:
        raise ValueError(
            "--resume contains an unknown Atari actor loss mode: "
            f"{saved_actor_loss_mode!r}"
        )
    if saved_actor_loss_mode != actor_loss_mode:
        raise ValueError(
            "--actor-loss-mode must match the saved E1 checkpoint "
            f"({saved_actor_loss_mode})"
        )
    expected_initialization = _economic_initialization(args)
    saved_initialization = model_economic_initialization(
        model,
        default_mean=(
            BUYER_INIT_MEAN if args.role == BUYER else SELLER_INIT_MEAN
        ),
        default_concentration=(
            BUYER_INIT_CONCENTRATION
            if args.role == BUYER
            else SELLER_INIT_CONCENTRATION
        ),
    )
    if saved_initialization != expected_initialization:
        raise ValueError(
            "economic-head initialization arguments must match the saved E1 "
            f"checkpoint ({saved_initialization})"
        )
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
    provenance = getattr(model, "atari_e1_source_provenance", None)
    if provenance is None:
        raise ValueError("E1 resume is missing its E0b source provenance")
    _validate_e0b_source(provenance)
    current_sha256 = _checkpoint_sha256(args.e0b_checkpoint)
    if current_sha256 != provenance["sha256"]:
        raise ValueError(
            "--e0b-checkpoint differs from the source bound into this E1 "
            f"run ({current_sha256} != {provenance['sha256']})"
        )
    attach_atari_training_contract(
        model,
        actor_loss_mode=actor_loss_mode,
        economic_init_mean=expected_initialization["mean"],
        economic_init_concentration=expected_initialization["concentration"],
    )
    if _checkpoint_sha256(resume_source["path"]) != resume_source["sha256"]:
        raise RuntimeError("E1 resume checkpoint changed while it was loaded")
    resume_source["training_total_timesteps"] = int(model.num_timesteps)
    _attach_sampler_contract(
        model,
        args,
        resume_source=resume_source,
        preserve_existing_sampler=bool(getattr(args, "eval_only", False)),
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


def canonical_evaluation_args(args):
    """Return an evaluation-only copy using the preregistered uniform law."""

    evaluation_args = copy(args)
    evaluation_args.e1_sampler_mode = UNIFORM_E1_SAMPLER
    return evaluation_args


def evaluate_response(model, args):
    """Evaluate random commitments and a paired fixed-context grid."""

    evaluation_args = canonical_evaluation_args(args)
    random_evaluation = evaluate_model(
        model,
        lambda episode: make_env(
            evaluation_args, seed=args.seed + 300_000 + episode
        ),
        episodes=args.eval_episodes,
    )
    random_evaluation["summary"].update(_trade_diagnostics(
        random_evaluation["episode_rows"],
        gameplay_horizon=args.gameplay_horizon,
    ))
    fixed_rows = []
    fixed_evaluations = []
    for value in args.fixed_eval_values:
        fixed = float(value)
        context = np.full(5, fixed, dtype=np.float32)
        result = evaluate_model(
            model,
            lambda episode, context=context: make_env(
                evaluation_args,
                # Reuse the same ALE/no-op and event-schedule seeds at every
                # opponent value.  The fixed grid is therefore genuinely
                # paired; only the commitment changes across rows.
                seed=args.seed + 400_000 + episode,
                context_sampler=lambda rng, context=context: context,
            ),
            episodes=args.fixed_eval_episodes,
        )
        result["summary"].update(_trade_diagnostics(
            result["episode_rows"],
            gameplay_horizon=args.gameplay_horizon,
        ))
        fixed_rows.append({
            "opponent_value": fixed,
            **result["summary"],
        })
        fixed_evaluations.append({
            "opponent_value": fixed,
            **result,
        })
    return {
        "summary": random_evaluation["summary"],
        "random": random_evaluation,
        "fixed_contexts": fixed_rows,
        "fixed_context_evaluations": fixed_evaluations,
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
    parser.add_argument("--event-tail-steps", type=int, default=0)
    parser.add_argument("--fixed-event-steps", type=str)
    parser.add_argument(
        "--e1-sampler-mode",
        choices=E1_SAMPLER_MODES,
        default=UNIFORM_E1_SAMPLER,
        help=(
            "episode sampler used only for training; evaluation always uses "
            "the canonical uniform sampler"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--pretrained-lr-scale", type=float, default=0.1)
    parser.add_argument(
        "--actor-loss-mode",
        choices=ACTOR_LOSS_MODES,
        default=STANDARD_ACTOR_LOSS_MODE,
    )
    parser.add_argument("--buyer-init-mean", type=float, default=BUYER_INIT_MEAN)
    parser.add_argument(
        "--buyer-init-concentration",
        type=float,
        default=BUYER_INIT_CONCENTRATION,
    )
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float)
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
    if args.gameplay_horizon < NUM_TRADE_EVENTS:
        parser.error("--gameplay-horizon must be at least five")
    if args.event_tail_steps < 0:
        parser.error("--event-tail-steps must be nonnegative")
    if (
            args.gameplay_horizon - args.event_tail_steps
            < NUM_TRADE_EVENTS
    ):
        parser.error("E1 event window must contain at least five steps")
    if args.e1_sampler_mode == TEMPORAL_MIX_E1_SAMPLER:
        if args.gameplay_horizon != TEMPORAL_MIX_GAMEPLAY_HORIZON:
            parser.error(
                f"--e1-sampler-mode {TEMPORAL_MIX_E1_SAMPLER} requires "
                f"--gameplay-horizon {TEMPORAL_MIX_GAMEPLAY_HORIZON}"
            )
        if args.event_tail_steps != 0:
            parser.error(
                f"--e1-sampler-mode {TEMPORAL_MIX_E1_SAMPLER} requires "
                "--event-tail-steps 0"
            )
        if args.fixed_event_steps is not None:
            parser.error(
                f"--e1-sampler-mode {TEMPORAL_MIX_E1_SAMPLER} cannot be "
                "combined with --fixed-event-steps"
            )
        if args.eval_only:
            parser.error(
                "--eval-only always uses canonical uniform sampling; omit "
                "the temporal sampler flag"
            )
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.n_steps != transitions:
        parser.error(f"--n-steps must equal one full E1 episode ({transitions})")
    if args.batch_size <= 0 or args.batch_size > buffer_size:
        parser.error("--batch-size must lie in [1, n_steps * num_envs]")
    if buffer_size % args.batch_size:
        parser.error("--batch-size must divide n_steps * num_envs exactly")
    if (
            args.actor_loss_mode == PHASE_BALANCED_ACTOR_LOSS_MODE
            and args.batch_size != buffer_size
    ):
        parser.error(
            "phase-balanced Atari PPO requires one full-rollout minibatch "
            "(batch-size = n-steps * num-envs)"
        )
    if (
            not math.isfinite(args.buyer_init_mean)
            or not 0.0 < args.buyer_init_mean < 1.0
    ):
        parser.error("--buyer-init-mean must lie in (0, 1)")
    if (
            not math.isfinite(args.buyer_init_concentration)
            or args.buyer_init_concentration <= 0.0
    ):
        parser.error("--buyer-init-concentration must be positive")
    if args.target_kl is not None and (
            not math.isfinite(args.target_kl) or args.target_kl <= 0.0
    ):
        parser.error("--target-kl must be positive")
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
        / (
            f"meta_{args.role}_e1_ppo{_run_variant_suffix(args)}"
            f"_seed{args.seed}.zip"
        )
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
        source_provenance = dict(model.atari_e1_source_provenance)
        if run is not None:
            run.config.update(
                {
                    "e0b_source_provenance": source_provenance,
                    "e1_sampler_provenance": dict(
                        model.atari_e1_sampler_provenance
                    ),
                    "e1_sampler_history": list(
                        model.atari_e1_sampler_history
                    ),
                    "e1_resume_source_provenance": getattr(
                        model, "atari_e1_resume_source_provenance", None
                    ),
                },
                allow_val_change=True,
            )
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
        evaluation["provenance"] = {
            "e0b_source": source_provenance,
            "training_sampler": dict(model.atari_e1_sampler_provenance),
            "training_sampler_history": list(model.atari_e1_sampler_history),
            "resume_source": getattr(
                model, "atari_e1_resume_source_provenance", None
            ),
            "evaluation_sampler": e1_sampler_provenance(
                UNIFORM_E1_SAMPLER,
                gameplay_horizon=args.gameplay_horizon,
                event_tail_steps=args.event_tail_steps,
            ),
            "actor_loss_mode": model_actor_loss_mode(model),
            "target_kl": getattr(model, "target_kl", None),
            "economic_head_initialization": model_economic_initialization(
                model,
                default_mean=(
                    BUYER_INIT_MEAN if args.role == BUYER else SELLER_INIT_MEAN
                ),
                default_concentration=(
                    BUYER_INIT_CONCENTRATION
                    if args.role == BUYER
                    else SELLER_INIT_CONCENTRATION
                ),
            ),
        }
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
