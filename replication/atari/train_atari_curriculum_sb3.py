"""Train the clean E0a/E0b composite Atari policy with Stable-Baselines3."""

import argparse
import json
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
    json_path,
    make_vec_env,
    step_checkpoint_path,
    target_checkpoint_path,
    target_selection_path,
    validation_log_path,
    write_json,
)
from stackelberg_pomdp.atari.curriculum_env import (
    AtariCurriculumConfig,
    AtariCurriculumEnv,
)
from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_SEED_OFFSET = 1_000_000
CONFIRMATION_SEED_OFFSET = 2_000_000
CONFIRMATION_SEED_STRIDE = 10_000
FINAL_EVALUATION_SEED_OFFSET = 9_000_000


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


def e0a_target_metrics(
        evaluation,
        *,
        required_episodes,
        gameplay_horizon,
        target_reward=5.0,
        target_mean_reward=4.8,
        target_reward_rate=0.90,
        target_shots=5.0,
        target_mean_shots=4.95,
        target_fired_rate=0.95,
        target_final_ammo=0.05,
):
    """Score a deterministic suite against the reliable-five E0a gate."""

    rows = list(evaluation.get("episode_rows", ()))
    rewards = []
    shots = []
    final_ammo = []
    accounting_errors = []
    episode_lengths = []
    payments = []
    return_errors = []
    for row in rows:
        reward = float(
            row["game_reward"]
            if "game_reward" in row
            else row["evaluation_return"]
        )
        fired = float(row.get("shots_fired", row.get("shots_fired_total", 0.0)))
        ammo = float(row.get("final_ammo", row.get("ammo_remaining", 0.0)))
        error = abs(float(row.get("bullet_accounting_error", math.inf)))
        rewards.append(reward)
        shots.append(fired)
        final_ammo.append(ammo)
        accounting_errors.append(error)
        episode_lengths.append(float(row.get("evaluation_steps", math.inf)))
        payments.append(abs(float(row.get("payments", row.get("payment", 0.0)))))
        return_errors.append(abs(float(row["evaluation_return"]) - reward))
    episodes = len(rows)
    reward_rate = float(np.mean(
        np.asarray(rewards) >= float(target_reward) - 1.0e-9
    )) if rewards else 0.0
    fired_rate = float(np.mean(
        np.asarray(shots) >= float(target_shots) - 1.0e-9
    )) if shots else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else math.nan
    mean_shots = float(np.mean(shots)) if shots else math.nan
    mean_ammo = float(np.mean(final_ammo)) if final_ammo else math.nan
    max_accounting_error = (
        float(np.max(accounting_errors)) if accounting_errors else math.inf
    )
    complete_horizons = bool(
        episode_lengths
        and all(
            abs(length - int(gameplay_horizon)) <= 1.0e-9
            for length in episode_lengths
        )
    )
    exact_economics = bool(
        payments
        and max(payments) <= 1.0e-9
        and max(return_errors) <= 1.0e-9
    )
    passed = bool(
        episodes == int(required_episodes)
        and complete_horizons
        and exact_economics
        and max_accounting_error <= 1.0e-9
        and mean_reward >= float(target_mean_reward) - 1.0e-9
        and reward_rate >= float(target_reward_rate) - 1.0e-9
        and mean_shots >= float(target_mean_shots) - 1.0e-9
        and fired_rate >= float(target_fired_rate) - 1.0e-9
        and mean_ammo <= float(target_final_ammo) + 1.0e-9
    )
    return {
        "passed": passed,
        "episodes": episodes,
        "complete_horizons": complete_horizons,
        "exact_economics": exact_economics,
        "mean_game_reward": mean_reward,
        "min_game_reward": float(np.min(rewards)) if rewards else math.nan,
        "reward_target_rate": reward_rate,
        "mean_shots_fired": mean_shots,
        "min_shots_fired": float(np.min(shots)) if shots else math.nan,
        "fired_all_rate": fired_rate,
        "mean_final_ammo": mean_ammo,
        "max_final_ammo": float(np.max(final_ammo)) if final_ammo else math.nan,
        "max_abs_bullet_accounting_error": max_accounting_error,
        "max_abs_payment": float(np.max(payments)) if payments else math.inf,
        "max_abs_return_error": (
            float(np.max(return_errors)) if return_errors else math.inf
        ),
        "target_reward": float(target_reward),
        "target_mean_reward": float(target_mean_reward),
        "target_reward_rate": float(target_reward_rate),
        "target_shots": float(target_shots),
        "target_mean_shots": float(target_mean_shots),
        "target_fired_rate": float(target_fired_rate),
        "target_final_ammo": float(target_final_ammo),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("e0a", "e0b"), default="e0a")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000_000,
        help="maximum total training timesteps, including a resumed prefix",
    )
    parser.add_argument("--gameplay-horizon", type=int, default=200)
    parser.add_argument("--event-tail-steps", type=int, default=0)
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
    parser.add_argument("--checkpoint-every", type=int, default=400_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--target-eval-every",
        type=int,
        default=400_000,
        help="E0a deterministic validation interval; zero disables selection",
    )
    parser.add_argument("--target-eval-episodes", type=int, default=20)
    parser.add_argument("--target-confirm-episodes", type=int, default=100)
    parser.add_argument("--target-consecutive-passes", type=int, default=2)
    parser.add_argument("--target-reward", type=float, default=5.0)
    parser.add_argument("--target-mean-reward", type=float, default=4.8)
    parser.add_argument("--target-reward-rate", type=float, default=0.90)
    parser.add_argument("--target-shots", type=float, default=5.0)
    parser.add_argument("--target-mean-shots", type=float, default=4.95)
    parser.add_argument("--target-fired-rate", type=float, default=0.95)
    parser.add_argument("--target-final-ammo", type=float, default=0.05)
    parser.add_argument(
        "--target-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop E0a after repeated screens and independent confirmation",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-id")
    parser.add_argument(
        "--wandb-resume", choices=("allow", "must", "never"), default="allow"
    )
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
    if args.event_tail_steps < 0:
        parser.error("--event-tail-steps must be nonnegative")
    if (
            args.stage == "e0b"
            and args.gameplay_horizon - args.event_tail_steps
            < NUM_TRADE_EVENTS
    ):
        parser.error("E0b event window must contain at least five steps")
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
    if args.target_eval_every < 0:
        parser.error("--target-eval-every must be nonnegative")
    if args.target_eval_episodes <= 0:
        parser.error("--target-eval-episodes must be positive")
    if args.target_confirm_episodes <= 0:
        parser.error("--target-confirm-episodes must be positive")
    if args.target_consecutive_passes <= 0:
        parser.error("--target-consecutive-passes must be positive")
    if not 0.0 <= args.target_reward_rate <= 1.0:
        parser.error("--target-reward-rate must lie in [0, 1]")
    if not 0.0 <= args.target_fired_rate <= 1.0:
        parser.error("--target-fired-rate must lie in [0, 1]")
    if args.target_reward < 0.0 or args.target_mean_reward < 0.0:
        parser.error("reward targets must be nonnegative")
    if args.target_shots < 0.0 or args.target_mean_shots < 0.0:
        parser.error("shot targets must be nonnegative")
    if not 0.0 <= args.target_final_ammo <= 5.0:
        parser.error("--target-final-ammo must lie in [0, 5]")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.stage == "e0a" and args.target_eval_every > 0:
        if args.target_eval_every % expected_batch:
            parser.error(
                "--target-eval-every must be divisible by one complete "
                f"vector rollout ({expected_batch})"
            )
        if args.timesteps % expected_batch:
            parser.error(
                "--timesteps must be divisible by one complete vector rollout "
                f"({expected_batch})"
            )
    if args.wandb_id and not args.wandb:
        parser.error("--wandb-id requires W&B logging")
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


def make_training_callback(args, *, wandb_run=None):
    return EpisodeCheckpointCallback(
        checkpoint=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        wandb_run=wandb_run,
        resume=bool(args.resume),
    )


def _named_checkpoint(path, suffix):
    path = checkpoint_path(path)
    return path.with_name(f"{path.stem}_{suffix}.zip")


def _selector_state_path(path):
    return json_path(path, "target_state.json")


def _best_selection_path(path):
    return json_path(path, "best_selection.json")


def _metric_key(metrics):
    return (
        float(metrics["reward_target_rate"]),
        float(metrics["mean_game_reward"]),
        float(metrics["fired_all_rate"]),
        float(metrics["mean_shots_fired"]),
        -float(metrics["mean_final_ammo"]),
    )


class E0ATargetSelector:
    """Auditable two-screen plus independent-confirmation selector."""

    def __init__(self, args, *, wandb_run=None):
        self.args = args
        self.wandb_run = wandb_run
        self.validation_log = validation_log_path(args.checkpoint)
        self.state_path = _selector_state_path(args.checkpoint)
        self.best_checkpoint = _named_checkpoint(args.checkpoint, "best")
        self.selected_checkpoint = target_checkpoint_path(args.checkpoint)
        self.state = {
            "consecutive_screen_passes": 0,
            "confirmation_attempts": 0,
            "best": None,
            "selected": None,
        }
        if args.resume and self.state_path.is_file():
            with self.state_path.open("r", encoding="utf-8") as handle:
                self.state.update(json.load(handle))
        else:
            self.validation_log.parent.mkdir(parents=True, exist_ok=True)
            self.validation_log.open("w", encoding="utf-8").close()
        self._save_state()

    def _save_state(self):
        write_json(self.state_path, self.state)

    def _evaluate(self, checkpoint, *, episodes, seed_start):
        candidate = ScaledLearningRatePPO.load(
            checkpoint, device=self.args.device
        )
        try:
            evaluation = evaluate_model(
                candidate,
                lambda episode: make_env(
                    self.args, seed=int(seed_start) + episode
                ),
                episodes=episodes,
            )
        except Exception:
            del candidate
            raise
        return candidate, evaluation

    def _metrics(self, evaluation, *, required_episodes):
        return e0a_target_metrics(
            evaluation,
            required_episodes=required_episodes,
            gameplay_horizon=self.args.gameplay_horizon,
            target_reward=self.args.target_reward,
            target_mean_reward=self.args.target_mean_reward,
            target_reward_rate=self.args.target_reward_rate,
            target_shots=self.args.target_shots,
            target_mean_shots=self.args.target_mean_shots,
            target_fired_rate=self.args.target_fired_rate,
            target_final_ammo=self.args.target_final_ammo,
        )

    def _record(
            self,
            *,
            kind,
            step,
            checkpoint,
            seed_start,
            evaluation,
            metrics,
            output,
    ):
        record = {
            "kind": kind,
            "total_timesteps": int(step),
            "checkpoint_path": str(checkpoint),
            "seed_start": int(seed_start),
            "seed_end": int(seed_start) + int(metrics["episodes"]) - 1,
            **metrics,
        }
        write_json(output, {"selection": record, **evaluation})
        with self.validation_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.wandb_run is not None:
            payload = {
                f"{kind}/{key}": value
                for key, value in record.items()
                if isinstance(value, (bool, int, float, np.number))
            }
            # Validation and confirmation can occur at the same policy clock as
            # the final training episode.  Their namespace-specific timestep is
            # the x-axis; using a fresh W&B history row avoids dropped metrics.
            self.wandb_run.log(payload)
        print({f"target_{kind}": record}, flush=True)
        return record

    def evaluate_checkpoint(self, checkpoint, *, step):
        candidate, evaluation = self._evaluate(
            checkpoint,
            episodes=self.args.target_eval_episodes,
            seed_start=self.args.seed + VALIDATION_SEED_OFFSET,
        )
        metrics = self._metrics(
            evaluation, required_episodes=self.args.target_eval_episodes
        )
        record = self._record(
            kind="validation",
            step=step,
            checkpoint=checkpoint,
            seed_start=self.args.seed + VALIDATION_SEED_OFFSET,
            evaluation=evaluation,
            metrics=metrics,
            output=json_path(checkpoint, "validation.json"),
        )

        previous_best = self.state.get("best")
        if previous_best is None or _metric_key(metrics) > tuple(
                previous_best["metric_key"]
        ):
            candidate.save(self.best_checkpoint)
            self.state["best"] = {
                **record,
                "metric_key": list(_metric_key(metrics)),
                "best_checkpoint_path": str(self.best_checkpoint),
            }
            write_json(_best_selection_path(self.args.checkpoint), self.state["best"])
            if self.wandb_run is not None:
                self.wandb_run.summary.update({
                    "best/total_timesteps": int(step),
                    "best/mean_game_reward": metrics["mean_game_reward"],
                    "best/reward_target_rate": metrics["reward_target_rate"],
                    "best/checkpoint_path": str(self.best_checkpoint),
                })

        if metrics["passed"]:
            self.state["consecutive_screen_passes"] += 1
        else:
            self.state["consecutive_screen_passes"] = 0

        confirmed = False
        if (
                self.state.get("selected") is None
                and
                self.state["consecutive_screen_passes"]
                >= self.args.target_consecutive_passes
        ):
            attempt = int(self.state["confirmation_attempts"])
            confirmation_seed = (
                self.args.seed
                + CONFIRMATION_SEED_OFFSET
                + attempt * CONFIRMATION_SEED_STRIDE
            )
            self.state["confirmation_attempts"] = attempt + 1
            confirmation = evaluate_model(
                candidate,
                lambda episode: make_env(
                    self.args, seed=confirmation_seed + episode
                ),
                episodes=self.args.target_confirm_episodes,
            )
            confirmation_metrics = self._metrics(
                confirmation,
                required_episodes=self.args.target_confirm_episodes,
            )
            confirmation_record = self._record(
                kind="confirmation",
                step=step,
                checkpoint=checkpoint,
                seed_start=confirmation_seed,
                evaluation=confirmation,
                metrics=confirmation_metrics,
                output=json_path(
                    checkpoint, f"confirmation_{attempt + 1}.json"
                ),
            )
            if confirmation_metrics["passed"]:
                candidate.save(self.selected_checkpoint)
                self.state["selected"] = {
                    **confirmation_record,
                    "selected_checkpoint_path": str(self.selected_checkpoint),
                    "screening_checkpoint_path": str(checkpoint),
                }
                write_json(
                    target_selection_path(self.args.checkpoint),
                    self.state["selected"],
                )
                if self.wandb_run is not None:
                    self.wandb_run.summary.update({
                        "target/passed": True,
                        "target/total_timesteps": int(step),
                        "target/checkpoint_path": str(self.selected_checkpoint),
                    })
                confirmed = True
            else:
                self.state["consecutive_screen_passes"] = 0

        self._save_state()
        del candidate
        return bool(confirmed and self.args.target_stop)

    def finish_at_cap(self, step):
        self.state["stop_reason"] = (
            "target_confirmed"
            if self.state.get("selected") is not None
            else "maximum_total_timesteps"
        )
        if self.wandb_run is not None and self.state.get("selected") is None:
            self.wandb_run.summary.update({
                "target/passed": False,
                "target/stop_reason": "maximum_total_timesteps",
                "target/total_timesteps": int(step),
            })
        self._save_state()

    def final_checkpoint(self):
        if self.state.get("selected") is not None:
            return self.selected_checkpoint
        if self.state.get("best") is not None:
            return self.best_checkpoint
        return checkpoint_path(self.args.checkpoint)


def train_e0a_to_target(model, args, *, wandb_run=None):
    """Train in rollout-aligned chunks and evaluate only after PPO updates."""

    callback = make_training_callback(args, wandb_run=wandb_run)
    selector = E0ATargetSelector(args, wandb_run=wandb_run)
    if selector.state.get("selected") is not None and args.target_stop:
        return selector.final_checkpoint(), True
    rollout_size = int(args.n_steps) * int(args.num_envs)
    first_chunk = True
    stopped_on_target = False
    while model.num_timesteps < args.timesteps:
        next_evaluation = (
            (int(model.num_timesteps) // args.target_eval_every) + 1
        ) * args.target_eval_every
        chunk_end = min(int(args.timesteps), next_evaluation)
        chunk_size = chunk_end - int(model.num_timesteps)
        if chunk_size <= 0 or chunk_size % rollout_size:
            raise RuntimeError(
                "E0a training chunk does not align with a complete vector rollout"
            )
        updates_before = int(model._n_updates)
        model.learn(
            total_timesteps=chunk_size,
            callback=callback,
            reset_num_timesteps=(first_chunk and not bool(args.resume)),
        )
        first_chunk = False
        if int(model.num_timesteps) != chunk_end:
            raise RuntimeError(
                f"E0a chunk ended at {model.num_timesteps}, expected {chunk_end}"
            )
        if int(model._n_updates) <= updates_before:
            raise RuntimeError("E0a chunk returned before a PPO update")

        evaluated_checkpoint = step_checkpoint_path(
            args.checkpoint, model.num_timesteps
        )
        model.save(evaluated_checkpoint)
        model.save(args.checkpoint)
        if selector.evaluate_checkpoint(
                evaluated_checkpoint, step=model.num_timesteps
        ):
            stopped_on_target = True
            selector.state["stop_reason"] = "target_confirmed"
            selector._save_state()
            break

    if not stopped_on_target:
        selector.finish_at_cap(model.num_timesteps)
    return selector.final_checkpoint(), stopped_on_target


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
        selected_checkpoint = checkpoint_path(
            args.resume if args.eval_only else args.checkpoint
        )
        stopped_on_target = False
        if not args.eval_only:
            remaining_timesteps = int(args.timesteps) - int(model.num_timesteps)
            if remaining_timesteps <= 0:
                raise ValueError(
                    "--timesteps is a total cap and must exceed the resumed "
                    f"checkpoint clock ({model.num_timesteps})"
                )
            if args.stage == "e0a" and args.target_eval_every > 0:
                selected_checkpoint, stopped_on_target = train_e0a_to_target(
                    model, args, wandb_run=run
                )
            else:
                model.learn(
                    total_timesteps=remaining_timesteps,
                    callback=make_training_callback(args, wandb_run=run),
                    reset_num_timesteps=not bool(args.resume),
                )
                model.save(args.checkpoint)
                selected_checkpoint = checkpoint_path(args.checkpoint)

        evaluation_model = model
        selected_checkpoint = checkpoint_path(selected_checkpoint)
        if selected_checkpoint != checkpoint_path(
                args.resume if args.eval_only else args.checkpoint
        ):
            evaluation_model = ScaledLearningRatePPO.load(
                selected_checkpoint, device=args.device
            )
        evaluation = evaluate_model(
            evaluation_model,
            lambda episode: make_env(
                args, seed=args.seed + FINAL_EVALUATION_SEED_OFFSET + episode
            ),
            episodes=args.eval_episodes,
        )
        evaluation["selection"] = {
            "stage": args.stage,
            "training_total_timesteps": int(model.num_timesteps),
            "selected_checkpoint_timesteps": int(
                evaluation_model.num_timesteps
            ),
            "selected_checkpoint_path": str(selected_checkpoint),
            "stopped_on_target": bool(stopped_on_target),
            "final_evaluation_seed_start": int(
                args.seed + FINAL_EVALUATION_SEED_OFFSET
            ),
            "final_evaluation_seed_end": int(
                args.seed + FINAL_EVALUATION_SEED_OFFSET + args.eval_episodes - 1
            ),
        }
        if run is not None:
            run.summary.update({
                "selection/checkpoint_path": str(selected_checkpoint),
                "selection/checkpoint_timesteps": int(
                    evaluation_model.num_timesteps
                ),
                "selection/stopped_on_target": bool(stopped_on_target),
                "selection/training_total_timesteps": int(model.num_timesteps),
            })
        finish_run(
            run,
            checkpoint=selected_checkpoint,
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
