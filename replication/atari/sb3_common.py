"""Shared SB3, evaluation, checkpoint, and W&B utilities for clean Atari."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import time

from gym import spaces
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stackelberg_pomdp.atari.protocol import ACTION_CREDIT, NUM_TRADE_EVENTS


WANDB_PROJECT = "StackPOMDP"
WANDB_GROUP = "atari_clean_curriculum"
STANDARD_ACTOR_LOSS_MODE = "standard"
PHASE_BALANCED_ACTOR_LOSS_MODE = "balanced"
ACTOR_LOSS_MODES = (
    STANDARD_ACTOR_LOSS_MODE,
    PHASE_BALANCED_ACTOR_LOSS_MODE,
)
ACTOR_LOSS_MODE_ATTRIBUTE = "atari_actor_loss_mode"
ECONOMIC_INIT_ATTRIBUTE = "atari_economic_head_initialization"
OPTIMIZER_METRICS_ATTRIBUTE = "atari_last_optimizer_metrics"


class ScaledLearningRatePPO(PPO):
    """PPO that preserves optional per-parameter-group learning-rate scales."""

    def _update_learning_rate(self, optimizers):
        base_rate = float(self.lr_schedule(self._current_progress_remaining))
        self.logger.record("train/learning_rate", base_rate)
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = base_rate * float(group.get("lr_scale", 1.0))


def model_actor_loss_mode(model):
    """Return the stored objective, treating old checkpoints as standard PPO."""

    return str(getattr(
        model, ACTOR_LOSS_MODE_ATTRIBUTE, STANDARD_ACTOR_LOSS_MODE
    ))


def ppo_class_for_actor_loss_mode(mode):
    mode = str(mode)
    if mode == STANDARD_ACTOR_LOSS_MODE:
        return ScaledLearningRatePPO
    if mode == PHASE_BALANCED_ACTOR_LOSS_MODE:
        return PhaseBalancedPPO
    raise ValueError(f"unknown Atari actor loss mode: {mode!r}")


def model_economic_initialization(model, *, default_mean, default_concentration):
    """Return a checkpoint's initialization contract with legacy defaults."""

    raw = getattr(model, ECONOMIC_INIT_ATTRIBUTE, None)
    if raw is None:
        return {
            "mean": float(default_mean),
            "concentration": float(default_concentration),
        }
    if not isinstance(raw, dict):
        raise ValueError("Atari economic initialization metadata must be a mapping")
    try:
        mean = float(raw["mean"])
        concentration = float(raw["concentration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Atari economic initialization metadata is incomplete"
        ) from error
    if (
            not np.isfinite(mean)
            or not np.isfinite(concentration)
            or not 0.0 < mean < 1.0
            or concentration <= 0.0
    ):
        raise ValueError("Atari economic initialization metadata is invalid")
    return {"mean": mean, "concentration": concentration}


def attach_atari_training_contract(
        model,
        *,
        actor_loss_mode,
        economic_init_mean,
        economic_init_concentration,
):
    """Persist the actor objective and economic-head initialization in SB3."""

    mode = str(actor_loss_mode)
    if mode not in ACTOR_LOSS_MODES:
        raise ValueError(f"unknown Atari actor loss mode: {mode!r}")
    mean = float(economic_init_mean)
    concentration = float(economic_init_concentration)
    if not np.isfinite(mean) or not 0.0 < mean < 1.0:
        raise ValueError("economic initialization mean must lie in (0, 1)")
    if not np.isfinite(concentration) or concentration <= 0.0:
        raise ValueError("economic initialization concentration must be positive")
    setattr(model, ACTOR_LOSS_MODE_ATTRIBUTE, mode)
    setattr(model, ECONOMIC_INIT_ATTRIBUTE, {
        "mean": mean,
        "concentration": concentration,
    })


def _masked_mean(values, mask):
    weights = mask.to(dtype=values.dtype)
    return th.sum(values * weights) / th.clamp(th.sum(weights), min=1.0)


def phase_balanced_actor_terms(
        *,
        advantages,
        log_prob,
        old_log_prob,
        entropy,
        action_credit,
        clip_range,
):
    """Return PPO actor terms averaged within each credited Atari phase."""

    advantages = advantages.reshape(-1)
    log_prob = log_prob.reshape(-1)
    old_log_prob = old_log_prob.reshape(-1)
    if action_credit.ndim < 2 or action_credit.shape[-1] != 2:
        raise ValueError(
            "Atari action credit must have a final [game, economic] axis"
        )
    credit = action_credit.float().reshape(-1, 2)
    rows = advantages.shape[0]
    if (
            log_prob.shape[0] != rows
            or old_log_prob.shape[0] != rows
            or credit.shape[0] != rows
    ):
        raise ValueError("phase-balanced PPO tensors have inconsistent row counts")
    if not bool(th.all((credit == 0.0) | (credit == 1.0)).item()):
        raise ValueError("Atari action credit must contain exact binary gates")
    if bool(th.any(th.sum(credit, dim=1) > 1.0).item()):
        raise ValueError("an Atari transition cannot credit both actor heads")

    game_mask = credit[:, 0] > 0.5
    economic_mask = credit[:, 1] > 0.5
    if not bool(th.any(game_mask).item()):
        raise ValueError(
            "phase-balanced Atari PPO requires at least one gameplay row"
        )
    if not bool(th.any(economic_mask).item()):
        raise ValueError(
            "phase-balanced Atari PPO requires at least one economic row"
        )
    inactive_mask = ~(game_mask | economic_mask)
    ratio = th.exp(log_prob - old_log_prob)
    unclipped = advantages * ratio
    clipped = advantages * th.clamp(
        ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)
    )
    surrogate = th.minimum(unclipped, clipped)
    game_policy_loss = -_masked_mean(surrogate, game_mask)
    economic_policy_loss = -_masked_mean(surrogate, economic_mask)

    entropy_rows = log_prob if entropy is None else -entropy.reshape(-1)
    if entropy_rows.shape[0] != rows:
        raise ValueError("phase-balanced entropy has an inconsistent row count")
    game_entropy_loss = _masked_mean(entropy_rows, game_mask)
    economic_entropy_loss = _masked_mean(entropy_rows, economic_mask)

    clipped_rows = (th.abs(ratio - 1.0) > float(clip_range)).float()
    game_clip_fraction = _masked_mean(clipped_rows, game_mask)
    economic_clip_fraction = _masked_mean(clipped_rows, economic_mask)
    log_ratio = log_prob - old_log_prob
    kl_rows = (th.exp(log_ratio) - 1.0) - log_ratio
    game_approx_kl = _masked_mean(kl_rows, game_mask)
    economic_approx_kl = _masked_mean(kl_rows, economic_mask)

    return {
        "policy_loss": game_policy_loss + economic_policy_loss,
        "game_policy_loss": game_policy_loss,
        "economic_policy_loss": economic_policy_loss,
        "entropy_loss": game_entropy_loss + economic_entropy_loss,
        "game_entropy_loss": game_entropy_loss,
        "economic_entropy_loss": economic_entropy_loss,
        "game_clip_fraction": game_clip_fraction,
        "economic_clip_fraction": economic_clip_fraction,
        "game_approx_kl": game_approx_kl,
        "economic_approx_kl": economic_approx_kl,
        "max_approx_kl": th.maximum(game_approx_kl, economic_approx_kl),
        "game_active_rows": th.sum(game_mask),
        "economic_active_rows": th.sum(economic_mask),
        "inactive_actor_rows": th.sum(inactive_mask),
    }


class PhaseBalancedPPO(ScaledLearningRatePPO):
    """PPO with independent active-row means for the two Atari actor heads."""

    def train(self):
        full_rollout_rows = (
            int(self.rollout_buffer.buffer_size)
            * int(self.rollout_buffer.n_envs)
        )
        if int(self.batch_size) != full_rollout_rows:
            raise ValueError(
                "phase-balanced Atari PPO requires one full-rollout "
                "minibatch"
            )
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )

        metrics = defaultdict(list)
        continue_training = True
        last_loss = None
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (
                        (advantages - advantages.mean())
                        / (advantages.std() + 1.0e-8)
                    )
                actor = phase_balanced_actor_terms(
                    advantages=advantages,
                    log_prob=log_prob,
                    old_log_prob=rollout_data.old_log_prob,
                    entropy=entropy,
                    action_credit=rollout_data.observations[ACTION_CREDIT],
                    clip_range=clip_range,
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                loss = (
                    actor["policy_loss"]
                    + self.ent_coef * actor["entropy_loss"]
                    + self.vf_coef * value_loss
                )
                last_loss = loss

                for name in (
                        "policy_loss",
                        "game_policy_loss",
                        "economic_policy_loss",
                        "entropy_loss",
                        "game_entropy_loss",
                        "economic_entropy_loss",
                        "game_clip_fraction",
                        "economic_clip_fraction",
                        "game_approx_kl",
                        "economic_approx_kl",
                        "max_approx_kl",
                        "game_active_rows",
                        "economic_active_rows",
                        "inactive_actor_rows",
                ):
                    metrics[name].append(float(
                        actor[name].detach().cpu().item()
                    ))
                metrics["value_loss"].append(float(value_loss.item()))

                approx_kl = float(actor["max_approx_kl"].detach().cpu().item())
                if (
                        self.target_kl is not None
                        and approx_kl > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to a head "
                            f"KL of {approx_kl:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record(
            "train/entropy_loss", np.mean(metrics["entropy_loss"])
        )
        self.logger.record(
            "train/policy_gradient_loss", np.mean(metrics["policy_loss"])
        )
        self.logger.record(
            "train/value_loss", np.mean(metrics["value_loss"])
        )
        self.logger.record(
            "train/approx_kl", np.mean(metrics["max_approx_kl"])
        )
        self.logger.record(
            "train/max_head_approx_kl", np.mean(metrics["max_approx_kl"])
        )
        self.logger.record(
            "train/clip_fraction",
            np.mean([
                np.mean(metrics["game_clip_fraction"]),
                np.mean(metrics["economic_clip_fraction"]),
            ]),
        )
        if last_loss is not None:
            self.logger.record("train/loss", float(last_loss.item()))
        self.logger.record("train/explained_variance", explained_var)
        for name in (
                "game_policy_loss",
                "economic_policy_loss",
                "game_entropy_loss",
                "economic_entropy_loss",
                "game_clip_fraction",
                "economic_clip_fraction",
                "game_approx_kl",
                "economic_approx_kl",
                "game_active_rows",
                "economic_active_rows",
                "inactive_actor_rows",
        ):
            self.logger.record(f"train/{name}", np.mean(metrics[name]))
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item()
            )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        optimizer_metrics = {
            "train/optimizer_total_timesteps": int(self.num_timesteps),
            "train/actor_loss_mode": PHASE_BALANCED_ACTOR_LOSS_MODE,
            "train/policy_gradient_loss": float(np.mean(
                metrics["policy_loss"]
            )),
            "train/value_loss": float(np.mean(metrics["value_loss"])),
            "train/entropy_loss": float(np.mean(metrics["entropy_loss"])),
            "train/approx_kl": float(np.mean(metrics["max_approx_kl"])),
            "train/max_head_approx_kl": float(np.mean(
                metrics["max_approx_kl"]
            )),
            "train/clip_fraction": float(np.mean([
                np.mean(metrics["game_clip_fraction"]),
                np.mean(metrics["economic_clip_fraction"]),
            ])),
        }
        for name in (
                "game_policy_loss",
                "economic_policy_loss",
                "game_entropy_loss",
                "economic_entropy_loss",
                "game_clip_fraction",
                "economic_clip_fraction",
                "game_approx_kl",
                "economic_approx_kl",
                "game_active_rows",
                "economic_active_rows",
                "inactive_actor_rows",
        ):
            optimizer_metrics[f"train/{name}"] = float(np.mean(metrics[name]))
        setattr(self, OPTIMIZER_METRICS_ATTRIBUTE, optimizer_metrics)


def make_vec_env(factory, *, num_envs, start_method="spawn"):
    constructors = [
        (lambda rank=rank: factory(rank))
        for rank in range(int(num_envs))
    ]
    if len(constructors) == 1:
        return DummyVecEnv(constructors)
    return SubprocVecEnv(constructors, start_method=start_method)


def checkpoint_path(path):
    result = Path(path).expanduser().resolve()
    result.parent.mkdir(parents=True, exist_ok=True)
    return result if result.suffix == ".zip" else result.with_suffix(".zip")


def step_checkpoint_path(path, step):
    path = checkpoint_path(path)
    return path.with_name(f"{path.stem}_step{int(step)}.zip")


def json_path(path, suffix="evaluation.json"):
    path = checkpoint_path(path)
    return path.with_name(f"{path.stem}.{suffix}")


def training_log_path(path):
    path = checkpoint_path(path)
    return path.with_name(f"{path.stem}.training.jsonl")


def validation_log_path(path):
    path = checkpoint_path(path)
    return path.with_name(f"{path.stem}.validation.jsonl")


def target_checkpoint_path(path):
    path = checkpoint_path(path)
    return path.with_name(f"{path.stem}_target.zip")


def target_selection_path(path):
    path = checkpoint_path(path)
    return path.with_name(f"{path.stem}.target_selection.json")


def fixed_context_csv_path(path):
    path = checkpoint_path(path)
    return path.with_name(f"{path.stem}.fixed_contexts.csv")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def summarize_episodes(rows):
    numeric = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (bool, int, float, np.number)):
                numeric[key].append(float(value))
    summary = {
        f"mean_{key}": float(np.mean(values))
        for key, values in numeric.items()
        if values
    }
    return {"episodes": len(rows), **summary}


def _episode_trade_time_metrics(episode):
    """Summarize one episode's realized trades by normalized gameplay time."""

    events = list(episode.get("events", ()))
    if not events:
        return {}
    gameplay_horizon = float(episode.get("gameplay_transitions", 0.0))
    if gameplay_horizon <= 0.0:
        raise ValueError("trade timing metrics require a positive gameplay horizon")
    bins = {
        "early": (0.0, 1.0 / 3.0),
        "middle": (1.0 / 3.0, 2.0 / 3.0),
        "late": (2.0 / 3.0, 1.0 + 1.0e-12),
    }
    result = {}
    for name, (low, high) in bins.items():
        selected = [
            event
            for event in events
            if low <= float(event["game_step"]) / gameplay_horizon < high
        ]
        result[f"{name}_trade_events"] = len(selected)
        if selected:
            result[f"{name}_acceptance_rate"] = float(np.mean([
                float(event["accepted"]) for event in selected
            ]))
            result[f"{name}_mean_price"] = float(np.mean([
                float(event["price"]) for event in selected
            ]))
            result[f"{name}_mean_threshold"] = float(np.mean([
                float(event["threshold"]) for event in selected
            ]))
    return result


def evaluate_model(model, env_factory, *, episodes, use_action_cache=False):
    """Run deterministic complete episodes and return terminal info rows."""

    rows = []
    if use_action_cache:
        model.policy.fix_policy_actions()
    for episode in range(int(episodes)):
        if hasattr(model.policy, "clear_obs_action_map"):
            model.policy.clear_obs_action_map()
        env = env_factory(episode)
        try:
            observation = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            terminal_info = {}
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, done, terminal_info = env.step(action)
                total_reward += float(reward)
                steps += 1
            row = {
                "evaluation_episode": episode,
                "evaluation_return": total_reward,
                "evaluation_steps": steps,
                **_jsonable(terminal_info),
            }
            row.pop("episode", None)
            rows.append(row)
        finally:
            env.close()
    return {"summary": summarize_episodes(rows), "episode_rows": rows}


def init_wandb(args, *, stage, checkpoint):
    if not getattr(args, "wandb", False):
        return None
    import wandb

    run_id = getattr(args, "wandb_id", None)
    resume = getattr(args, "wandb_resume", None) if run_id else None
    run = wandb.init(
        project=getattr(args, "wandb_project", WANDB_PROJECT),
        group=getattr(args, "wandb_group", WANDB_GROUP),
        job_type=f"atari_{stage}",
        name=(
            getattr(args, "wandb_name", None)
            or f"atari_{stage}_{Path(checkpoint).stem}"
        ),
        id=run_id,
        resume=resume,
        allow_val_change=bool(run_id),
        config={
            **vars(args),
            "algorithm": "PPO",
            "stage": stage,
            "checkpoint_path": str(checkpoint),
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "architecture": "clean_composite_atari_v2",
        },
    )
    for namespace in ("train", "validation", "confirmation", "eval"):
        run.define_metric(f"{namespace}/total_timesteps")
        run.define_metric(
            f"{namespace}/*",
            step_metric=f"{namespace}/total_timesteps",
        )
    return run


class EpisodeCheckpointCallback(BaseCallback):
    """Log completed episodes and checkpoint only at outer boundaries."""

    def __init__(
            self,
            *,
            checkpoint,
            checkpoint_every,
            seed,
            wandb_run=None,
            resume=False,
    ):
        super().__init__()
        self.checkpoint = checkpoint_path(checkpoint)
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.seed = int(seed)
        self.wandb_run = wandb_run
        self.resume = bool(resume)
        self.training_log = training_log_path(self.checkpoint)
        self.next_checkpoint = self.checkpoint_every
        self.episode_count = 0
        self.started = None
        self.started_timesteps = 0
        self.last_optimizer_metrics_step = 0
        self._initialized = False

    def _init_callback(self):
        if not self._initialized:
            self.started = time.time()
            self.started_timesteps = int(getattr(
                self.model, "num_timesteps", self.num_timesteps
            ))
            self.last_optimizer_metrics_step = self.started_timesteps
            self.training_log.parent.mkdir(parents=True, exist_ok=True)
            if not self.resume:
                self.training_log.open("w", encoding="utf-8").close()
            elif self.training_log.is_file():
                # A process can advance beyond its last durable checkpoint before
                # it is interrupted.  Keep only rows represented by the resumed
                # checkpoint so the local trace remains monotone and auditable.
                current_step = int(getattr(
                    self.model, "num_timesteps", self.num_timesteps
                ))
                retained = []
                retained_episode_count = 0
                with self.training_log.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if int(row.get("train/total_timesteps", 0)) <= current_step:
                            retained.append(line)
                            if row.get("record_kind", "episode") == "episode":
                                retained_episode_count += 1
                temporary = self.training_log.with_suffix(
                    self.training_log.suffix + ".tmp"
                )
                with temporary.open("w", encoding="utf-8") as handle:
                    handle.writelines(retained)
                temporary.replace(self.training_log)
                self.episode_count = retained_episode_count
            self._initialized = True
        current_step = int(getattr(
            self.model, "num_timesteps", self.num_timesteps
        ))
        self.next_checkpoint = (
            (current_step // self.checkpoint_every) + 1
        ) * self.checkpoint_every

    def _log_optimizer_metrics(self):
        raw = getattr(self.model, OPTIMIZER_METRICS_ATTRIBUTE, None)
        if raw is None:
            return
        if not isinstance(raw, dict):
            raise ValueError("stored Atari optimizer metrics must be a mapping")
        try:
            optimizer_step = int(raw["train/optimizer_total_timesteps"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "stored Atari optimizer metrics have no valid timestep"
            ) from error
        if optimizer_step <= self.last_optimizer_metrics_step:
            return
        payload = {
            **raw,
            "train/total_timesteps": optimizer_step,
        }
        if self.wandb_run is not None:
            # Episode logging may already have committed W&B's internal row at
            # this environment step.  Let W&B advance its internal clock while
            # the explicit metric remains the exact scientific x-axis.
            self.wandb_run.log(payload)
        local_payload = {
            **payload,
            "record_kind": "optimizer",
            "seed": self.seed,
            "algorithm": "PPO",
            "checkpoint_path": str(self.checkpoint),
        }
        with self.training_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(local_payload, sort_keys=True) + "\n")
        self.last_optimizer_metrics_step = optimizer_step

    def _on_rollout_start(self):
        self._log_optimizer_metrics()

    def _on_training_end(self):
        self._log_optimizer_metrics()

    def _on_step(self):
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        infos = self.locals.get("infos", [])
        wandb_payloads = []
        for row, done in enumerate(dones):
            if not done:
                continue
            self.episode_count += 1
            info = dict(infos[row])
            episode = dict(info.get("episode", {}))
            role = episode.get("controlled_role", episode.get("leader_role"))

            def role_metric(name, default=0.0):
                if name in episode:
                    return episode[name]
                if role in ("buyer", "seller"):
                    role_name = f"{role}_{name}"
                    if role_name in episode:
                        return episode[role_name]
                # Backward-compatible fallback for old buyer-only summaries.
                return episode.get(f"buyer_{name}", default)

            game_reward = float(role_metric("game_reward"))
            shots_fired = float(role_metric("shots_fired"))
            final_ammo = float(role_metric("final_ammo"))
            true_game_over_resets = float(role_metric(
                "true_game_over_resets"
            ))
            true_game_over_reset_rate = float(role_metric(
                "true_game_over_reset_rate"
            ))
            time_limit_resets = float(role_metric("time_limit_resets"))
            real_terminal_resets = float(role_metric("real_terminal_resets"))
            life_resets = float(role_metric("life_resets"))
            game_over_before_fifth = float(role_metric(
                "true_game_over_before_fifth_event"
            ))
            payload = {
                "train/episode": self.episode_count,
                "train/episode_reward": float(
                    episode.get("r", info.get("episode_reward", 0.0))
                ),
                "train/episode_length": int(
                    episode.get("l", info.get("outer_transition_count", 0))
                ),
                "train/game_reward": game_reward,
                "train/payments": float(episode.get("payments", 0.0)),
                "train/purchases": float(episode.get("purchases", 0.0)),
                "train/shots_fired": shots_fired,
                "train/final_ammo": final_ammo,
                "train/reward_per_bullet": (
                    game_reward / shots_fired if shots_fired > 0.0 else 0.0
                ),
                "train/life_resets": life_resets,
                "train/real_terminal_resets": real_terminal_resets,
                "train/true_game_over_resets": true_game_over_resets,
                "train/true_game_over_reset_rate": true_game_over_reset_rate,
                "train/time_limit_resets": time_limit_resets,
                "train/true_game_over_before_fifth_event": (
                    game_over_before_fifth
                ),
                "train/total_timesteps": int(self.num_timesteps),
                "train/learning_rate": float(
                    self.model.lr_schedule(self.model._current_progress_remaining)
                ),
                "train/fps_wall": float(
                    (self.num_timesteps - self.started_timesteps)
                    / max(time.time() - self.started, 1.0e-9)
                ),
            }
            for key in (
                "seller_reward",
                "buyer_reward",
                "seller_game_reward",
                "buyer_game_reward",
                "seller_shots_fired",
                "buyer_shots_fired",
                "seller_final_ammo",
                "buyer_final_ammo",
                "seller_life_resets",
                "buyer_life_resets",
                "seller_real_terminal_resets",
                "buyer_real_terminal_resets",
                "seller_true_game_over_resets",
                "buyer_true_game_over_resets",
                "seller_true_game_over_reset_rate",
                "buyer_true_game_over_reset_rate",
                "seller_time_limit_resets",
                "buyer_time_limit_resets",
                "seller_true_game_over_before_fifth_event",
                "buyer_true_game_over_before_fifth_event",
                "any_true_game_over_before_fifth_event",
                "cache_hits",
            ):
                if key in episode:
                    payload[f"train/{key}"] = float(episode[key])
            for event in episode.get("events", ()):
                event_index = int(event["event_index"]) + 1
                for key in (
                    "game_step",
                    "price",
                    "threshold",
                    "accepted",
                    "seller_ammo_before",
                    "buyer_ammo_before",
                    "seller_ammo_after",
                    "buyer_ammo_after",
                    "seller_true_game_over_resets",
                    "buyer_true_game_over_resets",
                    "seller_time_limit_resets",
                    "buyer_time_limit_resets",
                ):
                    if key in event:
                        payload[
                            f"train/event_{event_index}/{key}"
                        ] = float(event[key])
            payload.update({
                f"train/{key}": value
                for key, value in _episode_trade_time_metrics(episode).items()
            })
            payload.update({
                f"train/{key}": float(value)
                for key, value in episode.items()
                if key.startswith("e1_")
                and isinstance(value, (bool, int, float, np.number))
            })
            wandb_payloads.append(payload)
            local_payload = {
                **payload,
                "record_kind": "episode",
                "seed": self.seed,
                "algorithm": "PPO",
                "checkpoint_path": str(self.checkpoint),
            }
            for key in (
                    "e1_sampler_mode",
                    "e1_schedule_stratum",
                    "e1_context_stratum",
            ):
                if key in episode:
                    local_payload[f"train/{key}"] = str(episode[key])
            with self.training_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(local_payload, sort_keys=True) + "\n")

        if self.wandb_run is not None and wandb_payloads:
            keys = {
                key
                for payload in wandb_payloads
                for key, value in payload.items()
                if isinstance(value, (bool, int, float, np.number))
            }
            aggregate = {
                key: float(np.mean([
                    payload[key]
                    for payload in wandb_payloads
                    if key in payload
                ]))
                for key in keys
            }
            aggregate["train/episode"] = int(self.episode_count)
            aggregate["train/vector_episodes"] = len(wandb_payloads)
            one_hot_prefixes = (
                "train/e1_schedule_stratum_one_hot_",
                "train/e1_context_stratum_one_hot_",
            )
            for key in sorted(keys):
                if not key.startswith(one_hot_prefixes):
                    continue
                count_key = key.replace("_one_hot_", "_vector_count_", 1)
                aggregate[count_key] = float(sum(
                    payload.get(key, 0.0) for payload in wandb_payloads
                ))
            self.wandb_run.log(aggregate, step=int(self.num_timesteps))

        if np.any(dones) and self.num_timesteps >= self.next_checkpoint:
            path = step_checkpoint_path(self.checkpoint, self.num_timesteps)
            self.model.save(path)
            self.model.save(self.checkpoint)
            while self.next_checkpoint <= self.num_timesteps:
                self.next_checkpoint += self.checkpoint_every
        return True


def finish_run(run, *, checkpoint, evaluation, total_timesteps):
    write_json(json_path(checkpoint), evaluation)
    if run is not None:
        summary = {
            f"eval/{key}": value
            for key, value in evaluation["summary"].items()
        }
        summary["eval/total_timesteps"] = int(total_timesteps)
        for row in evaluation.get("fixed_contexts", ()):
            value = float(row["opponent_value"])
            for key, item in row.items():
                if key != "opponent_value" and isinstance(
                        item, (bool, int, float, np.number)
                ):
                    summary[
                        f"eval/fixed_{value:.2f}/{key}"
                    ] = float(item)
        # The last episode, validation, and final evaluation can share one
        # training clock.  Let W&B advance its internal history row while the
        # explicit eval/total_timesteps metric remains the scientific x-axis.
        run.log(summary)
        run.finish()


def e0_episode_transitions(stage, gameplay_horizon):
    if stage == "e0a":
        return int(gameplay_horizon)
    if stage == "e0b":
        return int(gameplay_horizon) + NUM_TRADE_EVENTS
    raise ValueError(f"unknown E0 stage: {stage!r}")


def e1_episode_transitions(gameplay_horizon):
    return int(gameplay_horizon) + NUM_TRADE_EVENTS


def e2_episode_transitions(gameplay_horizon):
    return int(gameplay_horizon) + 2 * NUM_TRADE_EVENTS


__all__ = [
    "ACTOR_LOSS_MODES",
    "ACTOR_LOSS_MODE_ATTRIBUTE",
    "EpisodeCheckpointCallback",
    "ECONOMIC_INIT_ATTRIBUTE",
    "OPTIMIZER_METRICS_ATTRIBUTE",
    "PHASE_BALANCED_ACTOR_LOSS_MODE",
    "PhaseBalancedPPO",
    "ScaledLearningRatePPO",
    "STANDARD_ACTOR_LOSS_MODE",
    "WANDB_GROUP",
    "WANDB_PROJECT",
    "attach_atari_training_contract",
    "checkpoint_path",
    "e0_episode_transitions",
    "e1_episode_transitions",
    "e2_episode_transitions",
    "evaluate_model",
    "fixed_context_csv_path",
    "finish_run",
    "init_wandb",
    "json_path",
    "make_vec_env",
    "model_actor_loss_mode",
    "model_economic_initialization",
    "phase_balanced_actor_terms",
    "ppo_class_for_actor_loss_mode",
    "target_checkpoint_path",
    "target_selection_path",
    "training_log_path",
    "validation_log_path",
    "write_csv",
    "write_json",
]
