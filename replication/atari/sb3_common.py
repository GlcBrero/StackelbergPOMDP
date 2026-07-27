"""Shared SB3, evaluation, checkpoint, and W&B utilities for clean Atari."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS


WANDB_PROJECT = "StackPOMDP"
WANDB_GROUP = "atari_clean_curriculum"


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
        name=getattr(args, "wandb_name", None) or f"atari_{stage}_seed{args.seed}",
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
        self._initialized = False

    def _init_callback(self):
        if not self._initialized:
            self.started = time.time()
            self.started_timesteps = int(getattr(
                self.model, "num_timesteps", self.num_timesteps
            ))
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
                with self.training_log.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if int(row.get("train/total_timesteps", 0)) <= current_step:
                            retained.append(line)
                temporary = self.training_log.with_suffix(
                    self.training_log.suffix + ".tmp"
                )
                with temporary.open("w", encoding="utf-8") as handle:
                    handle.writelines(retained)
                temporary.replace(self.training_log)
                self.episode_count = len(retained)
            self._initialized = True
        current_step = int(getattr(
            self.model, "num_timesteps", self.num_timesteps
        ))
        self.next_checkpoint = (
            (current_step // self.checkpoint_every) + 1
        ) * self.checkpoint_every

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
                ):
                    if key in event:
                        payload[
                            f"train/event_{event_index}/{key}"
                        ] = float(event[key])
            wandb_payloads.append(payload)
            local_payload = {
                **payload,
                "seed": self.seed,
                "algorithm": "PPO",
                "checkpoint_path": str(self.checkpoint),
            }
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
    "EpisodeCheckpointCallback",
    "ScaledLearningRatePPO",
    "WANDB_GROUP",
    "WANDB_PROJECT",
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
    "target_checkpoint_path",
    "target_selection_path",
    "training_log_path",
    "validation_log_path",
    "write_csv",
    "write_json",
]
