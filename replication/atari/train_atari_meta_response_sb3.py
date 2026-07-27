"""Train and evaluate an Atari economic meta-response with native SB3 PPO.

The Atari controller is copied from the canonical E0 checkpoint, frozen, and
verified bit-for-bit throughout training.  PPO therefore learns only a small
economic response head (plus its critic): buyer thresholds in response to a
five-price context, or seller prices in response to a five-threshold context.

Examples
--------
Buyer smoke test without Weights & Biases::

    python -m replication.atari.train_atari_meta_response_sb3 \
        --role buyer --timesteps 25 --n-steps 5 --batch-size 5 \
        --gameplay-horizon 20 --event-tail-steps 5 \
        --fixed-event-steps 0,3,6,9,12 --eval-episodes-per-context 1 \
        --random-eval-episodes 1 --no-wandb

Normal local buyer run, visible in the ``StackPOMDP`` W&B project::

    python -m replication.atari.train_atari_meta_response_sb3 \
        --role buyer --timesteps 1000000 --wandb
"""

import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
import os
from pathlib import Path
import site
import sys
import tempfile
import time

import numpy as np
import torch as th


# Avoid the known incompatible user-site Pillow install before importing SB3.
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stackpomdp-matplotlib")
)
os.environ.setdefault("WANDB_START_METHOD", "thread")
_user_site = str(Path(site.getusersitepackages()).resolve())
sys.path[:] = [
    entry
    for entry in sys.path
    if not entry or str(Path(entry).resolve()) != _user_site
]

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
    MetaEconomicResponseEnv,
    NUM_TRADE_EVENTS,
)
from stackelberg_pomdp.atari.stackpomdp_policy import (
    StackPOMDPAtariEconomicPolicy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPRODUCIBILITY_SOURCE_FILES = (
    "replication/atari/train_atari_meta_response_sb3.py",
    "stackelberg_pomdp/atari/stackpomdp_env.py",
    "stackelberg_pomdp/atari/stackpomdp_policy.py",
)
DEFAULT_E0_CHECKPOINT = (
    PROJECT_ROOT
    / "replication/atari/checkpoints/sb3/"
    / "space_invaders_e0_ppo_seed1_10m_best.zip"
)
WANDB_PROJECT = "StackPOMDP"
WANDB_GROUP = "atari_stackpomdp"
WANDB_JOB_TYPES = {
    BUYER: "meta_buyer_response",
    SELLER: "meta_seller_response",
}
PROTECTED_PREFIXES = ("features_extractor.", "game_action_net.")
SELECTION_METRIC = "random_context_summary.mean_controlled_reward"
SELECTION_RULE = (
    "maximize_paired_random_context_mean_controlled_reward_"
    "ties_keep_earliest_timestep_v1"
)


FIXED_CONTEXTS = (
    ("all_0p00", (0.00, 0.00, 0.00, 0.00, 0.00)),
    ("all_0p10", (0.10, 0.10, 0.10, 0.10, 0.10)),
    ("all_0p25", (0.25, 0.25, 0.25, 0.25, 0.25)),
    ("all_0p50", (0.50, 0.50, 0.50, 0.50, 0.50)),
    ("all_0p75", (0.75, 0.75, 0.75, 0.75, 0.75)),
    ("all_1p00", (1.00, 1.00, 1.00, 1.00, 1.00)),
    ("increasing", (0.00, 0.25, 0.50, 0.75, 1.00)),
    ("decreasing", (1.00, 0.75, 0.50, 0.25, 0.00)),
)


EPISODE_NUMERIC_FIELDS = (
    "trade_opportunities",
    "bullets_arrived",
    "purchases",
    "payments",
    "seller_game_reward",
    "buyer_game_reward",
    "seller_reward",
    "buyer_reward",
    "seller_shots_fired",
    "buyer_shots_fired",
    "seller_final_ammo",
    "buyer_final_ammo",
    "seller_life_resets",
    "buyer_life_resets",
    "seller_bullet_error",
    "buyer_bullet_error",
    "seller_payoff_error",
    "buyer_payoff_error",
)


def checkpoint_with_zip(path):
    """Return an absolute SB3 checkpoint path with a ``.zip`` suffix."""

    path = Path(path).expanduser().resolve()
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def evaluation_with_json(path):
    """Return an absolute evaluation path with a ``.json`` suffix."""

    path = Path(path).expanduser().resolve()
    return path if path.suffix == ".json" else path.with_suffix(".json")


def _tagged_path(path, tag):
    """Insert an archive tag without breaking ``.evaluation.json`` names."""

    path = Path(path).expanduser().resolve()
    evaluation_suffix = ".evaluation.json"
    if path.name.endswith(evaluation_suffix):
        prefix = path.name[:-len(evaluation_suffix)]
        return path.with_name(f"{prefix}_{tag}{evaluation_suffix}")
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def step_checkpoint_path(path, timesteps):
    return _tagged_path(checkpoint_with_zip(path), f"step{int(timesteps)}")


def step_evaluation_path(path, timesteps):
    return _tagged_path(evaluation_with_json(path), f"step{int(timesteps)}")


def best_checkpoint_path(path):
    return _tagged_path(checkpoint_with_zip(path), "best")


def best_evaluation_path(path):
    return _tagged_path(evaluation_with_json(path), "best")


def selection_manifest_path(path):
    path = checkpoint_with_zip(path)
    return path.with_name(f"{path.stem}.selection.json")


def resolved_checkpoint(path):
    """Resolve an existing SB3 checkpoint with or without its zip suffix."""

    path = Path(path).expanduser().resolve()
    candidates = (path, Path(f"{path}.zip"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"checkpoint does not exist: {path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file_sha256s():
    """Hash the exact uncommitted sources that define a meta-response run."""

    result = {}
    for relative_path in REPRODUCIBILITY_SOURCE_FILES:
        source_path = PROJECT_ROOT / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"reproducibility source does not exist: {source_path}"
            )
        result[relative_path] = sha256_file(source_path)
    return result


def source_bundle_sha256(source_hashes):
    """Return one order-independent fingerprint for the source hash map."""

    digest = hashlib.sha256()
    for relative_path, source_hash in sorted(source_hashes.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(source_hash).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def reproducibility_source_metadata():
    source_hashes = source_file_sha256s()
    return {
        "source_files_sha256": source_hashes,
        "source_bundle_sha256": source_bundle_sha256(source_hashes),
        "source_provenance_scope": (
            "meta_trainer_bilateral_env_and_composite_policy_at_write_time"
        ),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _step_label(timesteps):
    timesteps = int(timesteps)
    if timesteps >= 1_000_000 and timesteps % 1_000_000 == 0:
        return f"{timesteps // 1_000_000}m"
    if timesteps >= 1_000 and timesteps % 1_000 == 0:
        return f"{timesteps // 1_000}k"
    return str(timesteps)


def parse_event_steps(raw):
    if raw is None or str(raw).strip() == "":
        return None
    values = tuple(
        int(value.strip())
        for value in str(raw).split(",")
        if value.strip()
    )
    if len(values) != NUM_TRADE_EVENTS:
        raise ValueError(
            f"--fixed-event-steps must contain {NUM_TRADE_EVENTS} integers"
        )
    return values


def bilateral_config(args, *, seed):
    return BilateralAtariConfig(
        seed=int(seed),
        gameplay_horizon=int(args.gameplay_horizon),
        event_tail_steps=int(args.event_tail_steps),
        num_trade_events=NUM_TRADE_EVENTS,
        seller_game_reward_scale=0.1,
        buyer_game_reward_scale=1.0,
        noop_max=int(args.noop_max),
        frame_skip=4,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        max_frames=int(args.max_frames),
        rom_path=args.rom_path,
        fixed_event_steps=args.fixed_event_steps,
    ).resolved()


def make_training_env(args, rank):
    seed = int(args.seed + 10_000 * rank)
    config = bilateral_config(args, seed=seed)
    game_checkpoint = str(args.game_checkpoint)
    role = str(args.role)

    def constructor():
        return MetaEconomicResponseEnv(
            controlled_role=role,
            game_checkpoint=game_checkpoint,
            config=config,
        )

    return constructor


def make_training_vec_env(args):
    constructors = [
        make_training_env(args, rank) for rank in range(args.num_envs)
    ]
    if args.num_envs == 1:
        return DummyVecEnv(constructors)
    return SubprocVecEnv(constructors, start_method=args.start_method)


def protected_gameplay_state(policy):
    """Clone all tensors that implement the frozen E0 gameplay mapping."""

    if not bool(policy.gameplay_ready.item()):
        raise RuntimeError("cannot snapshot an uninitialized E0 controller")
    state = policy.state_dict()
    protected = {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if name.startswith(PROTECTED_PREFIXES)
    }
    if not protected:
        raise RuntimeError("no frozen gameplay tensors were found")
    return protected


def assert_gameplay_unchanged(policy, expected, *, expected_fingerprint):
    """Fail immediately if optimizer or serialization changed frozen E0."""

    if policy.gameplay_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "frozen E0 fingerprint changed: "
            f"{policy.gameplay_fingerprint} != {expected_fingerprint}"
        )
    current = policy.state_dict()
    changed = [
        name
        for name, tensor in expected.items()
        if name not in current
        or not th.equal(current[name].detach().cpu(), tensor)
    ]
    if changed:
        raise RuntimeError(
            "frozen Atari gameplay drifted during economic training: "
            f"{changed[:3]}"
        )
    trainable_gameplay = [
        name
        for name, parameter in policy.named_parameters()
        if name.startswith(PROTECTED_PREFIXES) and parameter.requires_grad
    ]
    if trainable_gameplay:
        raise RuntimeError(
            "frozen Atari tensors became trainable: "
            f"{trainable_gameplay[:3]}"
        )


def _new_model(args, vec_env):
    model = PPO(
        StackPOMDPAtariEconomicPolicy,
        vec_env,
        policy_kwargs={
            "economic_role": args.role,
            "trade_events": NUM_TRADE_EVENTS,
            "visual_features": 512,
            "ammo_features": 32,
            "market_features": 16,
            "economic_hidden": args.economic_hidden,
            "critic_hidden": args.critic_hidden,
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
    provenance = model.policy.load_frozen_gameplay_checkpoint(
        args.game_checkpoint,
        device=args.device,
    )
    print(json.dumps({
        "event": "initialized_frozen_e0",
        "role": args.role,
        **provenance,
    }, sort_keys=True), flush=True)
    return model


def _resumed_model(args, vec_env):
    model = PPO.load(
        str(args.resume),
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
    if not isinstance(model.policy, StackPOMDPAtariEconomicPolicy):
        raise TypeError(
            "--resume must be a StackPOMDPAtariEconomicPolicy checkpoint"
        )
    if model.policy.economic_role != args.role:
        raise ValueError(
            "resume role mismatch: "
            f"{model.policy.economic_role!r} != {args.role!r}"
        )
    expected_fingerprint = sha256_file(args.game_checkpoint)
    if model.policy.gameplay_fingerprint != expected_fingerprint:
        raise ValueError(
            "resume checkpoint was built from a different E0 controller: "
            f"{model.policy.gameplay_fingerprint} != {expected_fingerprint}"
        )
    for parameter_group in model.policy.optimizer.param_groups:
        parameter_group["lr"] = float(args.learning_rate)
    print(json.dumps({
        "event": "resumed_meta_response",
        "role": args.role,
        "resume": str(args.resume),
        "starting_timesteps": int(model.num_timesteps),
        "e0_sha256": expected_fingerprint,
    }, sort_keys=True), flush=True)
    return model


def build_or_load_model(args, vec_env):
    return _resumed_model(args, vec_env) if args.resume else _new_model(
        args, vec_env
    )


def _episode_row(
        info,
        *,
        split,
        context_label,
        episode_index,
        actions,
        gameplay_horizon,
):
    row = {
        "split": split,
        "context_label": context_label,
        "episode": int(episode_index),
        "controlled_role": info["controlled_role"],
        "controlled_reward": float(
            info[f"{info['controlled_role']}_reward"]
        ),
        "gameplay_horizon": int(gameplay_horizon),
    }
    for key in EPISODE_NUMERIC_FIELDS:
        row[key] = info[key]
    context = tuple(float(value) for value in info["opponent_context"])
    event_steps = tuple(int(value) for value in info["event_steps"])
    for event_index in range(NUM_TRADE_EVENTS):
        row[f"context_{event_index + 1}"] = context[event_index]
        row[f"response_{event_index + 1}"] = float(actions[event_index])
        row[f"event_step_{event_index + 1}"] = event_steps[event_index]
    row["events"] = _jsonable(info["events"])
    return row


def run_evaluation_episode(model, env, *, split, context_label, episode_index):
    if hasattr(model.policy, "clear_obs_action_map"):
        model.policy.clear_obs_action_map()
    observation = env.reset()
    done = False
    actions = []
    episode_return = 0.0
    info = None
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(2)
        actions.append(float(action[1]))
        observation, reward, done, info = env.step(action)
        episode_return += float(reward)
    if len(actions) != NUM_TRADE_EVENTS:
        raise RuntimeError(
            f"evaluation produced {len(actions)} rather than five responses"
        )
    row = _episode_row(
        info,
        split=split,
        context_label=context_label,
        episode_index=episode_index,
        actions=actions,
        gameplay_horizon=env.config.gameplay_horizon,
    )
    if not np.isclose(row["controlled_reward"], episode_return, atol=1e-6):
        raise RuntimeError(
            "environment reward disagrees with terminal accounting: "
            f"{episode_return} != {row['controlled_reward']}"
        )
    return row


def aggregate_rows(rows):
    if not rows:
        return {"episodes": 0}
    fields = ("controlled_reward",) + EPISODE_NUMERIC_FIELDS
    result = {"episodes": len(rows)}
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        result[f"mean_{field}"] = float(values.mean())
        result[f"std_{field}"] = float(values.std())
    responses = np.asarray([
        [row[f"response_{index + 1}"] for index in range(NUM_TRADE_EVENTS)]
        for row in rows
    ], dtype=np.float64)
    contexts = np.asarray([
        [row[f"context_{index + 1}"] for index in range(NUM_TRADE_EVENTS)]
        for row in rows
    ], dtype=np.float64)
    result["mean_response"] = float(responses.mean())
    result["mean_context"] = float(contexts.mean())
    result["mean_response_vector"] = responses.mean(axis=0).tolist()

    # Trade timing is diagnostic only: the actor still receives event identity,
    # not the realized clock time.  These statistics test whether willingness
    # to trade changes as useful gameplay time runs out.
    events = [event for row in rows for event in row["events"]]
    horizons = {
        int(row["gameplay_horizon"])
        for row in rows
    }
    if len(horizons) != 1:
        raise ValueError("evaluation rows disagree on gameplay horizon")
    horizon = float(next(iter(horizons)))
    for event_index in range(NUM_TRADE_EVENTS):
        indexed = [
            event for event in events
            if int(event["event_index"]) == event_index
        ]
        steps = np.asarray(
            [event["game_step"] for event in indexed], dtype=np.float64
        )
        accepted = np.asarray(
            [event["accepted"] for event in indexed], dtype=np.float64
        )
        suffix = event_index + 1
        result[f"event_{suffix}_count"] = int(len(indexed))
        result[f"mean_event_{suffix}_step"] = float(steps.mean())
        result[f"mean_event_{suffix}_normalized_time"] = float(
            (steps / horizon).mean()
        )
        result[f"event_{suffix}_acceptance_rate"] = float(accepted.mean())

    normalized_times = np.asarray(
        [event["game_step"] / horizon for event in events],
        dtype=np.float64,
    )
    accepted_mask = np.asarray(
        [event["accepted"] for event in events], dtype=bool
    )
    for label, mask in (
        ("accepted", accepted_mask),
        ("rejected", ~accepted_mask),
    ):
        result[f"{label}_event_count"] = int(mask.sum())
        result[f"mean_{label}_normalized_game_time"] = (
            float(normalized_times[mask].mean()) if mask.any() else None
        )
    for label, lower, upper in (
        ("early", 0.0, 1.0 / 3.0),
        ("middle", 1.0 / 3.0, 2.0 / 3.0),
        ("late", 2.0 / 3.0, np.inf),
    ):
        bin_mask = (normalized_times >= lower) & (normalized_times < upper)
        result[f"{label}_event_count"] = int(bin_mask.sum())
        result[f"{label}_acceptance_rate"] = (
            float(accepted_mask[bin_mask].mean()) if bin_mask.any() else None
        )
    return result


def evaluate_response(model, args, *, eval_seed=None):
    """Evaluate deterministic response actions on fixed and random contexts."""

    eval_seed = int(args.eval_seed if eval_seed is None else eval_seed)
    fixed_rows = []
    fixed_table = []
    for context_index, (label, context) in enumerate(FIXED_CONTEXTS):
        context_array = np.asarray(context, dtype=np.float32)

        def sampler(_rng, context_array=context_array):
            return np.array(context_array, copy=True)

        config = bilateral_config(
            args, seed=eval_seed + 10_000 * context_index
        )
        env = MetaEconomicResponseEnv(
            controlled_role=args.role,
            game_checkpoint=str(args.game_checkpoint),
            config=config,
            context_sampler=sampler,
        )
        rows = []
        try:
            for episode in range(args.eval_episodes_per_context):
                rows.append(run_evaluation_episode(
                    model,
                    env,
                    split="fixed",
                    context_label=label,
                    episode_index=episode,
                ))
        finally:
            env.close()
        fixed_rows.extend(rows)
        fixed_table.append({
            "context_label": label,
            "context": list(context),
            **aggregate_rows(rows),
        })

    random_env = MetaEconomicResponseEnv(
        controlled_role=args.role,
        game_checkpoint=str(args.game_checkpoint),
        config=bilateral_config(args, seed=eval_seed + 900_001),
    )
    random_rows = []
    try:
        for episode in range(args.random_eval_episodes):
            random_rows.append(run_evaluation_episode(
                model,
                random_env,
                split="random",
                context_label="uniform_0_1",
                episode_index=episode,
            ))
    finally:
        random_env.close()

    result = {
        "metadata": {
            "algorithm": "SB3-PPO",
            "architecture": "frozen_e0_plus_meta_economic_response_v1",
            "controlled_role": args.role,
            "opponent_context": (
                "five_prices" if args.role == BUYER else "five_thresholds"
            ),
            "game_checkpoint": str(args.game_checkpoint),
            "game_checkpoint_sha256": sha256_file(args.game_checkpoint),
            "model_timesteps": int(model.num_timesteps),
            "seed": int(args.seed),
            "eval_seed": eval_seed,
            "deterministic": True,
            "gameplay_horizon": int(args.gameplay_horizon),
            "trade_events": NUM_TRADE_EVENTS,
            "gamma": 1.0,
            "gae_lambda": 1.0,
        },
        "fixed_context_table": fixed_table,
        "random_context_summary": aggregate_rows(random_rows),
        "episodes": fixed_rows + random_rows,
    }
    return _jsonable(result)


def selection_score(result):
    """Return the paired random-context score used for model selection."""

    score = float(
        result["random_context_summary"]["mean_controlled_reward"]
    )
    if not np.isfinite(score):
        raise ValueError(f"non-finite {SELECTION_METRIC}: {score}")
    return score


def add_evaluation_provenance(
        result,
        args,
        *,
        checkpoint_path,
        evaluation_kind,
):
    """Record the exact policy/E0/role inputs behind an evaluation bundle."""

    checkpoint_path = resolved_checkpoint(checkpoint_path)
    metadata = result.setdefault("metadata", {})
    metadata.update({
        "response_role": args.role,
        "evaluation_kind": str(evaluation_kind),
        "evaluated_checkpoint": str(checkpoint_path),
        "evaluated_checkpoint_sha256": sha256_file(checkpoint_path),
        "resume_source": str(args.resume) if args.resume else None,
        "selection_metric": SELECTION_METRIC,
        "selection_rule": SELECTION_RULE,
        "selection_eval_seed": int(args.eval_seed),
        "selection_metric_value": selection_score(result),
        **reproducibility_source_metadata(),
    })
    return _jsonable(result)


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key == "events" or key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_evaluation(path, result):
    path = evaluation_with_json(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    fixed_path = path.with_suffix(".fixed_contexts.csv")
    fixed_rows = []
    for entry in result["fixed_context_table"]:
        row = {
            key: value
            for key, value in entry.items()
            if key not in {"context", "mean_response_vector"}
        }
        for index, value in enumerate(entry["context"]):
            row[f"context_{index + 1}"] = value
        for index, value in enumerate(entry["mean_response_vector"]):
            row[f"mean_response_{index + 1}"] = value
        fixed_rows.append(row)
    _write_csv(fixed_path, fixed_rows)

    episode_path = path.with_suffix(".episodes.csv")
    _write_csv(episode_path, result["episodes"])

    event_path = path.with_suffix(".events.csv")
    event_rows = []
    for episode in result["episodes"]:
        for event in episode["events"]:
            event_rows.append({
                "split": episode["split"],
                "context_label": episode["context_label"],
                "episode": episode["episode"],
                "controlled_role": episode["controlled_role"],
                **event,
            })
    _write_csv(event_path, event_rows)
    return {
        "json": path,
        "fixed_contexts_csv": fixed_path,
        "episodes_csv": episode_path,
        "events_csv": event_path,
    }


def evaluation_wandb_metrics(result):
    metrics = {
        f"evaluation/random/{key}": value
        for key, value in result["random_context_summary"].items()
        if isinstance(value, (int, float, bool))
    }
    for entry in result["fixed_context_table"]:
        label = entry["context_label"]
        for key in (
            "mean_controlled_reward",
            "mean_purchases",
            "mean_payments",
            "mean_seller_game_reward",
            "mean_buyer_game_reward",
            "mean_seller_shots_fired",
            "mean_buyer_shots_fired",
            "mean_seller_final_ammo",
            "mean_buyer_final_ammo",
            "mean_response",
        ):
            metrics[f"evaluation/fixed/{label}/{key}"] = entry[key]
    metrics["total_timesteps"] = int(result["metadata"]["model_timesteps"])
    return metrics


class MetaResponseTrainingCallback(BaseCallback):
    """Log complete economic/game accounting and save drift-checked models."""

    def __init__(
            self,
            args,
            checkpoint_path,
            protected_state,
            expected_fingerprint,
            wandb_run=None,
    ):
        super().__init__(verbose=0)
        self.args = args
        self.checkpoint_path = Path(checkpoint_path)
        self.protected_state = protected_state
        self.expected_fingerprint = expected_fingerprint
        self.wandb_run = wandb_run
        self.recent = defaultdict(lambda: deque(maxlen=100))
        self.episodes = 0
        self.started = time.time()
        self.starting_timesteps = 0
        self.next_log = 0
        self.next_checkpoint = 0
        self.next_evaluation = 0
        self.best_score = -np.inf
        self.best_timestep = None
        self.best_checkpoint = best_checkpoint_path(self.checkpoint_path)
        self.best_evaluation = best_evaluation_path(self.args.output)
        self.selection_manifest = selection_manifest_path(
            self.checkpoint_path
        )

    def _next_boundary(self, interval):
        if interval <= 0:
            return sys.maxsize
        return (int(self.num_timesteps) // interval + 1) * interval

    def _on_training_start(self):
        self.starting_timesteps = int(self.num_timesteps)
        self._restore_selection()
        self.next_log = self._next_boundary(self.args.log_every)
        self.next_checkpoint = self._next_boundary(
            self.args.checkpoint_every
        )
        self.next_evaluation = self._next_boundary(self.args.eval_every)

    def _record_terminal_infos(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))
        for done, info in zip(dones, infos):
            if not done or "controlled_role" not in info:
                continue
            self.episodes += 1
            role = info["controlled_role"]
            values = {
                "episode_reward": info[f"{role}_reward"],
                **{
                    field: info[field] for field in EPISODE_NUMERIC_FIELDS
                },
            }
            buyer_shots = max(int(info["buyer_shots_fired"]), 1)
            seller_shots = max(int(info["seller_shots_fired"]), 1)
            transfers = max(int(info["purchases"]), 1)
            values.update({
                "buyer_game_reward_per_shot": (
                    float(info["buyer_game_reward"]) / buyer_shots
                ),
                "seller_game_reward_per_shot": (
                    float(info["seller_game_reward"]) / seller_shots
                ),
                "buyer_net_reward_per_transfer": (
                    float(info["buyer_reward"]) / transfers
                ),
            })
            for key, value in values.items():
                self.recent[key].append(float(value))

    def _training_metrics(self):
        elapsed = max(time.time() - self.started, 1.0e-9)
        metrics = {
            f"train/{key}": float(np.mean(values))
            for key, values in self.recent.items()
            if values
        }
        metrics.update({
            "total_timesteps": int(self.num_timesteps),
            "train/episodes": int(self.episodes),
            "train/learning_rate": float(
                self.model.policy.optimizer.param_groups[0]["lr"]
            ),
            "train/steps_per_second": float(
                (int(self.num_timesteps) - self.starting_timesteps) / elapsed
            ),
            "train/seed": int(self.args.seed),
        })
        return metrics

    def _log(self, metrics):
        print(json.dumps(_jsonable(metrics), sort_keys=True), flush=True)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=int(self.num_timesteps))

    def _save_checkpoint(self, path):
        assert_gameplay_unchanged(
            self.model.policy,
            self.protected_state,
            expected_fingerprint=self.expected_fingerprint,
        )
        path = checkpoint_with_zip(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"checkpoint={path}", flush=True)
        return path

    def _assert_and_save(self):
        return self._save_checkpoint(self.checkpoint_path)

    def _restore_selection(self):
        """Resume the same paired model-selection sequence when available."""

        if not self.args.resume or not self.selection_manifest.is_file():
            return
        payload = json.loads(self.selection_manifest.read_text())
        expected = {
            "controlled_role": self.args.role,
            "game_checkpoint_sha256": self.expected_fingerprint,
            "selection_metric": SELECTION_METRIC,
            "selection_rule": SELECTION_RULE,
            "selection_eval_seed": int(self.args.eval_seed),
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "resume selection metadata does not match this run: "
                f"{mismatches}"
            )
        best_checkpoint = resolved_checkpoint(payload["best_checkpoint"])
        best_evaluation = evaluation_with_json(payload["best_evaluation"])
        if not best_evaluation.is_file():
            raise FileNotFoundError(
                f"recorded best evaluation is missing: {best_evaluation}"
            )
        self.best_score = float(payload["best_score"])
        self.best_timestep = int(payload["best_timestep"])
        self.best_checkpoint = best_checkpoint
        self.best_evaluation = best_evaluation
        print(json.dumps({
            "event": "restored_meta_response_selection",
            "best_score": self.best_score,
            "best_timestep": self.best_timestep,
            "best_checkpoint": str(self.best_checkpoint),
        }, sort_keys=True), flush=True)

    def _is_better(self, score, timestep):
        if score > self.best_score:
            return True
        return bool(
            score == self.best_score
            and (
                self.best_timestep is None
                or int(timestep) < self.best_timestep
            )
        )

    def _promote_best(
            self,
            result,
            *,
            source_checkpoint,
            source_evaluation,
            score,
            timestep,
    ):
        self._save_checkpoint(self.best_checkpoint)
        best_result = json.loads(json.dumps(_jsonable(result)))
        best_result["metadata"].update({
            "evaluation_kind": "best",
            "evaluated_checkpoint": str(self.best_checkpoint),
            "evaluated_checkpoint_sha256": sha256_file(
                self.best_checkpoint
            ),
            "selected_from_checkpoint": str(source_checkpoint),
            "selected_from_evaluation": str(source_evaluation),
            "selected_as_best": True,
        })
        best_paths = write_evaluation(self.best_evaluation, best_result)
        self.best_score = float(score)
        self.best_timestep = int(timestep)
        manifest = {
            "schema_version": 1,
            "controlled_role": self.args.role,
            "seed": int(self.args.seed),
            "game_checkpoint": str(self.args.game_checkpoint),
            "game_checkpoint_sha256": self.expected_fingerprint,
            "selection_metric": SELECTION_METRIC,
            "selection_rule": SELECTION_RULE,
            "selection_eval_seed": int(self.args.eval_seed),
            "best_score": self.best_score,
            "best_timestep": self.best_timestep,
            "best_checkpoint": str(self.best_checkpoint),
            "best_checkpoint_sha256": sha256_file(self.best_checkpoint),
            "best_evaluation": str(best_paths["json"]),
            "best_fixed_contexts_csv": str(
                best_paths["fixed_contexts_csv"]
            ),
            "best_episodes_csv": str(best_paths["episodes_csv"]),
            "best_events_csv": str(best_paths["events_csv"]),
            "selected_from_checkpoint": str(source_checkpoint),
            "selected_from_evaluation": str(source_evaluation),
            "source_bundle_sha256": best_result["metadata"][
                "source_bundle_sha256"
            ],
            "source_files_sha256": best_result["metadata"][
                "source_files_sha256"
            ],
        }
        _write_json_atomic(self.selection_manifest, manifest)
        if self.wandb_run is not None:
            self.wandb_run.summary.update({
                "best/controlled_role": self.args.role,
                "best/selection_metric": SELECTION_METRIC,
                "best/selection_rule": SELECTION_RULE,
                "best/selection_eval_seed": int(self.args.eval_seed),
                "best/random_context_mean_controlled_reward": self.best_score,
                "best/timestep": self.best_timestep,
                "best/checkpoint": str(self.best_checkpoint),
                "best/checkpoint_sha256": manifest[
                    "best_checkpoint_sha256"
                ],
                "best/evaluation": str(best_paths["json"]),
                "best/fixed_contexts_csv": str(
                    best_paths["fixed_contexts_csv"]
                ),
                "best/episodes_csv": str(best_paths["episodes_csv"]),
                "best/events_csv": str(best_paths["events_csv"]),
                "best/game_checkpoint_sha256": self.expected_fingerprint,
                "best/source_bundle_sha256": manifest[
                    "source_bundle_sha256"
                ],
                "best/source_files_sha256": manifest[
                    "source_files_sha256"
                ],
            })

    def record_evaluation(
            self,
            result,
            *,
            checkpoint_path,
            evaluation_path,
            evaluation_kind,
    ):
        """Write one evaluation bundle and update deterministic best state."""

        checkpoint_path = resolved_checkpoint(checkpoint_path)
        evaluation_path = evaluation_with_json(evaluation_path)
        result = add_evaluation_provenance(
            result,
            self.args,
            checkpoint_path=checkpoint_path,
            evaluation_kind=evaluation_kind,
        )
        score = selection_score(result)
        timestep = int(result["metadata"]["model_timesteps"])
        is_best = self._is_better(score, timestep)
        result["metadata"].update({
            "selected_as_best": bool(is_best),
            "best_score_before_evaluation": (
                None if not np.isfinite(self.best_score) else self.best_score
            ),
            "best_timestep_before_evaluation": self.best_timestep,
        })
        output_paths = write_evaluation(evaluation_path, result)
        if is_best:
            self._promote_best(
                result,
                source_checkpoint=checkpoint_path,
                source_evaluation=output_paths["json"],
                score=score,
                timestep=timestep,
            )
        metrics = evaluation_wandb_metrics(result)
        metrics.update({
            "evaluation/selection/random_context_mean_controlled_reward": (
                score
            ),
            "evaluation/selection/is_best": int(is_best),
            "evaluation/selection/best_score": float(self.best_score),
            "evaluation/selection/eval_seed": int(self.args.eval_seed),
        })
        self._log(metrics)
        if self.wandb_run is not None:
            self.wandb_run.summary.update({
                "latest/evaluation_kind": str(evaluation_kind),
                "latest/checkpoint": str(checkpoint_path),
                "latest/evaluation": str(output_paths["json"]),
                "latest/random_context_mean_controlled_reward": score,
                "selection/metric": SELECTION_METRIC,
                "selection/rule": SELECTION_RULE,
                "selection/eval_seed": int(self.args.eval_seed),
                "selection/paired_contexts_and_schedules": True,
            })
        return result, output_paths

    def _periodic_evaluation(self):
        timesteps = int(self.num_timesteps)
        checkpoint_path = step_checkpoint_path(
            self.checkpoint_path, timesteps
        )
        evaluation_path = step_evaluation_path(self.args.output, timesteps)
        self._save_checkpoint(checkpoint_path)
        result = evaluate_response(
            self.model,
            self.args,
            # Paired random contexts and event schedules are essential for a
            # low-variance comparison between successive checkpoints.
            eval_seed=self.args.eval_seed,
        )
        return self.record_evaluation(
            result,
            checkpoint_path=checkpoint_path,
            evaluation_path=evaluation_path,
            evaluation_kind="periodic",
        )

    def _on_step(self):
        self._record_terminal_infos()
        if self.num_timesteps >= self.next_log:
            self._log(self._training_metrics())
            while self.next_log <= self.num_timesteps:
                self.next_log += self.args.log_every
        if self.num_timesteps >= self.next_checkpoint:
            self._assert_and_save()
            while self.next_checkpoint <= self.num_timesteps:
                self.next_checkpoint += self.args.checkpoint_every
        if self.num_timesteps >= self.next_evaluation:
            self._periodic_evaluation()
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += self.args.eval_every
        return True


def init_wandb(args):
    if not args.wandb:
        return None
    import wandb

    source_metadata = reproducibility_source_metadata()
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        name=args.wandb_name,
        config=_jsonable({
            **vars(args),
            "algorithm": "SB3-PPO",
            "architecture": "frozen_e0_plus_meta_economic_response_v1",
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "trade_events": NUM_TRADE_EVENTS,
            "context": (
                "five_prices" if args.role == BUYER else "five_thresholds"
            ),
            "gameplay_trainable": False,
            "gameplay_action": "deterministic_argmax",
            "economic_action": (
                "threshold" if args.role == BUYER else "price"
            ),
            "seller_game_reward_scale": 0.1,
            "buyer_game_reward_scale": 1.0,
            "clip_game_rewards": True,
            "checkpoint_path": str(args.checkpoint),
            "evaluation_path": str(args.output),
            "selection_metric": SELECTION_METRIC,
            "selection_rule": SELECTION_RULE,
            "selection_eval_seed": int(args.eval_seed),
            "selection_paired_contexts_and_schedules": True,
            **source_metadata,
        }),
    )
    run.define_metric("total_timesteps")
    run.define_metric("train/*", step_metric="total_timesteps")
    run.define_metric("evaluation/*", step_metric="total_timesteps")
    run.summary.update({
        "algorithm": "SB3-PPO",
        "controlled_role": args.role,
        "seed": int(args.seed),
        "checkpoint_path": str(args.checkpoint),
        "game_checkpoint": str(args.game_checkpoint),
        "game_checkpoint_sha256": sha256_file(args.game_checkpoint),
        **source_metadata,
    })
    print(f"wandb_url={run.url}", flush=True)
    return run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train a buyer-threshold or seller-price meta-response while "
            "keeping both Atari gameplay controllers frozen at E0."
        )
    )
    parser.add_argument("--role", choices=[BUYER, SELLER], default=BUYER)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--timesteps-are-target",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--game-checkpoint", default=str(DEFAULT_E0_CHECKPOINT))
    parser.add_argument("--resume")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "reload --resume, verify role/E0 provenance and frozen tensors, "
            "then regenerate evaluation artifacts without learning"
        ),
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--start-method",
        choices=["spawn", "forkserver", "fork"],
        default="spawn",
    )
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--economic-hidden", type=int, default=64)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--gameplay-horizon", type=int, default=200)
    parser.add_argument("--event-tail-steps", type=int, default=50)
    parser.add_argument(
        "--fixed-event-steps",
        help="optional five comma-separated outer gameplay steps",
    )
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=100_000)
    parser.add_argument("--rom-path")

    parser.add_argument("--eval-seed", type=int, default=200_001)
    parser.add_argument("--eval-episodes-per-context", type=int, default=3)
    parser.add_argument("--random-eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=10_000)

    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default="glcbrero")
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    parser.add_argument("--wandb-job-type")
    parser.add_argument("--wandb-name")
    return parser.parse_args(argv)


def validate_args(args):
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")
    if args.eval_only and args.checkpoint:
        raise ValueError(
            "--eval-only audits --resume directly; do not pass --checkpoint"
        )
    if args.eval_only and args.timesteps_are_target:
        raise ValueError(
            "--eval-only cannot be combined with --timesteps-are-target"
        )
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.n_steps <= 0 or args.batch_size <= 1 or args.n_epochs <= 0:
        raise ValueError("invalid PPO batch geometry")
    if args.n_steps % NUM_TRADE_EVENTS:
        raise ValueError(
            "--n-steps must be a multiple of five so rollouts end at an "
            "outer episode boundary"
        )
    rollout_size = args.n_steps * args.num_envs
    if args.batch_size > rollout_size or rollout_size % args.batch_size:
        raise ValueError(
            "n_steps * num_envs must be divisible by batch_size; "
            f"got {rollout_size} and {args.batch_size}"
        )
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.economic_hidden <= 0 or args.critic_hidden <= 0:
        raise ValueError("network widths must be positive")
    if args.noop_max <= 0 or args.max_frames <= 0:
        raise ValueError("--noop-max and --max-frames must be positive")
    if args.eval_episodes_per_context <= 0 or args.random_eval_episodes <= 0:
        raise ValueError("evaluation episode counts must be positive")
    if min(args.log_every, args.checkpoint_every, args.eval_every) < 0:
        raise ValueError("logging/checkpoint/evaluation intervals cannot be negative")
    args.fixed_event_steps = parse_event_steps(args.fixed_event_steps)
    # Resolve and validate the scientific horizon before any expensive ROM load.
    bilateral_config(args, seed=args.seed)
    args.game_checkpoint = resolved_checkpoint(args.game_checkpoint)
    if args.resume:
        args.resume = resolved_checkpoint(args.resume)

    if args.checkpoint:
        args.checkpoint = checkpoint_with_zip(args.checkpoint)
    elif args.resume:
        args.checkpoint = args.resume
    else:
        args.checkpoint = checkpoint_with_zip(
            PROJECT_ROOT
            / "replication/atari/checkpoints/stackpomdp"
            / (
                f"meta_{args.role}_response_ppo_seed{args.seed}_"
                f"{_step_label(args.timesteps)}.zip"
            )
        )
    args.output = evaluation_with_json(
        args.output
        or args.checkpoint.with_suffix(".evaluation.json")
    )
    args.wandb_job_type = (
        args.wandb_job_type or WANDB_JOB_TYPES[args.role]
    )
    if args.wandb_name is None:
        if args.eval_only:
            args.wandb_name = (
                f"atari_meta_{args.role}_response_eval_seed"
                f"{args.eval_seed}_local"
            )
        else:
            args.wandb_name = (
                f"atari_meta_{args.role}_response_ppo_seed{args.seed}_"
                f"{_step_label(args.timesteps)}_local"
            )
    return args


def _log_artifact(run, args, output_paths, model):
    if run is None:
        return
    import wandb

    evaluation_payload = json.loads(Path(output_paths["json"]).read_text())
    source_metadata = {
        key: evaluation_payload["metadata"][key]
        for key in (
            "source_files_sha256",
            "source_bundle_sha256",
            "source_provenance_scope",
        )
    }
    artifact = wandb.Artifact(
        f"atari-meta-{args.role}-response-seed{args.seed}",
        type="model",
        metadata={
            "role": args.role,
            "seed": args.seed,
            "timesteps": int(model.num_timesteps),
            "e0_sha256": model.policy.gameplay_fingerprint,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "selection_metric": SELECTION_METRIC,
            "selection_rule": SELECTION_RULE,
            "selection_eval_seed": int(args.eval_seed),
            "eval_only": bool(args.eval_only),
            **source_metadata,
        },
    )
    artifact.add_file(str(args.checkpoint))
    for path in output_paths.values():
        artifact.add_file(str(path))
    manifest = selection_manifest_path(args.checkpoint)
    if manifest.is_file():
        artifact.add_file(str(manifest))
    run.log_artifact(artifact)


def main(argv=None):
    args = validate_args(parse_args(argv))
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    vec_env = make_training_vec_env(args)
    model = None
    run = None
    callback = None
    try:
        model = build_or_load_model(args, vec_env)
        expected_fingerprint = sha256_file(args.game_checkpoint)
        protected = protected_gameplay_state(model.policy)
        assert_gameplay_unchanged(
            model.policy,
            protected,
            expected_fingerprint=expected_fingerprint,
        )
        run = init_wandb(args)
        if not args.eval_only:
            callback = MetaResponseTrainingCallback(
                args,
                args.checkpoint,
                protected,
                expected_fingerprint,
                wandb_run=run,
            )
            learn_timesteps = int(args.timesteps)
            if args.timesteps_are_target:
                learn_timesteps = max(
                    int(args.timesteps) - int(model.num_timesteps), 0
                )
                if learn_timesteps == 0:
                    raise ValueError(
                        "checkpoint has already reached --timesteps target"
                    )
            model.learn(
                total_timesteps=learn_timesteps,
                callback=callback,
                reset_num_timesteps=args.resume is None,
            )

        assert_gameplay_unchanged(
            model.policy,
            protected,
            expected_fingerprint=expected_fingerprint,
        )
        if args.eval_only:
            evaluated_checkpoint = args.resume
        else:
            model.save(str(args.checkpoint))
            evaluated_checkpoint = args.checkpoint
        result = evaluate_response(
            model,
            args,
            # Model selection and independent audits use the identical paired
            # contexts and event schedules.
            eval_seed=args.eval_seed,
        )
        if callback is not None:
            result, output_paths = callback.record_evaluation(
                result,
                checkpoint_path=evaluated_checkpoint,
                evaluation_path=args.output,
                evaluation_kind="final",
            )
        else:
            result = add_evaluation_provenance(
                result,
                args,
                checkpoint_path=evaluated_checkpoint,
                evaluation_kind="eval_only",
            )
            result["metadata"]["selected_as_best"] = False
            output_paths = write_evaluation(args.output, result)
            final_metrics = evaluation_wandb_metrics(result)
            print(json.dumps(final_metrics, sort_keys=True), flush=True)
            if run is not None:
                run.log(final_metrics, step=int(model.num_timesteps))
        if run is not None:
            run.summary.update({
                "checkpoint_path": str(args.checkpoint),
                "evaluated_checkpoint_path": str(evaluated_checkpoint),
                "evaluated_checkpoint_sha256": result["metadata"][
                    "evaluated_checkpoint_sha256"
                ],
                "evaluation_path": str(output_paths["json"]),
                "fixed_contexts_csv": str(
                    output_paths["fixed_contexts_csv"]
                ),
                "episodes_csv": str(output_paths["episodes_csv"]),
                "events_csv": str(output_paths["events_csv"]),
                "gameplay_checkpoint_sha256": expected_fingerprint,
                "gameplay_drift": False,
                "controlled_role": args.role,
                "eval_only": bool(args.eval_only),
                "selection_metric": SELECTION_METRIC,
                "selection_rule": SELECTION_RULE,
                "selection_eval_seed": int(args.eval_seed),
                "selection_paired_contexts_and_schedules": True,
                "source_bundle_sha256": result["metadata"][
                    "source_bundle_sha256"
                ],
                "source_files_sha256": result["metadata"][
                    "source_files_sha256"
                ],
                "source_provenance_scope": result["metadata"][
                    "source_provenance_scope"
                ],
                **{
                    f"random/{key}": value
                    for key, value in result["random_context_summary"].items()
                    if isinstance(value, (int, float, bool))
                },
            })
            _log_artifact(run, args, output_paths, model)
    finally:
        if run is not None:
            run.finish()
        if model is not None and model.get_env() is not None:
            model.get_env().close()
        else:
            vec_env.close()


if __name__ == "__main__":
    main()
