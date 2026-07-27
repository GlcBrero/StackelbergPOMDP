"""Train and evaluate a full-trajectory Atari StackPOMDP leader with SB3 PPO.

The leader makes five event-indexed commitments during five explicit
zero-reward query steps.  The policy-level cache reuses each complete queried
action when its actor-identical trade observation recurs in a fresh reward
game against a frozen meta-response.  PPO receives all five queries, all 200
gameplay decisions, and all five paused trades.  Both players' Space Invaders
actions come from the canonical, immutable E0 checkpoint; PPO trains only the
leader's scalar price/threshold head and value network.

Examples
--------
Seller leader against a trained meta-buyer::

    python -m replication.atari.train_atari_stackpomdp_leader_sb3 \
      --leader-role seller \
      --response-checkpoint replication/atari/checkpoints/stackpomdp/meta_buyer.zip

Buyer leader against a trained meta-seller::

    python -m replication.atari.train_atari_stackpomdp_leader_sb3 \
      --leader-role buyer \
      --response-checkpoint replication/atari/checkpoints/stackpomdp/meta_seller.zip

Use ``--eval-only --resume CHECKPOINT`` to regenerate deterministic JSON and
CSV evaluation artifacts without further training.
"""

import argparse
from collections import deque
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


os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stackpomdp-matplotlib")
)
os.environ.setdefault("WANDB_START_METHOD", "thread")
# This machine has an incompatible Pillow in the user site.  PYTHONNOUSERSITE
# must normally be set before interpreter startup, so also remove the already
# populated entry when this module is launched directly.
_user_site = str(Path(site.getusersitepackages()).resolve())
sys.path[:] = [
    entry
    for entry in sys.path
    if not entry or str(Path(entry).resolve()) != _user_site
]

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    NUM_TRADE_EVENTS,
    SELLER,
    BilateralAtariConfig,
)
from stackelberg_pomdp.atari.stackpomdp_full_leader_env import (
    FullTraceStackPOMDPAtariLeaderEnv,
)
from stackelberg_pomdp.atari.stackpomdp_policy import (
    StackPOMDPAtariEconomicPolicy,
)
from stackelberg_pomdp.callbacks import FixPolicyActionsCallback


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCIBILITY_SOURCE_FILES = (
    "replication/atari/train_atari_stackpomdp_leader_sb3.py",
    "stackelberg_pomdp/atari/stackpomdp_env.py",
    "stackelberg_pomdp/atari/stackpomdp_full_leader_env.py",
    "stackelberg_pomdp/atari/stackpomdp_policy.py",
    "stackelberg_pomdp/atari/query_trace.py",
)
CANONICAL_E0 = (
    REPOSITORY_ROOT
    / "replication/atari/checkpoints/sb3/"
    "space_invaders_e0_ppo_seed1_10m_best.zip"
)
WANDB_PROJECT = "StackPOMDP"
WANDB_GROUP = "atari_stackpomdp"
PROTECTED_PREFIXES = ("features_extractor.", "game_action_net.")


def full_episode_transitions(gameplay_horizon):
    """Stored transitions in one no-skipping outer StackPOMDP episode."""

    return int(gameplay_horizon) + 2 * NUM_TRADE_EVENTS


def checkpoint_with_zip(path):
    path = Path(path).expanduser().resolve()
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def resolve_existing_checkpoint(path, label):
    candidate = Path(path).expanduser().resolve()
    alternatives = (candidate, Path(f"{candidate}.zip"))
    for alternative in alternatives:
        if alternative.is_file():
            return alternative
    raise FileNotFoundError(f"{label} checkpoint does not exist: {candidate}")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file_sha256s():
    """Hash the exact uncommitted sources that define a leader run."""

    result = {}
    for relative_path in REPRODUCIBILITY_SOURCE_FILES:
        source_path = REPOSITORY_ROOT / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"reproducibility source does not exist: {source_path}"
            )
        result[relative_path] = file_sha256(source_path)
    return result


def source_bundle_sha256(source_hashes):
    """Return one order-independent fingerprint for a source hash mapping."""

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
    }


def frozen_tensor_fingerprint(policy):
    """Hash every protected tensor so training drift can be detected exactly."""

    digest = hashlib.sha256()
    protected = 0
    for name, tensor in sorted(policy.state_dict().items()):
        if not name.startswith(PROTECTED_PREFIXES):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
        protected += 1
    if protected == 0:
        raise RuntimeError("policy has no protected E0 gameplay tensors")
    return digest.hexdigest()


def assert_frozen_gameplay(policy, expected_tensor_fingerprint):
    if not bool(policy.gameplay_ready.item()):
        raise RuntimeError("leader policy has no initialized E0 gameplay branch")
    actual = frozen_tensor_fingerprint(policy)
    if actual != expected_tensor_fingerprint:
        raise RuntimeError(
            "frozen E0 gameplay tensors drifted during economic training: "
            f"{actual} != {expected_tensor_fingerprint}"
        )
    for module in (policy.features_extractor, policy.game_action_net):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("a protected E0 gameplay parameter is trainable")


def _parse_fixed_event_steps(raw):
    if raw is None:
        return None
    values = tuple(
        int(value.strip()) for value in str(raw).split(",") if value.strip()
    )
    if len(values) != NUM_TRADE_EVENTS:
        raise ValueError("--fixed-event-steps must contain exactly five integers")
    return values


def bilateral_config(args, *, seed):
    return BilateralAtariConfig(
        seed=int(seed),
        gameplay_horizon=int(args.gameplay_horizon),
        event_tail_steps=int(args.event_tail_steps),
        num_trade_events=NUM_TRADE_EVENTS,
        seller_game_reward_scale=0.1,
        buyer_game_reward_scale=1.0,
        noop_max=30,
        frame_skip=4,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        max_frames=int(args.max_frames),
        rom_path=args.rom_path,
        fixed_event_steps=args.fixed_event_steps,
    ).resolved()


def make_leader_env(args, *, seed):
    return FullTraceStackPOMDPAtariLeaderEnv(
        leader_role=args.leader_role,
        response_checkpoint=args.response_checkpoint,
        game_checkpoint=args.game_checkpoint,
        config=bilateral_config(args, seed=seed),
        device=args.device,
    )


def make_training_vec_env(args):
    constructors = []
    for rank in range(args.num_envs):
        env_seed = args.seed + 10_000 * rank

        def constructor(env_seed=env_seed):
            return Monitor(make_leader_env(args, seed=env_seed))

        constructors.append(constructor)
    if args.num_envs == 1:
        return DummyVecEnv(constructors)
    return SubprocVecEnv(constructors, start_method=args.start_method)


def expected_follower_role(leader_role):
    return BUYER if leader_role == SELLER else SELLER


def validate_response_checkpoint(args):
    """Fail before training if the response or frozen gameplay is mismatched."""

    response = PPO.load(args.response_checkpoint, device=args.device)
    try:
        policy = response.policy
        expected_role = expected_follower_role(args.leader_role)
        actual_role = getattr(policy, "economic_role", None)
        if actual_role != expected_role:
            raise ValueError(
                f"{args.leader_role} leader requires a {expected_role} "
                f"meta-response, got {actual_role!r}"
            )
        gameplay_ready = getattr(policy, "gameplay_ready", None)
        if gameplay_ready is None or not bool(gameplay_ready.item()):
            raise ValueError("meta-response has no initialized E0 gameplay branch")
        expected_gameplay = file_sha256(args.game_checkpoint)
        actual_gameplay = getattr(policy, "gameplay_fingerprint", None)
        if actual_gameplay != expected_gameplay:
            raise ValueError(
                "meta-response and requested E0 checkpoint differ: "
                f"{actual_gameplay} != {expected_gameplay}"
            )
        return {
            "response_role": actual_role,
            "response_gameplay_sha256": actual_gameplay,
        }
    finally:
        del response


def _new_model(args, vec_env):
    model = PPO(
        StackPOMDPAtariEconomicPolicy,
        vec_env,
        policy_kwargs={
            "economic_role": args.leader_role,
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
        args.game_checkpoint, device=args.device
    )
    print(json.dumps({
        "initialized_frozen_gameplay": provenance,
        "leader_role": args.leader_role,
        "trainable_component": "economic_head_and_value_only",
    }, sort_keys=True), flush=True)
    return model


def _resumed_model(args, vec_env):
    model = PPO.load(
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
    if not isinstance(model.policy, StackPOMDPAtariEconomicPolicy):
        raise TypeError("--resume is not an Atari StackPOMDP economic policy")
    if model.policy.economic_role != args.leader_role:
        raise ValueError(
            f"resume role {model.policy.economic_role!r} does not match "
            f"--leader-role {args.leader_role!r}"
        )
    expected_gameplay = file_sha256(args.game_checkpoint)
    if model.policy.gameplay_fingerprint != expected_gameplay:
        raise ValueError(
            "resume checkpoint was initialized from a different E0 policy: "
            f"{model.policy.gameplay_fingerprint} != {expected_gameplay}"
        )
    model.policy.clear_obs_action_map()
    return model


def build_or_load_model(args, vec_env):
    model = _resumed_model(args, vec_env) if args.resume else _new_model(args, vec_env)
    model.policy.clear_obs_action_map()
    fingerprint = frozen_tensor_fingerprint(model.policy)
    assert_frozen_gameplay(model.policy, fingerprint)
    return model, fingerprint


def _scalar_episode_row(info, episode_index):
    events = tuple(info["events"])
    leader_actions = tuple(info["query_actions"])
    follower_actions = tuple(info["follower_actions"])
    row = {
        "episode": int(episode_index),
        "leader_role": info["leader_role"],
        "follower_role": info["follower_role"],
        "query_trace_sha256": info["query_trace_sha256"],
        "outer_transition_count": int(info["outer_transition_count"]),
        "query_transitions": int(info["query_transitions"]),
        "gameplay_transitions": int(info["gameplay_transitions"]),
        "trade_transitions": int(info["trade_transitions"]),
        "cache_hits": int(info["cache_hits"]),
    }
    scalar_fields = (
        "trade_opportunities",
        "bullets_arrived",
        "purchases",
        "payments",
        "seller_game_reward",
        "buyer_game_reward",
        "seller_reward",
        "buyer_reward",
        "leader_reward",
        "seller_shots_fired",
        "buyer_shots_fired",
        "seller_final_ammo",
        "buyer_final_ammo",
        "seller_bullet_error",
        "buyer_bullet_error",
        "seller_payoff_error",
        "buyer_payoff_error",
    )
    row.update({field: info[field] for field in scalar_fields})
    for event_index, event in enumerate(events):
        suffix = event_index + 1
        row[f"event_step_{suffix}"] = int(event["game_step"])
        row[f"price_{suffix}"] = float(event["price"])
        row[f"threshold_{suffix}"] = float(event["threshold"])
        row[f"accepted_{suffix}"] = int(event["accepted"])
        row[f"leader_game_action_{suffix}"] = float(
            leader_actions[event_index][0]
        )
        row[f"leader_action_{suffix}"] = float(
            leader_actions[event_index][1]
        )
        row[f"follower_game_action_{suffix}"] = float(
            follower_actions[event_index][0]
        )
        row[f"follower_action_{suffix}"] = float(
            follower_actions[event_index][1]
        )
    return row


def _json_episode(info, episode_index):
    row = _scalar_episode_row(info, episode_index)
    row["event_steps"] = [int(value) for value in info["event_steps"]]
    row["query_actions"] = [
        [float(value) for value in action]
        for action in info["query_actions"]
    ]
    row["follower_actions"] = [
        [float(value) for value in action]
        for action in info["follower_actions"]
    ]
    row["events"] = [dict(event) for event in info["events"]]
    return row


def _mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def trade_timing_diagnostics(events, gameplay_horizon):
    """Summarize when offers occur and when they are accepted.

    Time uses the same normalization supplied to the privileged critic:
    ``game_step / gameplay_horizon``.  Empty conditional groups are retained
    as ``None`` in JSON, but are omitted from scalar W&B logging.
    """

    horizon = float(gameplay_horizon)
    if horizon <= 0.0:
        raise ValueError("gameplay_horizon must be positive")
    events = tuple(events)
    result = {}
    for event_index in range(NUM_TRADE_EVENTS):
        indexed = [
            event
            for event in events
            if int(event["event_index"]) == event_index
        ]
        suffix = event_index + 1
        result[f"event_{suffix}_count"] = int(len(indexed))
        if indexed:
            steps = np.asarray(
                [event["game_step"] for event in indexed], dtype=np.float64
            )
            accepted = np.asarray(
                [event["accepted"] for event in indexed], dtype=np.float64
            )
            result[f"mean_event_{suffix}_step"] = float(steps.mean())
            result[f"mean_event_{suffix}_normalized_time"] = float(
                (steps / horizon).mean()
            )
            result[f"event_{suffix}_acceptance_rate"] = float(
                accepted.mean()
            )
        else:
            result[f"mean_event_{suffix}_step"] = None
            result[f"mean_event_{suffix}_normalized_time"] = None
            result[f"event_{suffix}_acceptance_rate"] = None

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
        bin_mask = (
            (normalized_times >= lower) & (normalized_times < upper)
        )
        result[f"{label}_event_count"] = int(bin_mask.sum())
        result[f"{label}_acceptance_rate"] = (
            float(accepted_mask[bin_mask].mean()) if bin_mask.any() else None
        )
    return result


def summarize_evaluation(rows, gameplay_horizon):
    scalar_fields = (
        "leader_reward",
        "seller_reward",
        "buyer_reward",
        "seller_game_reward",
        "buyer_game_reward",
        "payments",
        "purchases",
        "seller_shots_fired",
        "buyer_shots_fired",
        "seller_final_ammo",
        "buyer_final_ammo",
        "seller_bullet_error",
        "buyer_bullet_error",
        "seller_payoff_error",
        "buyer_payoff_error",
    )
    summary = {"episodes": len(rows), "deterministic": True}
    for field in scalar_fields:
        summary[f"mean_{field}"] = _mean(rows, field)
    summary["std_leader_reward"] = float(
        np.std([float(row["leader_reward"]) for row in rows])
    )
    summary["min_leader_reward"] = float(
        min(float(row["leader_reward"]) for row in rows)
    )
    summary["max_leader_reward"] = float(
        max(float(row["leader_reward"]) for row in rows)
    )
    all_prices = []
    all_thresholds = []
    all_acceptances = []
    for event_index in range(NUM_TRADE_EVENTS):
        suffix = event_index + 1
        prices = [float(row[f"price_{suffix}"]) for row in rows]
        thresholds = [float(row[f"threshold_{suffix}"]) for row in rows]
        accepted = [float(row[f"accepted_{suffix}"]) for row in rows]
        summary[f"event_{suffix}_mean_price"] = float(np.mean(prices))
        summary[f"event_{suffix}_mean_threshold"] = float(np.mean(thresholds))
        summary[f"event_{suffix}_acceptance_rate"] = float(np.mean(accepted))
        all_prices.extend(prices)
        all_thresholds.extend(thresholds)
        all_acceptances.extend(accepted)
    summary["mean_price"] = float(np.mean(all_prices))
    summary["mean_threshold"] = float(np.mean(all_thresholds))
    summary["acceptance_rate"] = float(np.mean(all_acceptances))
    events = []
    for row in rows:
        for event_index in range(NUM_TRADE_EVENTS):
            suffix = event_index + 1
            events.append({
                "event_index": event_index,
                "game_step": int(row[f"event_step_{suffix}"]),
                "accepted": bool(row[f"accepted_{suffix}"]),
            })
    summary.update(trade_timing_diagnostics(events, gameplay_horizon))
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation(path, result):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    episode_rows = [
        {key: value for key, value in episode.items() if not isinstance(value, list)}
        for episode in result["episodes"]
    ]
    episodes_path = path.with_suffix(".episodes.csv")
    write_csv(episodes_path, episode_rows)

    event_rows = []
    for episode in result["episodes"]:
        for event in episode["events"]:
            event_rows.append({
                "episode": episode["episode"],
                "leader_role": episode["leader_role"],
                "follower_role": episode["follower_role"],
                "leader_reward": episode["leader_reward"],
                "seller_reward": episode["seller_reward"],
                "buyer_reward": episode["buyer_reward"],
                **event,
            })
    events_path = path.with_suffix(".trade_events.csv")
    write_csv(events_path, event_rows)
    return episodes_path, events_path


def evaluate_model(model, args, *, checkpoint, episodes=None, output=None):
    evaluation_episodes = int(episodes or args.eval_episodes)
    env = make_leader_env(args, seed=args.eval_seed)
    rows = []
    expected_steps = full_episode_transitions(args.gameplay_horizon)
    model.policy.fix_policy_actions()
    try:
        for episode_index in range(evaluation_episodes):
            model.policy.clear_obs_action_map()
            observation = env.reset()
            done = False
            final_info = None
            step_count = 0
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, done, info = env.step(action)
                step_count += 1
                if done:
                    final_info = info
            if step_count != expected_steps:
                raise RuntimeError(
                    "outer episode stored the wrong number of transitions: "
                    f"{step_count} != {expected_steps}"
                )
            rows.append(_json_episode(final_info, episode_index))
    finally:
        model.policy.clear_obs_action_map()
        env.close()

    source_metadata = reproducibility_source_metadata()
    result = {
        "schema_version": 1,
        "algorithm": "SB3-PPO",
        "architecture": "full_trace_frozen_e0_plus_economic_head",
        "leader_role": args.leader_role,
        "follower_role": expected_follower_role(args.leader_role),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "response_checkpoint": str(args.response_checkpoint),
        "game_checkpoint": str(args.game_checkpoint),
        "gameplay_sha256": model.policy.gameplay_fingerprint,
        "evaluation_seed": int(args.eval_seed),
        **source_metadata,
        "summary": summarize_evaluation(rows, args.gameplay_horizon),
        "episodes": rows,
    }
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else Path(checkpoint).expanduser().resolve().with_suffix(".evaluation.json")
    )
    episodes_path, events_path = write_evaluation(output_path, result)
    print(json.dumps({
        "evaluation": str(output_path),
        "episodes_csv": str(episodes_path),
        "trade_events_csv": str(events_path),
        "summary": result["summary"],
    }, sort_keys=True), flush=True)
    return result, output_path, episodes_path, events_path


def _episode_metrics(infos, gameplay_horizon):
    completed = [info for info in infos if "leader_reward" in info]
    if not completed:
        return {}

    def mean(field):
        return float(np.mean([float(info[field]) for info in completed]))

    metrics = {}
    scalar_fields = (
        "leader_reward",
        "seller_reward",
        "buyer_reward",
        "seller_game_reward",
        "buyer_game_reward",
        "payments",
        "purchases",
        "trade_opportunities",
        "bullets_arrived",
        "seller_shots_fired",
        "buyer_shots_fired",
        "seller_final_ammo",
        "buyer_final_ammo",
        "seller_bullet_error",
        "buyer_bullet_error",
        "seller_payoff_error",
        "buyer_payoff_error",
    )
    metrics.update({f"train/{field}": mean(field) for field in scalar_fields})
    prices = [float(event["price"]) for info in completed for event in info["events"]]
    thresholds = [
        float(event["threshold"]) for info in completed for event in info["events"]
    ]
    accepted = [
        float(event["accepted"]) for info in completed for event in info["events"]
    ]
    metrics.update({
        "train/mean_price": float(np.mean(prices)),
        "train/mean_threshold": float(np.mean(thresholds)),
        "train/acceptance_rate": float(np.mean(accepted)),
        "train/episode_length": mean("outer_transition_count"),
        "train/query_transitions": mean("query_transitions"),
        "train/gameplay_transitions": mean("gameplay_transitions"),
        "train/trade_transitions": mean("trade_transitions"),
        "train/policy_cache_hits": mean("cache_hits"),
    })
    for event_index in range(NUM_TRADE_EVENTS):
        event_group = [info["events"][event_index] for info in completed]
        suffix = event_index + 1
        metrics[f"train/event_{suffix}_price"] = float(
            np.mean([event["price"] for event in event_group])
        )
        metrics[f"train/event_{suffix}_threshold"] = float(
            np.mean([event["threshold"] for event in event_group])
        )
        metrics[f"train/event_{suffix}_acceptance_rate"] = float(
            np.mean([event["accepted"] for event in event_group])
        )
    timing = trade_timing_diagnostics(
        [event for info in completed for event in info["events"]],
        gameplay_horizon,
    )
    metrics.update({
        f"train/{key}": value
        for key, value in timing.items()
        if value is not None
    })
    return metrics


class LeaderTrainingCallback(BaseCallback):
    def __init__(
            self,
            args,
            checkpoint_path,
            frozen_fingerprint,
            wandb_run=None,
    ):
        super().__init__(verbose=0)
        self.args = args
        self.checkpoint_path = checkpoint_path
        self.best_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_best{checkpoint_path.suffix}"
        )
        self.frozen_fingerprint = frozen_fingerprint
        self.wandb_run = wandb_run
        self.recent_episodes = deque(maxlen=100)
        self.next_log = args.log_every
        self.next_checkpoint = args.checkpoint_every
        self.next_evaluation = args.eval_every
        self.best_leader_reward = -np.inf
        self.started = time.time()
        self.starting_timesteps = 0
        self.final_artifacts = None

    def _on_training_start(self):
        self.starting_timesteps = int(self.num_timesteps)
        for attribute, interval in (
                ("next_log", self.args.log_every),
                ("next_checkpoint", self.args.checkpoint_every),
                ("next_evaluation", self.args.eval_every),
        ):
            setattr(
                self,
                attribute,
                (int(self.num_timesteps) // interval + 1) * interval,
            )

    def _collect_terminal_infos(self):
        for info in self.locals.get("infos", []):
            if "leader_reward" in info:
                self.recent_episodes.append(info)

    def _log(self, metrics):
        metrics.update({
            "total_timesteps": int(self.num_timesteps),
            "train/learning_rate": float(
                self.model.policy.optimizer.param_groups[0]["lr"]
            ),
            "train/seed": int(self.args.seed),
        })
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=int(self.num_timesteps))
            self.wandb_run.summary.update(metrics)

    def _rolling_metrics(self):
        metrics = _episode_metrics(
            tuple(self.recent_episodes), self.args.gameplay_horizon
        )
        elapsed = max(time.time() - self.started, 1.0e-9)
        metrics["train/steps_per_second"] = float(
            max(int(self.num_timesteps) - self.starting_timesteps, 0) / elapsed
        )
        return metrics

    def _save(self, path):
        assert_frozen_gameplay(self.model.policy, self.frozen_fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        if self.wandb_run is not None:
            self.wandb_run.summary["checkpoint_path"] = str(path)
        print(f"checkpoint={path}", flush=True)

    def _evaluate(self, checkpoint_path, output=None):
        assert_frozen_gameplay(self.model.policy, self.frozen_fingerprint)
        result, output_path, episodes_path, events_path = evaluate_model(
            self.model,
            self.args,
            checkpoint=checkpoint_path,
            output=output,
        )
        metrics = {
            f"evaluation/{key}": value
            for key, value in result["summary"].items()
            if isinstance(value, (int, float, bool))
        }
        self._log(metrics)
        mean_reward = float(result["summary"]["mean_leader_reward"])
        if mean_reward > self.best_leader_reward:
            self.best_leader_reward = mean_reward
            self._save(self.best_path)
            best_output = self.best_path.with_suffix(".evaluation.json")
            best_result = dict(result)
            best_result["checkpoint"] = str(self.best_path)
            best_episode_path, best_event_path = write_evaluation(
                best_output, best_result
            )
            if self.wandb_run is not None:
                self.wandb_run.summary.update({
                    "best/mean_leader_reward": mean_reward,
                    "best/checkpoint": str(self.best_path),
                    "best/evaluation": str(best_output),
                    "best/episodes_csv": str(best_episode_path),
                    "best/trade_events_csv": str(best_event_path),
                })
        return result, output_path, episodes_path, events_path

    def _on_step(self):
        self._collect_terminal_infos()
        episode_finished = any(
            bool(value) for value in self.locals.get("dones", [])
        )
        if self.num_timesteps >= self.next_log:
            self._log(self._rolling_metrics())
            while self.next_log <= self.num_timesteps:
                self.next_log += self.args.log_every
        if self.num_timesteps >= self.next_checkpoint and episode_finished:
            self._save(self.checkpoint_path)
            while self.next_checkpoint <= self.num_timesteps:
                self.next_checkpoint += self.args.checkpoint_every
        # Evaluation clears and repopulates the policy cache, so it may only
        # run after the training environment has completed its outer episode.
        if self.num_timesteps >= self.next_evaluation and episode_finished:
            step_path = self.checkpoint_path.with_name(
                f"{self.checkpoint_path.stem}_step{int(self.num_timesteps)}.zip"
            )
            self._save(step_path)
            self._evaluate(step_path)
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += self.args.eval_every
        return True

    def _on_training_end(self):
        self._save(self.checkpoint_path)
        output = self.args.output or self.checkpoint_path.with_suffix(
            ".evaluation.json"
        )
        self.final_artifacts = self._evaluate(
            self.checkpoint_path, output=output
        )
        if self.wandb_run is not None:
            _, output_path, episodes_path, events_path = self.final_artifacts
            self.wandb_run.summary.update({
                "checkpoint_path": str(self.checkpoint_path),
                "evaluation_path": str(output_path),
                "evaluation_episodes_csv": str(episodes_path),
                "evaluation_trade_events_csv": str(events_path),
            })
        assert_frozen_gameplay(self.model.policy, self.frozen_fingerprint)


def make_training_callback(
        args,
        checkpoint_path,
        frozen_fingerprint,
        *,
        wandb_run=None,
):
    """Pair experiment logging with the mandatory policy action cache."""

    leader_callback = LeaderTrainingCallback(
        args,
        checkpoint_path,
        frozen_fingerprint,
        wandb_run=wandb_run,
    )
    callbacks = CallbackList([
        FixPolicyActionsCallback(),
        leader_callback,
    ])
    return leader_callback, callbacks


def _wandb_config(args, checkpoint_path, response_metadata):
    return {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **response_metadata,
        "algorithm": "SB3-PPO",
        "architecture": "full_trace_frozen_e0_plus_economic_head",
        "checkpoint_path": str(checkpoint_path),
        "follower_role": expected_follower_role(args.leader_role),
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "query_steps": NUM_TRADE_EVENTS,
        "query_rewards": 0.0,
        "query_steps_in_rollout_buffer": True,
        "gameplay_steps_in_rollout_buffer": int(args.gameplay_horizon),
        "trade_steps_in_rollout_buffer": NUM_TRADE_EVENTS,
        "outer_episode_transitions": full_episode_transitions(
            args.gameplay_horizon
        ),
        "exclude_from_buffer": False,
        "leader_action_cached_by_environment": False,
        "policy_action_cache": True,
        "policy_action_cache_scope": "complete_actor_observation_and_full_action",
        "gameplay_trainable": False,
        "gameplay_policy": "E0 deterministic argmax",
        "seller_game_reward_scale": 0.1,
        "buyer_game_reward_scale": 1.0,
        "payments_immediate": True,
        "clip_game_rewards": True,
        "trade_opportunities": NUM_TRADE_EVENTS,
        "exogenous_event_timing": True,
        **reproducibility_source_metadata(),
    }


def init_wandb(args, checkpoint_path, response_metadata):
    if not args.wandb:
        return None
    import wandb

    config = _wandb_config(args, checkpoint_path, response_metadata)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type=args.wandb_job_type or f"{args.leader_role}_leader",
        name=args.wandb_name,
        config=config,
    )
    run.define_metric("total_timesteps")
    run.define_metric("train/*", step_metric="total_timesteps")
    run.define_metric("evaluation/*", step_metric="total_timesteps")
    run.summary.update({
        "algorithm": "SB3-PPO",
        "seed": int(args.seed),
        "leader_role": args.leader_role,
        "checkpoint_path": str(checkpoint_path),
        "response_checkpoint": str(args.response_checkpoint),
        "game_checkpoint": str(args.game_checkpoint),
        "source_bundle_sha256": config["source_bundle_sha256"],
        "source_files_sha256": config["source_files_sha256"],
    })
    print(f"wandb_url={run.url}", flush=True)
    return run


def _default_checkpoint(args):
    return (
        REPOSITORY_ROOT
        / "replication/atari/checkpoints/stackpomdp"
        / (
            f"space_invaders_{args.leader_role}_leader_ppo_"
            f"seed{args.seed}_{args.timesteps}.zip"
        )
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-role", choices=[SELLER, BUYER], required=True)
    parser.add_argument("--response-checkpoint", required=True)
    parser.add_argument("--game-checkpoint", default=str(CANONICAL_E0))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--timesteps-are-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="when resuming, treat --timesteps as the desired final timestep",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", help="final deterministic evaluation JSON")
    parser.add_argument("--resume")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--start-method",
        choices=["spawn", "forkserver", "fork"],
        default="spawn",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=210,
        help=(
            "rollout steps per environment; must equal five queries plus "
            "--gameplay-horizon plus five paused trades"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=210)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--economic-hidden", type=int, default=64)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--gameplay-horizon", type=int, default=200)
    parser.add_argument("--event-tail-steps", type=int, default=50)
    parser.add_argument("--fixed-event-steps")
    parser.add_argument("--max-frames", type=int, default=100_000)
    parser.add_argument("--rom-path")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=200_003)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--device", default="cpu")
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
    positive = (
        "timesteps",
        "num_envs",
        "n_steps",
        "batch_size",
        "n_epochs",
        "economic_hidden",
        "critic_hidden",
        "gameplay_horizon",
        "event_tail_steps",
        "max_frames",
        "eval_episodes",
        "eval_every",
        "checkpoint_every",
        "log_every",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size <= 1:
        raise ValueError("--batch-size must exceed one")
    expected_steps = full_episode_transitions(args.gameplay_horizon)
    if args.n_steps != expected_steps:
        raise ValueError(
            "--n-steps must equal one complete no-skipping outer episode "
            f"(five queries + {args.gameplay_horizon} gameplay + five "
            f"trades = {expected_steps}); got {args.n_steps}"
        )
    rollout_size = args.n_steps * args.num_envs
    if rollout_size % args.batch_size:
        raise ValueError(
            "n_steps * num_envs must be divisible by batch_size; "
            f"got {rollout_size} and {args.batch_size}"
        )
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.entropy_coeff < 0.0:
        raise ValueError("--entropy-coeff must be nonnegative")
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")

    args.fixed_event_steps = _parse_fixed_event_steps(args.fixed_event_steps)
    args.game_checkpoint = str(
        resolve_existing_checkpoint(args.game_checkpoint, "E0 gameplay")
    )
    args.response_checkpoint = str(
        resolve_existing_checkpoint(args.response_checkpoint, "meta-response")
    )
    if args.resume:
        args.resume = str(resolve_existing_checkpoint(args.resume, "resume"))
    if args.rom_path:
        args.rom_path = str(Path(args.rom_path).expanduser().resolve())
    # Resolve and validate all schedule invariants before loading Atari.
    bilateral_config(args, seed=args.seed)


def _default_wandb_name(args):
    return (
        f"atari_stackpomdp_{args.leader_role}_leader_ppo_"
        f"seed{args.seed}_{args.timesteps}_local"
    )


def _log_artifact(run, args, checkpoint_path, artifacts, *, total_timesteps):
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(
        f"atari-stackpomdp-{args.leader_role}-leader-seed{args.seed}",
        type="model",
        metadata={
            "algorithm": "SB3-PPO",
            "leader_role": args.leader_role,
            "seed": int(args.seed),
            "timesteps": int(total_timesteps),
            "gameplay_sha256": artifacts[0].get("gameplay_sha256"),
            "source_bundle_sha256": artifacts[0].get(
                "source_bundle_sha256"
            ),
            "source_files_sha256": artifacts[0].get(
                "source_files_sha256"
            ),
        },
    )
    for path in (checkpoint_path, *artifacts[1:]):
        path = Path(path)
        if path.is_file():
            artifact.add_file(str(path))
    run.log_artifact(artifact)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if args.wandb_name is None:
        args.wandb_name = _default_wandb_name(args)
    checkpoint_path = checkpoint_with_zip(args.checkpoint or _default_checkpoint(args))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    response_metadata = validate_response_checkpoint(args)

    if args.eval_only:
        model = PPO.load(args.resume, device=args.device)
        try:
            if not isinstance(model.policy, StackPOMDPAtariEconomicPolicy):
                raise TypeError("--resume is not an Atari StackPOMDP policy")
            if model.policy.economic_role != args.leader_role:
                raise ValueError("--leader-role does not match --resume")
            expected_gameplay = file_sha256(args.game_checkpoint)
            if model.policy.gameplay_fingerprint != expected_gameplay:
                raise ValueError(
                    "evaluation checkpoint was initialized from a different "
                    "E0 gameplay policy"
                )
            expected = frozen_tensor_fingerprint(model.policy)
            assert_frozen_gameplay(model.policy, expected)
            evaluate_model(
                model,
                args,
                checkpoint=args.resume,
                output=args.output,
            )
        finally:
            del model
        return

    vec_env = make_training_vec_env(args)
    model = None
    run = None
    try:
        model, frozen_fingerprint = build_or_load_model(args, vec_env)
        run = init_wandb(args, checkpoint_path, response_metadata)
        leader_callback, callback = make_training_callback(
            args,
            checkpoint_path,
            frozen_fingerprint,
            wandb_run=run,
        )
        learn_timesteps = int(args.timesteps)
        if args.timesteps_are_target:
            learn_timesteps = max(
                int(args.timesteps) - int(model.num_timesteps), 0
            )
            if learn_timesteps == 0:
                raise ValueError(
                    "resume checkpoint has already reached --timesteps target"
                )
        model.learn(
            total_timesteps=learn_timesteps,
            callback=callback,
            reset_num_timesteps=args.resume is None,
        )
        assert_frozen_gameplay(model.policy, frozen_fingerprint)
        if run is not None and leader_callback.final_artifacts is not None:
            run.summary["total_timesteps"] = int(model.num_timesteps)
            run.summary["gameplay_tensor_fingerprint"] = frozen_fingerprint
            _log_artifact(
                run,
                args,
                checkpoint_path,
                leader_callback.final_artifacts,
                total_timesteps=model.num_timesteps,
            )
    finally:
        if run is not None:
            run.finish()
        if model is not None and model.get_env() is not None:
            model.get_env().close()
        else:
            vec_env.close()


if __name__ == "__main__":
    main()
