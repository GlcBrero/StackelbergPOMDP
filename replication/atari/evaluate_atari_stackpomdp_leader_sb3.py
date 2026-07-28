"""Screen, select, and confirm clean Atari E2 leader checkpoints.

Every candidate is evaluated deterministically against one explicitly supplied,
frozen, opposite-role E1 response on the same 20 episode seeds.  A candidate is
eligible for selection only when every episode satisfies the complete E2
protocol: five event-only queries, 200 gameplay transitions, five actor-identical
cached trade replays, exact bilateral accounting, and one deterministic query
trace/commitment.  The best eligible checkpoint is copied to a collision-safe
alias and then confirmed on 100 disjoint seeds.

Selection ties are resolved, in order, by higher median leader payoff, higher
minimum leader payoff, lower payoff standard deviation, fewer training
timesteps, and finally lexicographically smaller checkpoint SHA-256.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stackpomdp-matplotlib")
)

from replication.atari.sb3_common import (
    ACTOR_LOSS_MODES,
    STANDARD_ACTOR_LOSS_MODE,
    ScaledLearningRatePPO,
    model_actor_loss_mode,
    model_economic_initialization,
    write_csv,
    write_json,
)
from replication.atari.train_atari_stackpomdp_leader_sb3 import (
    E2_PROVENANCE_ATTRIBUTE,
    e2_implementation_provenance,
    validate_e2_provenance_manifest,
)
from stackelberg_pomdp.atari.core import default_rom_path
from stackelberg_pomdp.atari.meta_response import make_stackpomdp_atari_leader_env
from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTION_MASK,
    ACTOR_STATE,
    CACHED_TRADE_REPLAY,
    GAMEPLAY,
    IMAGE,
    LEADER_QUERY,
    NUM_TRADE_EVENTS,
    actor_observation,
    canonical_leader_state,
)
from stackelberg_pomdp.atari.schedule import ExactFiveEventSchedule
from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
)
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "replication/atari/results/e2_selections"
PROTOCOL_ATOL = 1.0e-6
EVALUATOR_NAME = "clean_atari_e2_selector_v1"
CANONICAL_GAMEPLAY_HORIZON = 200
SCREEN_EPISODES = 20
CONFIRMATION_EPISODES = 100
E1_BUYER_INIT_MEAN = 0.95
E1_BUYER_INIT_CONCENTRATION = 10.0
E1_SELLER_INIT_MEAN = 0.5
E1_SELLER_INIT_CONCENTRATION = 2.0
E2_INIT_MEAN = 0.5
E2_INIT_CONCENTRATION = 2.0
SELECTION_RULE = (
    "exclude any checkpoint with a screen protocol violation",
    "maximize mean leader payoff",
    "maximize median leader payoff",
    "maximize minimum leader payoff",
    "minimize leader-payoff standard deviation",
    "prefer fewer training timesteps",
    "prefer lexicographically smaller checkpoint SHA-256",
)


def opposite_role(role):
    if role == SELLER:
        return BUYER
    if role == BUYER:
        return SELLER
    raise ValueError(f"unknown Atari role: {role!r}")


def _checkpoint_path(raw, *, label="checkpoint"):
    path = Path(raw).expanduser()
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")


def environment_config(args):
    """Return the exact actor-visible and economic environment contract."""

    fixed = getattr(args, "fixed_event_steps", None)
    rom_path = getattr(args, "rom_path", None)
    rom = (
        Path(rom_path).expanduser().resolve()
        if rom_path is not None
        else Path(default_rom_path()).resolve()
    )
    if not rom.is_file():
        raise FileNotFoundError(f"Space Invaders ROM does not exist: {rom}")
    rom_hash = checkpoint_sha256(rom)
    return {
        "leader_role": str(args.leader_role),
        "follower_role": opposite_role(args.leader_role),
        "gameplay_horizon": int(args.gameplay_horizon),
        "event_tail_steps": int(args.event_tail_steps),
        "fixed_event_steps": None if fixed is None else list(fixed),
        "seller_game_reward_scale": 0.1,
        "buyer_game_reward_scale": 1.0,
        "noop_max": int(args.noop_max),
        "frame_skip": 4,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
        "max_frames": int(args.max_frames),
        "rom_path": str(rom),
        "rom_sha256": rom_hash,
        "deterministic_actor": True,
        "leader_action_cache": True,
        "leader_economic_input": "event_only",
        "response_economic_input": "full",
        "response_algorithm": "frozen_meta_policy",
    }


def environment_config_sha256(config):
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def _load_model(path, *, device, expected_sha256=None):
    before = checkpoint_sha256(path)
    if expected_sha256 is not None and before != expected_sha256:
        raise RuntimeError("checkpoint bytes changed before model loading")
    model = ScaledLearningRatePPO.load(str(path), device=device)
    if checkpoint_sha256(path) != before:
        raise RuntimeError("checkpoint bytes changed while the model was loaded")
    return model


def load_e2_checkpoint(path, *, leader_role, device="cpu"):
    """Load and validate one deterministic E2 leader candidate."""

    model = _load_model(path, device=device)
    policy = model.policy
    if not isinstance(policy, StackPOMDPAtariPolicy):
        raise TypeError(f"{path} is not a clean composite Atari checkpoint")
    if policy.economic_role != leader_role:
        raise ValueError(
            f"{path} is a {policy.economic_role!r} policy, not {leader_role!r}"
        )
    if policy.economic_input_mode != "event_only":
        raise ValueError(
            f"{path} is not an event-only E2 leader "
            f"(economic_input_mode={policy.economic_input_mode!r})"
        )
    return model


def validate_candidate_provenance(model, *, response_hash, config):
    """Bind an E2 model to its exact E1 response and reward-game protocol."""

    raw_manifest = getattr(model, E2_PROVENANCE_ATTRIBUTE, None)
    if raw_manifest is None:
        raise ValueError("E2 candidate has no embedded provenance manifest")
    manifest = validate_e2_provenance_manifest(raw_manifest)
    scientific = manifest.get("scientific_config", {})
    recorded_environment = scientific.get("environment", {})
    recorded_protocol = scientific.get("protocol", {})
    recorded_optimization = scientific.get("optimization", {})
    recorded_initialization = scientific.get("initialization", {})
    recorded_response = manifest.get("artifacts", {}).get(
        "frozen_response", {}
    )
    expected_environment = {
        "gameplay_horizon": config["gameplay_horizon"],
        "event_tail_steps": config["event_tail_steps"],
        "fixed_event_steps": config["fixed_event_steps"],
        "seller_game_reward_scale": config["seller_game_reward_scale"],
        "buyer_game_reward_scale": config["buyer_game_reward_scale"],
        "noop_max": config["noop_max"],
        "frame_skip": config["frame_skip"],
        "frame_stack": config["frame_stack"],
        "episodic_life": config["episodic_life"],
        "clip_game_rewards": config["clip_game_rewards"],
        "max_frames": config["max_frames"],
        "rom_sha256": config["rom_sha256"],
    }
    environment_mismatches = {
        key: {
            "expected": expected,
            "recorded": recorded_environment.get(key),
        }
        for key, expected in expected_environment.items()
        if recorded_environment.get(key) != expected
    }
    if environment_mismatches:
        raise ValueError(
            "E2 candidate environment provenance does not match evaluation: "
            f"{environment_mismatches}"
        )
    expected_protocol = {
        "trade_events": NUM_TRADE_EVENTS,
        "query_transitions": NUM_TRADE_EVENTS,
        "cached_trade_replays": NUM_TRADE_EVENTS,
        "outer_episode_transitions": (
            config["gameplay_horizon"] + 2 * NUM_TRADE_EVENTS
        ),
        "policy_action_cache": True,
        "leader_economic_input": "event_only",
        "response_economic_input": "full",
        "response_algorithm": "frozen_meta_policy",
    }
    protocol_mismatches = {
        key: {"expected": expected, "recorded": recorded_protocol.get(key)}
        for key, expected in expected_protocol.items()
        if recorded_protocol.get(key) != expected
    }
    if protocol_mismatches:
        raise ValueError(
            "E2 candidate protocol provenance does not match evaluation: "
            f"{protocol_mismatches}"
        )
    if scientific.get("leader_role") != config["leader_role"]:
        raise ValueError("E2 candidate provenance has the wrong leader role")
    if scientific.get("follower_role") != config["follower_role"]:
        raise ValueError("E2 candidate provenance has the wrong follower role")
    if scientific.get("implementation") != e2_implementation_provenance():
        raise ValueError(
            "E2 candidate implementation provenance does not match the "
            "current evaluator runtime"
        )
    policy = model.policy
    actual_actor_loss_mode = model_actor_loss_mode(model)
    actual_economic_initialization = model_economic_initialization(
        model,
        default_mean=E2_INIT_MEAN,
        default_concentration=E2_INIT_CONCENTRATION,
    )
    recorded_actor_loss_mode = str(recorded_optimization.get(
        "actor_loss_mode", STANDARD_ACTOR_LOSS_MODE
    ))
    if recorded_actor_loss_mode not in ACTOR_LOSS_MODES:
        raise ValueError("E2 candidate provenance has an unknown actor loss mode")
    if actual_actor_loss_mode not in ACTOR_LOSS_MODES:
        raise ValueError("E2 candidate checkpoint has an unknown actor loss mode")
    if actual_actor_loss_mode != recorded_actor_loss_mode:
        raise ValueError(
            "E2 candidate actor loss mode does not match its provenance"
        )
    recorded_economic_initialization = {
        "mean": float(recorded_initialization.get(
            "economic_head_beta_mean", E2_INIT_MEAN
        )),
        "concentration": float(recorded_initialization.get(
            "economic_head_beta_concentration", E2_INIT_CONCENTRATION
        )),
    }
    if actual_economic_initialization != recorded_economic_initialization:
        raise ValueError(
            "E2 candidate economic initialization does not match its provenance"
        )
    actual_leader_policy = {
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "economic_role": policy.economic_role,
        "economic_input_mode": policy.economic_input_mode,
        "visual_features": int(policy.visual_features),
        "state_features": int(policy.state_features),
        "economic_hidden": int(policy.economic_hidden),
        "critic_hidden": int(policy.critic_hidden),
        "pretrained_lr_scale": float(policy.pretrained_lr_scale),
        "game_action_count": int(policy.game_action_count),
        "actor_loss_mode": actual_actor_loss_mode,
        "economic_head_initialization": actual_economic_initialization,
    }
    recorded_leader_policy = dict(scientific.get("leader_policy", {}))
    recorded_leader_policy.setdefault(
        "actor_loss_mode", recorded_actor_loss_mode
    )
    recorded_leader_policy.setdefault(
        "economic_head_initialization", recorded_economic_initialization
    )
    if recorded_leader_policy != actual_leader_policy:
        raise ValueError(
            "E2 candidate policy architecture does not match its provenance"
        )
    recorded_target_kl = recorded_optimization.get("target_kl")
    actual_target_kl = getattr(model, "target_kl", None)
    for label, value in (
            ("provenance", recorded_target_kl),
            ("checkpoint", actual_target_kl),
    ):
        if value is not None and (
                not np.isfinite(float(value)) or float(value) <= 0.0
        ):
            raise ValueError(f"E2 candidate {label} has an invalid target KL")
    normalized_recorded_target_kl = (
        None if recorded_target_kl is None else float(recorded_target_kl)
    )
    normalized_actual_target_kl = (
        None if actual_target_kl is None else float(actual_target_kl)
    )
    if normalized_actual_target_kl != normalized_recorded_target_kl:
        raise ValueError(
            "E2 candidate target KL does not match its provenance"
        )
    if recorded_response.get("sha256") != response_hash:
        raise ValueError(
            "E2 candidate was trained against different frozen E1 response bytes"
        )
    response_policy = recorded_response.get("policy", {})
    if (
            response_policy.get("economic_role") != config["follower_role"]
            or response_policy.get("economic_input_mode") != "full"
    ):
        raise ValueError("E2 candidate provenance has an invalid E1 response role")
    return manifest


def load_e1_response(
        path,
        *,
        leader_role,
        device="cpu",
        expected_sha256=None,
):
    """Load and validate the explicitly supplied opposite-role E1 response."""

    model = _load_model(
        path, device=device, expected_sha256=expected_sha256
    )
    policy = model.policy
    expected = opposite_role(leader_role)
    if not isinstance(policy, StackPOMDPAtariPolicy):
        raise TypeError(f"{path} is not a clean composite Atari checkpoint")
    if policy.economic_role != expected:
        raise ValueError(
            f"{leader_role} leader requires a {expected} response, got "
            f"{policy.economic_role!r}"
        )
    if policy.economic_input_mode != "full":
        raise ValueError("the frozen E1 response must use the full actor state")
    policy.set_training_mode(False)
    for parameter in policy.parameters():
        parameter.requires_grad = False
    return model


def make_e2_env(args, *, seed, response_model):
    config = BilateralAtariConfig(
        seed=int(seed),
        gameplay_horizon=int(args.gameplay_horizon),
        event_tail_steps=int(args.event_tail_steps),
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
    )
    return make_stackpomdp_atari_leader_env(
        leader_role=args.leader_role,
        response_checkpoint=args.response_checkpoint,
        config=config,
        response_model_factory=lambda path, device: response_model,
        device=args.device,
    )


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


def actor_observation_sha256(observation):
    """Hash exactly the fields entering the actor and action-cache key."""

    digest = hashlib.sha256()
    for name, value in actor_observation(observation).items():
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _action_list(action):
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"E2 policy produced an invalid full action: {action!r}")
    return [float(value) for value in values]


def evaluate_e2_model(
        model,
        env_factory,
        *,
        episodes,
        seed_start,
        checkpoint_path,
        checkpoint_hash,
        response_hash,
        config_hash,
        provenance_fingerprint,
        phase,
):
    """Run deterministic episodes while retaining complete protocol rows."""

    if not hasattr(model.policy, "fix_policy_actions"):
        raise TypeError("E2 leader policy does not implement action caching")
    model.policy.fix_policy_actions()
    episode_rows = []
    transition_rows = []
    decision_rows = []

    for episode in range(int(episodes)):
        if hasattr(model.policy, "clear_obs_action_map"):
            model.policy.clear_obs_action_map()
        evaluation_seed = int(seed_start) + episode
        env = env_factory(episode)
        try:
            observation = env.reset()
            done = False
            total_reward = 0.0
            transition_index = 0
            terminal_info = {}
            while not done:
                observation_hash = actor_observation_sha256(observation)
                action_credit = _jsonable(np.asarray(observation[ACTION_CREDIT]))
                actor_state_values = _jsonable(np.asarray(
                    observation[ACTOR_STATE]
                ))
                action_mask_values = _jsonable(np.asarray(
                    observation[ACTION_MASK]
                ))
                actor_image_nonzero = int(np.count_nonzero(
                    np.asarray(observation[IMAGE])
                ))
                action, _ = model.predict(observation, deterministic=True)
                requested_action = _action_list(action)
                observation, reward, done, terminal_info = env.step(action)
                reward = float(reward)
                total_reward += reward
                info = _jsonable(dict(terminal_info))
                substep = str(info.get("substep_type", ""))
                row = {
                    "phase": phase,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "response_sha256": response_hash,
                    "environment_config_sha256": config_hash,
                    "e2_provenance_fingerprint": provenance_fingerprint,
                    "evaluation_episode": episode,
                    "evaluation_seed": evaluation_seed,
                    "transition_index": transition_index,
                    "substep_type": substep,
                    "reward": reward,
                    "cumulative_return": total_reward,
                    "done": bool(done),
                    "is_reward_phase": bool(info.get("is_reward_phase", False)),
                    "reward_generated": bool(info.get("reward_generated", False)),
                    "emulator_advanced": bool(info.get("emulator_advanced", False)),
                    "action_credit": action_credit,
                    "actor_state": actor_state_values,
                    "action_mask": action_mask_values,
                    "actor_image_nonzero": actor_image_nonzero,
                    "actor_observation_sha256": observation_hash,
                    "requested_action": requested_action,
                    "query_index": info.get("query_index"),
                    "event_index": info.get("event_index"),
                    "cache_hit": bool(info.get("cache_hit", False)),
                    "leader_executed_action": info.get("leader_executed_action"),
                    "follower_action": info.get("follower_action"),
                    "game_step": info.get("game_step"),
                    "next_event": info.get("next_event"),
                }
                transition_rows.append(row)
                if substep in (LEADER_QUERY, CACHED_TRADE_REPLAY):
                    decision_rows.append(dict(row))
                transition_index += 1

            terminal = _jsonable(dict(terminal_info))
            nested_episode = terminal.get("episode")
            episode_rows.append({
                "phase": phase,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "response_sha256": response_hash,
                "environment_config_sha256": config_hash,
                "e2_provenance_fingerprint": provenance_fingerprint,
                "evaluation_episode": episode,
                "evaluation_seed": evaluation_seed,
                "evaluation_return": total_reward,
                "evaluation_steps": transition_index,
                "terminal_episode_summary": nested_episode,
                **{key: value for key, value in terminal.items() if key != "episode"},
            })
        finally:
            env.close()

    return {
        "episode_rows": episode_rows,
        "transition_rows": transition_rows,
        "decision_rows": decision_rows,
    }


def _number(value):
    if isinstance(value, (bool, int, float, np.number)):
        return float(value)
    return None


def _violation(row, field, expected, actual):
    return {
        "evaluation_episode": int(row.get("evaluation_episode", -1)),
        "evaluation_seed": int(row.get("evaluation_seed", -1)),
        "field": str(field),
        "expected": expected,
        "actual": actual,
    }


def _close(left, right, *, atol=PROTOCOL_ATOL):
    return bool(abs(float(left) - float(right)) <= float(atol))


def leader_outcome_summary(rows):
    rows = list(rows)
    if not rows:
        return {"episodes": 0, "mean_leader_payoff": None}
    payoff = np.asarray([float(row["leader_reward"]) for row in rows])
    return {
        "episodes": len(rows),
        "mean_leader_payoff": float(np.mean(payoff)),
        "median_leader_payoff": float(np.median(payoff)),
        "std_leader_payoff": float(np.std(payoff)),
        "min_leader_payoff": float(np.min(payoff)),
        "max_leader_payoff": float(np.max(payoff)),
        "mean_seller_payoff": float(np.mean([
            float(row["seller_reward"]) for row in rows
        ])),
        "mean_buyer_payoff": float(np.mean([
            float(row["buyer_reward"]) for row in rows
        ])),
        "mean_payments": float(np.mean([
            float(row["payments"]) for row in rows
        ])),
        "mean_purchases": float(np.mean([
            float(row["purchases"]) for row in rows
        ])),
        "mean_seller_game_reward": float(np.mean([
            float(row["seller_game_reward"]) for row in rows
        ])),
        "mean_buyer_game_reward": float(np.mean([
            float(row["buyer_game_reward"]) for row in rows
        ])),
    }


def _audit_episode(
        row,
        *,
        leader_role,
        gameplay_horizon,
        event_tail_steps,
        fixed_event_steps,
        seller_game_reward_scale,
        buyer_game_reward_scale,
):
    expected_reward = gameplay_horizon + NUM_TRADE_EVENTS
    expected_outer = expected_reward + NUM_TRADE_EVENTS
    exact = {
        "evaluation_steps": expected_outer,
        "outer_transition_count": expected_outer,
        "query_transitions": NUM_TRADE_EVENTS,
        "gameplay_transitions": gameplay_horizon,
        "trade_transitions": NUM_TRADE_EVENTS,
        "reward_transition_count": expected_reward,
        "cache_hits": NUM_TRADE_EVENTS,
        "bullets_arrived": NUM_TRADE_EVENTS,
        "seller_emulator_step_calls": gameplay_horizon,
        "buyer_emulator_step_calls": gameplay_horizon,
        "response_algorithm": "frozen_meta_policy",
        "leader_role": leader_role,
        "follower_role": opposite_role(leader_role),
    }
    violations = []
    for field, expected in exact.items():
        if row.get(field) != expected:
            violations.append(_violation(row, field, expected, row.get(field)))

    for field in (
            "seller_bullet_error",
            "buyer_bullet_error",
            "seller_payoff_error",
            "buyer_payoff_error",
    ):
        actual = _number(row.get(field))
        if actual is None or abs(actual) > PROTOCOL_ATOL:
            violations.append(_violation(row, field, 0.0, row.get(field)))

    numeric = {}
    for field in (
            "evaluation_return",
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
    ):
        numeric[field] = _number(row.get(field))
        if numeric[field] is None:
            violations.append(_violation(row, field, "numeric", row.get(field)))
    if all(numeric[field] is not None for field in (
            "evaluation_return", "leader_reward"
    )) and not _close(numeric["evaluation_return"], numeric["leader_reward"]):
        violations.append(_violation(
            row, "evaluation_return == leader_reward",
            numeric["leader_reward"], numeric["evaluation_return"],
        ))
    role_reward = numeric.get(f"{leader_role}_reward")
    if (
            numeric["leader_reward"] is not None
            and role_reward is not None
            and not _close(numeric["leader_reward"], role_reward)
    ):
        violations.append(_violation(
            row, "leader_reward == role reward", role_reward,
            numeric["leader_reward"],
        ))
    if all(numeric[field] is not None for field in (
            "seller_reward", "seller_game_reward", "payments"
    )):
        expected = (
            seller_game_reward_scale * numeric["seller_game_reward"]
            + numeric["payments"]
        )
        if not _close(numeric["seller_reward"], expected):
            violations.append(_violation(
                row, "seller payoff identity", expected,
                numeric["seller_reward"],
            ))
    if all(numeric[field] is not None for field in (
            "buyer_reward", "buyer_game_reward", "payments"
    )):
        expected = (
            buyer_game_reward_scale * numeric["buyer_game_reward"]
            - numeric["payments"]
        )
        if not _close(numeric["buyer_reward"], expected):
            violations.append(_violation(
                row, "buyer payoff identity", expected,
                numeric["buyer_reward"],
            ))
    if all(numeric[field] is not None for field in (
            "seller_shots_fired", "purchases", "seller_final_ammo"
    )):
        accounted = (
            numeric["seller_shots_fired"]
            + numeric["purchases"]
            + numeric["seller_final_ammo"]
        )
        if not _close(accounted, NUM_TRADE_EVENTS):
            violations.append(_violation(
                row, "seller bullet identity", NUM_TRADE_EVENTS, accounted
            ))
    if all(numeric[field] is not None for field in (
            "buyer_shots_fired", "buyer_final_ammo", "purchases"
    )):
        accounted = numeric["buyer_shots_fired"] + numeric["buyer_final_ammo"]
        if not _close(accounted, numeric["purchases"]):
            violations.append(_violation(
                row, "buyer bullet identity", numeric["purchases"], accounted
            ))

    event_stop = gameplay_horizon - event_tail_steps
    try:
        event_steps = tuple(int(value) for value in row.get("event_steps", ()))
    except (TypeError, ValueError):
        event_steps = ()
    valid_schedule = bool(
        len(event_steps) == NUM_TRADE_EVENTS
        and tuple(sorted(set(event_steps))) == event_steps
        and event_steps[0] >= 0
        and event_steps[-1] < event_stop
    )
    if not valid_schedule:
        violations.append(_violation(
            row, "event_steps",
            f"five sorted distinct steps in [0, {event_stop})",
            row.get("event_steps"),
        ))
    elif fixed_event_steps is not None and event_steps != tuple(fixed_event_steps):
        violations.append(_violation(
            row, "event_steps", list(fixed_event_steps), list(event_steps)
        ))

    events = row.get("events")
    if not isinstance(events, list) or len(events) != NUM_TRADE_EVENTS:
        violations.append(_violation(
            row, "events", f"{NUM_TRADE_EVENTS} trade rows", events
        ))
        events = []
    accepted_count = 0
    payment_sum = 0.0
    for index, event in enumerate(events):
        expected_event = event_steps[index] if valid_schedule else None
        if int(event.get("event_index", -1)) != index:
            violations.append(_violation(
                row, f"events[{index}].event_index", index,
                event.get("event_index"),
            ))
        if expected_event is not None and int(event.get("game_step", -1)) != expected_event:
            violations.append(_violation(
                row, f"events[{index}].game_step", expected_event,
                event.get("game_step"),
            ))
        price = _number(event.get("price"))
        threshold = _number(event.get("threshold"))
        accepted = event.get("accepted")
        if (
                price is None or threshold is None
                or not 0.0 <= price <= 1.0
                or not 0.0 <= threshold <= 1.0
                or not isinstance(accepted, bool)
        ):
            violations.append(_violation(
                row, f"events[{index}].trade", "valid [0,1] trade", event
            ))
            continue
        expected_acceptance = bool(price <= threshold)
        if accepted != expected_acceptance:
            violations.append(_violation(
                row, f"events[{index}].accepted", expected_acceptance, accepted
            ))
        accepted_count += int(accepted)
        payment_sum += price if accepted else 0.0
        seller_delta = int(event["seller_ammo_after"]) - int(event["seller_ammo_before"])
        buyer_delta = int(event["buyer_ammo_after"]) - int(event["buyer_ammo_before"])
        expected_seller_delta = -1 if accepted else 0
        expected_buyer_delta = 1 if accepted else 0
        if seller_delta != expected_seller_delta:
            violations.append(_violation(
                row, f"events[{index}].seller_ammo_delta",
                expected_seller_delta, seller_delta,
            ))
        if buyer_delta != expected_buyer_delta:
            violations.append(_violation(
                row, f"events[{index}].buyer_ammo_delta",
                expected_buyer_delta, buyer_delta,
            ))
    if numeric["purchases"] is not None and not _close(
            accepted_count, numeric["purchases"]
    ):
        violations.append(_violation(
            row, "purchases == accepted trades", accepted_count,
            numeric["purchases"],
        ))
    if numeric["payments"] is not None and not _close(
            payment_sum, numeric["payments"]
    ):
        violations.append(_violation(
            row, "payments == accepted prices", payment_sum,
            numeric["payments"],
        ))

    query_actions = row.get("query_actions")
    commitment = row.get("leader_commitment")
    follower_actions = row.get("follower_actions")
    valid_query_actions = bool(
        isinstance(query_actions, list)
        and len(query_actions) == NUM_TRADE_EVENTS
        and all(isinstance(action, list) and len(action) == 2 for action in query_actions)
    )
    if not valid_query_actions:
        violations.append(_violation(
            row, "query_actions", "five full actions", query_actions
        ))
    if not isinstance(commitment, list) or len(commitment) != NUM_TRADE_EVENTS:
        violations.append(_violation(
            row, "leader_commitment", "five economic actions", commitment
        ))
    elif valid_query_actions and not np.allclose(
            np.asarray(commitment, dtype=float),
            np.asarray(query_actions, dtype=float)[:, 1],
            rtol=0.0,
            atol=PROTOCOL_ATOL,
    ):
        violations.append(_violation(
            row, "leader_commitment == query economic actions",
            np.asarray(query_actions, dtype=float)[:, 1].tolist(), commitment,
        ))
    valid_follower_actions = bool(
        isinstance(follower_actions, list)
        and len(follower_actions) == NUM_TRADE_EVENTS
        and all(
            isinstance(action, list) and len(action) == 2
            for action in follower_actions
        )
    )
    if not valid_follower_actions:
        violations.append(_violation(
            row, "follower_actions", "five full trade actions", follower_actions
        ))
    if valid_query_actions and valid_follower_actions and len(events) == 5:
        for index, event in enumerate(events):
            leader_economic = float(query_actions[index][1])
            follower_economic = float(follower_actions[index][1])
            if leader_role == SELLER:
                leader_field, follower_field = "price", "threshold"
            else:
                leader_field, follower_field = "threshold", "price"
            if not _close(leader_economic, event[leader_field]):
                violations.append(_violation(
                    row, f"events[{index}].{leader_field} == leader query",
                    leader_economic, event[leader_field],
                ))
            if not _close(follower_economic, event[follower_field]):
                violations.append(_violation(
                    row, f"events[{index}].{follower_field} == response action",
                    follower_economic, event[follower_field],
                ))
    trace = row.get("query_trace_sha256")
    if not isinstance(trace, str) or re.fullmatch(r"[0-9a-f]{64}", trace) is None:
        violations.append(_violation(
            row, "query_trace_sha256", "64 lowercase hex characters", trace
        ))
    nested = row.get("terminal_episode_summary")
    if not isinstance(nested, dict):
        violations.append(_violation(
            row, "terminal_episode_summary", "terminal episode mapping", nested
        ))
    else:
        if nested.get("l") != expected_outer:
            violations.append(_violation(
                row, "terminal episode length", expected_outer, nested.get("l")
            ))
        if (
                numeric["leader_reward"] is not None
                and _number(nested.get("r")) is not None
                and not _close(nested["r"], numeric["leader_reward"])
        ):
            violations.append(_violation(
                row, "terminal episode return", numeric["leader_reward"],
                nested.get("r"),
            ))
    return violations


def _audit_transitions(rows, decisions, episode_row, *, gameplay_horizon):
    episode = int(episode_row["evaluation_episode"])
    selected = [row for row in rows if int(row["evaluation_episode"]) == episode]
    selected_decisions = [
        row for row in decisions if int(row["evaluation_episode"]) == episode
    ]
    expected_outer = gameplay_horizon + 2 * NUM_TRADE_EVENTS
    violations = []
    if len(selected) != expected_outer:
        violations.append(_violation(
            episode_row, "retained transition rows", expected_outer, len(selected)
        ))
        return violations
    indices = [int(row["transition_index"]) for row in selected]
    if indices != list(range(expected_outer)):
        violations.append(_violation(
            episode_row, "transition indices", list(range(expected_outer)), indices
        ))
    counts = defaultdict(int)
    for row in selected:
        counts[row["substep_type"]] += 1
    expected_counts = {
        LEADER_QUERY: NUM_TRADE_EVENTS,
        GAMEPLAY: gameplay_horizon,
        CACHED_TRADE_REPLAY: NUM_TRADE_EVENTS,
    }
    for substep, expected in expected_counts.items():
        if counts[substep] != expected:
            violations.append(_violation(
                episode_row, f"transition count: {substep}", expected,
                counts[substep],
            ))
    if [row["substep_type"] for row in selected[:NUM_TRADE_EVENTS]] != [
            LEADER_QUERY
    ] * NUM_TRADE_EVENTS:
        violations.append(_violation(
            episode_row, "first five transitions", [LEADER_QUERY] * 5,
            [row["substep_type"] for row in selected[:5]],
        ))
    for index, row in enumerate(selected[:NUM_TRADE_EVENTS]):
        if row.get("query_index") != index:
            violations.append(_violation(
                episode_row, f"query transition {index} index", index,
                row.get("query_index"),
            ))
        expected_state = canonical_leader_state(index)
        if not np.array_equal(
                np.asarray(row.get("actor_state"), dtype=np.float32),
                expected_state,
        ):
            violations.append(_violation(
                episode_row, f"query transition {index} canonical actor state",
                expected_state.tolist(), row.get("actor_state"),
            ))
        if row.get("actor_image_nonzero") != 0:
            violations.append(_violation(
                episode_row, f"query transition {index} dummy image", 0,
                row.get("actor_image_nonzero"),
            ))
    reward_rows = selected[NUM_TRADE_EVENTS:]
    gameplay_seen = 0
    replay_seen = 0
    event_steps = tuple(int(value) for value in episode_row.get("event_steps", ()))
    for row in reward_rows:
        if row["substep_type"] == CACHED_TRADE_REPLAY:
            if row.get("event_index") != replay_seen:
                violations.append(_violation(
                    episode_row, "cached replay event order", replay_seen,
                    row.get("event_index"),
                ))
            if (
                    len(event_steps) == NUM_TRADE_EVENTS
                    and replay_seen < NUM_TRADE_EVENTS
                    and gameplay_seen != event_steps[replay_seen]
            ):
                violations.append(_violation(
                    episode_row, f"cached replay {replay_seen} gameplay position",
                    event_steps[replay_seen], gameplay_seen,
                ))
            replay_seen += 1
        elif row["substep_type"] == GAMEPLAY:
            gameplay_seen += 1
    if gameplay_seen != gameplay_horizon or replay_seen != NUM_TRADE_EVENTS:
        violations.append(_violation(
            episode_row, "reward-phase gameplay/trade progression",
            [gameplay_horizon, NUM_TRADE_EVENTS],
            [gameplay_seen, replay_seen],
        ))
    for index, row in enumerate(selected):
        expected_done = index == len(selected) - 1
        if bool(row["done"]) != expected_done:
            violations.append(_violation(
                episode_row, f"transition {index} done", expected_done,
                row["done"],
            ))
    transition_return = float(sum(float(row["reward"]) for row in selected))
    if not _close(transition_return, episode_row["evaluation_return"]):
        violations.append(_violation(
            episode_row, "sum transition rewards", episode_row["evaluation_return"],
            transition_return,
        ))
    for row in selected:
        substep = row["substep_type"]
        if not row["reward_generated"]:
            violations.append(_violation(
                episode_row, f"transition {row['transition_index']} reward_generated",
                True, False,
            ))
        expected_reward_phase = substep != LEADER_QUERY
        if row["is_reward_phase"] != expected_reward_phase:
            violations.append(_violation(
                episode_row, f"transition {row['transition_index']} is_reward_phase",
                expected_reward_phase, row["is_reward_phase"],
            ))
        expected_credit = {
            LEADER_QUERY: [0.0, 1.0],
            GAMEPLAY: [1.0, 0.0],
            CACHED_TRADE_REPLAY: [0.0, 0.0],
        }.get(substep)
        if expected_credit is not None and not np.array_equal(
                np.asarray(row["action_credit"], dtype=float), expected_credit
        ):
            violations.append(_violation(
                episode_row, f"transition {row['transition_index']} action_credit",
                expected_credit, row["action_credit"],
            ))
        if substep == LEADER_QUERY:
            if float(row["reward"]) != 0.0 or row["emulator_advanced"]:
                violations.append(_violation(
                    episode_row, f"query transition {row['transition_index']}",
                    "zero reward and no emulator advance", row,
                ))
        if substep == CACHED_TRADE_REPLAY and (
                not row["cache_hit"] or row["emulator_advanced"]
        ):
            violations.append(_violation(
                episode_row, f"cached trade transition {row['transition_index']}",
                "cache hit and no emulator advance", row,
            ))

    if len(selected_decisions) != 2 * NUM_TRADE_EVENTS:
        violations.append(_violation(
            episode_row, "economic decision rows", 2 * NUM_TRADE_EVENTS,
            len(selected_decisions),
        ))
        return violations
    queries = {
        int(row["query_index"]): row
        for row in selected_decisions
        if row["substep_type"] == LEADER_QUERY
    }
    replays = {
        int(row["event_index"]): row
        for row in selected_decisions
        if row["substep_type"] == CACHED_TRADE_REPLAY
    }
    if set(queries) != set(range(NUM_TRADE_EVENTS)):
        violations.append(_violation(
            episode_row, "query indices", list(range(NUM_TRADE_EVENTS)),
            sorted(queries),
        ))
    if set(replays) != set(range(NUM_TRADE_EVENTS)):
        violations.append(_violation(
            episode_row, "cached replay indices", list(range(NUM_TRADE_EVENTS)),
            sorted(replays),
        ))
    for event in sorted(set(queries) & set(replays)):
        query = queries[event]
        replay = replays[event]
        expected_state = canonical_leader_state(event)
        if not np.array_equal(
                np.asarray(replay.get("actor_state"), dtype=np.float32),
                expected_state,
        ):
            violations.append(_violation(
                episode_row, f"event {event} cached canonical actor state",
                expected_state.tolist(), replay.get("actor_state"),
            ))
        if replay.get("actor_image_nonzero") != 0:
            violations.append(_violation(
                episode_row, f"event {event} cached dummy image", 0,
                replay.get("actor_image_nonzero"),
            ))
        if query["actor_observation_sha256"] != replay["actor_observation_sha256"]:
            violations.append(_violation(
                episode_row, f"event {event} actor observation cache identity",
                query["actor_observation_sha256"],
                replay["actor_observation_sha256"],
            ))
        if not np.allclose(
                query["requested_action"], replay["requested_action"],
                rtol=0.0, atol=PROTOCOL_ATOL,
        ):
            violations.append(_violation(
                episode_row, f"event {event} requested action cache identity",
                query["requested_action"], replay["requested_action"],
            ))
        executed = replay.get("leader_executed_action")
        if executed is None or not np.allclose(
                query["requested_action"], executed,
                rtol=0.0, atol=PROTOCOL_ATOL,
        ):
            violations.append(_violation(
                episode_row, f"event {event} executed action cache identity",
                query["requested_action"], executed,
            ))
    return violations


def audit_e2_protocol(
        evaluation,
        *,
        required_episodes,
        leader_role,
        gameplay_horizon,
        event_tail_steps=0,
        fixed_event_steps=None,
        seller_game_reward_scale=0.1,
        buyer_game_reward_scale=1.0,
):
    """Enforce every transition, trace, trade, bullet, and payoff identity."""

    episode_rows = list(evaluation.get("episode_rows", ()))
    transition_rows = list(evaluation.get("transition_rows", ()))
    decision_rows = list(evaluation.get("decision_rows", ()))
    violations = []
    if len(episode_rows) != int(required_episodes):
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "episode_count",
            "expected": int(required_episodes),
            "actual": len(episode_rows),
        })
    for row in episode_rows:
        violations.extend(_audit_episode(
            row,
            leader_role=leader_role,
            gameplay_horizon=int(gameplay_horizon),
            event_tail_steps=int(event_tail_steps),
            fixed_event_steps=fixed_event_steps,
            seller_game_reward_scale=float(seller_game_reward_scale),
            buyer_game_reward_scale=float(buyer_game_reward_scale),
        ))
        violations.extend(_audit_transitions(
            transition_rows, decision_rows, row,
            gameplay_horizon=int(gameplay_horizon),
        ))

    traces = sorted({str(row.get("query_trace_sha256")) for row in episode_rows})
    commitments = sorted({
        _canonical_json_bytes(row.get("leader_commitment")).decode("ascii")
        for row in episode_rows
    })
    query_actions = sorted({
        _canonical_json_bytes(row.get("query_actions")).decode("ascii")
        for row in episode_rows
    })
    if len(traces) != 1:
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "deterministic query trace",
            "expected": "one trace SHA-256 across all episodes",
            "actual": traces,
        })
    if len(commitments) != 1:
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "deterministic leader commitment",
            "expected": "one five-action commitment across all episodes",
            "actual": commitments,
        })
    if len(query_actions) != 1:
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "deterministic full query actions",
            "expected": "one full five-action trace across all episodes",
            "actual": query_actions,
        })
    return {
        "passed": not violations,
        "episodes": len(episode_rows),
        "required_episodes": int(required_episodes),
        "expected_outer_transitions": int(gameplay_horizon) + 10,
        "expected_query_transitions": NUM_TRADE_EVENTS,
        "expected_gameplay_transitions": int(gameplay_horizon),
        "expected_cached_trade_replays": NUM_TRADE_EVENTS,
        "single_query_trace": len(traces) == 1,
        "single_leader_commitment": len(commitments) == 1,
        "single_full_query_action_trace": len(query_actions) == 1,
        "query_trace_sha256": traces[0] if len(traces) == 1 else None,
        "leader_commitment": (
            json.loads(commitments[0]) if len(commitments) == 1 else None
        ),
        "full_query_actions": (
            json.loads(query_actions[0]) if len(query_actions) == 1 else None
        ),
        "violations": violations,
    }


def evaluate_checkpoint(
        model,
        checkpoint,
        *,
        args,
        response_model,
        response_hash,
        config,
        config_hash,
        episodes,
        seed_start,
        phase,
):
    checkpoint = Path(checkpoint).resolve()
    digest = checkpoint_sha256(checkpoint)
    provenance = validate_candidate_provenance(
        model, response_hash=response_hash, config=config
    )
    actor_loss_mode = model_actor_loss_mode(model)
    economic_initialization = model_economic_initialization(
        model,
        default_mean=E2_INIT_MEAN,
        default_concentration=E2_INIT_CONCENTRATION,
    )
    evaluation = evaluate_e2_model(
        model,
        lambda episode: make_e2_env(
            args, seed=int(seed_start) + int(episode),
            response_model=response_model,
        ),
        episodes=episodes,
        seed_start=seed_start,
        checkpoint_path=checkpoint,
        checkpoint_hash=digest,
        response_hash=response_hash,
        config_hash=config_hash,
        provenance_fingerprint=provenance["fingerprint_sha256"],
        phase=phase,
    )
    protocol = audit_e2_protocol(
        evaluation,
        required_episodes=episodes,
        leader_role=args.leader_role,
        gameplay_horizon=config["gameplay_horizon"],
        event_tail_steps=config["event_tail_steps"],
        fixed_event_steps=config["fixed_event_steps"],
        seller_game_reward_scale=config["seller_game_reward_scale"],
        buyer_game_reward_scale=config["buyer_game_reward_scale"],
    )
    return {
        "checkpoint_id": checkpoint.stem,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "response_checkpoint_sha256": response_hash,
        "environment_config_sha256": config_hash,
        "e2_provenance_fingerprint": provenance["fingerprint_sha256"],
        "e2_provenance_manifest": provenance,
        "training_total_timesteps": int(getattr(model, "num_timesteps", 0)),
        "economic_role": model.policy.economic_role,
        "economic_input_mode": model.policy.economic_input_mode,
        "actor_loss_mode": actor_loss_mode,
        "target_kl": getattr(model, "target_kl", None),
        "economic_head_initialization": economic_initialization,
        "phase": phase,
        "seed_start": int(seed_start),
        "seed_end": int(seed_start) + int(episodes) - 1,
        "summary": leader_outcome_summary(evaluation["episode_rows"]),
        "protocol": protocol,
        **evaluation,
    }


def validate_common_screen(results):
    """Require all candidates to see identical seeds and event schedules."""

    results = list(results)
    if not results:
        raise ValueError("selection requires at least one checkpoint result")
    def seed_schedule(result):
        rows = result["episode_rows"]
        seeds = [int(row["evaluation_seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError("candidate screen contains duplicate evaluation seeds")
        expected = set(range(
            int(result["seed_start"]), int(result["seed_end"]) + 1
        ))
        if set(seeds) != expected:
            raise RuntimeError(
                "candidate screen is missing its exact contiguous seed range"
            )
        return {
            int(row["evaluation_seed"]): tuple(
                int(value) for value in row["event_steps"]
            )
            for row in rows
        }

    reference = seed_schedule(results[0])
    reference_response = results[0]["response_checkpoint_sha256"]
    reference_config = results[0]["environment_config_sha256"]
    reference_provenance = results[0]["e2_provenance_fingerprint"]
    for result in results[1:]:
        if result["response_checkpoint_sha256"] != reference_response:
            raise RuntimeError("candidate screens used different E1 responses")
        if result["environment_config_sha256"] != reference_config:
            raise RuntimeError("candidate screens used different environment configs")
        if result["e2_provenance_fingerprint"] != reference_provenance:
            raise RuntimeError(
                "candidate screens used different E2 scientific provenance"
            )
        current = seed_schedule(result)
        if set(current) != set(reference):
            raise RuntimeError("candidate screens do not share evaluation seeds")
        for seed in sorted(reference):
            if current[seed] != reference[seed]:
                raise RuntimeError(
                    "candidate screens received different event schedules for "
                    f"seed {seed}: {reference[seed]} != {current[seed]}"
                )
    return {
        "passed": True,
        "episodes": len(reference),
        "seed_schedule_pairs": [
            {"evaluation_seed": seed, "event_steps": list(reference[seed])}
            for seed in sorted(reference)
        ],
    }


def _selection_key(result):
    summary = result["summary"]
    return (
        -float(summary["mean_leader_payoff"]),
        -float(summary["median_leader_payoff"]),
        -float(summary["min_leader_payoff"]),
        float(summary["std_leader_payoff"]),
        int(result["training_total_timesteps"]),
        str(result["checkpoint_sha256"]),
    )


def rank_candidates(results):
    """Rank valid checkpoints using the documented role-neutral rule."""

    valid = sorted(
        (result for result in results if result["protocol"]["passed"]),
        key=_selection_key,
    )
    ranks = {result["checkpoint_sha256"]: index + 1 for index, result in enumerate(valid)}
    rows = []
    for result in results:
        rows.append({
            "rank": ranks.get(result["checkpoint_sha256"]),
            "eligible": bool(result["protocol"]["passed"]),
            "checkpoint_id": result["checkpoint_id"],
            "checkpoint_path": result["checkpoint_path"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "e2_provenance_fingerprint": result.get(
                "e2_provenance_fingerprint"
            ),
            "training_total_timesteps": result["training_total_timesteps"],
            "actor_loss_mode": result.get(
                "actor_loss_mode", STANDARD_ACTOR_LOSS_MODE
            ),
            "target_kl": result.get("target_kl"),
            "economic_head_initialization": result.get(
                "economic_head_initialization",
                {"mean": E2_INIT_MEAN, "concentration": E2_INIT_CONCENTRATION},
            ),
            **result["summary"],
            "protocol_violation_count": len(result["protocol"]["violations"]),
        })
    rows.sort(key=lambda row: (
        row["rank"] is None,
        row["rank"] if row["rank"] is not None else 10 ** 9,
        row["checkpoint_sha256"],
    ))
    return {
        "selection_rule": list(SELECTION_RULE),
        "eligible_checkpoints": len(valid),
        "selected_checkpoint_sha256": (
            valid[0]["checkpoint_sha256"] if valid else None
        ),
        "ranking_rows": rows,
    }


def atomic_copy_no_overwrite(source, destination):
    """Copy exact bytes to a new alias without ever replacing a prior file."""

    source = Path(source).resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.suffix != ".zip":
        destination = destination.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite selected checkpoint alias: {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite selected checkpoint alias: {destination}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    source_hash = checkpoint_sha256(source)
    destination_hash = checkpoint_sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError("selected checkpoint copy failed its SHA-256 check")
    return {
        "source_checkpoint_path": str(source),
        "selected_checkpoint_path": str(destination),
        "checkpoint_sha256": destination_hash,
        "copy_verified": True,
    }


def confirmation_matches_screen(screen_result, confirmation_result):
    screen = screen_result["protocol"]
    confirmation = confirmation_result["protocol"]
    checks = {
        "protocol_passed": bool(confirmation["passed"]),
        "query_trace_matches_screen": (
            confirmation["query_trace_sha256"] == screen["query_trace_sha256"]
        ),
        "leader_commitment_matches_screen": (
            confirmation["leader_commitment"] == screen["leader_commitment"]
        ),
        "full_query_actions_match_screen": (
            confirmation["full_query_actions"] == screen["full_query_actions"]
        ),
        "checkpoint_hash_matches_screen": (
            confirmation_result["checkpoint_sha256"]
            == screen_result["checkpoint_sha256"]
        ),
        "provenance_fingerprint_matches_screen": (
            confirmation_result["e2_provenance_fingerprint"]
            == screen_result["e2_provenance_fingerprint"]
        ),
    }
    return {"passed": all(checks.values()), **checks}


def _ranges_overlap(start_a, count_a, start_b, count_b):
    end_a = int(start_a) + int(count_a) - 1
    end_b = int(start_b) + int(count_b) - 1
    return max(int(start_a), int(start_b)) <= min(end_a, end_b)


def run_selection(args):
    """Run the common screen, alias its winner, and independently confirm it."""

    if int(args.screen_episodes) != SCREEN_EPISODES:
        raise ValueError(
            f"clean E2 selection requires exactly {SCREEN_EPISODES} episodes"
        )
    if int(args.confirmation_episodes) != CONFIRMATION_EPISODES:
        raise ValueError(
            "clean E2 confirmation requires exactly "
            f"{CONFIRMATION_EPISODES} episodes"
        )
    if _ranges_overlap(
            args.screen_seed_start,
            args.screen_episodes,
            args.confirmation_seed_start,
            args.confirmation_episodes,
    ):
        raise ValueError("screen and confirmation seed ranges must be disjoint")
    if int(args.gameplay_horizon) != CANONICAL_GAMEPLAY_HORIZON:
        raise ValueError(
            "clean E2 selection requires the canonical 200-step gameplay "
            "horizon"
        )

    checkpoints = tuple(
        _checkpoint_path(path, label="E2 candidate") for path in args.checkpoint
    )
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("E2 candidate paths must be unique")
    hashes = [checkpoint_sha256(path) for path in checkpoints]
    if len(set(hashes)) != len(hashes):
        raise ValueError("E2 candidates must contain distinct checkpoint bytes")
    response_path = _checkpoint_path(
        args.response_checkpoint, label="frozen E1 response"
    )
    response_hash = checkpoint_sha256(response_path)
    args.response_checkpoint = str(response_path)
    config = environment_config(args)
    config_hash = environment_config_sha256(config)
    response_model = load_e1_response(
        response_path,
        leader_role=args.leader_role,
        device=args.device,
        expected_sha256=response_hash,
    )
    response_is_buyer = response_model.policy.economic_role == BUYER
    response_actor_loss_mode = model_actor_loss_mode(response_model)
    if response_actor_loss_mode not in ACTOR_LOSS_MODES:
        raise ValueError("frozen E1 response has an unknown actor loss mode")
    response_target_kl = getattr(response_model, "target_kl", None)
    if response_target_kl is not None:
        response_target_kl = float(response_target_kl)
        if not np.isfinite(response_target_kl) or response_target_kl <= 0.0:
            raise ValueError("frozen E1 response has an invalid target KL")
    response_initialization = model_economic_initialization(
        response_model,
        default_mean=(
            E1_BUYER_INIT_MEAN if response_is_buyer else E1_SELLER_INIT_MEAN
        ),
        default_concentration=(
            E1_BUYER_INIT_CONCENTRATION
            if response_is_buyer
            else E1_SELLER_INIT_CONCENTRATION
        ),
    )
    response_metadata = {
        "checkpoint_path": str(response_path),
        "checkpoint_sha256": response_hash,
        "economic_role": response_model.policy.economic_role,
        "economic_input_mode": response_model.policy.economic_input_mode,
        "training_total_timesteps": int(
            getattr(response_model, "num_timesteps", 0)
        ),
        "actor_loss_mode": response_actor_loss_mode,
        "target_kl": response_target_kl,
        "economic_head_initialization": response_initialization,
        "frozen": True,
        "deterministic": True,
    }

    screen_results = []
    for checkpoint in checkpoints:
        model = load_e2_checkpoint(
            checkpoint, leader_role=args.leader_role, device=args.device
        )
        try:
            screen_results.append(evaluate_checkpoint(
                model,
                checkpoint,
                args=args,
                response_model=response_model,
                response_hash=response_hash,
                config=config,
                config_hash=config_hash,
                episodes=args.screen_episodes,
                seed_start=args.screen_seed_start,
                phase="screen",
            ))
        finally:
            del model
    common_screen = validate_common_screen(screen_results)
    ranking = rank_candidates(screen_results)
    selected_hash = ranking["selected_checkpoint_sha256"]
    if selected_hash is None:
        return {
            "schema_version": 1,
            "evaluator": EVALUATOR_NAME,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment_config": config,
            "environment_config_sha256": config_hash,
            "response_checkpoint": str(response_path),
            "response_checkpoint_sha256": response_hash,
            "response_metadata": response_metadata,
            "screen": {
                "episodes_per_checkpoint": int(args.screen_episodes),
                "seed_start": int(args.screen_seed_start),
                "seed_end": int(args.screen_seed_start) + args.screen_episodes - 1,
                "common_seed_schedule_check": common_screen,
                "checkpoint_results": screen_results,
            },
            "selection": ranking,
            "selected_alias": None,
            "confirmation": None,
            "passed": False,
        }

    selected = next(
        result for result in screen_results
        if result["checkpoint_sha256"] == selected_hash
    )
    selected_alias = atomic_copy_no_overwrite(
        selected["checkpoint_path"], args.selected_checkpoint
    )
    alias_model = load_e2_checkpoint(
        selected_alias["selected_checkpoint_path"],
        leader_role=args.leader_role,
        device=args.device,
    )
    try:
        confirmed = evaluate_checkpoint(
            alias_model,
            selected_alias["selected_checkpoint_path"],
            args=args,
            response_model=response_model,
            response_hash=response_hash,
            config=config,
            config_hash=config_hash,
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
            phase="confirmation",
        )
    finally:
        del alias_model
    confirmation_check = confirmation_matches_screen(selected, confirmed)
    return {
        "schema_version": 1,
        "evaluator": EVALUATOR_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_config": config,
        "environment_config_sha256": config_hash,
        "response_checkpoint": str(response_path),
        "response_checkpoint_sha256": response_hash,
        "response_metadata": response_metadata,
        "screen": {
            "episodes_per_checkpoint": int(args.screen_episodes),
            "seed_start": int(args.screen_seed_start),
            "seed_end": int(args.screen_seed_start) + args.screen_episodes - 1,
            "common_seed_schedule_check": common_screen,
            "checkpoint_results": screen_results,
        },
        "selection": ranking,
        "selected_alias": selected_alias,
        "confirmation": {
            "episodes": int(args.confirmation_episodes),
            "seed_start": int(args.confirmation_seed_start),
            "seed_end": (
                int(args.confirmation_seed_start) + args.confirmation_episodes - 1
            ),
            "disjoint_from_screen": True,
            "result": confirmed,
            "checks": confirmation_check,
        },
        "passed": bool(confirmation_check["passed"]),
    }


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")


def default_run_name(report):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    role = report["environment_config"]["leader_role"]
    seed_start = report["screen"]["seed_start"]
    seed_end = report["screen"]["seed_end"]
    return f"e2_{role}_selection_seed{seed_start}-{seed_end}_{stamp}"


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _csv_row(row):
    return {key: _csv_value(value) for key, value in row.items()}


def _event_rows(results):
    rows = []
    for result in results:
        for episode in result["episode_rows"]:
            for event in episode.get("events", ()):
                rows.append({
                    "phase": result["phase"],
                    "checkpoint_path": result["checkpoint_path"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "response_sha256": episode["response_sha256"],
                    "environment_config_sha256": episode[
                        "environment_config_sha256"
                    ],
                    "e2_provenance_fingerprint": episode.get(
                        "e2_provenance_fingerprint"
                    ),
                    "evaluation_episode": episode["evaluation_episode"],
                    "evaluation_seed": episode["evaluation_seed"],
                    **event,
                })
    return rows


def _artifact_paths(output_dir, stem, *, has_confirmation):
    paths = {
        "report_json": output_dir / f"{stem}.json",
        "ranking_csv": output_dir / f"{stem}.ranking.csv",
        "screen_episodes_csv": output_dir / f"{stem}.screen.episodes.csv",
        "screen_transitions_csv": output_dir / f"{stem}.screen.transitions.csv",
        "screen_decisions_csv": output_dir / f"{stem}.screen.decisions.csv",
        "screen_events_csv": output_dir / f"{stem}.screen.events.csv",
    }
    if has_confirmation:
        paths.update({
            "confirmation_episodes_csv": (
                output_dir / f"{stem}.confirmation.episodes.csv"
            ),
            "confirmation_transitions_csv": (
                output_dir / f"{stem}.confirmation.transitions.csv"
            ),
            "confirmation_decisions_csv": (
                output_dir / f"{stem}.confirmation.decisions.csv"
            ),
            "confirmation_events_csv": (
                output_dir / f"{stem}.confirmation.events.csv"
            ),
        })
    return paths


def write_selection_artifacts(report, *, output_dir, run_name=None):
    """Write all rows under dedicated names and refuse every collision."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(run_name or default_run_name(report))
    if not stem:
        raise ValueError("selection run name cannot be empty")
    paths = _artifact_paths(
        output_dir, stem, has_confirmation=report.get("confirmation") is not None
    )
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing E2 selection artifacts: "
            + ", ".join(collisions)
        )

    screen_results = report["screen"]["checkpoint_results"]
    screen_episodes = [
        row for result in screen_results for row in result["episode_rows"]
    ]
    screen_transitions = [
        row for result in screen_results for row in result["transition_rows"]
    ]
    screen_decisions = [
        row for result in screen_results for row in result["decision_rows"]
    ]
    write_csv(paths["ranking_csv"], [
        _csv_row(row) for row in report["selection"]["ranking_rows"]
    ])
    write_csv(paths["screen_episodes_csv"], [
        _csv_row(row) for row in screen_episodes
    ])
    write_csv(paths["screen_transitions_csv"], [
        _csv_row(row) for row in screen_transitions
    ])
    write_csv(paths["screen_decisions_csv"], [
        _csv_row(row) for row in screen_decisions
    ])
    write_csv(paths["screen_events_csv"], [
        _csv_row(row) for row in _event_rows(screen_results)
    ])

    if report.get("confirmation") is not None:
        confirmed = report["confirmation"]["result"]
        write_csv(paths["confirmation_episodes_csv"], [
            _csv_row(row) for row in confirmed["episode_rows"]
        ])
        write_csv(paths["confirmation_transitions_csv"], [
            _csv_row(row) for row in confirmed["transition_rows"]
        ])
        write_csv(paths["confirmation_decisions_csv"], [
            _csv_row(row) for row in confirmed["decision_rows"]
        ])
        write_csv(paths["confirmation_events_csv"], [
            _csv_row(row) for row in _event_rows([confirmed])
        ])

    artifacts = {key: str(path) for key, path in paths.items()}
    report = {**report, "artifacts": artifacts}
    write_json(paths["report_json"], report)
    return report


def _parse_event_steps(raw):
    if raw is None:
        return None
    values = tuple(int(value.strip()) for value in str(raw).split(","))
    if len(values) != NUM_TRADE_EVENTS:
        raise ValueError("--fixed-event-steps requires five comma-separated steps")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-role", choices=(BUYER, SELLER), required=True)
    parser.add_argument("--response-checkpoint", required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="E2 step checkpoint; repeat for every candidate in the common screen",
    )
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--screen-episodes", type=int, default=SCREEN_EPISODES)
    parser.add_argument("--screen-seed-start", type=int, default=4_000_001)
    parser.add_argument(
        "--confirmation-episodes", type=int, default=CONFIRMATION_EPISODES
    )
    parser.add_argument("--confirmation-seed-start", type=int, default=5_000_001)
    parser.add_argument("--gameplay-horizon", type=int, default=200)
    parser.add_argument("--event-tail-steps", type=int, default=0)
    parser.add_argument("--fixed-event-steps")
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=100_000)
    parser.add_argument("--rom-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)

    if args.screen_episodes <= 0:
        parser.error("--screen-episodes must be positive")
    if args.confirmation_episodes <= 0:
        parser.error("--confirmation-episodes must be positive")
    if args.screen_episodes != SCREEN_EPISODES:
        parser.error(
            f"clean E2 selection requires exactly {SCREEN_EPISODES} "
            "common screening episodes"
        )
    if args.confirmation_episodes != CONFIRMATION_EPISODES:
        parser.error(
            f"clean E2 confirmation requires exactly {CONFIRMATION_EPISODES} "
            "episodes"
        )
    if args.screen_seed_start < 0 or args.confirmation_seed_start < 0:
        parser.error("evaluation seeds must be nonnegative")
    if _ranges_overlap(
            args.screen_seed_start, args.screen_episodes,
            args.confirmation_seed_start, args.confirmation_episodes,
    ):
        parser.error("screen and confirmation seed ranges must be disjoint")
    if args.gameplay_horizon < NUM_TRADE_EVENTS:
        parser.error("--gameplay-horizon must be at least five")
    if args.gameplay_horizon != CANONICAL_GAMEPLAY_HORIZON:
        parser.error(
            "clean E2 selection requires the canonical 200-step gameplay "
            "horizon (210 outer transitions)"
        )
    if not 0 <= args.event_tail_steps < args.gameplay_horizon:
        parser.error("--event-tail-steps must lie in [0, gameplay_horizon)")
    if args.gameplay_horizon - args.event_tail_steps < NUM_TRADE_EVENTS:
        parser.error("the E2 event window must contain at least five steps")
    if args.noop_max < 0:
        parser.error("--noop-max must be nonnegative")
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    try:
        args.fixed_event_steps = _parse_event_steps(args.fixed_event_steps)
        if args.fixed_event_steps is not None:
            ExactFiveEventSchedule(
                gameplay_horizon=args.gameplay_horizon,
                tail_steps=args.event_tail_steps,
                fixed_event_steps=args.fixed_event_steps,
            )
    except ValueError as error:
        parser.error(str(error))
    selected = Path(args.selected_checkpoint).expanduser()
    if selected.suffix != ".zip":
        selected = selected.with_suffix(".zip")
    args.selected_checkpoint = str(selected.resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    if Path(args.selected_checkpoint).exists():
        raise FileExistsError(
            "refusing to overwrite selected checkpoint alias: "
            f"{args.selected_checkpoint}"
        )
    if args.run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_name = (
            f"e2_{args.leader_role}_selection_seed"
            f"{args.screen_seed_start}-"
            f"{args.screen_seed_start + args.screen_episodes - 1}_{stamp}"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    planned_stem = _slug(args.run_name)
    if not planned_stem:
        raise ValueError("selection run name cannot be empty")
    planned = _artifact_paths(
        output_dir, planned_stem, has_confirmation=True
    )
    collisions = [str(path) for path in planned.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing E2 selection artifacts: "
            + ", ".join(collisions)
        )
    report = run_selection(args)
    report = write_selection_artifacts(
        report, output_dir=args.output_dir, run_name=args.run_name
    )
    print({
        "passed": report["passed"],
        "selected_checkpoint": (
            None if report["selected_alias"] is None
            else report["selected_alias"]["selected_checkpoint_path"]
        ),
        "selected_checkpoint_sha256": report["selection"][
            "selected_checkpoint_sha256"
        ],
        "report": report["artifacts"]["report_json"],
    }, flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
