"""Screen, select, and confirm clean Atari E2 leader checkpoints.

Every candidate is evaluated deterministically against one explicitly supplied,
frozen, opposite-role E1 response on the same 20 episode seeds.  A candidate is
eligible for selection only when every episode satisfies the complete E2
protocol: five event-only queries, 200 gameplay transitions, five actor-identical
cached trade replays, exact bilateral accounting, one deterministic query
trace/commitment, and this experiment's predeclared interior-surplus hypothesis:
paired dominance over the all-zero and all-one economic commitments.  This is
a diagnostic hypothesis for the five-bullet design, not a universal equilibrium
condition.  Only the screen-selected top candidate is confirmed on 100 disjoint
seeds, and no collision-safe selected alias is created unless it passes.

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

from stackelberg_pomdp.atari.training import (
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
    e2_implementation_provenance_compatible,
    validate_e2_gameplay_actor_freeze,
    validate_e2_provenance_manifest,
)
from stackelberg_pomdp.atari.envs.space_invaders import default_rom_path
from stackelberg_pomdp.atari.wrappers.meta_follower import (
    make_stackpomdp_atari_leader_env,
)
from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTION_MASK,
    ACTOR_STATE,
    CACHED_TRADE_REPLAY,
    EVENT_SLICE,
    GAMEPLAY,
    IMAGE,
    LEADER_QUERY,
    NUM_TRADE_EVENTS,
    TRADE_MODE_INDEX,
    actor_observation,
    canonical_leader_state,
)
from stackelberg_pomdp.atari.sampling import ExactFiveEventSchedule
from stackelberg_pomdp.atari.envs.bilateral import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
)
from stackelberg_pomdp.atari.policies.composite import (
    ATARI_POLICY_PROVENANCE_ID,
    StackPOMDPAtariPolicy,
    canonical_atari_policy_provenance_id,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "replication/atari/results/e2_selections"
PROTOCOL_ATOL = 1.0e-6
EVALUATOR_NAME = "clean_atari_e2_selector_v2"
CANONICAL_GAMEPLAY_HORIZON = 200
SCREEN_EPISODES = 20
CONFIRMATION_EPISODES = 100
ECONOMIC_INTERVENTION_SCHEMA = "clean_atari_e2_commitment_intervention_v1"
ECONOMIC_CONTROL_COMMITMENTS = {
    "all_zero": (0.0,) * NUM_TRADE_EVENTS,
    "all_one": (1.0,) * NUM_TRADE_EVENTS,
}
ECONOMIC_GATE_MIN_FACTUAL_PAYOFF = 0.25
ECONOMIC_GATE_MIN_MEAN_ADVANTAGE = 0.25
ECONOMIC_GATE_MIN_MEDIAN_ADVANTAGE = 0.0
ECONOMIC_GATE_MIN_WIN_RATE = 0.60
ECONOMIC_GATE_MIN_TOTAL_SHOTS = 4.0
# The vendored Space Invaders ROM exposes the minimal actions
# NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE in this order.  Economic
# query/replay observations have no ammunition, so every FIRE-containing
# coordinate must be infeasible while the three non-FIRE coordinates remain
# available.  Keeping this contract explicit lets the retained rows certify
# the actor-visible event-only observation without consulting hidden env state.
CANONICAL_EVENT_ACTION_MASK = (1.0, 0.0, 1.0, 1.0, 0.0, 0.0)
ECONOMIC_GATE_HYPOTHESIS = (
    "predeclared five-bullet interior-surplus hypothesis: the learned "
    "commitment outperforms both endpoint commitments; this is diagnostic "
    "for this experiment and is not a universal equilibrium condition"
)
E1_BUYER_INIT_MEAN = 0.95
E1_BUYER_INIT_CONCENTRATION = 10.0
E1_SELLER_INIT_MEAN = 0.5
E1_SELLER_INIT_CONCENTRATION = 2.0
E2_INIT_MEAN = 0.5
E2_INIT_CONCENTRATION = 2.0
SELECTION_RULE = (
    "exclude any checkpoint with a screen protocol violation",
    "exclude any checkpoint that fails this experiment's predeclared five-bullet interior-surplus endpoint-dominance hypothesis",
    "maximize mean leader payoff",
    "maximize median leader payoff",
    "maximize minimum leader payoff",
    "minimize leader-payoff standard deviation",
    "prefer fewer training timesteps",
    "prefer lexicographically smaller checkpoint SHA-256",
    "confirm only the screen-preselected top candidate on disjoint seeds",
    "create the selected alias only after confirmation passes",
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


def load_e2_checkpoint(
        path,
        *,
        leader_role,
        device="cpu",
        expected_sha256=None,
):
    """Load and validate one deterministic E2 leader candidate."""

    path = Path(path).resolve()
    model = _load_model(
        path, device=device, expected_sha256=expected_sha256
    )
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
    loaded_hash = checkpoint_sha256(path)
    if expected_sha256 is not None and loaded_hash != expected_sha256:
        raise RuntimeError("E2 checkpoint bytes changed after model validation")
    model.e2_evaluation_loaded_checkpoint = {
        "path": str(path),
        "sha256": loaded_hash,
    }
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
    if not e2_implementation_provenance_compatible(
            scientific.get("implementation"), e2_implementation_provenance()
    ):
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
        "policy_class": (
            ATARI_POLICY_PROVENANCE_ID
            if isinstance(policy, StackPOMDPAtariPolicy)
            else f"{type(policy).__module__}.{type(policy).__qualname__}"
        ),
        "economic_role": policy.economic_role,
        "economic_input_mode": policy.economic_input_mode,
        "visual_features": int(policy.visual_features),
        "state_features": int(policy.state_features),
        "economic_hidden": int(policy.economic_hidden),
        "critic_hidden": int(policy.critic_hidden),
        "pretrained_lr_scale": float(policy.pretrained_lr_scale),
        "gameplay_actor_frozen": bool(getattr(
            policy, "gameplay_actor_frozen", False
        )),
        "game_action_count": int(policy.game_action_count),
        "actor_loss_mode": actual_actor_loss_mode,
        "economic_head_initialization": actual_economic_initialization,
    }
    recorded_leader_policy = dict(scientific.get("leader_policy", {}))
    if "policy_class" in recorded_leader_policy:
        recorded_leader_policy["policy_class"] = (
            canonical_atari_policy_provenance_id(
                recorded_leader_policy["policy_class"]
            )
        )
    recorded_leader_policy.setdefault("gameplay_actor_frozen", False)
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
    validate_e2_gameplay_actor_freeze(model, initialize=False)
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


def _normalized_commitment(values):
    if values is None:
        return None
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
            result.shape != (NUM_TRADE_EVENTS,)
            or not np.all(np.isfinite(result))
            or not np.all((0.0 <= result) & (result <= 1.0))
    ):
        raise ValueError(
            "economic commitment override must contain five finite values "
            "in [0, 1]"
        )
    return tuple(float(value) for value in result)


def _canonical_economic_transition(observation):
    """Identify a canonical E2 query/replay without using hidden env state."""

    credit = np.asarray(observation[ACTION_CREDIT], dtype=np.float32).reshape(-1)
    state = np.asarray(observation[ACTOR_STATE], dtype=np.float32).reshape(-1)
    if state.shape[0] <= max(TRADE_MODE_INDEX, EVENT_SLICE.stop - 1):
        raise RuntimeError("E2 actor state is too short for an event identity")
    trade_mode = bool(np.isclose(
        state[TRADE_MODE_INDEX], 1.0, atol=0.0, rtol=0.0
    ))
    query_credit = np.array_equal(credit, np.asarray([0.0, 1.0]))
    replay_credit = np.array_equal(credit, np.asarray([0.0, 0.0]))
    if not trade_mode:
        if query_credit:
            raise RuntimeError("economic actor credit appeared outside trade mode")
        return None
    event_values = state[EVENT_SLICE]
    active = np.flatnonzero(np.isclose(event_values, 1.0, atol=0.0, rtol=0.0))
    if (
            active.shape != (1,)
            or not np.array_equal(
                event_values,
                np.eye(NUM_TRADE_EVENTS, dtype=np.float32)[active[0]],
            )
    ):
        raise RuntimeError("canonical E2 economic state has an invalid event one-hot")
    event_index = int(active[0])
    expected_state = canonical_leader_state(event_index)
    if not np.array_equal(state, expected_state):
        raise RuntimeError("canonical E2 economic state is not event-only")
    if query_credit:
        return LEADER_QUERY, event_index
    if replay_credit:
        return CACHED_TRADE_REPLAY, event_index
    raise RuntimeError("trade-mode E2 state has invalid action credit")


def apply_economic_commitment_override(observation, action, commitment):
    """Apply one deterministic commitment to both queries and cache replays.

    The policy's Atari coordinate is never changed.  The returned decision tag
    lets the evaluator prove that exactly five queries and five corresponding
    reward-trade replays were intervened on, while gameplay was untouched.
    """

    policy_action = _action_list(action)
    normalized = _normalized_commitment(commitment)
    if normalized is None:
        return policy_action, None
    decision = _canonical_economic_transition(observation)
    if decision is None:
        return policy_action, None
    kind, event_index = decision
    effective = list(policy_action)
    effective[1] = float(normalized[event_index])
    return effective, (kind, event_index)


def economic_intervention_manifest(
        *,
        intervention_id,
        commitment,
        checkpoint_hash,
        response_hash,
        config_hash,
        provenance_fingerprint,
):
    normalized = _normalized_commitment(commitment)
    body = {
        "schema": ECONOMIC_INTERVENTION_SCHEMA,
        "intervention_id": str(intervention_id),
        "target": "leader_economic_commitment",
        "economic_commitment": (
            None if normalized is None else list(normalized)
        ),
        "apply_to": [LEADER_QUERY, CACHED_TRADE_REPLAY],
        "preserve_candidate_game_coordinate": True,
        "no_direct_gameplay_action_override": True,
        "fresh_follower_response_episode": True,
        "checkpoint_sha256": str(checkpoint_hash),
        "response_sha256": str(response_hash),
        "environment_config_sha256": str(config_hash),
        "e2_provenance_fingerprint": str(provenance_fingerprint),
    }
    return {
        **body,
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
    }


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
        economic_intervention,
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
            override_counts = {
                LEADER_QUERY: np.zeros(NUM_TRADE_EVENTS, dtype=np.int64),
                CACHED_TRADE_REPLAY: np.zeros(
                    NUM_TRADE_EVENTS, dtype=np.int64
                ),
            }
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
                policy_action = _action_list(action)
                requested_action, overridden = apply_economic_commitment_override(
                    observation,
                    policy_action,
                    economic_intervention["economic_commitment"],
                )
                if overridden is not None:
                    override_kind, override_event = overridden
                    override_counts[override_kind][override_event] += 1
                observation, reward, done, terminal_info = env.step(
                    np.asarray(requested_action, dtype=np.float32)
                )
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
                    "economic_intervention_id": economic_intervention[
                        "intervention_id"
                    ],
                    "economic_intervention_sha256": economic_intervention[
                        "manifest_sha256"
                    ],
                    "economic_commitment_override": economic_intervention[
                        "economic_commitment"
                    ],
                    "economic_override_applied": overridden is not None,
                    "economic_override_kind": (
                        None if overridden is None else overridden[0]
                    ),
                    "economic_override_event": (
                        None if overridden is None else overridden[1]
                    ),
                    "policy_action_before_intervention": policy_action,
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
                "economic_intervention_id": economic_intervention[
                    "intervention_id"
                ],
                "economic_intervention_sha256": economic_intervention[
                    "manifest_sha256"
                ],
                "economic_commitment_override": economic_intervention[
                    "economic_commitment"
                ],
                "economic_override_query_applications": int(
                    np.sum(override_counts[LEADER_QUERY])
                ),
                "economic_override_replay_applications": int(
                    np.sum(override_counts[CACHED_TRADE_REPLAY])
                ),
                "economic_override_query_event_counts": override_counts[
                    LEADER_QUERY
                ].tolist(),
                "economic_override_replay_event_counts": override_counts[
                    CACHED_TRADE_REPLAY
                ].tolist(),
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
    purchases = np.asarray([float(row["purchases"]) for row in rows])
    payments = np.asarray([float(row["payments"]) for row in rows])
    seller_shots = np.asarray([
        float(row["seller_shots_fired"]) for row in rows
    ])
    buyer_shots = np.asarray([
        float(row["buyer_shots_fired"]) for row in rows
    ])
    retained_bullets = NUM_TRADE_EVENTS - purchases
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
        "mean_payments": float(np.mean(payments)),
        "mean_purchases": float(np.mean(purchases)),
        "mean_purchase_rate": float(
            np.mean(purchases) / NUM_TRADE_EVENTS
        ),
        "mean_accepted_price": float(
            np.sum(payments) / np.sum(purchases)
        ) if float(np.sum(purchases)) > 0.0 else 0.0,
        "mean_seller_shots_fired": float(np.mean(seller_shots)),
        "mean_buyer_shots_fired": float(np.mean(buyer_shots)),
        "mean_total_shots_fired": float(np.mean(seller_shots + buyer_shots)),
        "buyer_purchased_bullet_utilization": float(
            np.sum(buyer_shots) / np.sum(purchases)
        ) if float(np.sum(purchases)) > 0.0 else 0.0,
        "mean_seller_retained_bullets": float(np.mean(retained_bullets)),
        "seller_retained_bullet_utilization": float(
            np.sum(seller_shots) / np.sum(retained_bullets)
        ) if float(np.sum(retained_bullets)) > 0.0 else 0.0,
        "mean_seller_final_ammo": float(np.mean([
            float(row["seller_final_ammo"]) for row in rows
        ])),
        "mean_buyer_final_ammo": float(np.mean([
            float(row["buyer_final_ammo"]) for row in rows
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
    intervention_id = row.get("economic_intervention_id")
    override = row.get("economic_commitment_override")
    if not isinstance(intervention_id, str) or not intervention_id:
        violations.append(_violation(
            row, "economic_intervention_id", "nonempty string", intervention_id
        ))
    intervention_sha = row.get("economic_intervention_sha256")
    if (
            not isinstance(intervention_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", intervention_sha) is None
    ):
        violations.append(_violation(
            row,
            "economic_intervention_sha256",
            "64 lowercase hex characters",
            intervention_sha,
        ))
    normalized_override = None
    try:
        normalized_override = _normalized_commitment(override)
    except ValueError:
        violations.append(_violation(
            row,
            "economic_commitment_override",
            "null or five values in [0,1]",
            override,
        ))
    expected_applications = 0 if normalized_override is None else NUM_TRADE_EVENTS
    for field in (
            "economic_override_query_applications",
            "economic_override_replay_applications",
    ):
        if row.get(field) != expected_applications:
            violations.append(_violation(
                row, field, expected_applications, row.get(field)
            ))
    expected_event_counts = [
        0 if normalized_override is None else 1
    ] * NUM_TRADE_EVENTS
    for field in (
            "economic_override_query_event_counts",
            "economic_override_replay_event_counts",
    ):
        if row.get(field) != expected_event_counts:
            violations.append(_violation(
                row, field, expected_event_counts, row.get(field)
            ))
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
    if normalized_override is not None:
        expected_commitment = list(normalized_override)
        if (
                not isinstance(commitment, list)
                or len(commitment) != NUM_TRADE_EVENTS
                or not np.allclose(
                np.asarray(commitment, dtype=float),
                np.asarray(expected_commitment, dtype=float),
                rtol=0.0,
                atol=PROTOCOL_ATOL,
                )
        ):
            violations.append(_violation(
                row,
                "leader commitment == economic override",
                expected_commitment,
                commitment,
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


def _audit_transitions(
        rows,
        decisions,
        episode_row,
        *,
        gameplay_horizon,
        leader_role,
):
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
    try:
        override = _normalized_commitment(
            episode_row.get("economic_commitment_override")
        )
    except ValueError:
        override = None
    for row in selected:
        substep = row["substep_type"]
        for field in (
                "economic_intervention_id",
                "economic_intervention_sha256",
                "economic_commitment_override",
        ):
            if row.get(field) != episode_row.get(field):
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} {field}",
                    episode_row.get(field),
                    row.get(field),
                ))
        try:
            policy_action = _action_list(
                row.get("policy_action_before_intervention")
            )
            requested_action = _action_list(row.get("requested_action"))
        except RuntimeError:
            violations.append(_violation(
                episode_row,
                f"transition {row['transition_index']} intervention action",
                "two finite full actions",
                {
                    "policy": row.get("policy_action_before_intervention"),
                    "requested": row.get("requested_action"),
                },
            ))
            policy_action = requested_action = None
        if (
                policy_action is not None
                and not _close(policy_action[0], requested_action[0])
        ):
            violations.append(_violation(
                episode_row,
                f"transition {row['transition_index']} preserved game action",
                policy_action[0],
                requested_action[0],
            ))
        expected_override = bool(
            override is not None
            and substep in (LEADER_QUERY, CACHED_TRADE_REPLAY)
        )
        if bool(row.get("economic_override_applied")) != expected_override:
            violations.append(_violation(
                episode_row,
                f"transition {row['transition_index']} override application",
                expected_override,
                row.get("economic_override_applied"),
            ))
        if not expected_override:
            if (
                    policy_action is not None
                    and not np.array_equal(
                        np.asarray(requested_action, dtype=np.float64),
                        np.asarray(policy_action, dtype=np.float64),
                    )
            ):
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} "
                    "no economic override action identity",
                    policy_action,
                    requested_action,
                ))
            for field in (
                    "economic_override_kind", "economic_override_event"
            ):
                if row.get(field) is not None:
                    violations.append(_violation(
                        episode_row,
                        f"transition {row['transition_index']} {field}",
                        None,
                        row.get(field),
                    ))
        if expected_override and requested_action is not None:
            event_index = (
                row.get("query_index")
                if substep == LEADER_QUERY
                else row.get("event_index")
            )
            if row.get("economic_override_kind") != substep:
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} override kind",
                    substep,
                    row.get("economic_override_kind"),
                ))
            if row.get("economic_override_event") != event_index:
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} override event",
                    event_index,
                    row.get("economic_override_event"),
                ))
            if (
                    not isinstance(event_index, (int, np.integer))
                    or not 0 <= int(event_index) < NUM_TRADE_EVENTS
                    or not _close(requested_action[1], override[int(event_index)])
            ):
                expected_economic = (
                    override[int(event_index)]
                    if isinstance(event_index, (int, np.integer))
                    and 0 <= int(event_index) < NUM_TRADE_EVENTS
                    else "valid event-indexed override"
                )
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} overridden economics",
                    expected_economic,
                    requested_action[1],
                ))
        if substep in (LEADER_QUERY, CACHED_TRADE_REPLAY):
            try:
                action_mask = np.asarray(
                    row.get("action_mask"), dtype=np.float64
                ).reshape(-1)
            except (TypeError, ValueError):
                action_mask = np.asarray([], dtype=np.float64)
            required_mask = np.asarray(
                CANONICAL_EVENT_ACTION_MASK, dtype=np.float64
            )
            if not np.array_equal(action_mask, required_mask):
                violations.append(_violation(
                    episode_row,
                    f"transition {row['transition_index']} "
                    "canonical event-only action mask",
                    required_mask.tolist(),
                    row.get("action_mask"),
                ))
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
        if not np.allclose(
                query.get("policy_action_before_intervention"),
                replay.get("policy_action_before_intervention"),
                rtol=0.0,
                atol=PROTOCOL_ATOL,
        ):
            violations.append(_violation(
                episode_row,
                f"event {event} pre-intervention policy cache identity",
                query.get("policy_action_before_intervention"),
                replay.get("policy_action_before_intervention"),
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
        events = episode_row.get("events")
        event_row = (
            events[event]
            if isinstance(events, list) and event < len(events)
            else None
        )
        if isinstance(event_row, dict):
            if replay.get("game_step") != event_row.get("game_step"):
                violations.append(_violation(
                    episode_row,
                    f"event {event} replay game step",
                    event_row.get("game_step"),
                    replay.get("game_step"),
                ))
            price = _number(event_row.get("price"))
            accepted = event_row.get("accepted")
            replay_reward = _number(replay.get("reward"))
            if price is not None and isinstance(accepted, bool):
                signed_payment = price if accepted else 0.0
                if leader_role == BUYER:
                    signed_payment = -signed_payment
                if (
                        replay_reward is None
                        or not _close(replay_reward, signed_payment)
                ):
                    violations.append(_violation(
                        episode_row,
                        f"event {event} immediate {leader_role} trade reward",
                        signed_payment,
                        replay.get("reward"),
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
    expected_decision_rows = [
        row for row in transition_rows
        if row.get("substep_type") in (LEADER_QUERY, CACHED_TRADE_REPLAY)
    ]
    try:
        decisions_match_transitions = (
            _canonical_json_bytes(decision_rows)
            == _canonical_json_bytes(expected_decision_rows)
        )
    except (TypeError, ValueError):
        decisions_match_transitions = False
    if not decisions_match_transitions:
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "decision rows equal economic transition subset",
            "expected": expected_decision_rows,
            "actual": decision_rows,
        })

    first_query = next((
        row for row in transition_rows
        if row.get("substep_type") == LEADER_QUERY
    ), None)
    observed_event_mask = (
        None if first_query is None else first_query.get("action_mask")
    )
    try:
        normalized_event_mask = np.asarray(
            observed_event_mask, dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError):
        normalized_event_mask = np.asarray([], dtype=np.float64)
    if not np.array_equal(
            normalized_event_mask,
            np.asarray(CANONICAL_EVENT_ACTION_MASK, dtype=np.float64),
    ):
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "derived canonical event-only action mask",
            "expected": list(CANONICAL_EVENT_ACTION_MASK),
            "actual": observed_event_mask,
        })
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
            leader_role=leader_role,
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
        "canonical_event_action_mask": (
            normalized_event_mask.tolist()
            if np.array_equal(
                normalized_event_mask,
                np.asarray(CANONICAL_EVENT_ACTION_MASK, dtype=np.float64),
            )
            else None
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
        intervention_id="factual",
        economic_commitment_override=None,
        expected_checkpoint_sha256=None,
):
    checkpoint = Path(checkpoint).resolve()
    digest = checkpoint_sha256(checkpoint)
    if (
            expected_checkpoint_sha256 is not None
            and digest != expected_checkpoint_sha256
    ):
        raise RuntimeError("E2 candidate bytes changed before evaluation")
    loaded = getattr(model, "e2_evaluation_loaded_checkpoint", None)
    if not isinstance(loaded, dict) or loaded != {
            "path": str(checkpoint), "sha256": digest,
    }:
        raise RuntimeError(
            "E2 in-memory model is not bound to the reported checkpoint bytes"
        )
    provenance = validate_candidate_provenance(
        model, response_hash=response_hash, config=config
    )
    actor_loss_mode = model_actor_loss_mode(model)
    economic_initialization = model_economic_initialization(
        model,
        default_mean=E2_INIT_MEAN,
        default_concentration=E2_INIT_CONCENTRATION,
    )
    intervention = economic_intervention_manifest(
        intervention_id=intervention_id,
        commitment=economic_commitment_override,
        checkpoint_hash=digest,
        response_hash=response_hash,
        config_hash=config_hash,
        provenance_fingerprint=provenance["fingerprint_sha256"],
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
        economic_intervention=intervention,
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
    if checkpoint_sha256(checkpoint) != digest:
        raise RuntimeError("E2 candidate bytes changed during evaluation")
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
        "economic_intervention": intervention,
        "phase": phase,
        "seed_start": int(seed_start),
        "seed_end": int(seed_start) + int(episodes) - 1,
        "summary": leader_outcome_summary(evaluation["episode_rows"]),
        "protocol": protocol,
        **evaluation,
    }


def evaluate_endpoint_controls(
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
        expected_checkpoint_sha256,
):
    """Evaluate both canonical endpoint commitments on fresh matched episodes."""

    return [
        evaluate_checkpoint(
            model,
            checkpoint,
            args=args,
            response_model=response_model,
            response_hash=response_hash,
            config=config,
            config_hash=config_hash,
            episodes=episodes,
            seed_start=seed_start,
            phase=phase,
            intervention_id=intervention_id,
            economic_commitment_override=commitment,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        for intervention_id, commitment in ECONOMIC_CONTROL_COMMITMENTS.items()
    ]


def _gate_check(name, actual, relation, target):
    passed = {
        ">=": float(actual) >= float(target),
        ">": float(actual) > float(target),
    }[relation]
    return {
        "name": str(name),
        "actual": float(actual),
        "relation": relation,
        "target": float(target),
        "passed": bool(passed),
    }


def paired_economic_gate(factual, controls, *, required_episodes):
    """Require a deterministic leader to beat both endpoint commitments."""

    controls = list(controls)
    by_id = {
        result.get("economic_intervention", {}).get("intervention_id"): result
        for result in controls
    }
    expected_ids = set(ECONOMIC_CONTROL_COMMITMENTS)
    errors = []
    if len(by_id) != len(controls) or set(by_id) != expected_ids:
        errors.append({
            "field": "counterfactual_conditions",
            "expected": sorted(expected_ids),
            "actual": sorted(str(value) for value in by_id),
        })
    factual_manifest = factual.get("economic_intervention", {})
    if (
            factual_manifest.get("intervention_id") != "factual"
            or factual_manifest.get("economic_commitment") is not None
    ):
        errors.append({
            "field": "factual_intervention",
            "expected": {"intervention_id": "factual", "commitment": None},
            "actual": factual_manifest,
        })

    def check_episode_intervention(result, rows, label):
        manifest = result.get("economic_intervention", {})
        for seed, row in rows.items():
            expected = {
                "economic_intervention_id": manifest.get("intervention_id"),
                "economic_intervention_sha256": manifest.get(
                    "manifest_sha256"
                ),
                "economic_commitment_override": manifest.get(
                    "economic_commitment"
                ),
            }
            for field, value in expected.items():
                if row.get(field) != value:
                    errors.append({
                        "field": f"{label}.episode.{field}",
                        "evaluation_seed": seed,
                        "expected": value,
                        "actual": row.get(field),
                    })

    identity_fields = (
        "checkpoint_sha256",
        "response_checkpoint_sha256",
        "environment_config_sha256",
        "e2_provenance_fingerprint",
    )
    factual_rows = {
        int(row["evaluation_seed"]): row for row in factual.get("episode_rows", ())
    }
    check_episode_intervention(factual, factual_rows, "factual")
    if len(factual_rows) != int(required_episodes):
        errors.append({
            "field": "factual_seed_count",
            "expected": int(required_episodes),
            "actual": len(factual_rows),
        })
    expected_seeds = set(range(
        int(factual.get("seed_start", -1)),
        int(factual.get("seed_start", -1)) + int(required_episodes),
    ))
    if set(factual_rows) != expected_seeds:
        errors.append({
            "field": "factual_seeds",
            "expected": sorted(expected_seeds),
            "actual": sorted(factual_rows),
        })

    control_rows = {}
    for intervention_id in sorted(expected_ids & set(by_id)):
        result = by_id[intervention_id]
        for field in identity_fields:
            if result.get(field) != factual.get(field):
                errors.append({
                    "field": f"{intervention_id}.{field}",
                    "expected": factual.get(field),
                    "actual": result.get(field),
                })
        manifest = result.get("economic_intervention", {})
        expected_commitment = list(
            ECONOMIC_CONTROL_COMMITMENTS[intervention_id]
        )
        if manifest.get("economic_commitment") != expected_commitment:
            errors.append({
                "field": f"{intervention_id}.economic_commitment",
                "expected": expected_commitment,
                "actual": manifest.get("economic_commitment"),
            })
        rows = {
            int(row["evaluation_seed"]): row
            for row in result.get("episode_rows", ())
        }
        control_rows[intervention_id] = rows
        check_episode_intervention(result, rows, intervention_id)
        if set(rows) != expected_seeds:
            errors.append({
                "field": f"{intervention_id}.seeds",
                "expected": sorted(expected_seeds),
                "actual": sorted(rows),
            })
        for seed in sorted(expected_seeds & set(rows) & set(factual_rows)):
            factual_schedule = tuple(factual_rows[seed].get("event_steps", ()))
            control_schedule = tuple(rows[seed].get("event_steps", ()))
            if control_schedule != factual_schedule:
                errors.append({
                    "field": f"{intervention_id}.event_steps",
                    "evaluation_seed": seed,
                    "expected": list(factual_schedule),
                    "actual": list(control_schedule),
                })

    mechanics_passed = bool(
        factual.get("protocol", {}).get("passed")
        and len(controls) == len(expected_ids)
        and all(
            result.get("protocol", {}).get("passed") for result in controls
        )
        and not errors
    )
    paired_rows = []
    if set(control_rows) == expected_ids:
        for seed in sorted(
                expected_seeds
                & set(factual_rows)
                & set(control_rows["all_zero"])
                & set(control_rows["all_one"])
        ):
            factual_row = factual_rows[seed]
            zero_row = control_rows["all_zero"][seed]
            one_row = control_rows["all_one"][seed]
            factual_return = float(factual_row["leader_reward"])
            zero_return = float(zero_row["leader_reward"])
            one_return = float(one_row["leader_reward"])
            paired_rows.append({
                "checkpoint_path": factual["checkpoint_path"],
                "checkpoint_sha256": factual["checkpoint_sha256"],
                "response_sha256": factual["response_checkpoint_sha256"],
                "environment_config_sha256": factual[
                    "environment_config_sha256"
                ],
                "e2_provenance_fingerprint": factual[
                    "e2_provenance_fingerprint"
                ],
                "phase": factual["phase"],
                "evaluation_seed": seed,
                "event_steps": list(factual_row["event_steps"]),
                "factual_leader_payoff": factual_return,
                "all_zero_leader_payoff": zero_return,
                "all_one_leader_payoff": one_return,
                "factual_minus_all_zero": factual_return - zero_return,
                "factual_minus_all_one": factual_return - one_return,
                "factual_purchases": factual_row.get("purchases"),
                "factual_payments": factual_row.get("payments"),
                "factual_seller_shots_fired": factual_row.get(
                    "seller_shots_fired"
                ),
                "factual_buyer_shots_fired": factual_row.get(
                    "buyer_shots_fired"
                ),
            })
    if len(paired_rows) != int(required_episodes):
        errors.append({
            "field": "paired_episode_count",
            "expected": int(required_episodes),
            "actual": len(paired_rows),
        })
        mechanics_passed = False

    checks = []
    if paired_rows:
        factual_payoffs = np.asarray([
            row["factual_leader_payoff"] for row in paired_rows
        ], dtype=np.float64)
        checks.append(_gate_check(
            "factual mean leader payoff",
            np.mean(factual_payoffs),
            ">=",
            ECONOMIC_GATE_MIN_FACTUAL_PAYOFF,
        ))
        checks.append(_gate_check(
            "factual mean total bullets fired",
            np.mean([
                float(row["factual_seller_shots_fired"])
                + float(row["factual_buyer_shots_fired"])
                for row in paired_rows
            ]),
            ">=",
            ECONOMIC_GATE_MIN_TOTAL_SHOTS,
        ))
        for intervention_id in ("all_zero", "all_one"):
            differences = np.asarray([
                row[f"factual_minus_{intervention_id}"] for row in paired_rows
            ], dtype=np.float64)
            checks.extend((
                _gate_check(
                    f"mean paired advantage over {intervention_id}",
                    np.mean(differences),
                    ">=",
                    ECONOMIC_GATE_MIN_MEAN_ADVANTAGE,
                ),
                _gate_check(
                    f"median paired advantage over {intervention_id}",
                    np.median(differences),
                    ">=",
                    ECONOMIC_GATE_MIN_MEDIAN_ADVANTAGE,
                ),
                _gate_check(
                    f"paired win rate over {intervention_id}",
                    np.mean(differences > PROTOCOL_ATOL),
                    ">=",
                    ECONOMIC_GATE_MIN_WIN_RATE,
                ),
            ))
    return {
        "hypothesis": ECONOMIC_GATE_HYPOTHESIS,
        "scope": "current five-bullet Atari experiment only",
        "passed": bool(
            mechanics_passed and checks and all(row["passed"] for row in checks)
        ),
        "mechanics_passed": mechanics_passed,
        "required_episodes": int(required_episodes),
        "thresholds": {
            "minimum_factual_mean_leader_payoff": (
                ECONOMIC_GATE_MIN_FACTUAL_PAYOFF
            ),
            "minimum_mean_paired_advantage": (
                ECONOMIC_GATE_MIN_MEAN_ADVANTAGE
            ),
            "minimum_median_paired_advantage": (
                ECONOMIC_GATE_MIN_MEDIAN_ADVANTAGE
            ),
            "minimum_paired_win_rate": ECONOMIC_GATE_MIN_WIN_RATE,
            "minimum_factual_mean_total_bullets_fired": (
                ECONOMIC_GATE_MIN_TOTAL_SHOTS
            ),
        },
        "checks": checks,
        "errors": errors,
        "paired_rows": paired_rows,
        "condition_summaries": {
            "factual": factual.get("summary"),
            **{
                intervention_id: by_id[intervention_id].get("summary")
                for intervention_id in sorted(expected_ids & set(by_id))
            },
        },
    }


def attach_economic_gate(factual, controls, *, required_episodes):
    return {
        **factual,
        "counterfactual_controls": list(controls),
        "economic_gate": paired_economic_gate(
            factual, controls, required_episodes=required_episodes
        ),
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
        (
            result for result in results
            if result["protocol"]["passed"]
            and result.get("economic_gate", {}).get("passed", False)
        ),
        key=_selection_key,
    )
    ranks = {result["checkpoint_sha256"]: index + 1 for index, result in enumerate(valid)}
    rows = []
    for result in results:
        rows.append({
            "rank": ranks.get(result["checkpoint_sha256"]),
            "eligible": bool(
                result["protocol"]["passed"]
                and result.get("economic_gate", {}).get("passed", False)
            ),
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
            "economic_gate_passed": bool(
                result.get("economic_gate", {}).get("passed", False)
            ),
            "economic_gate_checks": result.get(
                "economic_gate", {}
            ).get("checks", []),
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
        "screen_ranked_checkpoint_sha256s": [
            result["checkpoint_sha256"] for result in valid
        ],
        "ranking_rows": rows,
    }


def atomic_copy_no_overwrite(source, destination):
    """Copy exact bytes to a new alias without ever replacing a prior file."""

    source = Path(source).resolve()
    destination = Path(destination).expanduser()
    if destination.suffix != ".zip":
        destination = destination.with_suffix(".zip")
    destination = destination.parent.resolve() / destination.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite selected checkpoint alias: {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    linked_identity = None
    copied_hash = None
    try:
        source_hash = checkpoint_sha256(source)
        shutil.copy2(source, temporary)
        copied_hash = checkpoint_sha256(temporary)
        if copied_hash != source_hash or checkpoint_sha256(source) != source_hash:
            raise RuntimeError(
                "source checkpoint changed while the selected alias was copied"
            )
        os.link(temporary, destination)
        linked = destination.stat()
        linked_identity = (
            linked.st_dev,
            linked.st_ino,
            linked.st_size,
            linked.st_mtime_ns,
        )
        destination_hash = checkpoint_sha256(destination)
        if destination_hash != source_hash:
            raise RuntimeError(
                "selected checkpoint copy failed its SHA-256 check"
            )
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite selected checkpoint alias: {destination}"
        ) from error
    except BaseException:
        if linked_identity is not None and copied_hash is not None:
            try:
                before = destination.stat()
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                current_hash = checkpoint_sha256(destination)
                after = destination.stat()
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if (
                        before_identity == linked_identity == after_identity
                        and current_hash == copied_hash
                ):
                    destination.unlink()
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source_checkpoint_path": str(source),
        "selected_checkpoint_path": str(destination),
        "checkpoint_sha256": destination_hash,
        "copy_verified": True,
        "created_by_selection_run": True,
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
        "economic_gate_passed": bool(
            confirmation_result.get("economic_gate", {}).get("passed", False)
        ),
    }
    screen_controls = {
        result["economic_intervention"]["intervention_id"]: result
        for result in screen_result.get("counterfactual_controls", ())
    }
    confirmation_controls = {
        result["economic_intervention"]["intervention_id"]: result
        for result in confirmation_result.get("counterfactual_controls", ())
    }
    checks["counterfactual_condition_set_matches_screen"] = bool(
        set(screen_controls) == set(ECONOMIC_CONTROL_COMMITMENTS)
        and set(confirmation_controls) == set(screen_controls)
    )
    if checks["counterfactual_condition_set_matches_screen"]:
        for intervention_id in sorted(screen_controls):
            screen_control = screen_controls[intervention_id]
            confirmation_control = confirmation_controls[intervention_id]
            for label, left, right in (
                    (
                        "intervention",
                        screen_control["economic_intervention"][
                            "manifest_sha256"
                        ],
                        confirmation_control["economic_intervention"][
                            "manifest_sha256"
                        ],
                    ),
                    (
                        "query_trace",
                        screen_control["protocol"]["query_trace_sha256"],
                        confirmation_control["protocol"][
                            "query_trace_sha256"
                        ],
                    ),
                    (
                        "commitment",
                        screen_control["protocol"]["leader_commitment"],
                        confirmation_control["protocol"][
                            "leader_commitment"
                        ],
                    ),
                    (
                        "full_query_actions",
                        screen_control["protocol"]["full_query_actions"],
                        confirmation_control["protocol"][
                            "full_query_actions"
                        ],
                    ),
            ):
                checks[
                    f"{intervention_id}_{label}_matches_screen"
                ] = left == right
    return {"passed": all(checks.values()), **checks}


def _ranges_overlap(start_a, count_a, start_b, count_b):
    end_a = int(start_a) + int(count_a) - 1
    end_b = int(start_b) + int(count_b) - 1
    return max(int(start_a), int(start_b)) <= min(end_a, end_b)


def assert_evaluation_inputs_unchanged(*, response_path, response_hash, config):
    """Fail if the frozen response or ROM changed during E2 evaluation."""

    if checkpoint_sha256(response_path) != response_hash:
        raise RuntimeError("frozen E1 response bytes changed during evaluation")
    if checkpoint_sha256(config["rom_path"]) != config["rom_sha256"]:
        raise RuntimeError("Space Invaders ROM bytes changed during evaluation")


def run_selection(args):
    """Screen, confirm only its top candidate, then alias only after a pass."""

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
    for checkpoint, expected_hash in zip(checkpoints, hashes):
        model = load_e2_checkpoint(
            checkpoint,
            leader_role=args.leader_role,
            device=args.device,
            expected_sha256=expected_hash,
        )
        try:
            factual = evaluate_checkpoint(
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
                expected_checkpoint_sha256=expected_hash,
            )
            controls = (
                evaluate_endpoint_controls(
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
                    expected_checkpoint_sha256=expected_hash,
                )
                if factual["protocol"]["passed"]
                else []
            )
            screen_results.append(attach_economic_gate(
                factual, controls, required_episodes=args.screen_episodes
            ))
        finally:
            del model
    common_screen = validate_common_screen(screen_results)
    ranking = rank_candidates(screen_results)
    screen_selected_hash = ranking["selected_checkpoint_sha256"]
    ranking = {
        **ranking,
        "screen_selected_checkpoint_sha256": screen_selected_hash,
    }
    if screen_selected_hash is None:
        assert_evaluation_inputs_unchanged(
            response_path=response_path,
            response_hash=response_hash,
            config=config,
        )
        return {
            "schema_version": 2,
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
            "confirmation_attempts": [],
            "passed": False,
        }

    selected = next(
        result for result in screen_results
        if result["checkpoint_sha256"] == screen_selected_hash
    )
    selected_alias = None
    confirmation_model = load_e2_checkpoint(
        selected["checkpoint_path"],
        leader_role=args.leader_role,
        device=args.device,
        expected_sha256=screen_selected_hash,
    )
    try:
        factual = evaluate_checkpoint(
            confirmation_model,
            selected["checkpoint_path"],
            args=args,
            response_model=response_model,
            response_hash=response_hash,
            config=config,
            config_hash=config_hash,
            episodes=args.confirmation_episodes,
            seed_start=args.confirmation_seed_start,
            phase="confirmation",
            expected_checkpoint_sha256=screen_selected_hash,
        )
        controls = (
            evaluate_endpoint_controls(
                confirmation_model,
                selected["checkpoint_path"],
                args=args,
                response_model=response_model,
                response_hash=response_hash,
                config=config,
                config_hash=config_hash,
                episodes=args.confirmation_episodes,
                seed_start=args.confirmation_seed_start,
                phase="confirmation",
                expected_checkpoint_sha256=screen_selected_hash,
            )
            if factual["protocol"]["passed"]
            else []
        )
        confirmed = attach_economic_gate(
            factual,
            controls,
            required_episodes=args.confirmation_episodes,
        )
    finally:
        del confirmation_model
    confirmation_check = confirmation_matches_screen(selected, confirmed)
    confirmation = {
        "screen_rank": 1,
        "episodes": int(args.confirmation_episodes),
        "seed_start": int(args.confirmation_seed_start),
        "seed_end": (
            int(args.confirmation_seed_start) + args.confirmation_episodes - 1
        ),
        "disjoint_from_screen": True,
        "result": confirmed,
        "checks": confirmation_check,
    }
    selected_hash = None
    if confirmation_check["passed"]:
        selected_alias = atomic_copy_no_overwrite(
            selected["checkpoint_path"], args.selected_checkpoint
        )
        if selected_alias["checkpoint_sha256"] != screen_selected_hash:
            rollback_new_selected_alias(
                {"selected_alias": selected_alias},
                expected_path=args.selected_checkpoint,
            )
            raise RuntimeError(
                "selected E2 checkpoint bytes changed after confirmation"
            )
        selected_hash = screen_selected_hash
    ranking = {
        **ranking,
        "selected_checkpoint_sha256": selected_hash,
    }
    assert_evaluation_inputs_unchanged(
        response_path=response_path,
        response_hash=response_hash,
        config=config,
    )
    return {
        "schema_version": 2,
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
        "confirmation": confirmation,
        "confirmation_attempts": [confirmation],
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
                    "economic_intervention_id": result.get(
                        "economic_intervention", {}
                    ).get("intervention_id"),
                    "economic_intervention_sha256": result.get(
                        "economic_intervention", {}
                    ).get("manifest_sha256"),
                    "economic_commitment_override": result.get(
                        "economic_intervention", {}
                    ).get("economic_commitment"),
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


def _counterfactual_results(results):
    return [
        control
        for result in results
        for control in result.get("counterfactual_controls", ())
    ]


def _condition_rows(results):
    rows = []
    for factual in results:
        conditions = [factual, *factual.get("counterfactual_controls", ())]
        for result in conditions:
            intervention = result.get("economic_intervention", {})
            rows.append({
                "phase": result.get("phase"),
                "checkpoint_path": result.get("checkpoint_path"),
                "checkpoint_sha256": result.get("checkpoint_sha256"),
                "response_sha256": result.get(
                    "response_checkpoint_sha256"
                ),
                "environment_config_sha256": result.get(
                    "environment_config_sha256"
                ),
                "e2_provenance_fingerprint": result.get(
                    "e2_provenance_fingerprint"
                ),
                "intervention_id": intervention.get("intervention_id"),
                "intervention_sha256": intervention.get("manifest_sha256"),
                "economic_commitment": intervention.get(
                    "economic_commitment"
                ),
                "protocol_passed": result.get("protocol", {}).get("passed"),
                **(result.get("summary") or {}),
            })
    return rows


def _paired_gate_rows(results):
    return [
        row
        for result in results
        for row in result.get("economic_gate", {}).get("paired_rows", ())
    ]


def _artifact_paths(output_dir, stem, *, has_confirmation):
    paths = {
        "report_json": output_dir / f"{stem}.json",
        "ranking_csv": output_dir / f"{stem}.ranking.csv",
        "screen_episodes_csv": output_dir / f"{stem}.screen.episodes.csv",
        "screen_transitions_csv": output_dir / f"{stem}.screen.transitions.csv",
        "screen_decisions_csv": output_dir / f"{stem}.screen.decisions.csv",
        "screen_events_csv": output_dir / f"{stem}.screen.events.csv",
        "screen_counterfactual_conditions_csv": (
            output_dir / f"{stem}.screen.counterfactual.conditions.csv"
        ),
        "screen_counterfactual_paired_csv": (
            output_dir / f"{stem}.screen.counterfactual.paired.csv"
        ),
        "screen_counterfactual_episodes_csv": (
            output_dir / f"{stem}.screen.counterfactual.episodes.csv"
        ),
        "screen_counterfactual_transitions_csv": (
            output_dir / f"{stem}.screen.counterfactual.transitions.csv"
        ),
        "screen_counterfactual_decisions_csv": (
            output_dir / f"{stem}.screen.counterfactual.decisions.csv"
        ),
        "screen_counterfactual_events_csv": (
            output_dir / f"{stem}.screen.counterfactual.events.csv"
        ),
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
            "confirmation_counterfactual_conditions_csv": (
                output_dir
                / f"{stem}.confirmation.counterfactual.conditions.csv"
            ),
            "confirmation_counterfactual_paired_csv": (
                output_dir / f"{stem}.confirmation.counterfactual.paired.csv"
            ),
            "confirmation_counterfactual_episodes_csv": (
                output_dir / f"{stem}.confirmation.counterfactual.episodes.csv"
            ),
            "confirmation_counterfactual_transitions_csv": (
                output_dir
                / f"{stem}.confirmation.counterfactual.transitions.csv"
            ),
            "confirmation_counterfactual_decisions_csv": (
                output_dir
                / f"{stem}.confirmation.counterfactual.decisions.csv"
            ),
            "confirmation_counterfactual_events_csv": (
                output_dir / f"{stem}.confirmation.counterfactual.events.csv"
            ),
        })
    return paths


def _selection_artifact_tables(report):
    """Return nonempty CSV tables; absent counterfactuals stay absent."""

    screen_results = report["screen"]["checkpoint_results"]
    screen_controls = _counterfactual_results(screen_results)
    tables = {
        "ranking_csv": report["selection"]["ranking_rows"],
        "screen_episodes_csv": [
            row for result in screen_results for row in result["episode_rows"]
        ],
        "screen_transitions_csv": [
            row
            for result in screen_results
            for row in result["transition_rows"]
        ],
        "screen_decisions_csv": [
            row for result in screen_results for row in result["decision_rows"]
        ],
        "screen_events_csv": _event_rows(screen_results),
    }
    if screen_controls:
        tables.update({
            "screen_counterfactual_conditions_csv": _condition_rows(
                screen_results
            ),
            "screen_counterfactual_paired_csv": _paired_gate_rows(
                screen_results
            ),
            "screen_counterfactual_episodes_csv": [
                row
                for result in screen_controls
                for row in result["episode_rows"]
            ],
            "screen_counterfactual_transitions_csv": [
                row
                for result in screen_controls
                for row in result["transition_rows"]
            ],
            "screen_counterfactual_decisions_csv": [
                row
                for result in screen_controls
                for row in result["decision_rows"]
            ],
            "screen_counterfactual_events_csv": _event_rows(screen_controls),
        })

    confirmation_attempts = report.get("confirmation_attempts") or ()
    if confirmation_attempts:
        confirmation_results = [
            attempt["result"] for attempt in confirmation_attempts
        ]
        confirmation_controls = _counterfactual_results(
            confirmation_results
        )
        tables.update({
            "confirmation_episodes_csv": [
                row
                for result in confirmation_results
                for row in result["episode_rows"]
            ],
            "confirmation_transitions_csv": [
                row
                for result in confirmation_results
                for row in result["transition_rows"]
            ],
            "confirmation_decisions_csv": [
                row
                for result in confirmation_results
                for row in result["decision_rows"]
            ],
            "confirmation_events_csv": _event_rows(confirmation_results),
        })
        if confirmation_controls:
            tables.update({
                "confirmation_counterfactual_conditions_csv": _condition_rows(
                    confirmation_results
                ),
                "confirmation_counterfactual_paired_csv": _paired_gate_rows(
                    confirmation_results
                ),
                "confirmation_counterfactual_episodes_csv": [
                    row
                    for result in confirmation_controls
                    for row in result["episode_rows"]
                ],
                "confirmation_counterfactual_transitions_csv": [
                    row
                    for result in confirmation_controls
                    for row in result["transition_rows"]
                ],
                "confirmation_counterfactual_decisions_csv": [
                    row
                    for result in confirmation_controls
                    for row in result["decision_rows"]
                ],
                "confirmation_counterfactual_events_csv": _event_rows(
                    confirmation_controls
                ),
            })
    return {
        key: [_csv_row(row) for row in rows]
        for key, rows in tables.items()
        if rows
    }


def _artifact_file_identity(path):
    try:
        status = os.lstat(path)
    except (FileNotFoundError, OSError):
        return None
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
    )


def _same_artifact_file(path, identity):
    return _artifact_file_identity(path) == identity


def _publish_artifact_set(staged_paths, final_paths, *, lock_path):
    """Hard-link a staged set without overwriting, rolling back on failure.

    The per-stem lock serializes cooperating writers.  Hard links make each
    file publication atomic and inherently no-overwrite even if an external
    writer races after the collision check.  The report JSON is linked last and
    therefore serves as the completion marker for the whole artifact set.
    """

    lock_acquired = False
    attempted = []
    try:
        try:
            os.mkdir(lock_path)
        except FileExistsError as error:
            raise FileExistsError(
                "E2 artifact publication is already in progress for "
                f"{lock_path.name}"
            ) from error
        lock_acquired = True

        collisions = [
            str(path)
            for path in final_paths.values()
            if os.path.lexists(path)
        ]
        if collisions:
            raise FileExistsError(
                "refusing to overwrite existing E2 selection artifacts: "
                + ", ".join(collisions)
            )

        publication_order = [
            key for key in final_paths if key != "report_json"
        ] + ["report_json"]
        for key in publication_order:
            staged = staged_paths[key]
            identity = _artifact_file_identity(staged)
            if identity is None:
                raise RuntimeError(f"staged E2 artifact disappeared: {staged}")
            final = final_paths[key]
            os.link(staged, final, follow_symlinks=False)
            attempted.append((final, identity))

        mismatches = [
            str(path)
            for path, identity in attempted
            if not _same_artifact_file(path, identity)
        ]
        if mismatches:
            raise RuntimeError(
                "published E2 artifacts changed during publication: "
                + ", ".join(mismatches)
            )
    except BaseException:
        for path, identity in reversed(attempted):
            if _same_artifact_file(path, identity):
                try:
                    path.unlink()
                except (FileNotFoundError, OSError):
                    pass
        raise
    finally:
        if lock_acquired:
            try:
                Path(lock_path).rmdir()
            except (FileNotFoundError, OSError):
                pass


def write_selection_artifacts(report, *, output_dir, run_name=None):
    """Transactionally stage and publish one collision-safe artifact set."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(run_name or default_run_name(report))
    if not stem:
        raise ValueError("selection run name cannot be empty")
    all_paths = _artifact_paths(
        output_dir,
        stem,
        has_confirmation=bool(report.get("confirmation_attempts")),
    )
    tables = _selection_artifact_tables(report)
    final_paths = {
        key: all_paths[key]
        for key in tables
    }
    final_paths["report_json"] = all_paths["report_json"]
    artifacts = {key: str(path) for key, path in final_paths.items()}
    result = {**report, "artifacts": artifacts}

    stage_dir = Path(tempfile.mkdtemp(
        prefix=f".{stem}.stage-", dir=output_dir
    ))
    try:
        staged_paths = {
            key: stage_dir / path.name for key, path in final_paths.items()
        }
        for key, rows in tables.items():
            write_csv(staged_paths[key], rows)
        write_json(staged_paths["report_json"], result)
        missing = [
            str(path) for path in staged_paths.values() if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "E2 artifact staging did not create every declared file: "
                + ", ".join(missing)
            )
        _publish_artifact_set(
            staged_paths,
            final_paths,
            lock_path=output_dir / f".{stem}.publish.lock",
        )
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    return result


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
    args.selected_checkpoint = str(selected.parent.resolve() / selected.name)
    return args


def rollback_new_selected_alias(report, *, expected_path):
    """Remove only the exact alias created by this failed artifact write."""

    alias = (report or {}).get("selected_alias")
    if not isinstance(alias, dict) or not (
            alias.get("created_by_selection_run")
            and alias.get("copy_verified")
    ):
        return False
    path = Path(alias.get("selected_checkpoint_path", "")).resolve()
    if path != Path(expected_path).expanduser().resolve() or path.is_symlink():
        return False
    expected_hash = alias.get("checkpoint_sha256")
    if not isinstance(expected_hash, str):
        return False
    try:
        before = path.stat()
        if checkpoint_sha256(path) != expected_hash:
            return False
        after = path.stat()
        if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
        ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
        ):
            return False
        path.unlink()
    except (FileNotFoundError, OSError):
        return False
    return True


def main(argv=None):
    args = parse_args(argv)
    if os.path.lexists(args.selected_checkpoint):
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
    collisions = [
        str(path) for path in planned.values() if os.path.lexists(path)
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing E2 selection artifacts: "
            + ", ".join(collisions)
        )
    report = run_selection(args)
    try:
        report = write_selection_artifacts(
            report, output_dir=args.output_dir, run_name=args.run_name
        )
    except BaseException:
        rollback_new_selected_alias(
            report, expected_path=args.selected_checkpoint
        )
        raise
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
