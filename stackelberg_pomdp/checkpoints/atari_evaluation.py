"""Load Atari evaluation checkpoints and verify their saved scientific provenance."""

from pathlib import Path

import numpy as np

from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS
from stackelberg_pomdp.atari.training import (
    ACTOR_LOSS_MODES,
    STANDARD_ACTOR_LOSS_MODE,
    ScaledLearningRatePPO,
    model_actor_loss_mode,
    model_economic_initialization,
)
from stackelberg_pomdp.checkpoints.atari_provenance import (
    E2_PROVENANCE_ATTRIBUTE,
    e2_implementation_provenance,
    e2_implementation_provenance_compatible,
    validate_e2_gameplay_actor,
    validate_e2_provenance_manifest,
)
from stackelberg_pomdp.checkpoints.files import checkpoint_sha256
from stackelberg_pomdp.evaluation.atari.contracts import (
    E2_INIT_CONCENTRATION,
    E2_INIT_MEAN,
    opposite_role,
)
from stackelberg_pomdp.policies.atari import (
    ATARI_POLICY_PROVENANCE_ID,
    StackPOMDPAtariPolicy,
    canonical_atari_policy_provenance_id,
)


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
    validate_e2_gameplay_actor(model)
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
