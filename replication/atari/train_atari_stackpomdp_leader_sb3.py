"""Train a clean full-trajectory E2 Atari StackPOMDP leader with PPO.

The leader first answers five canonical event-only queries.  A fresh bilateral
game then executes 200 gameplay decisions and five actor-identical cached
trade replays against a frozen opposite-role E1 composite response.  The new
leader inherits the same-role E1 visual, state, and Atari-game actor modules;
its economic actor and stage-private critic are initialized from scratch.
"""

import argparse
import copy
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
from pathlib import Path
import re
import tempfile
import uuid


os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stackpomdp-matplotlib")
)
os.environ.setdefault("WANDB_START_METHOD", "thread")

from stable_baselines3 import PPO
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
    e2_episode_transitions,
    evaluate_model,
    finish_run,
    init_wandb,
    make_vec_env,
    model_actor_loss_mode,
    model_economic_initialization,
    ppo_class_for_actor_loss_mode,
    write_json,
)
from stackelberg_pomdp.atari.core import default_rom_path
from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS
from stackelberg_pomdp.atari.meta_response import (
    make_stackpomdp_atari_leader_env,
)
from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    SELLER,
    BilateralAtariConfig,
)
from stackelberg_pomdp.atari.stackpomdp_policy import (
    SELLER_SHARED_CONTEXT_BETA_V5,
    SELLER_TWO_BRANCH_BETA_V4,
    StackPOMDPAtariPolicy,
)
from stackelberg_pomdp.callbacks import FixPolicyActionsCallback


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
E2_PROVENANCE_SCHEMA = "stackelberg_pomdp.atari.e2_provenance"
E2_PROVENANCE_VERSION = 1
E2_PROVENANCE_ATTRIBUTE = "e2_provenance_manifest"
E2_PROTOCOL_IMPLEMENTATION = "clean_atari_stackpomdp_e2_v1"
E2_ACTOR_TRANSFER_MODULES = ("features_extractor", "game_action_net")
E2_ECONOMIC_INIT_MEAN = 0.5
E2_ECONOMIC_INIT_CONCENTRATION = 2.0
E1_ECONOMIC_ARCHITECTURE_ATTRIBUTE = (
    "atari_e1_economic_architecture_provenance"
)
E1_TRAINING_CODE_REVISION_ATTRIBUTE = (
    "atari_e1_threshold_residual_training_code_revision"
)
E1_DIRECT_THRESHOLD_INITIALIZATION_ATTRIBUTE = (
    "atari_e1_direct_threshold_initialization_provenance"
)
E1_TWO_BRANCH_INITIALIZATION_ATTRIBUTE = (
    "atari_e1_two_branch_initialization_provenance"
)
E1_SHARED_CONTEXT_INITIALIZATION_ATTRIBUTE = (
    "atari_e1_shared_context_initialization_provenance"
)
FROZEN_E1_SELLER_ARCHITECTURES = {
    SELLER_TWO_BRANCH_BETA_V4,
    SELLER_SHARED_CONTEXT_BETA_V5,
}
E2_IMPLEMENTATION_FILES = (
    "replication/atari/train_atari_stackpomdp_leader_sb3.py",
    "replication/atari/sb3_common.py",
    "stackelberg_pomdp/atari/core.py",
    "stackelberg_pomdp/atari/gameplay.py",
    "stackelberg_pomdp/atari/wrappers.py",
    "stackelberg_pomdp/atari/protocol.py",
    "stackelberg_pomdp/atari/schedule.py",
    "stackelberg_pomdp/atari/stackpomdp_env.py",
    "stackelberg_pomdp/atari/stackpomdp_policy.py",
    "stackelberg_pomdp/atari/meta_response.py",
    "stackelberg_pomdp/gym_envs/envs/base_envs.py",
    "stackelberg_pomdp/gym_envs/envs/wrappers.py",
    "stackelberg_pomdp/callbacks.py",
)
E2_PACKAGE_DISTRIBUTIONS = (
    "stable-baselines3",
    "torch",
    "gym",
    "numpy",
    "multi-agent-ale-py",
    "opencv-python",
)


def _actor_loss_mode(args):
    return str(getattr(args, "actor_loss_mode", STANDARD_ACTOR_LOSS_MODE))


def _value_slug(value):
    return format(float(value), ".6g").replace("-", "m").replace(".", "p")


def _run_variant_suffix(args):
    parts = []
    if _actor_loss_mode(args) != STANDARD_ACTOR_LOSS_MODE:
        parts.append(_actor_loss_mode(args))
    if args.target_kl is not None:
        parts.append(f"kl{_value_slug(args.target_kl)}")
    return "" if not parts else "_" + "_".join(parts)


def _scientific_config_with_legacy_defaults(config):
    """Add fields implied by pre-phase-balanced version-1 manifests."""

    result = _canonical_json_copy(config)
    actor_loss_mode = result.setdefault("optimization", {}).setdefault(
        "actor_loss_mode", STANDARD_ACTOR_LOSS_MODE
    )
    initialization = result.setdefault("initialization", {})
    economic_init_mean = initialization.setdefault(
        "economic_head_beta_mean", E2_ECONOMIC_INIT_MEAN
    )
    economic_init_concentration = initialization.setdefault(
        "economic_head_beta_concentration", E2_ECONOMIC_INIT_CONCENTRATION
    )
    leader_policy = result.setdefault("leader_policy", {})
    leader_policy.setdefault("actor_loss_mode", actor_loss_mode)
    leader_policy.setdefault(
        "economic_head_initialization",
        {
            "mean": economic_init_mean,
            "concentration": economic_init_concentration,
        },
    )
    return result


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


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_copy(value):
    """Return a JSON-only deep copy and reject non-finite numbers."""

    return json.loads(json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ))


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def e2_implementation_provenance():
    """Hash the implementation and runtime packages that define E2 behavior."""

    source_hashes = {}
    for relative in E2_IMPLEMENTATION_FILES:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"E2 implementation file is missing: {path}")
        source_hashes[relative] = _sha256_file(path)
    packages = {}
    for distribution in E2_PACKAGE_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = "unavailable"
    return _canonical_json_copy({
        "protocol_implementation": E2_PROTOCOL_IMPLEMENTATION,
        "python": platform.python_version(),
        "packages": packages,
        "source_sha256": source_hashes,
    })


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
        actor_loss_mode = model_actor_loss_mode(model)
        if actor_loss_mode not in ACTOR_LOSS_MODES:
            raise ValueError(
                f"{label} checkpoint has an unknown actor-loss mode: "
                f"{actor_loss_mode!r}"
            )
        legacy_buyer = (
            policy.economic_role == BUYER
            and policy.economic_input_mode == "full"
        )
        economic_initialization = model_economic_initialization(
            model,
            default_mean=0.95 if legacy_buyer else E2_ECONOMIC_INIT_MEAN,
            default_concentration=(
                10.0 if legacy_buyer else E2_ECONOMIC_INIT_CONCENTRATION
            ),
        )
        policy_metadata = {
            "policy_class": (
                f"{type(policy).__module__}.{type(policy).__qualname__}"
            ),
            "economic_role": policy.economic_role,
            "economic_input_mode": policy.economic_input_mode,
            "visual_features": int(policy.visual_features),
            "state_features": int(policy.state_features),
            "economic_hidden": int(policy.economic_hidden),
            "critic_hidden": int(policy.critic_hidden),
            "pretrained_lr_scale": float(policy.pretrained_lr_scale),
            "game_action_count": int(policy.game_action_count),
            "actor_loss_mode": actor_loss_mode,
            "economic_head_initialization": economic_initialization,
        }
        economic_architecture = None
        if bool(
                getattr(policy, "economic_threshold_residual", False)
                or getattr(policy, "economic_architecture", None) is not None
        ):
            economic_architecture = (
                policy.economic_architecture_provenance()
            )
            recorded_architecture = getattr(
                model, E1_ECONOMIC_ARCHITECTURE_ATTRIBUTE, None
            )
            if recorded_architecture != economic_architecture:
                raise ValueError(
                    f"{label} threshold-residual policy lacks exact saved "
                    "economic architecture provenance"
                )
            training_code_revision = getattr(
                model, E1_TRAINING_CODE_REVISION_ATTRIBUTE, None
            )
            if not isinstance(training_code_revision, str) or re.fullmatch(
                    r"[0-9a-f]{40}", training_code_revision
            ) is None:
                raise ValueError(
                    f"{label} threshold-residual policy lacks a full saved "
                    "training code revision"
                )
            policy_metadata["economic_architecture"] = economic_architecture
            policy_metadata["e1_training_code_revision"] = (
                training_code_revision
            )
            if bool(getattr(
                    policy,
                    "economic_threshold_residual_direct_input",
                    False,
            )):
                from replication.atari.train_atari_meta_response_sb3 import (
                    direct_threshold_initialization_provenance,
                )

                expected_initialization = (
                    direct_threshold_initialization_provenance(policy)
                )
                recorded_initialization = getattr(
                    model,
                    E1_DIRECT_THRESHOLD_INITIALIZATION_ATTRIBUTE,
                    None,
                )
                if recorded_initialization != expected_initialization:
                    raise ValueError(
                        f"{label} direct-threshold policy lacks exact saved "
                        "initialization provenance"
                    )
                policy_metadata["direct_threshold_initialization"] = (
                    expected_initialization
                )
            frozen_architecture = getattr(
                policy, "economic_architecture", None
            )
            if frozen_architecture in FROZEN_E1_SELLER_ARCHITECTURES:
                from replication.atari.train_atari_meta_response_sb3 import (
                    gameplay_actor_sha256,
                    shared_context_initialization_provenance,
                    two_branch_initialization_provenance,
                    validate_frozen_gameplay_actor,
                )

                if frozen_architecture == SELLER_TWO_BRANCH_BETA_V4:
                    initialization_key = "two_branch_initialization"
                    initialization_label = "seller-v4"
                    initialization_attribute = (
                        E1_TWO_BRANCH_INITIALIZATION_ATTRIBUTE
                    )
                    expected_initialization = (
                        two_branch_initialization_provenance(policy)
                    )
                else:
                    initialization_key = "shared_context_initialization"
                    initialization_label = "seller-v5"
                    initialization_attribute = (
                        E1_SHARED_CONTEXT_INITIALIZATION_ATTRIBUTE
                    )
                    expected_initialization = (
                        shared_context_initialization_provenance(policy)
                    )
                recorded_initialization = getattr(
                    model, initialization_attribute, None
                )
                if recorded_initialization != expected_initialization:
                    raise ValueError(
                        f"{label} {initialization_label} policy lacks exact saved "
                        "initialization provenance"
                    )
                validate_frozen_gameplay_actor(model)
                policy_metadata[initialization_key] = expected_initialization
                policy_metadata["frozen_gameplay_actor_sha256"] = (
                    gameplay_actor_sha256(policy)
                )
        result = {
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
            "economic_role": policy.economic_role,
            "economic_input_mode": policy.economic_input_mode,
            "actor_loss_mode": actor_loss_mode,
            "economic_head_initialization": economic_initialization,
            "policy_metadata": policy_metadata,
        }
        if economic_architecture is not None:
            result["economic_architecture"] = economic_architecture
            result["e1_training_code_revision"] = training_code_revision
            if "direct_threshold_initialization" in policy_metadata:
                result["direct_threshold_initialization"] = policy_metadata[
                    "direct_threshold_initialization"
                ]
            if "two_branch_initialization" in policy_metadata:
                result["two_branch_initialization"] = policy_metadata[
                    "two_branch_initialization"
                ]
            if "shared_context_initialization" in policy_metadata:
                result["shared_context_initialization"] = policy_metadata[
                    "shared_context_initialization"
                ]
        manifest = getattr(model, E2_PROVENANCE_ATTRIBUTE, None)
        if manifest is not None:
            result["e2_provenance_manifest"] = _canonical_json_copy(manifest)
        return result
    finally:
        del model


def _artifact_manifest_entry(metadata, *, label):
    """Strip paths from one checkpoint identity before fingerprinting it."""

    digest = metadata.get("sha256")
    policy = metadata.get("policy_metadata")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} metadata is missing a SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(
            f"{label} metadata has an invalid SHA-256 digest"
        ) from error
    if not isinstance(policy, dict):
        raise ValueError(f"{label} metadata is missing policy metadata")
    return _canonical_json_copy({
        "sha256": digest.lower(),
        "policy": policy,
    })


def e2_scientific_config(args):
    """Return every role, environment, protocol, and PPO choice bound to E2."""

    config = bilateral_config(args, seed=args.seed).resolved()
    rom_path = (
        Path(config.rom_path).expanduser().resolve()
        if config.rom_path is not None
        else Path(default_rom_path()).resolve()
    )
    if not rom_path.is_file():
        raise FileNotFoundError(f"Space Invaders ROM does not exist: {rom_path}")
    fixed_steps = (
        None
        if config.fixed_event_steps is None
        else [int(step) for step in config.fixed_event_steps]
    )
    return _canonical_json_copy({
        "stage": "e2",
        "leader_role": args.leader_role,
        "follower_role": follower_role(args.leader_role),
        "environment": {
            "seed": int(config.seed),
            "gameplay_horizon": int(config.gameplay_horizon),
            "event_tail_steps": int(config.event_tail_steps),
            "fixed_event_steps": fixed_steps,
            "seller_game_reward_scale": float(
                config.seller_game_reward_scale
            ),
            "buyer_game_reward_scale": float(config.buyer_game_reward_scale),
            "noop_max": int(config.noop_max),
            "frame_skip": int(config.frame_skip),
            "frame_stack": int(config.frame_stack),
            "episodic_life": bool(config.episodic_life),
            "clip_game_rewards": bool(config.clip_game_rewards),
            "max_frames": int(config.max_frames),
            # Bind the scientific artifact to ROM contents, not a
            # machine-specific absolute path.  The ordinary run config still
            # records the path used on that machine.
            "rom_sha256": _sha256_file(rom_path),
        },
        "protocol": {
            "trade_events": NUM_TRADE_EVENTS,
            "query_transitions": NUM_TRADE_EVENTS,
            "cached_trade_replays": NUM_TRADE_EVENTS,
            "outer_episode_transitions": e2_episode_transitions(
                config.gameplay_horizon
            ),
            "policy_action_cache": True,
            "leader_economic_input": "event_only",
            "response_economic_input": "full",
            "response_algorithm": "frozen_meta_policy",
        },
        "optimization": {
            "algorithm": "PPO",
            "actor_loss_mode": _actor_loss_mode(args),
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "start_method": str(args.start_method),
            "device": str(args.device),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "learning_rate": float(args.learning_rate),
            "pretrained_lr_scale": float(args.pretrained_lr_scale),
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "clip_range": float(args.clip_range),
            "entropy_coefficient": float(args.entropy_coeff),
            "value_coefficient": float(args.value_coefficient),
            "max_grad_norm": float(args.max_grad_norm),
            "normalize_advantage": True,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": (
                None if args.target_kl is None else float(args.target_kl)
            ),
            "stats_window_size": 100,
        },
        "leader_policy": {
            "policy_class": (
                "stackelberg_pomdp.atari.stackpomdp_policy."
                "StackPOMDPAtariPolicy"
            ),
            "economic_role": args.leader_role,
            "economic_input_mode": "event_only",
            "visual_features": 512,
            "state_features": 64,
            "economic_hidden": 64,
            "critic_hidden": 256,
            "pretrained_lr_scale": float(args.pretrained_lr_scale),
            "game_action_count": 6,
            "actor_loss_mode": _actor_loss_mode(args),
            "economic_head_initialization": {
                "mean": E2_ECONOMIC_INIT_MEAN,
                "concentration": E2_ECONOMIC_INIT_CONCENTRATION,
            },
        },
        "initialization": {
            "actor_transfer_modules": list(E2_ACTOR_TRANSFER_MODULES),
            "economic_actor_transferred": False,
            "critic_transferred": False,
            "critic_initialization": "fresh",
            "optimizer_initialization": "fresh",
            "economic_head_beta_mean": E2_ECONOMIC_INIT_MEAN,
            "economic_head_beta_concentration": E2_ECONOMIC_INIT_CONCENTRATION,
        },
        "implementation": e2_implementation_provenance(),
    })


def build_e2_provenance_manifest(
        args,
        *,
        response,
        leader_e1,
        run_lineage_id=None,
):
    """Bind a new E2 run to exact E1 actors and its scientific setup."""

    scientific_config = e2_scientific_config(args)
    artifacts = {
        "frozen_response": _artifact_manifest_entry(
            response, label="frozen E1 response"
        ),
        "same_role_e1_initialization": _artifact_manifest_entry(
            leader_e1, label="same-role E1 initialization"
        ),
    }
    identity = {
        "scientific_config": scientific_config,
        "artifacts": artifacts,
    }
    lineage = uuid.uuid4().hex if run_lineage_id is None else str(run_lineage_id)
    unsigned = {
        "schema": E2_PROVENANCE_SCHEMA,
        "version": E2_PROVENANCE_VERSION,
        "run_lineage_id": lineage,
        "scientific_identity_sha256": _canonical_sha256(identity),
        **identity,
    }
    manifest = _canonical_json_copy(unsigned)
    manifest["fingerprint_sha256"] = _canonical_sha256(manifest)
    return manifest


def validate_e2_provenance_manifest(manifest):
    """Validate the schema and its self-consistent canonical checksums."""

    if not isinstance(manifest, dict):
        raise ValueError("E2 checkpoint has no valid provenance manifest")
    result = _canonical_json_copy(manifest)
    if result.get("schema") != E2_PROVENANCE_SCHEMA:
        raise ValueError("E2 checkpoint provenance schema is unsupported")
    if result.get("version") != E2_PROVENANCE_VERSION:
        raise ValueError("E2 checkpoint provenance version is unsupported")
    fingerprint = result.pop("fingerprint_sha256", None)
    if fingerprint != _canonical_sha256(result):
        raise ValueError("E2 checkpoint provenance fingerprint is invalid")
    result["fingerprint_sha256"] = fingerprint
    lineage = result.get("run_lineage_id")
    if not isinstance(lineage, str) or len(lineage) != 32:
        raise ValueError("E2 checkpoint provenance lineage ID is invalid")
    try:
        int(lineage, 16)
    except ValueError as error:
        raise ValueError(
            "E2 checkpoint provenance lineage ID is invalid"
        ) from error
    identity = {
        "scientific_config": result.get("scientific_config"),
        "artifacts": result.get("artifacts"),
    }
    if result.get("scientific_identity_sha256") != _canonical_sha256(identity):
        raise ValueError("E2 scientific identity checksum is invalid")
    return result


def require_compatible_e2_provenance(
        manifest,
        args,
        *,
        response,
        resumed_leader=None,
):
    """Require the current response bytes and scientific setup to match E2."""

    manifest = validate_e2_provenance_manifest(manifest)
    expected_config = _scientific_config_with_legacy_defaults(
        e2_scientific_config(args)
    )
    recorded_config = _scientific_config_with_legacy_defaults(
        manifest.get("scientific_config", {})
    )
    if recorded_config != expected_config:
        raise ValueError(
            "E2 resume/evaluation scientific config does not match the "
            "checkpoint provenance"
        )
    current_response = _artifact_manifest_entry(
        response, label="frozen E1 response"
    )
    recorded_response = manifest.get("artifacts", {}).get("frozen_response")
    if recorded_response != current_response:
        raise ValueError(
            "E2 resume/evaluation requires the exact frozen response "
            "checkpoint bytes and policy metadata"
        )
    expected_roles = {
        "frozen_response": follower_role(args.leader_role),
        "same_role_e1_initialization": args.leader_role,
    }
    for name, role in expected_roles.items():
        artifact = manifest.get("artifacts", {}).get(name, {})
        policy = artifact.get("policy", {})
        if policy.get("economic_role") != role:
            raise ValueError(f"E2 provenance has the wrong role for {name}")
        if policy.get("economic_input_mode") != "full":
            raise ValueError(f"E2 provenance has the wrong actor mode for {name}")
    if resumed_leader is not None:
        actual_policy = resumed_leader.get("policy_metadata")
        expected_policy = expected_config["leader_policy"]
        if actual_policy != expected_policy:
            raise ValueError(
                "E2 resume/evaluation policy architecture does not match "
                "the checkpoint provenance"
            )
    return manifest


def attach_e2_provenance(model, manifest):
    """Attach an immutable-by-fingerprint manifest to every SB3 model save."""

    manifest = validate_e2_provenance_manifest(manifest)
    existing = getattr(model, E2_PROVENANCE_ATTRIBUTE, None)
    if existing is not None:
        existing = validate_e2_provenance_manifest(existing)
        if existing != manifest:
            raise ValueError("refusing to replace an E2 checkpoint's provenance")
    setattr(model, E2_PROVENANCE_ATTRIBUTE, copy.deepcopy(manifest))
    return manifest


def provenance_sidecar_path(checkpoint):
    path = checkpoint_path(checkpoint)
    return path.with_name(f"{path.stem}.provenance.json")


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
        manifest = leader.get("e2_provenance_manifest")
        if manifest is None:
            raise ValueError(
                "E2 resume/evaluation checkpoint has no provenance manifest"
            )
        manifest = require_compatible_e2_provenance(
            manifest,
            args,
            response=response,
            resumed_leader=leader,
        )
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
    if not args.resume:
        manifest = build_e2_provenance_manifest(
            args, response=response, leader_e1=leader
        )
    return {"response": response, "leader": leader, "manifest": manifest}


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
    expected_response_sha256 = getattr(
        args, "validated_response_sha256", None
    )

    def verified_response_factory(checkpoint, *, device):
        resolved = _existing_checkpoint(checkpoint, "frozen E1 response")
        before = _sha256_file(resolved)
        if before != expected_response_sha256:
            raise RuntimeError(
                "frozen E1 response bytes changed after provenance validation"
            )
        model = PPO.load(str(resolved), device=device)
        if _sha256_file(resolved) != before:
            raise RuntimeError("frozen E1 response changed while it was loaded")
        return model

    return make_stackpomdp_atari_leader_env(
        leader_role=args.leader_role,
        response_checkpoint=args.response_checkpoint,
        config=bilateral_config(args, seed=seed),
        response_model_factory=(
            None
            if expected_response_sha256 is None
            else verified_response_factory
        ),
        device=args.device,
    )


def _new_model(args, vec_env, *, provenance_manifest):
    actor_loss_mode = _actor_loss_mode(args)
    algorithm_class = ppo_class_for_actor_loss_mode(actor_loss_mode)
    model = algorithm_class(
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
        normalize_advantage=True,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=args.target_kl,
        stats_window_size=100,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    provenance = model.policy.load_actor_checkpoint(
        args.leader_e1_checkpoint,
        include_economic=False,
        device=args.device,
    )
    recorded_source = provenance_manifest["artifacts"][
        "same_role_e1_initialization"
    ]
    if provenance.get("sha256") != recorded_source.get("sha256"):
        raise RuntimeError(
            "same-role E1 checkpoint bytes changed between provenance "
            "validation and actor transfer"
        )
    source_policy = recorded_source.get("policy", {})
    source_parameterization = source_policy.get(
        "economic_architecture", {}
    ).get("parameterization")
    source_frozen_architecture = (
        source_parameterization
        if source_parameterization in FROZEN_E1_SELLER_ARCHITECTURES
        else None
    )
    if (
            provenance.get("source_economic_role")
            != source_policy.get("economic_role")
            or provenance.get("source_economic_input_mode")
            != source_policy.get("economic_input_mode")
            or bool(provenance.get(
                "source_economic_threshold_residual", False
            )) != (
                source_policy.get("economic_architecture", {}).get(
                    "parameterization"
                ) in {
                    "seller_threshold_residual_beta_v1",
                    "seller_direct_threshold_residual_beta_v3",
                }
            )
            or bool(provenance.get(
                "source_economic_threshold_residual_direct_input", False
            )) != (
                source_policy.get("economic_architecture", {}).get(
                    "parameterization"
                ) == "seller_direct_threshold_residual_beta_v3"
            )
            or provenance.get("source_economic_architecture") != (
                source_frozen_architecture
            )
    ):
        raise RuntimeError(
            "same-role E1 actor identity changed during E2 transfer"
        )
    if tuple(provenance.get("modules", ())) != E2_ACTOR_TRANSFER_MODULES:
        raise RuntimeError("E2 actor transfer used an unexpected module set")
    if provenance.get("critic_transferred") is not False:
        raise RuntimeError("E2 initialization must not transfer the E1 critic")
    model.policy.reset_economic_head(
        mean=E2_ECONOMIC_INIT_MEAN,
        concentration=E2_ECONOMIC_INIT_CONCENTRATION,
    )
    attach_atari_training_contract(
        model,
        actor_loss_mode=actor_loss_mode,
        economic_init_mean=E2_ECONOMIC_INIT_MEAN,
        economic_init_concentration=E2_ECONOMIC_INIT_CONCENTRATION,
    )
    model.policy.clear_obs_action_map()
    attach_e2_provenance(model, provenance_manifest)
    print({"actor_transfer": provenance, "fresh_economic_head": True}, flush=True)
    return model


def _resumed_model(args, vec_env, *, provenance_manifest):
    actor_loss_mode = _actor_loss_mode(args)
    algorithm_class = ppo_class_for_actor_loss_mode(actor_loss_mode)
    model = algorithm_class.load(
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
        normalize_advantage=True,
        use_sde=False,
        sde_sample_freq=-1,
        stats_window_size=100,
    )
    expected_resume_sha256 = getattr(args, "validated_resume_sha256", None)
    if (
            expected_resume_sha256 is not None
            and _sha256_file(_existing_checkpoint(args.resume, "E2 resume"))
            != expected_resume_sha256
    ):
        raise RuntimeError("E2 resume bytes changed while the model was loaded")
    if not isinstance(model.policy, StackPOMDPAtariPolicy):
        raise TypeError("--resume must contain the clean Atari composite policy")
    if model.policy.economic_role != args.leader_role:
        raise ValueError("--resume role does not match --leader-role")
    if model.policy.economic_input_mode != "event_only":
        raise ValueError("--resume is not an event-only E2 leader")
    saved_target_kl = getattr(model, "target_kl", None)
    if (
            (saved_target_kl is None) != (args.target_kl is None)
            or (
                saved_target_kl is not None
                and not math.isclose(
                    float(saved_target_kl),
                    float(args.target_kl),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
    ):
        raise ValueError(
            "--target-kl must match the saved E2 checkpoint "
            f"({saved_target_kl})"
        )
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
    saved_actor_loss_mode = model_actor_loss_mode(model)
    if saved_actor_loss_mode not in ACTOR_LOSS_MODES:
        raise ValueError(
            "--resume contains an unknown Atari actor loss mode: "
            f"{saved_actor_loss_mode!r}"
        )
    if saved_actor_loss_mode != actor_loss_mode:
        raise ValueError(
            "--actor-loss-mode must match the saved E2 checkpoint "
            f"({saved_actor_loss_mode})"
        )
    initialization = model_economic_initialization(
        model,
        default_mean=E2_ECONOMIC_INIT_MEAN,
        default_concentration=E2_ECONOMIC_INIT_CONCENTRATION,
    )
    expected_initialization = {
        "mean": E2_ECONOMIC_INIT_MEAN,
        "concentration": E2_ECONOMIC_INIT_CONCENTRATION,
    }
    if initialization != expected_initialization:
        raise ValueError(
            "E2 economic-head initialization does not match its checkpoint"
        )
    attach_atari_training_contract(
        model,
        actor_loss_mode=actor_loss_mode,
        economic_init_mean=initialization["mean"],
        economic_init_concentration=initialization["concentration"],
    )
    model.policy.clear_obs_action_map()
    attach_e2_provenance(model, provenance_manifest)
    return model


def build_model(args, vec_env, *, provenance_manifest):
    """Build E2 from E1 actors, or restore a complete in-progress E2 run."""

    builder = _resumed_model if args.resume else _new_model
    return builder(
        args, vec_env, provenance_manifest=provenance_manifest
    )


def result_checkpoint(args):
    """Return the checkpoint whose policy is evaluated and reported."""

    return checkpoint_path(args.resume if args.eval_only else args.checkpoint)


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
    parser.add_argument(
        "--actor-loss-mode",
        choices=ACTOR_LOSS_MODES,
        default=STANDARD_ACTOR_LOSS_MODE,
    )
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float)
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
    parser.add_argument(
        "--wandb-job-type",
        help="Optional explicit W&B job type for collision-safe pipelines.",
    )
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
    if (
            args.actor_loss_mode == PHASE_BALANCED_ACTOR_LOSS_MODE
            and args.batch_size != buffer_size
    ):
        parser.error(
            "phase-balanced Atari PPO requires one full-rollout minibatch "
            "(batch-size = n-steps * num-envs)"
        )
    if args.timesteps <= 0 and not args.eval_only:
        parser.error("--timesteps must be positive during training")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.entropy_coeff < 0.01:
        parser.error("E2 requires --entropy-coeff >= 0.01")
    if args.target_kl is not None and (
            not math.isfinite(args.target_kl) or args.target_kl <= 0.0
    ):
        parser.error("--target-kl must be positive")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    if not args.resume and args.leader_e1_checkpoint is None:
        parser.error("a new E2 run requires --leader-e1-checkpoint")

    default = (
        REPOSITORY_ROOT
        / "replication/atari/checkpoints/clean"
        / (
            f"leader_{args.leader_role}_e2_ppo{_run_variant_suffix(args)}"
            f"_seed{args.seed}.zip"
        )
    )
    args.checkpoint = str(checkpoint_path(args.checkpoint or default))
    return args


def main(argv=None):
    args = parse_args(argv)
    metadata = validate_stage_checkpoints(args)
    manifest = metadata["manifest"]
    args.validated_response_sha256 = metadata["response"]["sha256"]
    if args.resume:
        args.validated_resume_sha256 = metadata["leader"]["sha256"]
    reported_checkpoint = result_checkpoint(args)
    vec_env = make_vec_env(
        lambda rank: make_env(args, seed=args.seed + 10_000 * rank),
        num_envs=args.num_envs,
        start_method=args.start_method,
    )
    run = init_wandb(
        args,
        stage=f"e2_{args.leader_role}",
        checkpoint=reported_checkpoint,
    )
    if run is not None:
        run.config.update({
            "stage_checkpoints": {
                "response": metadata["response"],
                "leader": {
                    key: value
                    for key, value in metadata["leader"].items()
                    if key != "e2_provenance_manifest"
                },
            },
            "e2_provenance_manifest": manifest,
            "e2_provenance_fingerprint": manifest["fingerprint_sha256"],
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
        model = build_model(
            args, vec_env, provenance_manifest=manifest
        )
        if not args.eval_only:
            model.learn(
                total_timesteps=args.timesteps,
                callback=make_training_callback(args, wandb_run=run),
                reset_num_timesteps=not bool(args.resume),
            )
            model.save(args.checkpoint)
        evaluation = evaluate_leader(model, args)
        evaluation["e2_provenance_manifest"] = manifest
        evaluation["e2_provenance_fingerprint"] = (
            manifest["fingerprint_sha256"]
        )
        write_json(provenance_sidecar_path(reported_checkpoint), manifest)
        finish_run(
            run,
            checkpoint=reported_checkpoint,
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
