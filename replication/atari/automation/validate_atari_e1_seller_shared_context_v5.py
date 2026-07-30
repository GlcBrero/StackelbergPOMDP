#!/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python
"""Fail-closed smoke and preflight gates for the v5 Atari E1 seller.

Every supported protocol uses independent fresh-E0b, Uniform(0,1)^5 runs.
The 20,500-step smoke has execution-only gates.  A protocol-specific fresh
preflight adds the unchanged no-ALE learned-conditioning gate and a fresh
paired real-ALE full-versus-joint-ablation behavioral gate.  Only a gate
binding both results to one clean Git revision releases the separate formal
W&B run; no diagnostic checkpoint or v1-v4 recovery artifact is admissible.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari import probe_atari_e1_seller_shared_context as probe
from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.automation import (
    validate_atari_e1_seller_conditioning_recovery as shared,
)
from stackelberg_pomdp.atari.schedule import ExactFiveEventSchedule


SCHEMA_VERSION = 1
GATE_KIND = "stackpomdp.atari.e1_seller_shared_context_diagnostics_gate.v5"
SMOKE_KIND = "stackpomdp.atari.e1_seller_shared_context_mechanics_smoke.v5"
PREFLIGHT_KIND = (
    "stackpomdp.atari.e1_seller_shared_context_conditioning_preflight.v5"
)
ROLE = "seller"
ARCHITECTURE = "seller_shared_context_beta_v5"
SAMPLER_MODE = "uniform"
ACTOR_LOSS_MODE = "balanced"

POST_UPDATE_SNAPSHOT = "post_update"
RETAINED_PRE_UPDATE_SNAPSHOT = "retained_pre_update"
CHECKPOINT_SNAPSHOT_PHASES = (
    POST_UPDATE_SNAPSHOT,
    RETAINED_PRE_UPDATE_SNAPSHOT,
)
CANONICAL_ROLLOUT_TIMESTEPS = 820

FORMAL_TIMESTEPS = 2_000_800
CHECKPOINT_INTERVAL = 400_160
FORMAL_STEP_TIMESTEPS = (400_160, 800_320, 1_200_480, 1_600_640, 2_000_800)
WANDB_PROJECT = "StackPOMDP"
WANDB_GROUP = "atari_clean_curriculum"

PREFLIGHT_BEHAVIOR_EPISODES = 20
PREFLIGHT_FIXED_VALUES = tuple(index / 10.0 for index in range(11))
PREFLIGHT_FIXED_EVENT_STEPS = (20, 50, 80, 110, 140)

STANDARD_PROTOCOL = "standard_v1"
EXPOSURE_PROTOCOL = "exposure_v2"
EXPOSURE_V2_DIAGNOSTIC_REVISION = (
    "c4a7dcd92b621c0884f3dcef0b170961e1ec625b"
)
VALIDATOR_REPAIR_ALLOWED_PATHS = (
    "replication/atari/automation/atari_e1_seller_shared_context_v5_common.zsh",
    "replication/atari/automation/run_atari_clean_e1_seller_shared_context_v5.sh",
    "replication/atari/automation/validate_atari_e1_seller_shared_context_v5.py",
    "tests/test_atari_e1_seller_shared_context_v5_automation.py",
)
DEFAULT_PROTOCOL = STANDARD_PROTOCOL
PROTOCOL_CONFIGURATIONS = {
    STANDARD_PROTOCOL: {
        "token": "conditioning_recovery_v5_shared_context_v1",
        "source_kind": "seller_conditioning_recovery_v5_shared_context_v1",
        "smoke_timesteps": 20_500,
        "preflight_timesteps": 82_000,
        "behavior_seed_start": 10_400_001,
        "fixed_seed_start": 10_500_001,
    },
    EXPOSURE_PROTOCOL: {
        "token": "conditioning_recovery_v5_shared_context_exposure_v2",
        "source_kind": (
            "seller_conditioning_recovery_v5_shared_context_exposure_v2"
        ),
        "smoke_timesteps": 20_500,
        "preflight_timesteps": 400_160,
        # The longer-exposure protocol gets held-out rows that were not used
        # by the failed 82k v1 behavioral preflight.
        "behavior_seed_start": 11_400_001,
        "fixed_seed_start": 11_500_001,
    },
}


def protocol_configuration(name):
    """Return one immutable protocol contract by exact versioned name."""

    if name not in PROTOCOL_CONFIGURATIONS:
        raise ValueError(f"unknown v5 seller protocol: {name}")
    return dict(PROTOCOL_CONFIGURATIONS[name])


def configure_protocol(name):
    """Select exact names, exposure, and held-out seeds for this process."""

    configuration = protocol_configuration(name)
    global ACTIVE_PROTOCOL
    global SOURCE_KIND
    global SMOKE_TIMESTEPS
    global PREFLIGHT_TIMESTEPS
    global TOKEN
    global SMOKE_CHECKPOINT_NAME
    global PREFLIGHT_CHECKPOINT_NAME
    global PREFLIGHT_PROBE_NAME
    global PREFLIGHT_BEHAVIOR_NAME
    global FORMAL_CHECKPOINT_NAME
    global GATE_NAME
    global WANDB_JOB_TYPE
    global WANDB_NAME
    global PREFLIGHT_BEHAVIOR_SEED_START
    global PREFLIGHT_FIXED_SEED_START

    ACTIVE_PROTOCOL = name
    SOURCE_KIND = configuration["source_kind"]
    SMOKE_TIMESTEPS = configuration["smoke_timesteps"]
    PREFLIGHT_TIMESTEPS = configuration["preflight_timesteps"]
    TOKEN = configuration["token"]
    SMOKE_CHECKPOINT_NAME = (
        f"meta_seller_e1_ppo_balanced_{TOKEN}_seed1_"
        "uniform_mechanics_smoke.zip"
    )
    PREFLIGHT_CHECKPOINT_NAME = (
        f"meta_seller_e1_ppo_balanced_{TOKEN}_seed1_"
        "uniform_conditioning_preflight.zip"
    )
    PREFLIGHT_PROBE_NAME = (
        f"e1_seller_{TOKEN}_uniform_conditioning_preflight_probe.json"
    )
    PREFLIGHT_BEHAVIOR_NAME = (
        f"e1_seller_{TOKEN}_uniform_conditioning_preflight_behavior.json"
    )
    FORMAL_CHECKPOINT_NAME = (
        f"meta_seller_e1_ppo_balanced_{TOKEN}_seed1_uniform_formal.zip"
    )
    GATE_NAME = f"e1_seller_{TOKEN}_diagnostics_gate.json"
    WANDB_JOB_TYPE = f"atari_e1_seller_v5_{ACTIVE_PROTOCOL}_formal"
    WANDB_NAME = (
        f"atari_clean_e1_seller_{TOKEN}_uniform_seed1_2000800_local"
    )
    PREFLIGHT_BEHAVIOR_SEED_START = configuration["behavior_seed_start"]
    PREFLIGHT_FIXED_SEED_START = configuration["fixed_seed_start"]
    return configuration


configure_protocol(DEFAULT_PROTOCOL)

V1_NEGATIVE_EVIDENCE = {
    "checkpoint_sha256": (
        "6203035d3220dce8cd12fb45b9e32ad87d194e77cef4982e28e52ade34dacf8b"
    ),
    "behavior_report_sha256": (
        "c5902aa05419fce82bcb783535bd63eb0f057c90b20cda8031c259b167460278"
    ),
    "transfer_payoff_improvement": 0.0009462684392929077,
    "required_transfer_payoff_improvement": 0.05,
    "threshold_0_4_to_0_9_price_response": 0.050391235351562536,
    "required_threshold_0_4_to_0_9_price_response": 0.15,
}


def canonical_protocol_provenance():
    """Bind the sole preregistered difference between v1 and exposure v2."""

    configuration = protocol_configuration(ACTIVE_PROTOCOL)
    record = {
        "name": ACTIVE_PROTOCOL,
        "token": configuration["token"],
        "source_kind": configuration["source_kind"],
        "architecture_changed": False,
        "smoke_timesteps": configuration["smoke_timesteps"],
        "preflight_timesteps": configuration["preflight_timesteps"],
        "behavior_seed_start": configuration["behavior_seed_start"],
        "fixed_seed_start": configuration["fixed_seed_start"],
        "behavioral_gate_changed": False,
    }
    if ACTIVE_PROTOCOL == EXPOSURE_PROTOCOL:
        record.update({
            "only_preflight_exposure_and_holdout_seed_namespace_changed": True,
            "predecessor_protocol": STANDARD_PROTOCOL,
            "predecessor_negative_evidence": dict(V1_NEGATIVE_EVIDENCE),
            "fresh_from_canonical_e0b_not_resume": True,
            "expected_preflight_episodes": 1_952,
            "expected_preflight_rollout_iterations": 488,
            "expected_preflight_adam_step": 1_952,
        })
    else:
        record[
            "only_preflight_exposure_and_holdout_seed_namespace_changed"
        ] = False
    return record

CANONICAL_E0B_SHA256 = shared.CANONICAL_E0B_SHA256
CANONICAL_ROM_SHA256 = shared.CANONICAL_ROM_SHA256
CANONICAL_SELLER_RELEASE_SHA256 = shared.CANONICAL_SELLER_RELEASE_SHA256

_require = shared._require
sha256_file = shared.sha256_file
load_json = shared.load_json
atomic_write_json = shared.atomic_write_json
_same_path = shared._same_path
_canonical_json = shared._canonical_json

V5_LEARNING_RATES = {
    "train/learning_rate": 5.0e-4,
    "train/context_learning_rate": 2.0e-3,
    "train/critic_learning_rate": 1.0e-4,
}
V5_GRADIENT_NORM_KEYS = tuple(
    f"train/{group}_grad_norm_{when}"
    for group in ("seller_v5_live", "seller_v5_context", "seller_v5_critic")
    for when in ("pre", "post")
)
# ``clip_grad_norm_`` and the post-clip telemetry intentionally compute the
# same float32 norm by two different reduction paths.  Their roundoff can
# therefore differ by slightly more than 1e-6 even when clipping is correct.
# This bound exceeds the largest audited discrepancy (1.30e-6) while still
# rejecting any material norm increase or violation of the 0.5 cap.
V5_GRADIENT_NORM_ATOL = 2.0e-6


def canonical_sampler():
    return shared.canonical_uniform_sampler()


def canonical_architecture():
    value = probe.canonical_economic_architecture()
    _require(
        value.get("schema")
        == "stackpomdp.atari.economic_actor_architecture.v5",
        "v5 architecture schema changed",
    )
    _require(
        value.get("parameterization") == ARCHITECTURE,
        "v5 architecture parameterization changed",
    )
    return value


def canonical_initialization():
    value = trainer.shared_context_initialization_contract()
    _require(
        value.get("schema")
        == "stackpomdp.atari.e1_shared_context_initialization.v1",
        "v5 initialization schema changed",
    )
    _require(
        value.get("architecture_parameterization") == ARCHITECTURE,
        "v5 initialization parameterization changed",
    )
    return value


def canonical_training_config():
    return {
        "algorithm": "PPO",
        "actor_loss_mode": ACTOR_LOSS_MODE,
        "economic_head_initialization": {
            "mean": 0.5,
            "concentration": 2.0,
        },
        "target_kl": None,
        "seed": 1,
        "learning_rate": 5.0e-4,
        "n_steps": 205,
        "batch_size": 820,
        "n_epochs": 4,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "clip_range_at_start": 0.1,
        "entropy_coefficient": 0.01,
        "value_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "policy_class": (
            "stackelberg_pomdp.atari.stackpomdp_policy."
            "StackPOMDPAtariPolicy"
        ),
        "visual_features": 512,
        "state_features": 64,
        "economic_hidden": 64,
        "critic_hidden": 256,
        "pretrained_lr_scale": 0.1,
        "economic_architecture": ARCHITECTURE,
    }


def expected_optimizer_adam_step(timesteps, *, snapshot_phase):
    """Return the exact Adam clock for one canonical checkpoint snapshot.

    Stable-Baselines3 invokes ``EpisodeCheckpointCallback._on_step`` while it
    is collecting a rollout, before PPO optimizes that rollout.  A retained
    ``_stepN`` checkpoint therefore has observed all ``N`` transitions but is
    one four-epoch PPO update behind the post-``learn`` base checkpoint.  Both
    are intentional candidate snapshots and must remain distinguishable.
    """

    _require(
        snapshot_phase in CHECKPOINT_SNAPSHOT_PHASES,
        "unknown v5 checkpoint snapshot phase",
    )
    config = canonical_training_config()
    rollout_timesteps = CANONICAL_ROLLOUT_TIMESTEPS
    _require(
        int(config["batch_size"]) == rollout_timesteps,
        "v5 canonical PPO no longer uses one full-rollout minibatch",
    )
    epochs_per_rollout = int(config["n_epochs"])
    timesteps = int(timesteps)
    _require(
        timesteps > 0 and timesteps % rollout_timesteps == 0,
        "v5 checkpoint is not at a complete canonical rollout boundary",
    )
    completed_adam_steps = (
        timesteps // rollout_timesteps * epochs_per_rollout
    )
    if snapshot_phase == RETAINED_PRE_UPDATE_SNAPSHOT:
        completed_adam_steps -= epochs_per_rollout
    _require(
        completed_adam_steps >= 0,
        "v5 checkpoint snapshot precedes the first optimizer update",
    )
    return completed_adam_steps


def canonical_sampler_history():
    return [{
        "start_total_timesteps": 0,
        "sampler": canonical_sampler(),
        "inferred_for_legacy_checkpoint": False,
        "resume_sources": [],
    }]


def _git_revision(code_root):
    root = Path(code_root).expanduser().resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "v5 code revision is not a full lowercase SHA",
    )
    return revision


def _validate_diagnostic_revision_bridge(
        code_root, *, runtime_revision, evidence_revision,
):
    """Prove that reused diagnostics differ only by the validator repair."""

    if evidence_revision == runtime_revision:
        return {
            "applied": False,
            "evidence_revision": evidence_revision,
            "runtime_revision": runtime_revision,
        }
    _require(
        ACTIVE_PROTOCOL == EXPOSURE_PROTOCOL
        and evidence_revision == EXPOSURE_V2_DIAGNOSTIC_REVISION,
        "v5 diagnostic revision bridge is not preregistered",
    )
    root = Path(code_root).expanduser().resolve()
    ancestor = subprocess.run(
        [
            "git", "-C", str(root), "merge-base", "--is-ancestor",
            evidence_revision, runtime_revision,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        ancestor.returncode == 0,
        "v5 diagnostic revision is not an ancestor of the runtime",
    )
    changed = tuple(filter(None, subprocess.run(
        [
            "git", "-C", str(root), "diff", "--name-only",
            f"{evidence_revision}..{runtime_revision}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()))
    _require(
        changed == VALIDATOR_REPAIR_ALLOWED_PATHS,
        "v5 diagnostic/runtime revisions differ outside the exact "
        "validator-only repair",
    )
    return {
        "applied": True,
        "reason": "float32_gradient_norm_validator_tolerance",
        "evidence_revision": evidence_revision,
        "runtime_revision": runtime_revision,
        "changed_paths": list(changed),
        "training_implementation_changed": False,
        "gradient_norm_absolute_tolerance": V5_GRADIENT_NORM_ATOL,
    }


def _git_scoped_clean(code_root):
    root = Path(code_root).expanduser().resolve()
    module_root = Path(__file__).resolve().parents[3]
    _require(
        os.path.samefile(root, module_root),
        "v5 validator is not executing from --code-root",
    )
    paths = (
        "replication/atari/train_atari_meta_response_sb3.py",
        "replication/atari/evaluate_atari_meta_response_sb3.py",
        "replication/atari/probe_atari_e1_seller_conditioning.py",
        "replication/atari/probe_atari_e1_seller_shared_context.py",
        "replication/atari/sb3_common.py",
        "replication/atari/automation/atari_e1_seller_shared_context_v5_common.zsh",
        "replication/atari/automation/run_atari_clean_e1_seller_shared_context_v5.sh",
        "replication/atari/automation/run_atari_clean_e1_seller_shared_context_v5_exposure_v2.sh",
        "replication/atari/automation/validate_atari_e1_seller_shared_context_v5.py",
        "replication/atari/automation/atari_e2_pipeline_common.zsh",
        "replication/atari/automation/validate_atari_e1_seller_conditioning_recovery.py",
        "stackelberg_pomdp/atari",
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(not dirty, f"refusing dirty v5 seller code: {dirty}")


def _exact_name(path, expected, label):
    _require(Path(path).name == expected, f"{label} filename changed")


def _expected_sidecar(checkpoint, suffix):
    checkpoint = Path(checkpoint).expanduser().resolve()
    return checkpoint.with_name(f"{checkpoint.stem}.{suffix}")


def _load_metadata(checkpoint, *, e0b, expected_revision):
    checkpoint = Path(checkpoint).expanduser().resolve()
    model, metadata = evaluator.load_candidate(
        checkpoint,
        role=ROLE,
        e0b_sha256=CANONICAL_E0B_SHA256,
        device="cpu",
    )
    try:
        optimizer_step = shared._optimizer_step(model)
        resume_source = _canonical_json(getattr(
            model, "atari_e1_resume_source_provenance", None
        ))
        frozen = trainer.validate_frozen_gameplay_actor(model)
        _require(
            model.policy.economic_architecture_provenance()
            == canonical_architecture(),
            "loaded v5 policy architecture changed",
        )
    finally:
        del model
    metadata = _canonical_json(metadata)
    metadata["optimizer_adam_step"] = optimizer_step
    metadata["resume_source"] = resume_source
    metadata["frozen_gameplay_actor_sha256"] = frozen
    _require(
        metadata.get("e1_training_code_revision") == expected_revision,
        "v5 checkpoint was trained by another code revision",
    )
    _require(
        metadata.get("e0b_source_provenance", {}).get("sha256")
        == sha256_file(e0b)
        == CANONICAL_E0B_SHA256,
        "v5 checkpoint does not use canonical E0b bytes",
    )
    return metadata


def _validate_checkpoint_metadata(
        metadata, *, checkpoint, timesteps, expected_revision,
        snapshot_phase=POST_UPDATE_SNAPSHOT,
):
    _same_path(metadata.get("path"), checkpoint, label="v5 checkpoint")
    _require(
        metadata.get("training_timesteps") == timesteps,
        "v5 diagnostic clock changed",
    )
    _require(
        metadata.get("training_config") == canonical_training_config(),
        "v5 PPO/architecture configuration changed",
    )
    _require(
        metadata.get("economic_architecture") == canonical_architecture(),
        "v5 checkpoint architecture provenance changed",
    )
    _require(
        metadata.get("shared_context_initialization")
        == canonical_initialization(),
        "v5 checkpoint initialization provenance changed",
    )
    _require(
        metadata.get("atari_e1_sampler_provenance") == canonical_sampler()
        and metadata.get("atari_e1_sampler_history")
        == canonical_sampler_history(),
        "v5 diagnostic is not fresh independent Uniform(0,1)^5 training",
    )
    _require(
        metadata.get("resume_source") is None,
        "v5 diagnostic resumed another checkpoint",
    )
    expected_optimizer_step = expected_optimizer_adam_step(
        timesteps, snapshot_phase=snapshot_phase,
    )
    _require(
        metadata.get("optimizer_adam_step") == expected_optimizer_step,
        "v5 diagnostic optimizer clock changed",
    )
    _require(
        metadata.get("e1_training_code_revision") == expected_revision,
        "v5 diagnostic revision changed",
    )
    source = metadata.get("e0b_source_provenance", {})
    _require(
        source.get("frozen_gameplay_actor") is True
        and source.get("frozen_gameplay_actor_sha256")
        == metadata.get("frozen_gameplay_actor_sha256"),
        "v5 diagnostic gameplay actor was not frozen bit-exactly",
    )


def _validate_v5_optimizer_telemetry(path, *, expected_optimizers):
    """Require every v5 optimizer row to prove rates and group clipping."""

    path = Path(path).expanduser().resolve()
    optimizer_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_kind") != "optimizer":
                continue
            optimizer_rows += 1
            for key, expected in V5_LEARNING_RATES.items():
                actual = row.get(key)
                _require(
                    isinstance(actual, (int, float))
                    and not isinstance(actual, bool)
                    and math.isfinite(float(actual))
                    and float(actual) == expected,
                    f"v5 optimizer row {line_number} has invalid {key}",
                )
            _require(
                {key for key in V5_GRADIENT_NORM_KEYS if key in row}
                == set(V5_GRADIENT_NORM_KEYS),
                f"v5 optimizer row {line_number} lacks exact group norms",
            )
            for group in (
                    "seller_v5_live", "seller_v5_context",
                    "seller_v5_critic",
            ):
                pre = row[f"train/{group}_grad_norm_pre"]
                post = row[f"train/{group}_grad_norm_post"]
                _require(
                    all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and float(value) >= 0.0
                        for value in (pre, post)
                    ),
                    f"v5 optimizer row {line_number} has invalid {group} norms",
                )
                _require(
                    float(post) <= float(pre) + V5_GRADIENT_NORM_ATOL
                    and float(post) <= 0.5 + V5_GRADIENT_NORM_ATOL,
                    f"v5 optimizer row {line_number} did not clip {group} "
                    "independently at 0.5",
                )
    _require(
        optimizer_rows == expected_optimizers,
        "v5 telemetry has the wrong optimizer-row count",
    )
    return {
        "optimizer_rows": optimizer_rows,
        "learning_rates": dict(V5_LEARNING_RATES),
        "gradient_norm_keys": list(V5_GRADIENT_NORM_KEYS),
        "independent_group_gradient_clip_norm": 0.5,
    }


def _validate_trace(path, *, checkpoint, timesteps):
    trace = shared._validate_training_trace(
        path,
        mode=SAMPLER_MODE,
        first_step=820,
        last_step=timesteps,
        expected_episodes=timesteps // 205,
        expected_optimizers=timesteps // 820,
        checkpoint=checkpoint,
    )
    trace["seller_v5_optimizer_telemetry"] = (
        _validate_v5_optimizer_telemetry(
            path, expected_optimizers=timesteps // 820
        )
    )
    return trace


def _validate_evaluation(path, *, metadata, timesteps):
    record = shared._validate_evaluation(
        path,
        current=canonical_sampler(),
        history=canonical_sampler_history(),
        expected_step=timesteps,
        checkpoint_metadata=metadata,
        random_episodes=20,
        fixed_episodes=20,
    )
    value = load_json(path)
    provenance = value.get("provenance", {})
    _require(
        provenance.get("economic_architecture") == canonical_architecture(),
        "v5 diagnostic evaluation architecture changed",
    )
    _require(
        provenance.get("shared_context_initialization")
        == canonical_initialization(),
        "v5 diagnostic evaluation initialization changed",
    )
    _require(
        provenance.get("e1_training_code_revision")
        == metadata.get("e1_training_code_revision"),
        "v5 diagnostic evaluation revision changed",
    )
    return record


def validate_smoke(
        *, checkpoint, training_log, evaluation, e0b,
        expected_revision,
):
    checkpoint = Path(checkpoint).expanduser().resolve()
    _exact_name(checkpoint, SMOKE_CHECKPOINT_NAME, "v5 smoke checkpoint")
    _same_path(
        training_log,
        _expected_sidecar(checkpoint, "training.jsonl"),
        label="v5 smoke trace",
    )
    _same_path(
        evaluation,
        _expected_sidecar(checkpoint, "evaluation.json"),
        label="v5 smoke evaluation",
    )
    metadata = _load_metadata(
        checkpoint, e0b=e0b, expected_revision=expected_revision
    )
    _validate_checkpoint_metadata(
        metadata,
        checkpoint=checkpoint,
        timesteps=SMOKE_TIMESTEPS,
        expected_revision=expected_revision,
    )
    trace = _validate_trace(
        training_log, checkpoint=checkpoint, timesteps=SMOKE_TIMESTEPS
    )
    evaluation_record = _validate_evaluation(
        evaluation, metadata=metadata, timesteps=SMOKE_TIMESTEPS
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SMOKE_KIND,
        "passed": True,
        "source_kind": SOURCE_KIND,
        "execution_only_gate": True,
        "conditioning_or_behavior_gate_applied": False,
        "wandb_enabled": False,
        "fresh_e0b_initialization": True,
        "sampler": canonical_sampler(),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": metadata["sha256"],
            "training_timesteps": SMOKE_TIMESTEPS,
            "optimizer_adam_step": metadata["optimizer_adam_step"],
        },
        "training_trace": trace,
        "evaluation": evaluation_record,
        "economic_architecture": canonical_architecture(),
        "shared_context_initialization": canonical_initialization(),
        "code_revision": expected_revision,
    }


def _compare_probe(reported, inferred):
    reported = _canonical_json(reported)
    inferred = _canonical_json(inferred)
    reported.pop("created_utc", None)
    inferred.pop("created_utc", None)
    _require(
        reported == inferred,
        "v5 preflight probe differs from fresh checkpoint inference",
    )


def _validate_conditioning_probe(
        path, *, checkpoint, e0b, expected_revision,
):
    path = Path(path).expanduser().resolve()
    _exact_name(path, PREFLIGHT_PROBE_NAME, "v5 preflight probe")
    value = load_json(path)
    _require(value.get("probe") == probe.PROBE_NAME, "unknown v5 probe")
    _require(
        value.get("execution") == {
            "read_only_checkpoint": True,
            "ale_instantiated": False,
            "environment_steps": 0,
            "deterministic_statistics": [
                "full shared-context Beta mean",
                "current-skip-ablated Beta mean",
                "context-branch-ablated Beta mean",
                "context-and-current-skip-ablated Beta mean",
            ],
            "in_memory_ablation_restored": True,
            "device": "cpu",
        },
        "v5 probe execution contract changed",
    )
    _require(
        value.get("metadata_verification", {}).get("passed") is True,
        "v5 probe metadata verification failed",
    )
    record = value.get("checkpoint", {})
    _same_path(record.get("path"), checkpoint, label="v5 probe checkpoint")
    _require(
        record.get("sha256") == sha256_file(checkpoint)
        and record.get("training_timesteps") == PREFLIGHT_TIMESTEPS,
        "v5 probe checkpoint identity changed",
    )
    _require(
        record.get("economic_architecture") == canonical_architecture()
        and record.get("shared_context_initialization")
        == canonical_initialization(),
        "v5 probe architecture/initialization changed",
    )
    _require(
        record.get("e1_training_code_revision") == expected_revision,
        "v5 probe revision changed",
    )
    expected_checks = {
        "all_outputs_finite_and_in_unit_interval",
        "minimum_all_one_minus_all_zero_beta_mean_price",
        "minimum_current_coordinate_beta_mean_response",
        "largest_adjacent_threshold_price_reversal",
        "combined_context_and_skip_ablation_commitment_response",
        "conditioning_parameters_changed_from_exact_zero",
        "conditioning_parameters_have_positive_optimizer_steps",
        "in_memory_ablation_restored_parameters_exactly",
    }
    recomputed_gate = probe.shared_context_conditioning_gate(
        value.get("variants", {}),
        value.get("conditioning_learning_evidence", {}),
    )
    _require(
        value.get("warmup_gate") == recomputed_gate,
        "v5 conditioning gate does not recompute from raw variants",
    )
    _require(
        recomputed_gate.get("name") == probe.PROBE_GATE_NAME
        and set(recomputed_gate.get("checks", {})) == expected_checks,
        "v5 conditioning checks changed",
    )
    _require(
        recomputed_gate.get("passed") is True
        and all(
            check.get("passed") is True
            for check in recomputed_gate["checks"].values()
        ),
        "v5 conditioning preflight failed a preregistered gate",
    )
    inferred = probe.run_probe_from_checkpoints(
        checkpoint=checkpoint,
        e0b_checkpoint=e0b,
        device="cpu",
    )
    _compare_probe(value, inferred)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "probe": probe.PROBE_NAME,
        "gate": recomputed_gate,
        "no_ale": True,
        "environment_steps": 0,
    }


def _behavior_args(*, e0b, rom):
    return SimpleNamespace(
        role=ROLE,
        e0b_checkpoint=str(Path(e0b).expanduser().resolve()),
        gameplay_horizon=200,
        event_tail_steps=0,
        fixed_event_steps=None,
        grid_event_steps=PREFLIGHT_FIXED_EVENT_STEPS,
        fixed_eval_values=PREFLIGHT_FIXED_VALUES,
        e1_sampler_mode=SAMPLER_MODE,
        seed=1,
        start_method="spawn",
        noop_max=30,
        max_frames=100_000,
        rom_path=str(Path(rom).expanduser().resolve()),
        device="cpu",
    )


def _expected_event_rows(result, *, phase, context_ablation):
    rows = []
    for episode in result.get("episode_rows", []):
        for event in episode.get("events", []):
            rows.append({
                "phase": phase,
                "checkpoint_path": episode["checkpoint_path"],
                "checkpoint_sha256": episode["checkpoint_sha256"],
                "training_timesteps": episode["training_timesteps"],
                "evaluation_seed": episode["evaluation_seed"],
                "fifth_economic_override": episode.get(
                    "fifth_economic_override"
                ),
                "all_trade_economic_override": episode.get(
                    "all_trade_economic_override"
                ),
                "v5_context_ablation": bool(context_ablation),
                **event,
            })
    return _canonical_json(rows)


def _expected_behavior_schedule(seed, *, fixed_event_steps=None):
    if fixed_event_steps is not None:
        return tuple(int(step) for step in fixed_event_steps)
    wrapper_rng = np.random.default_rng(int(seed) + 74_711)
    # The externally supplied context is copied without consuming wrapper_rng.
    inner_seed = int(wrapper_rng.integers(0, 2 ** 31 - 1))
    return ExactFiveEventSchedule(
        gameplay_horizon=200,
        tail_steps=0,
    ).sample(np.random.default_rng(inner_seed))


def _validate_behavior_result(
        result, *, metadata, seeds, contexts, phase, context_ablation,
        fixed_event_steps=None,
):
    seeds = [int(seed) for seed in seeds]
    contexts = [tuple(float(value) for value in context) for context in contexts]
    rows = result.get("episode_rows", [])
    _require(
        len(rows) == len(seeds)
        and [row.get("evaluation_seed") for row in rows] == seeds,
        f"v5 behavioral {phase} seed schedule changed",
    )
    for index, (row, context) in enumerate(zip(rows, contexts)):
        _require(
            row.get("phase") == phase
            and row.get("checkpoint_sha256") == metadata["sha256"]
            and row.get("training_timesteps") == PREFLIGHT_TIMESTEPS,
            f"v5 behavioral {phase} episode identity changed",
        )
        _same_path(
            row.get("checkpoint_path"),
            metadata["path"],
            label=f"v5 behavioral {phase} checkpoint",
        )
        _require(
            len(row.get("opponent_commitment", [])) == 5
            and np.allclose(
                row["opponent_commitment"], context,
                rtol=0.0, atol=1.0e-7,
            ),
            f"v5 behavioral {phase} context {index} changed",
        )
        _require(
            bool(row.get("v5_context_ablation", False))
            is bool(context_ablation),
            f"v5 behavioral {phase} ablation flag changed",
        )
        expected_schedule = _expected_behavior_schedule(
            seeds[index], fixed_event_steps=fixed_event_steps
        )
        _require(
            tuple(row.get("event_steps", ())) == expected_schedule,
            f"v5 behavioral {phase} seeded schedule {index} changed",
        )
    violations = [
        violation
        for row in rows
        for violation in evaluator.audit_episode(row, role=ROLE)
    ]
    _require(
        not violations
        and result.get("protocol") == {"passed": True, "violations": []},
        f"v5 behavioral {phase} mechanics failed: {violations[:3]}",
    )
    _require(
        result.get("summary") == evaluator._summary(rows, role=ROLE),
        f"v5 behavioral {phase} summary does not recompute",
    )
    _require(
        result.get("event_rows") == _expected_event_rows(
            result, phase=phase, context_ablation=context_ablation
        ),
        f"v5 behavioral {phase} event rows do not recompute",
    )
    _require(
        len(result.get("event_rows", [])) == 5 * len(rows),
        f"v5 behavioral {phase} does not retain all five events",
    )


def run_behavioral_preflight(
        *, checkpoint, e0b, rom, output, expected_revision,
):
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    _exact_name(
        checkpoint, PREFLIGHT_CHECKPOINT_NAME,
        "v5 behavioral-preflight checkpoint",
    )
    _exact_name(
        output, PREFLIGHT_BEHAVIOR_NAME,
        "v5 behavioral-preflight report",
    )
    _require(
        not os.path.lexists(output),
        "refusing to overwrite immutable v5 behavioral preflight",
    )
    _require(
        sha256_file(e0b) == CANONICAL_E0B_SHA256,
        "v5 behavioral preflight requires canonical E0b",
    )
    _require(
        sha256_file(rom) == CANONICAL_ROM_SHA256,
        "v5 behavioral preflight requires canonical ROM",
    )
    metadata = _load_metadata(
        checkpoint, e0b=e0b, expected_revision=expected_revision
    )
    _validate_checkpoint_metadata(
        metadata,
        checkpoint=checkpoint,
        timesteps=PREFLIGHT_TIMESTEPS,
        expected_revision=expected_revision,
    )
    args = _behavior_args(e0b=e0b, rom=rom)
    environment = evaluator.environment_config(args)
    random_seeds = list(range(
        PREFLIGHT_BEHAVIOR_SEED_START,
        PREFLIGHT_BEHAVIOR_SEED_START + PREFLIGHT_BEHAVIOR_EPISODES,
    ))
    random_contexts = [evaluator.random_context(seed) for seed in random_seeds]
    fixed_seeds = list(range(
        PREFLIGHT_FIXED_SEED_START,
        PREFLIGHT_FIXED_SEED_START + PREFLIGHT_BEHAVIOR_EPISODES,
    ))
    before = sha256_file(checkpoint)
    model, loaded = evaluator.load_candidate(
        checkpoint,
        role=ROLE,
        e0b_sha256=CANONICAL_E0B_SHA256,
        device="cpu",
    )
    try:
        _require(
            loaded["sha256"] == before == metadata["sha256"],
            "v5 behavioral preflight loaded different checkpoint bytes",
        )
        full = evaluator.evaluate_rows(
            model,
            args,
            metadata,
            seeds=random_seeds,
            contexts=random_contexts,
            phase="preflight_random_full",
        )
        ablated = evaluator.evaluate_rows(
            model,
            args,
            metadata,
            seeds=random_seeds,
            contexts=random_contexts,
            phase="preflight_random_joint_context_ablated",
            v5_context_ablation=True,
        )
        fixed = evaluator.fixed_grid(
            model, args, metadata, seeds=fixed_seeds
        )
    finally:
        del model
    _require(
        sha256_file(checkpoint) == before,
        "v5 behavioral preflight changed checkpoint bytes",
    )
    gate = evaluator.seller_v5_behavioral_gate(
        random_result=full,
        fixed_results=fixed,
        ablated_random_result=ablated,
        forced_constant_results=None,
        formal=False,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": (
            "stackpomdp.atari.e1_seller_shared_context_behavioral_preflight.v5"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(gate.get("passed") is True),
        "source_kind": SOURCE_KIND,
        "formal": False,
        "checkpoint": metadata,
        "e0b": {
            "path": str(Path(e0b).expanduser().resolve()),
            "sha256": CANONICAL_E0B_SHA256,
        },
        "rom": {
            "path": str(Path(rom).expanduser().resolve()),
            "sha256": CANONICAL_ROM_SHA256,
        },
        "environment": environment,
        "protocol": {
            "random_episodes": PREFLIGHT_BEHAVIOR_EPISODES,
            "random_seed_start": PREFLIGHT_BEHAVIOR_SEED_START,
            "random_contexts": "five independent Uniform(0,1) values",
            "full_and_ablation_paired": True,
            "joint_context_ablation": (
                "context branch and current-threshold skip both disabled"
            ),
            "fixed_episodes_per_value": PREFLIGHT_BEHAVIOR_EPISODES,
            "fixed_seed_start": PREFLIGHT_FIXED_SEED_START,
            "fixed_values": list(PREFLIGHT_FIXED_VALUES),
            "fixed_event_steps": list(PREFLIGHT_FIXED_EVENT_STEPS),
            "forced_price_controls": False,
            "formal_gate": False,
        },
        "full_random": full,
        "joint_context_ablated_random": ablated,
        "fixed_contexts": fixed,
        "behavioral_gate": gate,
    }
    atomic_write_json(output, report)
    return validate_behavioral_preflight(
        output,
        checkpoint=checkpoint,
        e0b=e0b,
        rom=rom,
        expected_revision=expected_revision,
        require_pass=False,
    )


def validate_behavioral_preflight(
        path, *, checkpoint, e0b, rom, expected_revision, require_pass=True,
):
    path = Path(path).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    _exact_name(
        path, PREFLIGHT_BEHAVIOR_NAME,
        "v5 behavioral-preflight report",
    )
    value = load_json(path)
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("kind")
        == "stackpomdp.atari.e1_seller_shared_context_behavioral_preflight.v5"
        and value.get("source_kind") == SOURCE_KIND
        and value.get("formal") is False,
        "unknown v5 behavioral-preflight report",
    )
    _require(
        sha256_file(e0b) == CANONICAL_E0B_SHA256
        and value.get("e0b", {}).get("sha256") == CANONICAL_E0B_SHA256,
        "v5 behavioral-preflight E0b changed",
    )
    _require(
        sha256_file(rom) == CANONICAL_ROM_SHA256
        and value.get("rom", {}).get("sha256") == CANONICAL_ROM_SHA256,
        "v5 behavioral-preflight ROM changed",
    )
    _same_path(value["e0b"]["path"], e0b, label="behavioral E0b")
    _same_path(value["rom"]["path"], rom, label="behavioral ROM")
    metadata = _load_metadata(
        checkpoint, e0b=e0b, expected_revision=expected_revision
    )
    _validate_checkpoint_metadata(
        metadata,
        checkpoint=checkpoint,
        timesteps=PREFLIGHT_TIMESTEPS,
        expected_revision=expected_revision,
    )
    _require(
        value.get("checkpoint") == metadata,
        "v5 behavioral-preflight checkpoint metadata changed",
    )
    args = _behavior_args(e0b=e0b, rom=rom)
    _require(
        value.get("environment") == evaluator.environment_config(args),
        "v5 behavioral-preflight environment changed",
    )
    _require(
        value.get("protocol") == {
            "random_episodes": PREFLIGHT_BEHAVIOR_EPISODES,
            "random_seed_start": PREFLIGHT_BEHAVIOR_SEED_START,
            "random_contexts": "five independent Uniform(0,1) values",
            "full_and_ablation_paired": True,
            "joint_context_ablation": (
                "context branch and current-threshold skip both disabled"
            ),
            "fixed_episodes_per_value": PREFLIGHT_BEHAVIOR_EPISODES,
            "fixed_seed_start": PREFLIGHT_FIXED_SEED_START,
            "fixed_values": list(PREFLIGHT_FIXED_VALUES),
            "fixed_event_steps": list(PREFLIGHT_FIXED_EVENT_STEPS),
            "forced_price_controls": False,
            "formal_gate": False,
        },
        "v5 behavioral-preflight protocol changed",
    )
    random_seeds = list(range(
        PREFLIGHT_BEHAVIOR_SEED_START,
        PREFLIGHT_BEHAVIOR_SEED_START + PREFLIGHT_BEHAVIOR_EPISODES,
    ))
    random_contexts = [evaluator.random_context(seed) for seed in random_seeds]
    full = value.get("full_random", {})
    ablated = value.get("joint_context_ablated_random", {})
    _validate_behavior_result(
        full,
        metadata=metadata,
        seeds=random_seeds,
        contexts=random_contexts,
        phase="preflight_random_full",
        context_ablation=False,
    )
    _validate_behavior_result(
        ablated,
        metadata=metadata,
        seeds=random_seeds,
        contexts=random_contexts,
        phase="preflight_random_joint_context_ablated",
        context_ablation=True,
    )
    _require(
        [row.get("event_steps") for row in full["episode_rows"]]
        == [row.get("event_steps") for row in ablated["episode_rows"]],
        "v5 full and joint-ablation schedules are not paired",
    )
    fixed = value.get("fixed_contexts", [])
    _require(
        len(fixed) == len(PREFLIGHT_FIXED_VALUES),
        "v5 behavioral fixed grid is incomplete",
    )
    fixed_seeds = list(range(
        PREFLIGHT_FIXED_SEED_START,
        PREFLIGHT_FIXED_SEED_START + PREFLIGHT_BEHAVIOR_EPISODES,
    ))
    for result, fixed_value in zip(fixed, PREFLIGHT_FIXED_VALUES):
        _require(
            result.get("opponent_value") == fixed_value,
            "v5 behavioral fixed-grid value changed",
        )
        context = np.full(5, fixed_value, dtype=np.float32)
        _validate_behavior_result(
            result,
            metadata=metadata,
            seeds=fixed_seeds,
            contexts=[context] * len(fixed_seeds),
            phase=f"fixed_{fixed_value:.2f}",
            context_ablation=False,
            fixed_event_steps=PREFLIGHT_FIXED_EVENT_STEPS,
        )
    gate = evaluator.seller_v5_behavioral_gate(
        random_result=full,
        fixed_results=fixed,
        ablated_random_result=ablated,
        forced_constant_results=None,
        formal=False,
    )
    _require(
        value.get("behavioral_gate") == gate,
        "v5 behavioral gate does not recompute from episode/event rows",
    )
    passed = bool(gate.get("passed") is True)
    _require(
        value.get("passed") is passed,
        "v5 behavioral-preflight outcome changed",
    )
    if require_pass:
        _require(
            passed,
            f"v5 {PREFLIGHT_TIMESTEPS}-step real-ALE behavioral preflight failed",
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "passed": passed,
        "formal": False,
        "random_episodes": PREFLIGHT_BEHAVIOR_EPISODES,
        "fixed_episodes_per_value": PREFLIGHT_BEHAVIOR_EPISODES,
        "full_and_joint_ablation_paired": True,
        "behavioral_gate": gate,
    }


def validate_preflight(
        *, checkpoint, training_log, evaluation, conditioning_probe,
        behavioral_report, e0b, rom, expected_revision,
):
    checkpoint = Path(checkpoint).expanduser().resolve()
    _exact_name(
        checkpoint, PREFLIGHT_CHECKPOINT_NAME, "v5 preflight checkpoint"
    )
    _same_path(
        training_log,
        _expected_sidecar(checkpoint, "training.jsonl"),
        label="v5 preflight trace",
    )
    _same_path(
        evaluation,
        _expected_sidecar(checkpoint, "evaluation.json"),
        label="v5 preflight evaluation",
    )
    metadata = _load_metadata(
        checkpoint, e0b=e0b, expected_revision=expected_revision
    )
    _validate_checkpoint_metadata(
        metadata,
        checkpoint=checkpoint,
        timesteps=PREFLIGHT_TIMESTEPS,
        expected_revision=expected_revision,
    )
    trace = _validate_trace(
        training_log, checkpoint=checkpoint, timesteps=PREFLIGHT_TIMESTEPS
    )
    evaluation_record = _validate_evaluation(
        evaluation, metadata=metadata, timesteps=PREFLIGHT_TIMESTEPS
    )
    probe_record = _validate_conditioning_probe(
        conditioning_probe,
        checkpoint=checkpoint,
        e0b=e0b,
        expected_revision=expected_revision,
    )
    behavior_record = validate_behavioral_preflight(
        behavioral_report,
        checkpoint=checkpoint,
        e0b=e0b,
        rom=rom,
        expected_revision=expected_revision,
        require_pass=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PREFLIGHT_KIND,
        "passed": True,
        "source_kind": SOURCE_KIND,
        "execution_mechanics_gate_passed": True,
        "conditioning_gate_passed": True,
        "real_ale_behavioral_gate_passed": True,
        "formal_behavior_claim": False,
        "wandb_enabled": False,
        "fresh_e0b_initialization": True,
        "sampler": canonical_sampler(),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": metadata["sha256"],
            "training_timesteps": PREFLIGHT_TIMESTEPS,
            "optimizer_adam_step": metadata["optimizer_adam_step"],
        },
        "training_trace": trace,
        "evaluation": evaluation_record,
        "conditioning_probe": probe_record,
        "behavioral_preflight": behavior_record,
        "economic_architecture": canonical_architecture(),
        "shared_context_initialization": canonical_initialization(),
        "code_revision": expected_revision,
    }


def formal_candidates(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    stem = checkpoint.with_suffix("")
    return [
        str(stem.with_name(f"{stem.name}_step{step}.zip"))
        for step in FORMAL_STEP_TIMESTEPS
    ] + [str(checkpoint)]


def build_gate(args):
    _git_scoped_clean(args.code_root)
    revision = _git_revision(args.code_root)
    evidence_revision = args.evidence_code_revision or revision
    _require(
        isinstance(evidence_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", evidence_revision) is not None,
        "v5 diagnostic evidence revision is invalid",
    )
    revision_bridge = _validate_diagnostic_revision_bridge(
        args.code_root,
        runtime_revision=revision,
        evidence_revision=evidence_revision,
    )
    output = Path(args.output).expanduser().resolve()
    _exact_name(output, GATE_NAME, "v5 diagnostics gate")
    e0b = Path(args.e0b).expanduser().resolve()
    rom = Path(args.rom).expanduser().resolve()
    seller_release = Path(args.seller_release).expanduser().resolve()
    _require(
        sha256_file(e0b) == CANONICAL_E0B_SHA256,
        "v5 gate requires exact canonical E0b bytes",
    )
    _require(
        sha256_file(rom) == CANONICAL_ROM_SHA256,
        "v5 gate requires exact canonical ROM bytes",
    )
    shared.validate_seller_release(seller_release)
    _require(
        sha256_file(seller_release) == CANONICAL_SELLER_RELEASE_SHA256,
        "v5 gate requires exact canonical buyer-release bytes",
    )
    formal = Path(args.formal_checkpoint).expanduser().resolve()
    _exact_name(formal, FORMAL_CHECKPOINT_NAME, "v5 formal checkpoint")
    _require(
        not os.path.lexists(formal),
        "formal v5 checkpoint exists before its diagnostics gate",
    )
    smoke = validate_smoke(
        checkpoint=args.smoke_checkpoint,
        training_log=args.smoke_training_log,
        evaluation=args.smoke_evaluation,
        e0b=e0b,
        expected_revision=evidence_revision,
    )
    preflight = validate_preflight(
        checkpoint=args.preflight_checkpoint,
        training_log=args.preflight_training_log,
        evaluation=args.preflight_evaluation,
        conditioning_probe=args.preflight_probe,
        behavioral_report=args.preflight_behavior,
        e0b=e0b,
        rom=rom,
        expected_revision=evidence_revision,
    )
    _require(
        smoke["checkpoint"]["sha256"]
        != preflight["checkpoint"]["sha256"],
        "v5 smoke and conditioning preflight are not distinct runs",
    )
    _git_scoped_clean(args.code_root)
    _require(
        _git_revision(args.code_root) == revision,
        "v5 code revision changed while building diagnostics gate",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": GATE_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "source_kind": SOURCE_KIND,
        "exposure_protocol": canonical_protocol_provenance(),
        "code_revision": revision,
        "diagnostic_evidence_revision": evidence_revision,
        "validator_revision_bridge": revision_bridge,
        "legacy_recovery_artifacts_admitted": False,
        "prerequisites": {
            "e0b": {"path": str(e0b), "sha256": CANONICAL_E0B_SHA256},
            "rom": {"path": str(rom), "sha256": CANONICAL_ROM_SHA256},
            "buyer_release": {
                "path": str(seller_release),
                "sha256": CANONICAL_SELLER_RELEASE_SHA256,
            },
        },
        "economic_architecture": canonical_architecture(),
        "shared_context_initialization": canonical_initialization(),
        "training_config": canonical_training_config(),
        "sampler": canonical_sampler(),
        "smoke": smoke,
        "conditioning_preflight": preflight,
        "formal_release": {
            "allowed": True,
            "fresh_e0b_initialization": True,
            "resume": False,
            "additional_timesteps": FORMAL_TIMESTEPS,
            "expected_total_timesteps": FORMAL_TIMESTEPS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "checkpoint": str(formal),
            "candidate_paths": formal_candidates(formal),
            "wandb": {
                "enabled": True,
                "project": WANDB_PROJECT,
                "group": WANDB_GROUP,
                "job_type": WANDB_JOB_TYPE,
                "name": WANDB_NAME,
            },
        },
    }


def _validate_prerequisites(value):
    prerequisites = value.get("prerequisites", {})
    expected = {
        "e0b": CANONICAL_E0B_SHA256,
        "rom": CANONICAL_ROM_SHA256,
        "buyer_release": CANONICAL_SELLER_RELEASE_SHA256,
    }
    for key, digest in expected.items():
        record = prerequisites.get(key, {})
        _require(
            sha256_file(record.get("path", ""))
            == record.get("sha256")
            == digest,
            f"v5 gate prerequisite changed: {key}",
        )
    shared.validate_seller_release(
        prerequisites["buyer_release"]["path"]
    )


def validate_gate(path, *, code_root=None):
    path = Path(path).expanduser().resolve()
    _exact_name(path, GATE_NAME, "v5 diagnostics gate")
    value = load_json(path)
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("kind") == GATE_KIND
        and value.get("passed") is True,
        "unknown or failed v5 diagnostics gate",
    )
    _require(
        value.get("source_kind") == SOURCE_KIND
        and value.get("legacy_recovery_artifacts_admitted") is False,
        "v5 gate source namespace changed",
    )
    _require(
        value.get("exposure_protocol") == canonical_protocol_provenance(),
        "v5 gate exposure protocol changed",
    )
    revision = value.get("code_revision")
    _require(
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "v5 gate revision is invalid",
    )
    evidence_revision = value.get("diagnostic_evidence_revision", revision)
    _require(
        isinstance(evidence_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", evidence_revision) is not None,
        "v5 diagnostic evidence revision is invalid",
    )
    if code_root is not None:
        expected_bridge = _validate_diagnostic_revision_bridge(
            code_root,
            runtime_revision=revision,
            evidence_revision=evidence_revision,
        )
        _require(
            value.get("validator_revision_bridge") == expected_bridge,
            "v5 validator-only revision bridge changed",
        )
    _validate_prerequisites(value)
    _require(
        value.get("economic_architecture") == canonical_architecture()
        and value.get("shared_context_initialization")
        == canonical_initialization()
        and value.get("training_config") == canonical_training_config()
        and value.get("sampler") == canonical_sampler(),
        "v5 gate architecture/training protocol changed",
    )
    e0b = value["prerequisites"]["e0b"]["path"]
    rom = value["prerequisites"]["rom"]["path"]
    smoke_record = value.get("smoke", {})
    smoke = validate_smoke(
        checkpoint=smoke_record.get("checkpoint", {}).get("path"),
        training_log=smoke_record.get("training_trace", {}).get("path"),
        evaluation=smoke_record.get("evaluation", {}).get("path"),
        e0b=e0b,
        expected_revision=evidence_revision,
    )
    _require(smoke_record == smoke, "v5 smoke evidence changed")
    preflight_record = value.get("conditioning_preflight", {})
    preflight = validate_preflight(
        checkpoint=preflight_record.get("checkpoint", {}).get("path"),
        training_log=preflight_record.get("training_trace", {}).get("path"),
        evaluation=preflight_record.get("evaluation", {}).get("path"),
        conditioning_probe=preflight_record.get(
            "conditioning_probe", {}
        ).get("path"),
        behavioral_report=preflight_record.get(
            "behavioral_preflight", {}
        ).get("path"),
        e0b=e0b,
        rom=rom,
        expected_revision=evidence_revision,
    )
    _require(
        preflight_record == preflight,
        "v5 conditioning-preflight evidence changed",
    )
    formal = value.get("formal_release", {})
    _exact_name(
        formal.get("checkpoint", ""),
        FORMAL_CHECKPOINT_NAME,
        "v5 formal checkpoint",
    )
    _require(
        formal == {
            "allowed": True,
            "fresh_e0b_initialization": True,
            "resume": False,
            "additional_timesteps": FORMAL_TIMESTEPS,
            "expected_total_timesteps": FORMAL_TIMESTEPS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "checkpoint": formal["checkpoint"],
            "candidate_paths": formal_candidates(formal["checkpoint"]),
            "wandb": {
                "enabled": True,
                "project": WANDB_PROJECT,
                "group": WANDB_GROUP,
                "job_type": WANDB_JOB_TYPE,
                "name": WANDB_NAME,
            },
        },
        "v5 formal-release protocol changed",
    )
    if code_root is not None:
        _require(
            _git_revision(code_root) == revision,
            "runtime revision differs from v5 diagnostics gate",
        )
        _git_scoped_clean(code_root)
    return value


def write_or_validate_gate(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        return validate_gate(output, code_root=args.code_root)
    _require(
        not output.is_symlink(),
        "refusing symlink v5 diagnostics-gate output",
    )
    expected = build_gate(args)
    atomic_write_json(output, expected)
    return validate_gate(output, code_root=args.code_root)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOL_CONFIGURATIONS),
        default=DEFAULT_PROTOCOL,
        help=(
            "exact versioned exposure protocol; standard_v1 preserves the "
            "original 82k gate and exposure_v2 uses a fresh 400160-step gate"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("validate-smoke")
    for name in ("checkpoint", "training-log", "evaluation", "e0b"):
        smoke.add_argument(f"--{name}", required=True)
    smoke.add_argument("--code-revision", required=True)

    preflight = commands.add_parser("validate-preflight")
    for name in (
            "checkpoint", "training-log", "evaluation", "probe",
            "behavior-report", "e0b", "rom",
    ):
        preflight.add_argument(f"--{name}", required=True)
    preflight.add_argument("--code-revision", required=True)

    behavior = commands.add_parser("behavioral-preflight")
    for name in ("checkpoint", "e0b", "rom", "output"):
        behavior.add_argument(f"--{name}", required=True)
    behavior.add_argument("--code-revision", required=True)

    gate = commands.add_parser("gate")
    for name in (
            "seller-release", "e0b", "rom", "smoke-checkpoint",
            "smoke-training-log", "smoke-evaluation",
            "preflight-checkpoint", "preflight-training-log",
            "preflight-evaluation", "preflight-probe",
            "preflight-behavior", "formal-checkpoint", "code-root", "output",
    ):
        gate.add_argument(f"--{name}", required=True)
    gate.add_argument("--evidence-code-revision")

    validate = commands.add_parser("validate-gate")
    validate.add_argument("--gate", required=True)
    validate.add_argument("--code-root")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_protocol(args.protocol)
    if args.command == "validate-smoke":
        result = validate_smoke(
            checkpoint=args.checkpoint,
            training_log=args.training_log,
            evaluation=args.evaluation,
            e0b=args.e0b,
            expected_revision=args.code_revision,
        )
    elif args.command == "validate-preflight":
        result = validate_preflight(
            checkpoint=args.checkpoint,
            training_log=args.training_log,
            evaluation=args.evaluation,
            conditioning_probe=args.probe,
            behavioral_report=args.behavior_report,
            e0b=args.e0b,
            rom=args.rom,
            expected_revision=args.code_revision,
        )
    elif args.command == "behavioral-preflight":
        result = run_behavioral_preflight(
            checkpoint=args.checkpoint,
            e0b=args.e0b,
            rom=args.rom,
            output=args.output,
            expected_revision=args.code_revision,
        )
    elif args.command == "gate":
        result = write_or_validate_gate(args)
    else:
        result = validate_gate(args.gate, code_root=args.code_root)
    print(json.dumps({
        "command": args.command,
        "passed": bool(result.get("passed")),
        "kind": result.get("kind"),
        "value": result,
    }, sort_keys=True), flush=True)
    if args.command == "behavioral-preflight" and not result.get("passed"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
