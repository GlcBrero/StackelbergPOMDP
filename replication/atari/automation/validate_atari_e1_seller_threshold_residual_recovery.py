#!/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python
"""Fail-closed v2 Atari E1 seller threshold-residual recovery artifacts.

The ordinary 64-input economic head is retained.  Its learned Beta mean is
blended equally with the opponent threshold at the current trade event while
preserving concentration.  The fixed anchor is an inductive bias; a separate
gate requires nonzero learned base-head conditioning.  The v1 failed warm-up
is independently reloaded and recomputed before this family may be activated.
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

from replication.atari.automation import (
    validate_atari_e1_seller_conditioning_recovery as v1,
)
from replication.atari import (
    probe_atari_e1_seller_conditioning as v1_probe,
)
from replication.atari import (
    probe_atari_e1_seller_threshold_residual as v2_probe,
)


SCHEMA_VERSION = 1
ACTIVATION_KIND = (
    "stackpomdp.atari.e1_seller_conditioning_recovery_activation.v2"
)
FAMILY_KIND = "stackpomdp.atari.e1_seller_conditioning_recovery_family.v2"
GATE_KIND = "stackpomdp.atari.e1_seller_conditioning_recovery_gate.v2"
SOURCE_KIND = "seller_conditioning_recovery_v2_threshold_residual_v1"
EVALUATOR = v1.EVALUATOR
ROLE = v1.ROLE
ACTOR_LOSS_MODE = v1.ACTOR_LOSS_MODE
ALL_EQUAL_MODE = v1.ALL_EQUAL_MODE
UNIFORM_MODE = v1.UNIFORM_MODE
WARMUP_TIMESTEPS = v1.WARMUP_TIMESTEPS
PREFLIGHT_TIMESTEPS = 20_500
TARGET_ADDITIONAL_TIMESTEPS = v1.TARGET_ADDITIONAL_TIMESTEPS
FINAL_TIMESTEPS = v1.FINAL_TIMESTEPS
CHECKPOINT_INTERVAL = v1.CHECKPOINT_INTERVAL
TARGET_STEP_TIMESTEPS = v1.TARGET_STEP_TIMESTEPS
TARGET_CANDIDATE_TIMESTEPS = v1.TARGET_CANDIDATE_TIMESTEPS
SCREEN_SEED_START = v1.SCREEN_SEED_START
CONFIRMATION_SEED_START = v1.CONFIRMATION_SEED_START
FIXED_SEED_START = v1.FIXED_SEED_START
TIMING_SEED_START = v1.TIMING_SEED_START
MINIMUM_ENDPOINT_RESPONSE = v1.MINIMUM_ENDPOINT_RESPONSE
MAXIMUM_ADJACENT_REVERSAL = v1.MAXIMUM_ADJACENT_REVERSAL
CANONICAL_E0B_SHA256 = v1.CANONICAL_E0B_SHA256
CANONICAL_ROM_SHA256 = v1.CANONICAL_ROM_SHA256
CANONICAL_SELLER_RELEASE_SHA256 = v1.CANONICAL_SELLER_RELEASE_SHA256

ACTIVATION_NAME = (
    "e1_seller_conditioning_recovery_v2_threshold_residual_v1_activation.json"
)
FAMILY_NAME = (
    "e1_seller_conditioning_recovery_v2_threshold_residual_v1_family.json"
)
REPORT_NAME = (
    "e1_seller_conditioning_recovery_v2_threshold_residual_v1_"
    "all6_selector_v2.json"
)
GATE_NAME = f"{Path(REPORT_NAME).stem}.gate.json"
SELECTED_NAME = (
    "meta_seller_e1_ppo_balanced_conditioning_recovery_v2_"
    "threshold_residual_v1_seed1_selected.zip"
)
WARMUP_CHECKPOINT_NAME = (
    "meta_seller_e1_ppo_balanced_conditioning_recovery_v2_"
    "threshold_residual_v1_seed1_all_equal_warmup.zip"
)
TARGET_CHECKPOINT_NAME = (
    "meta_seller_e1_ppo_balanced_conditioning_recovery_v2_"
    "threshold_residual_v1_seed1_uniform_target.zip"
)
PROBE_ARTIFACT_NAME = (
    "e1_seller_conditioning_recovery_v2_threshold_residual_v1_"
    "warmup_probe.json"
)
PREFLIGHT_CHECKPOINT_NAME = (
    "meta_seller_e1_ppo_balanced_conditioning_recovery_v2_"
    "threshold_residual_v1_seed1_pure64_preflight.zip"
)
PREFLIGHT_PROBE_NAME = (
    "e1_seller_conditioning_recovery_v2_threshold_residual_v1_"
    "pure64_preflight_probe.json"
)

# The exact already-completed v1 negative experiment.  The v2 activation also
# recomputes every semantic check; these hashes prevent substituting another
# failed run after seeing its outcome.
V1_ACTIVATION_SHA256 = (
    "74f61bdff7bc480a2b1512ab685c91cd6fd0398eff34936f47be235db0e25366"
)
V1_CODE_REVISION = "ca57c6daeb1b7282fb8f91018474d1fae8c48752"
V1_WARMUP_SHA256 = (
    "e2534d78e76603ef258c9339cf45e7994f705476b22f5abaed9b8e4b002b1763"
)
V1_PROBE_SHA256 = (
    "451760ae545436dbcb8f3275a169ba3e466c13bee233e78bd9cf5ae59f4e1e0f"
)
V1_TRAINING_LOG_SHA256 = (
    "7f9022c61b19af5065a7f6211a0ac41fa438b2014143abf06c8fb607e40e46ee"
)
V1_EVALUATION_SHA256 = (
    "921953801671395a4998fe0ae892c6b331ee3b59d527e5221f469d0f75d11533"
)
LEARNING_RATE = 1.0e-4


_require = v1._require
sha256_file = v1.sha256_file
load_json = v1.load_json
atomic_write_json = v1.atomic_write_json
_same_path = v1._same_path
_canonical_json = v1._canonical_json
canonical_uniform_sampler = v1.canonical_uniform_sampler
canonical_all_equal_sampler = v1.canonical_all_equal_sampler
expected_target_candidates = v1.expected_target_candidates
_git_revision = v1._git_revision


def _git_scoped_clean(code_root):
    """Extend the v1 clean-code contract to the v2 probe module."""

    v1._git_scoped_clean(code_root)
    root = Path(code_root).expanduser().resolve()
    dirty = subprocess.run(
        [
            "git", "-C", str(root), "status", "--porcelain", "--",
            "replication/atari/"
            "probe_atari_e1_seller_threshold_residual.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(
        not dirty,
        f"v2 recovery activation requires clean residual probe: {dirty}",
    )


def canonical_economic_architecture():
    return v2_probe.canonical_economic_architecture()


def canonical_training_config():
    return {
        **v1.canonical_training_config(),
        "learning_rate": LEARNING_RATE,
        "economic_threshold_residual": True,
    }


def activation_learning_rate(activation):
    config = activation.get("protocol", {}).get("training_config", {})
    value = float(config.get("learning_rate", math.nan))
    _require(value == LEARNING_RATE, "v2 activation learning rate is invalid")
    _require(config == canonical_training_config(), "v2 activation training config changed")
    return value


def _exact_hash(path, expected, label):
    actual = sha256_file(path)
    _require(actual == expected, f"{label} is not the preregistered byte artifact")
    return actual


def _compare_fresh_probe(reported, inferred, *, label):
    reported = _canonical_json(reported)
    inferred = _canonical_json(inferred)
    reported.pop("created_utc", None)
    inferred.pop("created_utc", None)
    _require(reported == inferred, f"{label} does not match fresh checkpoint inference")


def _v1_prohibited_outputs(activation, warmup_checkpoint):
    activation_path = Path(activation["__path__"]).resolve()
    warmup_path = Path(warmup_checkpoint).resolve()
    paths = [
        *activation["protocol"]["target"]["candidate_paths"],
        activation_path.with_name(v1.FAMILY_NAME),
        activation_path.with_name(v1.REPORT_NAME),
        activation_path.with_name(v1.GATE_NAME),
        warmup_path.with_name(
            "meta_seller_e1_ppo_balanced_conditioning_recovery_v1_"
            "seed1_selected.zip"
        ),
    ]
    return [str(Path(path).expanduser().resolve()) for path in paths]


def validate_v1_warmup_failure(
        *, activation, warmup_checkpoint, probe, training_log, evaluation,
        device="cpu",
):
    """Recompute and bind the exact v1 negative result, not its JSON pass bit."""

    activation_path = Path(activation).expanduser().resolve()
    _exact_hash(activation_path, V1_ACTIVATION_SHA256, "v1 activation")
    activation_value = v1.validate_activation(activation_path)
    _require(
        activation_value["code_revision"] == V1_CODE_REVISION,
        "v1 failure activation uses another code revision",
    )
    activation_value = {**activation_value, "__path__": str(activation_path)}

    checkpoint_path = Path(warmup_checkpoint).expanduser().resolve()
    _same_path(
        str(checkpoint_path),
        activation_value["protocol"]["warmup"]["checkpoint"],
        label="v1 failed warm-up checkpoint",
    )
    _exact_hash(checkpoint_path, V1_WARMUP_SHA256, "v1 warm-up checkpoint")
    metadata = v1._load_candidate_metadata(
        checkpoint_path, activation=activation_value, device=device
    )
    metadata["resume_source"] = metadata.get("resume_source")
    v1._validate_warmup_metadata(metadata, activation_value)

    trace_path = Path(training_log).expanduser().resolve()
    _exact_hash(trace_path, V1_TRAINING_LOG_SHA256, "v1 warm-up trace")
    trace = v1._validate_training_trace(
        trace_path,
        mode=ALL_EQUAL_MODE,
        first_step=820,
        last_step=WARMUP_TIMESTEPS,
        expected_episodes=1_952,
        expected_optimizers=488,
        checkpoint=checkpoint_path,
    )
    evaluation_path = Path(evaluation).expanduser().resolve()
    _exact_hash(evaluation_path, V1_EVALUATION_SHA256, "v1 warm-up evaluation")
    evaluation_record = v1._validate_evaluation(
        evaluation_path,
        current=canonical_all_equal_sampler(),
        history=metadata["atari_e1_sampler_history"],
        expected_step=WARMUP_TIMESTEPS,
        checkpoint_metadata=metadata,
        random_episodes=20,
        fixed_episodes=5,
    )

    probe_path = Path(probe).expanduser().resolve()
    _same_path(
        str(probe_path),
        activation_value["protocol"]["warmup"]["conditioning_probe"]["path"],
        label="v1 failed warm-up probe",
    )
    _exact_hash(probe_path, V1_PROBE_SHA256, "v1 warm-up probe")
    reported = load_json(probe_path)
    _require(reported.get("probe") == v1.PROBE_NAME, "unknown v1 failure probe")
    checkpoint_record = reported.get("checkpoint", {})
    _same_path(checkpoint_record.get("path"), checkpoint_path, label="v1 probe checkpoint")
    _require(
        checkpoint_record.get("sha256") == V1_WARMUP_SHA256,
        "v1 probe is bound to another checkpoint",
    )
    all_equal = [
        row.get("event_beta_mean_prices")
        for row in reported.get("all_equal_thresholds", [])
    ]
    coordinate = reported.get("current_coordinate_only_sensitivity", {}).get(
        "rows", []
    )
    low = [row.get("low_beta_mean_price") for row in coordinate]
    high = [row.get("high_beta_mean_price") for row in coordinate]
    recomputed_gate = v1_probe.warmup_diagnostic_gate(all_equal, low, high)
    _require(
        reported.get("warmup_gate") == recomputed_gate,
        "v1 failure gate does not recompute from raw measurements",
    )
    _require(recomputed_gate.get("passed") is False, "v1 warm-up did not fail")
    checks = recomputed_gate.get("checks", {})
    _require(
        checks.get("all_outputs_finite_and_in_unit_interval", {}).get("passed")
        is True,
        "v1 failure was caused by invalid numerical output",
    )
    endpoint = checks.get(
        "minimum_all_one_minus_all_zero_beta_mean_price", {}
    )
    _require(
        endpoint.get("passed") is False
        and endpoint.get("required") == MINIMUM_ENDPOINT_RESPONSE
        and float(endpoint.get("actual")) < MINIMUM_ENDPOINT_RESPONSE,
        "v1 failure is not the preregistered conditioning shortfall",
    )
    non_endpoint = {
        name: check
        for name, check in checks.items()
        if name != "minimum_all_one_minus_all_zero_beta_mean_price"
    }
    _require(
        non_endpoint
        and all(check.get("passed") is True for check in non_endpoint.values()),
        "v1 warm-up failed a gate other than endpoint response",
    )
    inferred = v1_probe.run_probe_from_checkpoints(
        checkpoint=checkpoint_path,
        e0b_checkpoint=activation_value["e0b_source"]["path"],
        device="cpu",
    )
    _compare_fresh_probe(reported, inferred, label="v1 failure probe JSON")

    prohibited = _v1_prohibited_outputs(activation_value, checkpoint_path)
    present = [path for path in prohibited if os.path.lexists(path)]
    _require(not present, f"v1 target/selection artifacts exist: {present}")
    return {
        "passed": True,
        "v1_warmup_gate_passed": False,
        "activation": {
            "path": str(activation_path),
            "sha256": V1_ACTIVATION_SHA256,
            "code_revision": V1_CODE_REVISION,
        },
        "warmup_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": V1_WARMUP_SHA256,
            "training_timesteps": WARMUP_TIMESTEPS,
        },
        "probe": {
            "path": str(probe_path),
            "sha256": V1_PROBE_SHA256,
            "minimum_endpoint_response": float(endpoint["actual"]),
            "required_minimum_endpoint_response": MINIMUM_ENDPOINT_RESPONSE,
        },
        "training_trace": trace,
        "evaluation": evaluation_record,
        "prohibited_v1_outputs_absent": prohibited,
    }


def _validate_exact_names(
        *, output, warmup, warmup_probe, target, preflight=None,
        preflight_probe=None,
):
    _require(Path(output).name == ACTIVATION_NAME, "v2 activation name changed")
    _require(Path(warmup).name == WARMUP_CHECKPOINT_NAME, "v2 warm-up name changed")
    _require(Path(warmup_probe).name == PROBE_ARTIFACT_NAME, "v2 probe name changed")
    _require(Path(target).name == TARGET_CHECKPOINT_NAME, "v2 target name changed")
    if preflight is not None:
        _require(
            Path(preflight).name == PREFLIGHT_CHECKPOINT_NAME,
            "v2 pure-64 preflight checkpoint name changed",
        )
    if preflight_probe is not None:
        _require(
            Path(preflight_probe).name == PREFLIGHT_PROBE_NAME,
            "v2 pure-64 preflight probe name changed",
        )


def build_activation(args):
    _git_scoped_clean(args.code_root)
    code_revision = _git_revision(args.code_root)
    prior = validate_v1_warmup_failure(
        activation=args.v1_activation,
        warmup_checkpoint=args.v1_warmup_checkpoint,
        probe=args.v1_warmup_probe,
        training_log=args.v1_warmup_training_log,
        evaluation=args.v1_warmup_evaluation,
        device="cpu",
    )
    release = Path(args.seller_release).expanduser().resolve()
    v1.validate_seller_release(release)
    e0b = Path(args.e0b).expanduser().resolve()
    _require(sha256_file(e0b) == CANONICAL_E0B_SHA256, "v2 E0b is not canonical")
    rom = Path(args.rom).expanduser().resolve()
    _require(sha256_file(rom) == CANONICAL_ROM_SHA256, "v2 ROM is not canonical")
    warmup = Path(args.warmup_checkpoint).expanduser().resolve()
    warmup_probe = Path(args.warmup_probe).expanduser().resolve()
    target = Path(args.target_checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    preflight = validate_preflight_evidence(
        checkpoint=args.preflight_checkpoint,
        probe=args.preflight_probe,
        evaluation=args.preflight_evaluation,
        e0b=e0b,
        expected_code_revision=code_revision,
        device="cpu",
    )
    _validate_exact_names(
        output=output, warmup=warmup,
        warmup_probe=warmup_probe, target=target,
        preflight=args.preflight_checkpoint,
        preflight_probe=args.preflight_probe,
    )
    candidates = expected_target_candidates(target)
    _git_scoped_clean(args.code_root)
    _require(
        _git_revision(args.code_root) == code_revision,
        "v2 code revision changed while building activation",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ACTIVATION_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision,
        "activation_condition": {
            "formal_prior_seller_gate_failed": True,
            "seller_conditioning_recovery_v1_warmup_gate_failed": True,
            "v1_target_never_started": True,
        },
        "prerequisite_v1_failure": prior,
        "seller_release": {"path": str(release), "sha256": sha256_file(release)},
        "e0b_source": {"path": str(e0b), "sha256": CANONICAL_E0B_SHA256},
        "rom": {"path": str(rom), "sha256": CANONICAL_ROM_SHA256},
        "pure64_preflight": preflight,
        "controlled_change": {
            "only_architectural_change_from_v1": (
                "preserve the 64-input seller economic head, then blend its "
                "Beta mean equally with the current event threshold while "
                "preserving concentration"
            ),
            "economic_architecture": canonical_economic_architecture(),
            "new_observation_fields": [],
            "environment_unchanged": True,
            "samplers_unchanged": True,
            "ppo_hyperparameters_unchanged": True,
            "learning_rate": LEARNING_RATE,
            "fixed_anchor_is_inductive_bias": True,
            "learned_threshold_slope_claim": False,
        },
        "protocol": {
            "role": ROLE,
            "actor_loss_mode": ACTOR_LOSS_MODE,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "economic_architecture": canonical_economic_architecture(),
            "training_config": canonical_training_config(),
            "stage_order": ["all_equal_warmup", "independent_uniform_target"],
            "warmup": {
                "fresh_e0b_initialization": True,
                "resume": False,
                "sampler": canonical_all_equal_sampler(),
                "start_total_timesteps": 0,
                "additional_timesteps": WARMUP_TIMESTEPS,
                "expected_total_timesteps": WARMUP_TIMESTEPS,
                "checkpoint": str(warmup),
                "selectable": False,
                "conditioning_probe": {
                    "path": str(warmup_probe),
                    "probe": v2_probe.PROBE_NAME,
                    "gate": v2_probe.PROBE_GATE_NAME,
                    "gate_math": "v1_numeric_sanity_plus_learned_base_response",
                    "no_ale": True,
                    "environment_steps": 0,
                    "deterministic_statistics": [
                        "learned base Beta mean", "final residual Beta mean"
                    ],
                    "require_all_outputs_finite_and_unit": True,
                    "minimum_endpoint_response": MINIMUM_ENDPOINT_RESPONSE,
                    "minimum_current_coordinate_response": (
                        v2_probe.MINIMUM_CURRENT_COORDINATE_RESPONSE
                    ),
                    "minimum_learned_base_response": (
                        v2_probe.MINIMUM_LEARNED_BASE_ENDPOINT_RESPONSE
                    ),
                    "maximum_adjacent_reversal": MAXIMUM_ADJACENT_REVERSAL,
                    "require_pass": True,
                },
            },
            "target": {
                "resume_complete_model_optimizer_clock": True,
                "resume_checkpoint": str(warmup),
                "sampler": canonical_uniform_sampler(),
                "start_total_timesteps": WARMUP_TIMESTEPS,
                "additional_timesteps": TARGET_ADDITIONAL_TIMESTEPS,
                "expected_total_timesteps": FINAL_TIMESTEPS,
                "candidate_paths": candidates,
                "candidate_timesteps": list(TARGET_CANDIDATE_TIMESTEPS),
                "candidate_count": 6,
                "warmup_selectable": False,
            },
            "evaluation": {
                "screen": {"episodes": 20, "seed_start": SCREEN_SEED_START},
                "confirmation": {
                    "episodes": 100,
                    "seed_start": CONFIRMATION_SEED_START,
                    "policy": "screen_winner_only_no_fallback",
                    "fallback_allowed": False,
                },
                "fixed_grid": {
                    "episodes_per_value": 20,
                    "seed_start": FIXED_SEED_START,
                    "values": [index / 10 for index in range(11)],
                    "event_steps": [20, 50, 80, 110, 140],
                },
                "timing_seed_start_reserved": TIMING_SEED_START,
            },
        },
    }


def _prior_args(record):
    return {
        "activation": record["activation"]["path"],
        "warmup_checkpoint": record["warmup_checkpoint"]["path"],
        "probe": record["probe"]["path"],
        "training_log": record["training_trace"]["path"],
        "evaluation": record["evaluation"]["path"],
    }


def _validate_activation_contract(value):
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("kind") == ACTIVATION_KIND,
        "unknown threshold-residual recovery activation",
    )
    _require(
        value.get("activation_condition") == {
            "formal_prior_seller_gate_failed": True,
            "seller_conditioning_recovery_v1_warmup_gate_failed": True,
            "v1_target_never_started": True,
        },
        "v2 activation condition changed",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", str(value.get("code_revision", ""))),
        "v2 activation revision is invalid",
    )
    prior = validate_v1_warmup_failure(**_prior_args(
        value.get("prerequisite_v1_failure", {})
    ))
    _require(
        value.get("prerequisite_v1_failure") == prior,
        "v2 prerequisite failure record changed",
    )
    release = value.get("seller_release", {})
    _require(
        sha256_file(release.get("path", ""))
        == release.get("sha256")
        == CANONICAL_SELLER_RELEASE_SHA256,
        "v2 seller release changed",
    )
    v1.validate_seller_release(release["path"])
    e0b = value.get("e0b_source", {})
    _require(
        sha256_file(e0b.get("path", ""))
        == e0b.get("sha256")
        == CANONICAL_E0B_SHA256,
        "v2 E0b changed",
    )
    rom = value.get("rom", {})
    _require(
        sha256_file(rom.get("path", ""))
        == rom.get("sha256")
        == CANONICAL_ROM_SHA256,
        "v2 ROM changed",
    )
    preflight = value.get("pure64_preflight", {})
    expected_preflight = validate_preflight_evidence(
        checkpoint=preflight.get("checkpoint", {}).get("path", ""),
        probe=preflight.get("conditioning_probe", {}).get("path", ""),
        evaluation=preflight.get("evaluation", {}).get("path", ""),
        e0b=e0b["path"],
        expected_code_revision=value["code_revision"],
        device="cpu",
    )
    _require(
        preflight == expected_preflight,
        "v2 pure-64 real-ALE preflight evidence changed",
    )
    expected_change = {
        "only_architectural_change_from_v1": (
            "preserve the 64-input seller economic head, then blend its "
            "Beta mean equally with the current event threshold while "
            "preserving concentration"
        ),
        "economic_architecture": canonical_economic_architecture(),
        "new_observation_fields": [],
        "environment_unchanged": True,
        "samplers_unchanged": True,
        "ppo_hyperparameters_unchanged": True,
        "learning_rate": LEARNING_RATE,
        "fixed_anchor_is_inductive_bias": True,
        "learned_threshold_slope_claim": False,
    }
    _require(value.get("controlled_change") == expected_change, "v2 controlled change changed")
    protocol = value.get("protocol", {})
    learning_rate = activation_learning_rate(value)
    _require(
        value["controlled_change"]["learning_rate"] == learning_rate,
        "v2 controlled-change learning rate differs from training config",
    )
    _require(
        protocol.get("role") == ROLE
        and protocol.get("actor_loss_mode") == ACTOR_LOSS_MODE
        and protocol.get("checkpoint_interval") == CHECKPOINT_INTERVAL
        and protocol.get("economic_architecture") == canonical_economic_architecture(),
        "v2 protocol identity changed",
    )
    warmup = protocol.get("warmup", {})
    target = protocol.get("target", {})
    _validate_exact_names(
        output=ACTIVATION_NAME,
        warmup=warmup.get("checkpoint", ""),
        warmup_probe=warmup.get("conditioning_probe", {}).get("path", ""),
        target=target.get("candidate_paths", [""])[-1],
        preflight=preflight["checkpoint"]["path"],
        preflight_probe=preflight["conditioning_probe"]["path"],
    )
    _require(protocol.get("stage_order") == ["all_equal_warmup", "independent_uniform_target"], "v2 stage order changed")
    _require(warmup.get("fresh_e0b_initialization") is True and warmup.get("resume") is False, "v2 warm-up initialization changed")
    _require(warmup.get("sampler") == canonical_all_equal_sampler(), "v2 warm-up sampler changed")
    _require(warmup.get("start_total_timesteps") == 0 and warmup.get("additional_timesteps") == WARMUP_TIMESTEPS and warmup.get("expected_total_timesteps") == WARMUP_TIMESTEPS, "v2 warm-up clock changed")
    _require(warmup.get("selectable") is False, "v2 warm-up became selectable")
    expected_probe = {
        "path": warmup["conditioning_probe"]["path"],
        "probe": v2_probe.PROBE_NAME,
        "gate": v2_probe.PROBE_GATE_NAME,
        "gate_math": "v1_numeric_sanity_plus_learned_base_response",
        "no_ale": True,
        "environment_steps": 0,
        "deterministic_statistics": [
            "learned base Beta mean", "final residual Beta mean"
        ],
        "require_all_outputs_finite_and_unit": True,
        "minimum_endpoint_response": MINIMUM_ENDPOINT_RESPONSE,
        "minimum_current_coordinate_response": (
            v2_probe.MINIMUM_CURRENT_COORDINATE_RESPONSE
        ),
        "minimum_learned_base_response": (
            v2_probe.MINIMUM_LEARNED_BASE_ENDPOINT_RESPONSE
        ),
        "maximum_adjacent_reversal": MAXIMUM_ADJACENT_REVERSAL,
        "require_pass": True,
    }
    _require(warmup.get("conditioning_probe") == expected_probe, "v2 probe contract changed")
    _require(target.get("resume_complete_model_optimizer_clock") is True, "v2 target does not resume full state")
    _same_path(target.get("resume_checkpoint"), warmup.get("checkpoint"), label="v2 target resume")
    _require(target.get("sampler") == canonical_uniform_sampler(), "v2 target sampler changed")
    _require(target.get("start_total_timesteps") == WARMUP_TIMESTEPS and target.get("additional_timesteps") == TARGET_ADDITIONAL_TIMESTEPS and target.get("expected_total_timesteps") == FINAL_TIMESTEPS, "v2 target clock changed")
    _require(target.get("candidate_timesteps") == list(TARGET_CANDIDATE_TIMESTEPS), "v2 target candidates changed")
    _require(target.get("candidate_paths") == expected_target_candidates(target["candidate_paths"][-1]), "v2 candidate paths changed")
    _require(target.get("candidate_count") == 6 and target.get("warmup_selectable") is False, "v2 target selection contract changed")
    _require(str(Path(warmup["checkpoint"]).resolve()) not in target["candidate_paths"], "v2 warm-up entered target family")
    _require(protocol.get("evaluation") == {
        "screen": {"episodes": 20, "seed_start": SCREEN_SEED_START},
        "confirmation": {"episodes": 100, "seed_start": CONFIRMATION_SEED_START, "policy": "screen_winner_only_no_fallback", "fallback_allowed": False},
        "fixed_grid": {"episodes_per_value": 20, "seed_start": FIXED_SEED_START, "values": [index / 10 for index in range(11)], "event_steps": [20, 50, 80, 110, 140]},
        "timing_seed_start_reserved": TIMING_SEED_START,
    }, "v2 evaluation contract changed")
    return value


def validate_activation(path, *, code_root=None):
    path = Path(path).expanduser().resolve()
    _require(path.name == ACTIVATION_NAME, "v2 activation filename changed")
    value = _validate_activation_contract(load_json(path))
    if code_root is not None:
        _require(_git_revision(code_root) == value["code_revision"], "runtime revision differs from v2 activation")
        _git_scoped_clean(code_root)
    return value


def write_or_validate_activation(args):
    expected = build_activation(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        existing = validate_activation(output)
        for key in expected:
            if key != "created_utc":
                _require(existing.get(key) == expected.get(key), f"existing v2 activation differs in {key}")
        return existing
    atomic_write_json(output, expected)
    return validate_activation(output)


def _optimizer_step(model):
    return v1._optimizer_step(model)


def _load_candidate_metadata(path, *, activation, device):
    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    model, metadata = evaluator.load_candidate(
        path,
        role=ROLE,
        e0b_sha256=activation["e0b_source"]["sha256"],
        device=device,
    )
    try:
        expected = canonical_economic_architecture()
        _require(
            model.policy.economic_architecture_provenance() == expected,
            "loaded v2 candidate policy architecture changed",
        )
        _require(
            getattr(model, "atari_e1_economic_architecture_provenance", None)
            == expected,
            "loaded v2 candidate model architecture provenance changed",
        )
        metadata = _canonical_json(metadata)
        metadata["optimizer_adam_step"] = _optimizer_step(model)
        metadata["resume_source"] = _canonical_json(getattr(
            model, "atari_e1_resume_source_provenance", None
        ))
    finally:
        del model
    return metadata


def _expected_history(activation, warmup_digest):
    return v1._expected_history(activation, warmup_digest)


def _validate_architecture_metadata(metadata, activation):
    expected = canonical_economic_architecture()
    _require(metadata.get("economic_architecture") == expected, "candidate architecture metadata changed")
    _require(
        metadata.get("e1_training_code_revision")
        == activation.get("code_revision"),
        "candidate training code revision differs from activation",
    )
    activation_learning_rate(activation)
    _require(metadata.get("training_config") == canonical_training_config(), "candidate PPO/architecture config changed")


def _validate_warmup_metadata(metadata, activation):
    _same_path(metadata.get("path"), activation["protocol"]["warmup"]["checkpoint"], label="v2 warm-up checkpoint")
    _require(metadata.get("training_timesteps") == WARMUP_TIMESTEPS, "v2 warm-up clock changed")
    _require(metadata.get("atari_e1_sampler_provenance") == canonical_all_equal_sampler(), "v2 warm-up sampler changed")
    _require(metadata.get("atari_e1_sampler_history") == [{
        "start_total_timesteps": 0,
        "sampler": canonical_all_equal_sampler(),
        "inferred_for_legacy_checkpoint": False,
        "resume_sources": [],
    }], "v2 warm-up lineage changed")
    _require(metadata.get("resume_source") is None, "v2 warm-up did not start fresh")
    _require(metadata.get("optimizer_adam_step") == 1_952, "v2 warm-up optimizer clock changed")
    _validate_architecture_metadata(metadata, activation)


def _validate_target_metadata(metadata, *, expected_path, expected_step, activation, warmup_digest, final):
    _same_path(metadata.get("path"), expected_path, label="v2 target candidate")
    _require(metadata.get("training_timesteps") == expected_step, "v2 target clock changed")
    _require(metadata.get("atari_e1_sampler_provenance") == canonical_uniform_sampler(), "v2 target sampler changed")
    _require(metadata.get("atari_e1_sampler_history") == _expected_history(activation, warmup_digest), "v2 target lineage changed")
    resume = metadata.get("resume_source")
    _require(isinstance(resume, dict), "v2 target lacks direct resume provenance")
    _same_path(resume.get("path"), activation["protocol"]["warmup"]["checkpoint"], label="v2 target resume path")
    _require(resume.get("sha256") == warmup_digest and resume.get("training_total_timesteps") == WARMUP_TIMESTEPS, "v2 target resume source changed")
    expected_optimizer = expected_step // 820 * 4 - (0 if final else 4)
    _require(metadata.get("optimizer_adam_step") == expected_optimizer, "v2 target optimizer did not continue")
    _validate_architecture_metadata(metadata, activation)


def validate_warmup_probe(*, activation, probe, checkpoint=None, require_pass=False):
    activation = validate_activation(activation) if not isinstance(activation, dict) else _validate_activation_contract(activation)
    contract = activation["protocol"]["warmup"]["conditioning_probe"]
    path = Path(probe).expanduser().resolve()
    _same_path(str(path), contract["path"], label="v2 warm-up probe")
    value = load_json(path)
    _require(value.get("probe") == v2_probe.PROBE_NAME, "unknown v2 probe")
    execution = value.get("execution", {})
    _require(execution == {
        "read_only_checkpoint": True,
        "ale_instantiated": False,
        "environment_steps": 0,
        "deterministic_statistics": [
            "learned base Beta mean", "final residual Beta mean"
        ],
        "device": "cpu",
    }, "v2 probe execution contract changed")
    checkpoint_path = Path(checkpoint or activation["protocol"]["warmup"]["checkpoint"]).expanduser().resolve()
    record = value.get("checkpoint", {})
    _same_path(record.get("path"), checkpoint_path, label="v2 probe checkpoint")
    digest = sha256_file(checkpoint_path)
    _require(record.get("sha256") == digest and record.get("training_timesteps") == WARMUP_TIMESTEPS, "v2 probe checkpoint changed")
    _require(record.get("role") == ROLE and record.get("atari_e1_sampler_provenance") == canonical_all_equal_sampler(), "v2 probe role/sampler changed")
    _require(record.get("atari_e1_sampler_history") == [{
        "start_total_timesteps": 0,
        "sampler": canonical_all_equal_sampler(),
        "inferred_for_legacy_checkpoint": False,
        "resume_sources": [],
    }], "v2 probe lineage changed")
    _validate_architecture_metadata(record, activation)
    _require(value.get("metadata_verification", {}).get("economic_architecture") == canonical_economic_architecture(), "v2 probe architecture verification changed")
    _require(value.get("protocol", {}).get("economic_architecture") == canonical_economic_architecture(), "v2 probe protocol architecture changed")
    all_equal_rows = value.get("all_equal_thresholds", [])
    _require([row.get("threshold") for row in all_equal_rows] == [0.0, 0.25, 0.5, 0.75, 1.0], "v2 probe grid changed")
    final_all_equal = [
        row.get("event_final_residual_beta_mean_prices")
        for row in all_equal_rows
    ]
    base_all_equal = [
        row.get("event_learned_base_beta_mean_prices")
        for row in all_equal_rows
    ]
    coordinate = value.get("current_coordinate_only_sensitivity", {}).get("rows", [])
    _require([row.get("event_index") for row in coordinate] == list(range(5)), "v2 coordinate probe changed")
    final_low = [
        row.get("low_final_residual_beta_mean_price") for row in coordinate
    ]
    final_high = [
        row.get("high_final_residual_beta_mean_price") for row in coordinate
    ]
    base_low = [
        row.get("low_learned_base_beta_mean_price") for row in coordinate
    ]
    base_high = [
        row.get("high_learned_base_beta_mean_price") for row in coordinate
    ]
    gate = v2_probe.threshold_residual_warmup_gate(
        final_all_equal=final_all_equal,
        final_coordinate_low=final_low,
        final_coordinate_high=final_high,
        base_all_equal=base_all_equal,
        base_coordinate_low=base_low,
        base_coordinate_high=base_high,
    )
    _require(value.get("warmup_gate") == gate, "v2 probe gate does not recompute")
    inferred = v2_probe.run_probe_from_checkpoints(
        checkpoint=checkpoint_path,
        e0b_checkpoint=activation["e0b_source"]["path"],
        device="cpu",
    )
    _compare_fresh_probe(value, inferred, label="v2 probe JSON")
    passed = gate.get("passed") is True
    if require_pass:
        _require(passed, "v2 threshold-residual conditioning probe did not pass")
    endpoint = gate["checks"]["minimum_all_one_minus_all_zero_beta_mean_price"]
    reversal = gate["checks"]["largest_adjacent_threshold_price_reversal"]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "probe": v2_probe.PROBE_NAME,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "passed": passed,
        "no_ale": True,
        "environment_steps": 0,
        "minimum_endpoint_response": endpoint.get("actual"),
        "largest_adjacent_reversal": reversal.get("actual"),
        "minimum_learned_base_endpoint_response": gate["checks"][
            "minimum_learned_base_all_one_minus_all_zero_mean_response"
        ].get("actual"),
        "minimum_learned_base_coordinate_response": gate["checks"][
            "minimum_learned_base_current_coordinate_mean_response"
        ].get("actual"),
        "economic_architecture": canonical_economic_architecture(),
    }


def validate_warmup_checkpoint(*, activation, checkpoint, device="cpu"):
    activation_value = validate_activation(activation)
    metadata = _load_candidate_metadata(checkpoint, activation=activation_value, device=device)
    _validate_warmup_metadata(metadata, activation_value)
    return {"passed": True, "checkpoint_metadata": metadata}


def _validate_evaluation(path, *, current, history, expected_step, checkpoint_metadata, random_episodes, fixed_episodes):
    record = v1._validate_evaluation(
        path,
        current=current,
        history=history,
        expected_step=expected_step,
        checkpoint_metadata=checkpoint_metadata,
        random_episodes=random_episodes,
        fixed_episodes=fixed_episodes,
    )
    value = load_json(path)
    _require(value.get("provenance", {}).get("economic_architecture") == canonical_economic_architecture(), "v2 evaluation architecture provenance changed")
    revision = checkpoint_metadata.get("e1_training_code_revision")
    _require(
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision),
        "v2 checkpoint metadata lacks a training code revision",
    )
    _require(
        value.get("provenance", {}).get("e1_training_code_revision")
        == revision,
        "v2 evaluation and checkpoint training revisions differ",
    )
    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
    readiness = evaluator.seller_threshold_residual_behavioral_gate(
        random_result=value["random"],
        fixed_results=value["fixed_context_evaluations"],
        mechanics_verified=True,
    )
    return {
        **record,
        "economic_architecture": canonical_economic_architecture(),
        "behavioral_readiness": readiness,
    }


def validate_preflight_evidence(
        *, checkpoint, probe, evaluation, e0b, expected_code_revision,
        device="cpu",
):
    """Bind a fresh residual-only real-ALE smoke before formal activation."""

    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    probe_path = Path(probe).expanduser().resolve()
    _require(
        isinstance(expected_code_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", expected_code_revision),
        "v2 preflight expected code revision is invalid",
    )
    _validate_exact_names(
        output=ACTIVATION_NAME,
        warmup=WARMUP_CHECKPOINT_NAME,
        warmup_probe=PROBE_ARTIFACT_NAME,
        target=TARGET_CHECKPOINT_NAME,
        preflight=checkpoint_path,
        preflight_probe=probe_path,
    )
    e0b_metadata = evaluator.validate_e0b(e0b, device=device)
    _require(
        e0b_metadata["sha256"] == CANONICAL_E0B_SHA256,
        "v2 preflight used another E0b",
    )
    model, metadata = evaluator.load_candidate(
        checkpoint_path,
        role=ROLE,
        e0b_sha256=CANONICAL_E0B_SHA256,
        device=device,
    )
    try:
        metadata = _canonical_json(metadata)
        optimizer_step = _optimizer_step(model)
    finally:
        del model
    expected_history = [{
        "start_total_timesteps": 0,
        "sampler": canonical_all_equal_sampler(),
        "inferred_for_legacy_checkpoint": False,
        "resume_sources": [],
    }]
    _require(
        metadata.get("training_timesteps") == PREFLIGHT_TIMESTEPS,
        "v2 preflight clock changed",
    )
    _require(
        metadata.get("training_config") == canonical_training_config(),
        "v2 preflight PPO/architecture configuration changed",
    )
    _require(
        metadata.get("economic_architecture")
        == canonical_economic_architecture(),
        "v2 preflight is not the pure-64 residual architecture",
    )
    _require(
        metadata.get("e1_training_code_revision")
        == expected_code_revision,
        "v2 preflight checkpoint was produced by another code revision",
    )
    _require(
        metadata.get("atari_e1_sampler_provenance")
        == canonical_all_equal_sampler()
        and metadata.get("atari_e1_sampler_history") == expected_history,
        "v2 preflight sampler lineage changed",
    )
    _require(optimizer_step == 100, "v2 preflight optimizer clock changed")

    reported_probe = load_json(probe_path)
    inferred_probe = v2_probe.run_probe_from_checkpoints(
        checkpoint=checkpoint_path,
        e0b_checkpoint=e0b,
        device=device,
    )
    _compare_fresh_probe(
        reported_probe, inferred_probe, label="v2 pure-64 preflight probe"
    )
    _require(
        reported_probe.get("warmup_gate", {}).get("passed") is True,
        "v2 pure-64 preflight failed learned-base conditioning",
    )
    _require(
        reported_probe.get("checkpoint", {}).get(
            "e1_training_code_revision"
        ) == expected_code_revision,
        "v2 pure-64 preflight probe names another training revision",
    )
    evaluation_record = _validate_evaluation(
        evaluation,
        current=canonical_all_equal_sampler(),
        history=expected_history,
        expected_step=PREFLIGHT_TIMESTEPS,
        checkpoint_metadata=metadata,
        random_episodes=20,
        fixed_episodes=20,
    )
    _require(
        evaluation_record["behavioral_readiness"]["passed"] is True,
        "v2 pure-64 preflight failed real-ALE behavioral readiness",
    )
    return {
        "passed": True,
        "formal_training_family": False,
        "wandb_enabled": False,
        "fresh_e0b_initialization": True,
        "code_revision": expected_code_revision,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": metadata["sha256"],
            "training_timesteps": PREFLIGHT_TIMESTEPS,
            "optimizer_adam_step": optimizer_step,
        },
        "conditioning_probe": {
            "path": str(probe_path),
            "sha256": sha256_file(probe_path),
            "gate": reported_probe["warmup_gate"],
        },
        "evaluation": evaluation_record,
    }


def validate_warmup_stage(*, activation, probe, checkpoint, training_log, evaluation, device="cpu"):
    activation_value = validate_activation(activation)
    metadata = validate_warmup_checkpoint(
        activation=activation, checkpoint=checkpoint, device=device
    )["checkpoint_metadata"]
    probe_record = validate_warmup_probe(
        activation=activation_value, probe=probe,
        checkpoint=checkpoint, require_pass=True,
    )
    trace = v1._validate_training_trace(
        training_log, mode=ALL_EQUAL_MODE, first_step=820,
        last_step=WARMUP_TIMESTEPS, expected_episodes=1_952,
        expected_optimizers=488, checkpoint=checkpoint,
    )
    evaluation_record = _validate_evaluation(
        evaluation,
        current=canonical_all_equal_sampler(),
        history=metadata["atari_e1_sampler_history"],
        expected_step=WARMUP_TIMESTEPS,
        checkpoint_metadata=metadata,
        random_episodes=20,
        fixed_episodes=20,
    )
    _require(
        evaluation_record["behavioral_readiness"]["passed"] is True,
        "v2 warm-up failed the real-ALE behavioral-readiness gate",
    )
    return {
        "passed": True,
        "checkpoint_metadata": metadata,
        "conditioning_probe": probe_record,
        "training_trace": trace,
        "evaluation": evaluation_record,
    }


def build_training_family(args):
    activation_path = Path(args.activation).expanduser().resolve()
    activation = validate_activation(activation_path, code_root=args.code_root)
    warmup = _load_candidate_metadata(args.warmup_checkpoint, activation=activation, device=args.device)
    _validate_warmup_metadata(warmup, activation)
    warmup_digest = warmup["sha256"]
    probe = validate_warmup_probe(
        activation=activation, probe=args.warmup_probe,
        checkpoint=args.warmup_checkpoint, require_pass=True,
    )
    candidates_paths = [str(Path(path).expanduser().resolve()) for path in args.candidate]
    _require(candidates_paths == activation["protocol"]["target"]["candidate_paths"], "v2 family differs from activation")
    candidates = []
    for index, (path, step) in enumerate(zip(candidates_paths, TARGET_CANDIDATE_TIMESTEPS)):
        metadata = _load_candidate_metadata(path, activation=activation, device=args.device)
        _validate_target_metadata(
            metadata, expected_path=path, expected_step=step,
            activation=activation, warmup_digest=warmup_digest,
            final=index == 5,
        )
        candidates.append(metadata)
    hashes = [item["sha256"] for item in candidates]
    _require(len(set(hashes)) == 6, "v2 candidates are not byte-distinct")
    warmup_trace = v1._validate_training_trace(
        args.warmup_training_log, mode=ALL_EQUAL_MODE,
        first_step=820, last_step=WARMUP_TIMESTEPS,
        expected_episodes=1_952, expected_optimizers=488,
        checkpoint=args.warmup_checkpoint,
    )
    target_trace = v1._validate_training_trace(
        args.target_training_log, mode=UNIFORM_MODE,
        first_step=WARMUP_TIMESTEPS + 820, last_step=FINAL_TIMESTEPS,
        expected_episodes=9_760, expected_optimizers=2_440,
        checkpoint=args.target_checkpoint,
    )
    warmup_eval = _validate_evaluation(
        args.warmup_evaluation,
        current=canonical_all_equal_sampler(),
        history=warmup["atari_e1_sampler_history"],
        expected_step=WARMUP_TIMESTEPS,
        checkpoint_metadata=warmup,
        random_episodes=20,
        fixed_episodes=20,
    )
    _require(
        warmup_eval["behavioral_readiness"]["passed"] is True,
        "v2 warm-up failed the real-ALE behavioral-readiness gate",
    )
    target_eval = _validate_evaluation(
        args.target_evaluation,
        current=canonical_uniform_sampler(),
        history=candidates[0]["atari_e1_sampler_history"],
        expected_step=FINAL_TIMESTEPS,
        checkpoint_metadata=candidates[-1],
        random_episodes=100,
        fixed_episodes=20,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": FAMILY_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "activation": {"path": str(activation_path), "sha256": sha256_file(activation_path)},
        "economic_architecture": canonical_economic_architecture(),
        "warmup": {"selectable": False, "metadata": warmup, "conditioning_probe": probe, "training_trace": warmup_trace, "evaluation": warmup_eval},
        "target": {"candidate_metadata": candidates, "candidate_sha256": hashes, "training_trace": target_trace, "evaluation": target_eval},
    }


def _validate_family_contract(value, activation):
    _require(value.get("schema_version") == SCHEMA_VERSION and value.get("kind") == FAMILY_KIND and value.get("passed") is True, "unknown/failed v2 family")
    _require(value.get("economic_architecture") == canonical_economic_architecture(), "v2 family architecture changed")
    warmup = value.get("warmup", {})
    _require(warmup.get("selectable") is False, "v2 warm-up became selectable")
    metadata = warmup.get("metadata", {})
    _validate_warmup_metadata(metadata, activation)
    _require(sha256_file(metadata["path"]) == metadata["sha256"], "v2 warm-up bytes changed")
    probe = validate_warmup_probe(
        activation=activation,
        probe=warmup.get("conditioning_probe", {}).get("path", ""),
        checkpoint=metadata["path"], require_pass=True,
    )
    _require(warmup.get("conditioning_probe") == probe, "v2 family probe changed")
    candidates = value.get("target", {}).get("candidate_metadata", [])
    hashes = value.get("target", {}).get("candidate_sha256", [])
    _require(len(candidates) == len(hashes) == 6 and len(set(hashes)) == 6, "v2 family is not six candidates")
    paths = activation["protocol"]["target"]["candidate_paths"]
    _require([item.get("path") for item in candidates] == paths, "v2 family paths changed")
    for index, (item, path, step, digest) in enumerate(zip(candidates, paths, TARGET_CANDIDATE_TIMESTEPS, hashes)):
        _require(sha256_file(path) == item.get("sha256") == digest, "v2 candidate bytes changed")
        _validate_target_metadata(item, expected_path=path, expected_step=step, activation=activation, warmup_digest=metadata["sha256"], final=index == 5)
    target = value["target"]
    expected_warmup_trace = v1._validate_training_trace(warmup["training_trace"]["path"], mode=ALL_EQUAL_MODE, first_step=820, last_step=WARMUP_TIMESTEPS, expected_episodes=1_952, expected_optimizers=488, checkpoint=metadata["path"])
    _require(warmup.get("training_trace") == expected_warmup_trace, "v2 warm-up trace changed")
    expected_target_trace = v1._validate_training_trace(target["training_trace"]["path"], mode=UNIFORM_MODE, first_step=WARMUP_TIMESTEPS + 820, last_step=FINAL_TIMESTEPS, expected_episodes=9_760, expected_optimizers=2_440, checkpoint=candidates[-1]["path"])
    _require(target.get("training_trace") == expected_target_trace, "v2 target trace changed")
    warmup_eval = _validate_evaluation(warmup["evaluation"]["path"], current=canonical_all_equal_sampler(), history=metadata["atari_e1_sampler_history"], expected_step=WARMUP_TIMESTEPS, checkpoint_metadata=metadata, random_episodes=20, fixed_episodes=20)
    _require(warmup_eval["behavioral_readiness"]["passed"] is True, "v2 warm-up behavioral readiness changed")
    _require(warmup.get("evaluation") == warmup_eval, "v2 warm-up evaluation changed")
    target_eval = _validate_evaluation(target["evaluation"]["path"], current=canonical_uniform_sampler(), history=candidates[0]["atari_e1_sampler_history"], expected_step=FINAL_TIMESTEPS, checkpoint_metadata=candidates[-1], random_episodes=100, fixed_episodes=20)
    _require(target.get("evaluation") == target_eval, "v2 target evaluation changed")
    return value


def validate_training_family(path):
    path = Path(path).expanduser().resolve()
    _require(path.name == FAMILY_NAME, "v2 family filename changed")
    value = load_json(path)
    activation_record = value.get("activation", {})
    _require(sha256_file(activation_record.get("path", "")) == activation_record.get("sha256"), "v2 family activation changed")
    return _validate_family_contract(value, validate_activation(activation_record["path"]))


def write_training_family(args):
    output = Path(args.output).expanduser().resolve()
    _require(output.name == FAMILY_NAME, "v2 family output name changed")
    _require(not os.path.lexists(output), "refusing to overwrite v2 family")
    atomic_write_json(output, build_training_family(args))
    return validate_training_family(output)


def canonical_selector_environment(activation):
    return v1.canonical_selector_environment(activation)


def validate_selection(args):
    family = validate_training_family(args.family)
    report_path = Path(args.report).expanduser().resolve()
    _require(report_path.name == REPORT_NAME, "v2 selector report name changed")
    report = load_json(report_path)
    _require(report.get("role") == ROLE and report.get("evaluator") == EVALUATOR, "v2 selector identity changed")
    protocol = report.get("protocol", {})
    for key, expected in {
        "screen_episodes": 20,
        "screen_seed_start": SCREEN_SEED_START,
        "confirmation_episodes": 100,
        "confirmation_seed_start": CONFIRMATION_SEED_START,
        "fixed_context_episodes": 20,
        "fixed_context_seed_start": FIXED_SEED_START,
        "confirmation_policy": "screen_winner_only_no_fallback",
    }.items():
        _require(protocol.get(key) == expected, f"v2 selector protocol changed: {key}")
    activation = validate_activation(family["activation"]["path"])
    immutable = report.get("immutable_evaluation", {})
    _require(immutable.get("selector_code_revision") == activation["code_revision"], "v2 selector revision changed")
    _require(immutable.get("e0b_sha256") == CANONICAL_E0B_SHA256 and immutable.get("rom_sha256") == CANONICAL_ROM_SHA256, "v2 selector E0b/ROM changed")
    _require(report.get("environment") == canonical_selector_environment(activation), "v2 selector environment changed")
    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    screen = report.get("screen", {})
    common = screen.get("common_pairing", {})
    seeds = list(range(SCREEN_SEED_START, SCREEN_SEED_START + 20))
    contexts = [evaluator.random_context(seed) for seed in seeds]
    results = screen.get("results", [])
    hashes = family["target"]["candidate_sha256"]
    _require(len(results) == 6 and common.get("passed") is True, "v2 screen is incomplete")
    _require([row.get("metadata", {}).get("sha256") for row in results] == hashes, "v2 selector screened another family")
    _require([row.get("metadata", {}).get("training_timesteps") for row in results] == list(TARGET_CANDIDATE_TIMESTEPS), "v2 selector clocks changed")
    _require(immutable.get("candidate_sha256") == hashes, "v2 immutable candidate family changed")
    _require(common == evaluator.validate_common_screen(results, seeds=seeds, contexts=contexts), "v2 common screen does not recompute")
    expected_metadata = family["target"]["candidate_metadata"]
    for index, (result, family_metadata) in enumerate(zip(results, expected_metadata)):
        expected = {key: value for key, value in family_metadata.items() if key not in {"optimizer_adam_step", "resume_source"}}
        _require(result.get("metadata") == expected, f"v2 screen metadata {index} changed")
        _require(result["metadata"].get("economic_architecture") == canonical_economic_architecture(), "v2 screen architecture changed")
        rows = result.get("episode_rows", [])
        violations = [violation for row in rows for violation in evaluator.audit_episode(row, role=ROLE)]
        _require(len(rows) == 20 and not violations, f"v2 screen mechanics failed: {violations[:3]}")
        _require(result.get("protocol") == {"passed": True, "violations": []}, "v2 screen protocol changed")
        _require(result.get("summary") == evaluator._summary(rows, role=ROLE), "v2 screen summary changed")
    report_family = report.get("training_family", {})
    first = expected_metadata[0]
    activation_learning_rate(activation)
    _require(report_family.get("common_training_config") == canonical_training_config(), "v2 report training config changed")
    _require(report_family.get("common_economic_architecture") == canonical_economic_architecture(), "v2 report architecture changed")
    _require(
        report_family.get("common_e1_training_code_revision")
        == activation["code_revision"],
        "v2 report training revision differs from activation",
    )
    _require(report_family.get("common_sampler_provenance") == first["atari_e1_sampler_provenance"] and report_family.get("common_sampler_history") == first["atari_e1_sampler_history"], "v2 report sampler lineage changed")
    ranking = report.get("ranking", [])
    _, recomputed_ranking = evaluator.rank_candidates(results)
    _require(ranking == recomputed_ranking and len(ranking) == 6, "v2 ranking does not recompute")
    winner = ranking[0]["checkpoint_sha256"]
    attempts = report.get("confirmation_attempts", [])
    _require(len(attempts) == 1 and attempts[0].get("metadata", {}).get("sha256") == winner, "v2 confirmed a nonwinner")
    _require(attempts[0]["metadata"].get("economic_architecture") == canonical_economic_architecture(), "v2 confirmation architecture changed")
    random = attempts[0].get("random", {})
    random_rows = random.get("episode_rows", [])
    confirmation_seeds = list(range(CONFIRMATION_SEED_START, CONFIRMATION_SEED_START + 100))
    _require([row.get("evaluation_seed") for row in random_rows] == confirmation_seeds, "v2 confirmation seeds changed")
    expected_contexts = {seed: tuple(float(value) for value in evaluator.random_context(seed)) for seed in confirmation_seeds}
    _require({int(row.get("evaluation_seed")): tuple(row.get("opponent_commitment", [])) for row in random_rows} == expected_contexts, "v2 confirmation contexts changed")
    violations = [violation for row in random_rows for violation in evaluator.audit_episode(row, role=ROLE)]
    _require(not violations and random.get("protocol") == {"passed": True, "violations": []}, "v2 confirmation mechanics failed")
    _require(random.get("summary") == evaluator._summary(random_rows, role=ROLE), "v2 confirmation summary changed")
    fixed = attempts[0].get("fixed_contexts", [])
    _require(len(fixed) == 11, "v2 fixed grid incomplete")
    for index, row in enumerate(fixed):
        value = index / 10
        rows = row.get("episode_rows", [])
        _require(row.get("opponent_value") == value and [episode.get("evaluation_seed") for episode in rows] == list(range(FIXED_SEED_START, FIXED_SEED_START + 20)), "v2 fixed grid design changed")
        _require(all(episode.get("event_steps") == [20, 50, 80, 110, 140] and all(abs(float(item) - value) <= 1e-6 for item in episode.get("opponent_commitment", [])) for episode in rows), "v2 fixed contexts changed")
        violations = [violation for episode in rows for violation in evaluator.audit_episode(episode, role=ROLE)]
        _require(not violations and row.get("protocol") == {"passed": True, "violations": []}, "v2 fixed mechanics failed")
        _require(row.get("summary") == evaluator._summary(rows, role=ROLE), "v2 fixed summary changed")
    winner_model, winner_metadata = evaluator.load_candidate(
        attempts[0]["metadata"]["path"],
        role=ROLE,
        e0b_sha256=CANONICAL_E0B_SHA256,
        device="cpu",
    )
    try:
        conditioning_probe = v2_probe.collect_conditioning_report(winner_model)
    finally:
        del winner_model
    _require(
        winner_metadata["sha256"] == winner,
        "v2 winner bytes changed while recomputing residual probe",
    )
    _require(
        attempts[0].get("conditioning_probe") == conditioning_probe,
        "v2 winner conditioning probe does not recompute",
    )
    gate = evaluator.seller_threshold_residual_final_gate(
        random_result=random,
        fixed_results=fixed,
        conditioning_probe=conditioning_probe,
    )
    _require(attempts[0].get("behavioral_gate") == gate, "v2 behavioral gate does not recompute")
    selection = report.get("selection", {})
    _require(selection.get("fallback_allowed") is False and selection.get("screen_selected_checkpoint_sha256") == winner, "v2 selection permits fallback")
    selected = Path(args.selected).expanduser().resolve()
    _require(selected.name == SELECTED_NAME, "v2 selected alias name changed")
    if report.get("passed") is True:
        _require(gate.get("passed") is True and selection.get("selected_checkpoint_sha256") == winner, "v2 passing report has failed gate")
        _require(sha256_file(selected) == winner, "v2 selected bytes changed")
        _same_path(report.get("selected_alias", {}).get("pinned_path"), selected, label="v2 selected alias")
    else:
        _require(report.get("passed") is False and gate.get("passed") is False, "v2 report outcome is invalid")
        _require(report.get("selected_alias") is None and not os.path.lexists(selected), "failed v2 selector retained alias")
    _same_path(report.get("artifacts", {}).get("json"), report_path, label="v2 report artifact")
    return report


def build_selection_gate(args, report):
    _require(report.get("passed") is True, "only passing v2 selection opens E2")
    family_path = Path(args.family).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    selected = Path(args.selected).expanduser().resolve()
    family = validate_training_family(family_path)
    activation_path = Path(family["activation"]["path"]).resolve()
    activation = validate_activation(activation_path)
    winner = report["selection"]["screen_selected_checkpoint_sha256"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": GATE_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "role": ROLE,
        "actor_loss_mode": ACTOR_LOSS_MODE,
        "source_kind": SOURCE_KIND,
        "sampler_mode": UNIFORM_MODE,
        "economic_architecture": canonical_economic_architecture(),
        "e1_training_code_revision": activation["code_revision"],
        "report": {"path": str(report_path), "sha256": sha256_file(report_path), "evaluator": EVALUATOR},
        "selected_checkpoint": {"path": str(selected), "sha256": winner},
        "training_family": {"path": str(family_path), "sha256": sha256_file(family_path), "candidate_sha256": family["target"]["candidate_sha256"]},
        "activation": {"path": str(activation_path), "sha256": sha256_file(activation_path), "code_revision": activation["code_revision"]},
        "prerequisite_v1_failure": dict(activation["prerequisite_v1_failure"]),
        "seller_release": dict(activation["seller_release"]),
        "warmup_probe": dict(family["warmup"]["conditioning_probe"]),
        "selected_conditioning_probe": dict(
            report["confirmation_attempts"][0]["conditioning_probe"]
        ),
        "selected_behavioral_gate": dict(
            report["confirmation_attempts"][0]["behavioral_gate"]
        ),
        "selection": {
            "screen_selected_checkpoint_sha256": winner,
            "confirmed_checkpoint_sha256": report["confirmation_attempts"][0]["metadata"]["sha256"],
            "selected_checkpoint_sha256": report["selection"]["selected_checkpoint_sha256"],
            "confirmation_attempts": 1,
            "confirmation_policy": "screen_winner_only_no_fallback",
            "fallback_allowed": False,
        },
    }


def validate_selection_gate(path, *, family=None, report=None, selected=None):
    path = Path(path).expanduser().resolve()
    _require(path.name == GATE_NAME, "v2 gate filename changed")
    value = load_json(path)
    _require(value.get("schema_version") == SCHEMA_VERSION and value.get("kind") == GATE_KIND and value.get("passed") is True, "unknown/failed v2 gate")
    _require(value.get("role") == ROLE and value.get("actor_loss_mode") == ACTOR_LOSS_MODE and value.get("source_kind") == SOURCE_KIND and value.get("sampler_mode") == UNIFORM_MODE, "v2 gate identity changed")
    _require(value.get("economic_architecture") == canonical_economic_architecture(), "v2 gate architecture changed")
    family_path = Path(value["training_family"]["path"]).resolve()
    report_path = Path(value["report"]["path"]).resolve()
    selected_path = Path(value["selected_checkpoint"]["path"]).resolve()
    if family is not None: _same_path(str(family_path), family, label="v2 gate family")
    if report is not None: _same_path(str(report_path), report, label="v2 gate report")
    if selected is not None: _same_path(str(selected_path), selected, label="v2 gate selected")
    _require(sha256_file(family_path) == value["training_family"]["sha256"], "v2 gate family changed")
    _require(sha256_file(report_path) == value["report"]["sha256"], "v2 gate report changed")
    _require(sha256_file(selected_path) == value["selected_checkpoint"]["sha256"], "v2 gate selected changed")
    family_value = validate_training_family(family_path)
    _require(value["training_family"]["candidate_sha256"] == family_value["target"]["candidate_sha256"], "v2 gate candidates changed")
    activation = validate_activation(value["activation"]["path"])
    _require(sha256_file(value["activation"]["path"]) == value["activation"]["sha256"] and activation["code_revision"] == value["activation"]["code_revision"], "v2 gate activation changed")
    _require(value.get("prerequisite_v1_failure") == activation["prerequisite_v1_failure"], "v2 gate prerequisite changed")
    _require(value.get("seller_release") == activation["seller_release"], "v2 gate seller release changed")
    _require(value.get("warmup_probe") == family_value["warmup"]["conditioning_probe"], "v2 gate probe changed")
    report_value = validate_selection(SimpleNamespace(family=str(family_path), report=str(report_path), selected=str(selected_path)))
    expected = build_selection_gate(SimpleNamespace(family=str(family_path), report=str(report_path), selected=str(selected_path)), report_value)
    for key, expected_value in expected.items():
        if key != "created_utc":
            _require(value.get(key) == expected_value, f"v2 gate differs in {key}")
    return value


def write_or_validate_selection_gate(args, report):
    if report.get("passed") is False:
        if args.gate_output:
            _require(not os.path.lexists(args.gate_output), "failed v2 selector retained gate")
        return None
    _require(args.gate_output, "passing v2 selection requires gate output")
    output = Path(args.gate_output).expanduser().resolve()
    expected = build_selection_gate(args, report)
    if output.exists():
        existing = validate_selection_gate(output, family=args.family, report=args.report, selected=args.selected)
        for key, expected_value in expected.items():
            if key != "created_utc":
                _require(existing.get(key) == expected_value, f"existing v2 gate differs in {key}")
        return existing
    atomic_write_json(output, expected)
    return validate_selection_gate(output, family=args.family, report=args.report, selected=args.selected)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    activation = commands.add_parser("activation")
    for name in (
        "v1-activation", "v1-warmup-checkpoint", "v1-warmup-probe",
        "v1-warmup-training-log", "v1-warmup-evaluation",
        "seller-release", "e0b", "rom", "warmup-checkpoint",
        "warmup-probe", "target-checkpoint", "preflight-checkpoint",
        "preflight-probe", "preflight-evaluation", "code-root", "output",
    ):
        activation.add_argument(f"--{name}", required=True)
    validate = commands.add_parser("validate-activation")
    validate.add_argument("--activation", required=True)
    validate.add_argument("--code-root")
    failure = commands.add_parser("validate-v1-failure")
    for name in ("activation", "warmup-checkpoint", "probe", "training-log", "evaluation"):
        failure.add_argument(f"--{name}", required=True)
    failure.add_argument("--device", default="cpu")
    family = commands.add_parser("training-family")
    for name in ("activation", "code-root", "warmup-checkpoint", "warmup-probe", "target-checkpoint", "warmup-training-log", "target-training-log", "warmup-evaluation", "target-evaluation", "output"):
        family.add_argument(f"--{name}", required=True)
    family.add_argument("--candidate", action="append", required=True)
    family.add_argument("--device", default="cpu")
    validate_family = commands.add_parser("validate-training-family")
    validate_family.add_argument("--family", required=True)
    probe = commands.add_parser("validate-warmup-probe")
    probe.add_argument("--activation", required=True)
    probe.add_argument("--probe", required=True)
    probe.add_argument("--checkpoint")
    checkpoint = commands.add_parser("validate-warmup-checkpoint")
    checkpoint.add_argument("--activation", required=True)
    checkpoint.add_argument("--checkpoint", required=True)
    checkpoint.add_argument("--device", default="cpu")
    stage = commands.add_parser("validate-warmup-stage")
    for name in ("activation", "probe", "checkpoint", "training-log", "evaluation"):
        stage.add_argument(f"--{name}", required=True)
    stage.add_argument("--device", default="cpu")
    selection = commands.add_parser("selection")
    for name in ("family", "report", "selected"):
        selection.add_argument(f"--{name}", required=True)
    selection.add_argument("--gate-output")
    gate = commands.add_parser("validate-selection-gate")
    gate.add_argument("--gate", required=True)
    gate.add_argument("--family")
    gate.add_argument("--report")
    gate.add_argument("--selected")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "activation":
        result = write_or_validate_activation(args)
    elif args.command == "validate-activation":
        result = validate_activation(args.activation, code_root=args.code_root)
    elif args.command == "validate-v1-failure":
        result = validate_v1_warmup_failure(
            activation=args.activation,
            warmup_checkpoint=args.warmup_checkpoint,
            probe=args.probe,
            training_log=args.training_log,
            evaluation=args.evaluation,
            device=args.device,
        )
    elif args.command == "training-family":
        result = write_training_family(args)
    elif args.command == "validate-training-family":
        result = validate_training_family(args.family)
    elif args.command == "validate-warmup-probe":
        result = validate_warmup_probe(
            activation=args.activation, probe=args.probe,
            checkpoint=args.checkpoint, require_pass=False,
        )
    elif args.command == "validate-warmup-checkpoint":
        result = validate_warmup_checkpoint(
            activation=args.activation, checkpoint=args.checkpoint,
            device=args.device,
        )
    elif args.command == "validate-warmup-stage":
        result = validate_warmup_stage(
            activation=args.activation, probe=args.probe,
            checkpoint=args.checkpoint, training_log=args.training_log,
            evaluation=args.evaluation, device=args.device,
        )
    elif args.command == "selection":
        report = validate_selection(args)
        result = write_or_validate_selection_gate(args, report) if args.gate_output else report
        if result is None:
            result = report
    else:
        result = validate_selection_gate(
            args.gate, family=args.family, report=args.report,
            selected=args.selected,
        )
    print(json.dumps({
        "command": args.command,
        "passed": bool(result.get("passed", True)),
        "kind": result.get("kind"),
        "value": result,
    }, sort_keys=True), flush=True)
    if args.command == "validate-warmup-probe" and result.get("passed") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
