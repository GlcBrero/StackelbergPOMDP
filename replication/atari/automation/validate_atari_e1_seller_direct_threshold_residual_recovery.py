#!/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python
"""Versioned v3 profile for the shared residual-seller validator."""

from contextlib import contextmanager
from pathlib import Path
import subprocess

import numpy as np

from replication.atari import (
    probe_atari_e1_seller_direct_threshold_residual as v3_probe,
)
from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.automation import (
    validate_atari_e1_seller_threshold_residual_recovery as shared,
)


SCHEMA_VERSION = 1
ACTIVATION_KIND = (
    "stackpomdp.atari.e1_seller_conditioning_recovery_activation.v3"
)
FAMILY_KIND = "stackpomdp.atari.e1_seller_conditioning_recovery_family.v3"
GATE_KIND = "stackpomdp.atari.e1_seller_conditioning_recovery_gate.v3"
SOURCE_KIND = "seller_conditioning_recovery_v3_direct_threshold_residual_v1"
ACTIVATION_NAME = (
    "e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_"
    "activation.json"
)
FAMILY_NAME = (
    "e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_"
    "family.json"
)
REPORT_NAME = (
    "e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_"
    "all6_selector_v3.json"
)
GATE_NAME = f"{Path(REPORT_NAME).stem}.gate.json"
SELECTED_NAME = (
    "meta_seller_e1_ppo_balanced_conditioning_recovery_v3_"
    "direct_threshold_residual_v1_seed1_selected.zip"
)
WARMUP_CHECKPOINT_NAME = (
    "meta_seller_e1_ppo_balanced_conditioning_recovery_v3_"
    "direct_threshold_residual_v1_seed1_all_equal_warmup.zip"
)
TARGET_CHECKPOINT_NAME = (
    "meta_seller_e1_ppo_balanced_conditioning_recovery_v3_"
    "direct_threshold_residual_v1_seed1_uniform_target.zip"
)
PROBE_ARTIFACT_NAME = (
    "e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_"
    "warmup_probe.json"
)
PREFLIGHT_CHECKPOINT_NAME = (
    "meta_seller_e1_ppo_balanced_conditioning_recovery_v3_"
    "direct_threshold_residual_v1_seed1_direct65_preflight.zip"
)
PREFLIGHT_PROBE_NAME = (
    "e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_"
    "direct65_preflight_probe.json"
)
PREFLIGHT_RECORD_KEY = "direct65_preflight"


def canonical_economic_architecture():
    return v3_probe.canonical_economic_architecture()


def canonical_initialization():
    return trainer.direct_threshold_initialization_contract(
        state_features=64, economic_hidden=64
    )


def canonical_training_config():
    return {
        **shared.v1.canonical_training_config(),
        "learning_rate": shared.LEARNING_RATE,
        "economic_threshold_residual": True,
        "economic_threshold_residual_direct_input": True,
    }


def controlled_change_contract():
    return {
        "only_architectural_change_from_v2": (
            "append the current event threshold to the existing 64D seller "
            "economic base-head input; initialize only that new column to "
            "exact zero and retain the fixed equal-weight residual anchor"
        ),
        "economic_architecture": canonical_economic_architecture(),
        "direct_threshold_initialization": canonical_initialization(),
        "new_observation_fields": [],
        "environment_unchanged": True,
        "critic_input_unchanged": True,
        "samplers_unchanged": True,
        "ppo_hyperparameters_unchanged": True,
        "learning_rate": shared.LEARNING_RATE,
        "fixed_anchor_is_inductive_bias": True,
        "learned_base_threshold_slope_required": True,
    }


def probe_execution_statistics():
    return [
        "learned base Beta mean",
        "final residual Beta mean",
        "noncurrent-coordinate learned base leakage",
    ]


def conditioning_probe_contract(path):
    return {
        "path": str(path),
        "probe": v3_probe.PROBE_NAME,
        "gate": v3_probe.PROBE_GATE_NAME,
        "gate_math": (
            "residual_numeric_sanity_plus_learned_direct_response_and_"
            "current_vs_noncurrent_specificity"
        ),
        "no_ale": True,
        "environment_steps": 0,
        "deterministic_statistics": probe_execution_statistics(),
        "require_all_outputs_finite_and_unit": True,
        "minimum_endpoint_response": shared.MINIMUM_ENDPOINT_RESPONSE,
        "minimum_current_coordinate_response": (
            v3_probe.MINIMUM_CURRENT_COORDINATE_RESPONSE
        ),
        "minimum_learned_base_response": (
            v3_probe.MINIMUM_LEARNED_BASE_ENDPOINT_RESPONSE
        ),
        "minimum_learned_base_current_response": (
            v3_probe.MINIMUM_LEARNED_BASE_CURRENT_RESPONSE
        ),
        "minimum_current_vs_noncurrent_specificity_margin": (
            v3_probe.MINIMUM_CURRENT_VS_NONCURRENT_SPECIFICITY_MARGIN
        ),
        "maximum_adjacent_reversal": shared.MAXIMUM_ADJACENT_REVERSAL,
        "require_pass": True,
    }


def probe_gate_from_value(value):
    all_equal = value.get("all_equal_thresholds", [])
    coordinate = value.get(
        "current_coordinate_only_sensitivity", {}
    ).get("rows", [])
    leakage = value.get(
        "noncurrent_coordinate_leakage_control", {}
    ).get("rows", [])
    shared._require(
        [row.get("event_index") for row in leakage] == list(range(5)),
        "v3 noncurrent-coordinate control changed",
    )
    return v3_probe.direct_threshold_residual_warmup_gate(
        final_all_equal=[
            row.get("event_final_residual_beta_mean_prices")
            for row in all_equal
        ],
        final_coordinate_low=[
            row.get("low_final_residual_beta_mean_price")
            for row in coordinate
        ],
        final_coordinate_high=[
            row.get("high_final_residual_beta_mean_price")
            for row in coordinate
        ],
        base_all_equal=[
            row.get("event_learned_base_beta_mean_prices")
            for row in all_equal
        ],
        base_coordinate_low=[
            row.get("low_learned_base_beta_mean_price")
            for row in coordinate
        ],
        base_coordinate_high=[
            row.get("high_learned_base_beta_mean_price")
            for row in coordinate
        ],
        base_noncurrent_low=[
            row.get("low_learned_base_beta_mean_price") for row in leakage
        ],
        base_noncurrent_high=[
            row.get("high_learned_base_beta_mean_price") for row in leakage
        ],
    )


def probe_record_extras(gate):
    return {
        "minimum_learned_base_endpoint_response": gate["checks"][
            "minimum_learned_base_all_one_minus_all_zero_mean_response"
        ].get("actual"),
        "minimum_learned_base_coordinate_response": gate["checks"][
            "minimum_learned_base_current_coordinate_mean_response"
        ].get("actual"),
        "current_vs_noncurrent_specificity": gate[
            "current_vs_noncurrent_specificity"
        ],
    }


def validate_additional_architecture_metadata(metadata, activation):
    del activation
    shared._require(
        metadata.get("direct_threshold_initialization")
        == canonical_initialization(),
        "v3 candidate zero-column initialization provenance changed",
    )


def validate_additional_evaluation_provenance(value, checkpoint_metadata):
    shared._require(
        value.get("provenance", {}).get(
            "direct_threshold_initialization"
        ) == checkpoint_metadata.get("direct_threshold_initialization"),
        "v3 evaluation initialization provenance changed",
    )


def selection_gate_additional_fields():
    return {
        "direct_threshold_initialization": canonical_initialization(),
    }


_shared_git_scoped_clean = shared._git_scoped_clean


def _git_scoped_clean(code_root):
    _shared_git_scoped_clean(code_root)
    root = Path(code_root).expanduser().resolve()
    dirty = subprocess.run(
        [
            "git", "-C", str(root), "status", "--porcelain", "--",
            "replication/atari/"
            "probe_atari_e1_seller_direct_threshold_residual.py",
            "replication/atari/automation/"
            "validate_atari_e1_seller_direct_threshold_residual_recovery.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    shared._require(
        not dirty,
        f"v3 recovery activation requires clean v3 profile/probe: {dirty}",
    )


@contextmanager
def _profile():
    replacements = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "ACTIVATION_KIND": ACTIVATION_KIND,
        "FAMILY_KIND": FAMILY_KIND,
        "GATE_KIND": GATE_KIND,
        "SOURCE_KIND": SOURCE_KIND,
        "ACTIVATION_NAME": ACTIVATION_NAME,
        "FAMILY_NAME": FAMILY_NAME,
        "REPORT_NAME": REPORT_NAME,
        "GATE_NAME": GATE_NAME,
        "SELECTED_NAME": SELECTED_NAME,
        "WARMUP_CHECKPOINT_NAME": WARMUP_CHECKPOINT_NAME,
        "TARGET_CHECKPOINT_NAME": TARGET_CHECKPOINT_NAME,
        "PROBE_ARTIFACT_NAME": PROBE_ARTIFACT_NAME,
        "PREFLIGHT_CHECKPOINT_NAME": PREFLIGHT_CHECKPOINT_NAME,
        "PREFLIGHT_PROBE_NAME": PREFLIGHT_PROBE_NAME,
        "PREFLIGHT_RECORD_KEY": PREFLIGHT_RECORD_KEY,
        "v2_probe": v3_probe,
        "canonical_economic_architecture": canonical_economic_architecture,
        "canonical_training_config": canonical_training_config,
        "controlled_change_contract": controlled_change_contract,
        "probe_execution_statistics": probe_execution_statistics,
        "conditioning_probe_contract": conditioning_probe_contract,
        "probe_gate_from_value": probe_gate_from_value,
        "probe_record_extras": probe_record_extras,
        "validate_additional_architecture_metadata": (
            validate_additional_architecture_metadata
        ),
        "validate_additional_evaluation_provenance": (
            validate_additional_evaluation_provenance
        ),
        "selection_gate_additional_fields": (
            selection_gate_additional_fields
        ),
        "_git_scoped_clean": _git_scoped_clean,
    }
    original = {name: getattr(shared, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(shared, name, value)


def _call(name, *args, **kwargs):
    with _profile():
        return getattr(shared, name)(*args, **kwargs)


def validate_activation(*args, **kwargs):
    return _call("validate_activation", *args, **kwargs)


def validate_training_family(*args, **kwargs):
    return _call("validate_training_family", *args, **kwargs)


def validate_selection_gate(*args, **kwargs):
    return _call("validate_selection_gate", *args, **kwargs)


def write_or_validate_selection_gate(*args, **kwargs):
    return _call("write_or_validate_selection_gate", *args, **kwargs)


def main(argv=None):
    with _profile():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
