"""No-ALE probe for the v3 direct-threshold residual E1 seller."""

import argparse
from datetime import datetime, timezone
import re

import numpy as np

from replication.atari import probe_atari_e1_seller_conditioning as base_probe
from replication.atari import probe_atari_e1_seller_threshold_residual as v2_probe
from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.sb3_common import write_json
from stackelberg_pomdp.atari.stackpomdp_policy import (
    direct_threshold_residual_architecture_provenance,
)


PROBE_NAME = "clean_atari_e1_seller_direct_threshold_residual_probe_v3"
PROBE_GATE_NAME = (
    "clean_e1_seller_direct_threshold_residual_warmup_gate_v3"
)
ARCHITECTURE_SCHEMA = "stackpomdp.atari.economic_actor_architecture.v3"
PARAMETERIZATION = "seller_direct_threshold_residual_beta_v3"
MINIMUM_LEARNED_BASE_ENDPOINT_RESPONSE = 0.01
MINIMUM_CURRENT_COORDINATE_RESPONSE = 0.25
MINIMUM_LEARNED_BASE_CURRENT_RESPONSE = 0.01
MINIMUM_CURRENT_VS_NONCURRENT_SPECIFICITY_MARGIN = 0.005


def canonical_economic_architecture():
    return direct_threshold_residual_architecture_provenance(
        state_features=64
    )


def canonical_initialization_provenance(policy):
    return trainer.direct_threshold_initialization_provenance(policy)


def validate_loaded_seller(model, metadata, e0b_metadata):
    """Require exact v3 architecture, zero-column lineage, and code binding."""

    base = base_probe.validate_loaded_seller(model, metadata, e0b_metadata)
    policy = model.policy
    expected_architecture = canonical_economic_architecture()
    expected_initialization = canonical_initialization_provenance(policy)
    recorded_architecture = getattr(
        model, trainer.ECONOMIC_ARCHITECTURE_ATTRIBUTE, None
    )
    recorded_initialization = getattr(
        model, trainer.DIRECT_THRESHOLD_INITIALIZATION_ATTRIBUTE, None
    )
    training_code_revision = getattr(
        model, trainer.E1_TRAINING_CODE_REVISION_ATTRIBUTE, None
    )
    checks = dict(base["checks"])
    checks.update({
        "threshold_residual_enabled": bool(getattr(
            policy, "economic_threshold_residual", False
        )),
        "direct_threshold_input_enabled": bool(getattr(
            policy, "economic_threshold_residual_direct_input", False
        )),
        "direct_65_input_base_head": (
            int(policy.economic_head[0].in_features) == 65
        ),
        "policy_economic_architecture_exact": (
            policy.economic_architecture_provenance()
            == expected_architecture
        ),
        "model_economic_architecture_provenance_exact": (
            recorded_architecture == expected_architecture
        ),
        "metadata_economic_architecture_exact": (
            metadata.get("economic_architecture") == expected_architecture
        ),
        "policy_initialization_provenance_exact": (
            expected_initialization is not None
        ),
        "model_initialization_provenance_exact": (
            recorded_initialization == expected_initialization
        ),
        "metadata_initialization_provenance_exact": (
            metadata.get("direct_threshold_initialization")
            == expected_initialization
        ),
        "initialization_records_exact_zero_new_column": bool(
            expected_initialization
            and expected_initialization.get(
                "new_direct_column_initialized_exact_zero"
            ) is True
            and expected_initialization.get("new_direct_column_index") == 64
            and expected_initialization.get("first_linear_shape") == [64, 65]
        ),
        "training_config_enables_threshold_residual": (
            metadata.get("training_config", {}).get(
                "economic_threshold_residual"
            ) is True
        ),
        "training_config_enables_direct_input": (
            metadata.get("training_config", {}).get(
                "economic_threshold_residual_direct_input"
            ) is True
        ),
        "training_code_revision_is_full_sha": (
            isinstance(training_code_revision, str)
            and re.fullmatch(r"[0-9a-f]{40}", training_code_revision)
            is not None
        ),
        "metadata_training_code_revision_exact": (
            metadata.get("e1_training_code_revision")
            == training_code_revision
        ),
    })
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise ValueError(
            "seller checkpoint failed direct-threshold residual probe checks: "
            + ", ".join(failed)
        )
    return {
        "passed": True,
        "checks": {name: bool(value) for name, value in checks.items()},
        "economic_architecture": expected_architecture,
        "direct_threshold_initialization": expected_initialization,
        "e1_training_code_revision": training_code_revision,
    }


def direct_threshold_residual_warmup_gate(
        *, final_all_equal, final_coordinate_low, final_coordinate_high,
        base_all_equal, base_coordinate_low, base_coordinate_high,
        base_noncurrent_low, base_noncurrent_high,
):
    """Gate learned direct response and current-coordinate specificity."""

    gate = v2_probe.threshold_residual_warmup_gate(
        final_all_equal=final_all_equal,
        final_coordinate_low=final_coordinate_low,
        final_coordinate_high=final_coordinate_high,
        base_all_equal=base_all_equal,
        base_coordinate_low=base_coordinate_low,
        base_coordinate_high=base_coordinate_high,
    )
    base_current = (
        np.asarray(base_coordinate_high, dtype=np.float64)
        - np.asarray(base_coordinate_low, dtype=np.float64)
    )
    base_noncurrent = (
        np.asarray(base_noncurrent_high, dtype=np.float64)
        - np.asarray(base_noncurrent_low, dtype=np.float64)
    )
    current_minimum = float(np.min(base_current))
    leakage_maximum = float(np.max(np.abs(base_noncurrent)))
    specificity_margin = current_minimum - leakage_maximum
    finite = bool(
        np.all(np.isfinite(base_current))
        and np.all(np.isfinite(base_noncurrent))
    )
    specificity_passed = bool(
        finite
        and current_minimum >= MINIMUM_LEARNED_BASE_CURRENT_RESPONSE
        and specificity_margin
        >= MINIMUM_CURRENT_VS_NONCURRENT_SPECIFICITY_MARGIN
    )
    checks = dict(gate["checks"])
    checks["learned_base_current_vs_noncurrent_specificity_margin"] = {
        "actual": specificity_margin,
        "operator": ">=",
        "required": MINIMUM_CURRENT_VS_NONCURRENT_SPECIFICITY_MARGIN,
        "passed": specificity_passed,
        "minimum_current_response": current_minimum,
        "maximum_absolute_noncurrent_response": leakage_maximum,
    }
    return {
        **gate,
        "name": PROBE_GATE_NAME,
        "passed": bool(all(item["passed"] for item in checks.values())),
        "checks": checks,
        "architecture_parameterization": PARAMETERIZATION,
        "fixed_anchor_weight": 0.5,
        "fixed_anchor_is_inductive_bias": True,
        "learned_threshold_slope_claim": True,
        "per_event_learned_base_noncurrent_coordinate_response": (
            base_noncurrent.tolist()
        ),
        "current_vs_noncurrent_specificity": {
            "minimum_current_response": current_minimum,
            "maximum_absolute_noncurrent_response": leakage_maximum,
            "margin": specificity_margin,
        },
    }


def _arrays_from_v2_report(report):
    all_equal = report["all_equal_thresholds"]
    base_all_equal = np.asarray([
        row["event_learned_base_beta_mean_prices"] for row in all_equal
    ], dtype=np.float64)
    final_all_equal = np.asarray([
        row["event_final_residual_beta_mean_prices"] for row in all_equal
    ], dtype=np.float64)
    rows = report["current_coordinate_only_sensitivity"]["rows"]
    base_low = np.asarray([
        row["low_learned_base_beta_mean_price"] for row in rows
    ], dtype=np.float64)
    base_high = np.asarray([
        row["high_learned_base_beta_mean_price"] for row in rows
    ], dtype=np.float64)
    final_low = np.asarray([
        row["low_final_residual_beta_mean_price"] for row in rows
    ], dtype=np.float64)
    final_high = np.asarray([
        row["high_final_residual_beta_mean_price"] for row in rows
    ], dtype=np.float64)
    return (
        base_all_equal, final_all_equal, base_low, base_high,
        final_low, final_high,
    )


def collect_conditioning_report(model):
    """Measure endpoints, current response, and cyclic noncurrent leakage."""

    report = v2_probe.collect_conditioning_report(model)
    (
        base_all_equal, final_all_equal, base_low, base_high,
        final_low, final_high,
    ) = _arrays_from_v2_report(report)
    event_count = base_probe.NUM_TRADE_EVENTS
    noncurrent_low = np.empty(event_count, dtype=np.float64)
    noncurrent_high = np.empty(event_count, dtype=np.float64)
    leakage_rows = []
    for event_index in range(event_count):
        noncurrent_index = (event_index + 1) % event_count
        low_context = np.full(
            event_count,
            base_probe.COORDINATE_BASELINE_THRESHOLD,
            dtype=np.float32,
        )
        high_context = np.array(low_context, copy=True)
        low_context[noncurrent_index] = 0.0
        high_context[noncurrent_index] = 1.0
        low_values = base_probe.canonical_trade_observation(
            model.policy, event_index=event_index, thresholds=low_context
        )
        high_values = base_probe.canonical_trade_observation(
            model.policy, event_index=event_index, thresholds=high_context
        )
        noncurrent_low[event_index], _ = v2_probe.economic_beta_means(
            model, low_values
        )
        noncurrent_high[event_index], _ = v2_probe.economic_beta_means(
            model, high_values
        )
        leakage_rows.append({
            "event_index": event_index,
            "varied_noncurrent_index": noncurrent_index,
            "low_context": low_context.tolist(),
            "high_context": high_context.tolist(),
            "low_learned_base_beta_mean_price": noncurrent_low[event_index],
            "high_learned_base_beta_mean_price": noncurrent_high[event_index],
            "learned_base_sensitivity": (
                noncurrent_high[event_index] - noncurrent_low[event_index]
            ),
        })

    report["protocol"]["economic_architecture"] = (
        canonical_economic_architecture()
    )
    report["protocol"]["direct_threshold_initialization"] = (
        canonical_initialization_provenance(model.policy)
    )
    leakage = noncurrent_high - noncurrent_low
    report["noncurrent_coordinate_leakage_control"] = {
        "definition": (
            "cyclic next-coordinate threshold 1 minus 0 while the current "
            "threshold and all remaining thresholds stay at 0.5"
        ),
        "rows": leakage_rows,
        "learned_base": base_probe._finite_summary(leakage),
        "maximum_absolute_response": float(np.max(np.abs(leakage))),
    }
    report["warmup_gate"] = direct_threshold_residual_warmup_gate(
        final_all_equal=final_all_equal,
        final_coordinate_low=final_low,
        final_coordinate_high=final_high,
        base_all_equal=base_all_equal,
        base_coordinate_low=base_low,
        base_coordinate_high=base_high,
        base_noncurrent_low=noncurrent_low,
        base_noncurrent_high=noncurrent_high,
    )
    return report


def run_probe_from_checkpoints(*, checkpoint, e0b_checkpoint, device="cpu"):
    """Load, revalidate, and probe immutable bytes without constructing ALE."""

    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    candidate_path = evaluator.checkpoint_path(checkpoint)
    before = evaluator.checkpoint_sha256(candidate_path)
    e0b = evaluator.validate_e0b(e0b_checkpoint, device=device)
    model, metadata = evaluator.load_candidate(
        candidate_path,
        role=base_probe.SELLER_ROLE,
        e0b_sha256=e0b["sha256"],
        device=device,
    )
    try:
        verification = validate_loaded_seller(model, metadata, e0b)
        measurements = collect_conditioning_report(model)
    finally:
        del model
    after = evaluator.checkpoint_sha256(candidate_path)
    if after != before or metadata["sha256"] != before:
        raise RuntimeError("seller checkpoint bytes changed during the probe")
    return {
        "probe": PROBE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "read_only_checkpoint": True,
            "ale_instantiated": False,
            "environment_steps": 0,
            "deterministic_statistics": [
                "learned base Beta mean",
                "final residual Beta mean",
                "noncurrent-coordinate learned base leakage",
            ],
            "device": str(device),
        },
        "checkpoint": metadata,
        "e0b_source": e0b,
        "metadata_verification": verification,
        **measurements,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--e0b-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_probe_from_checkpoints(
        checkpoint=args.checkpoint,
        e0b_checkpoint=args.e0b_checkpoint,
        device=args.device,
    )
    if args.output:
        write_json(args.output, result)
    print(result)
    return 0 if result["warmup_gate"]["passed"] or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
