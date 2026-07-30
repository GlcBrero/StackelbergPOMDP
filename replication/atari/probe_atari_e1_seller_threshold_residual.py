"""No-ALE probe for the pure-64 E1 seller threshold-residual policy."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import torch as th
from torch.nn import functional as functional

from replication.atari import probe_atari_e1_seller_conditioning as base_probe
from replication.atari import train_atari_meta_response_sb3 as trainer
from replication.atari.sb3_common import write_json
from stackelberg_pomdp.atari.stackpomdp_policy import (
    BETA_PARAMETER_EPSILON,
    threshold_residual_architecture_provenance,
)


PROBE_NAME = "clean_atari_e1_seller_threshold_residual_probe_v1"
PROBE_GATE_NAME = "clean_e1_seller_threshold_residual_warmup_gate_v1"
ARCHITECTURE_SCHEMA = "stackpomdp.atari.economic_actor_architecture.v2"
MINIMUM_LEARNED_BASE_ENDPOINT_RESPONSE = 0.01
MINIMUM_CURRENT_COORDINATE_RESPONSE = 0.25


def canonical_economic_architecture():
    """Return the exact seller-only pure-64 architectural change."""

    return threshold_residual_architecture_provenance(state_features=64)


def validate_loaded_seller(model, metadata, e0b_metadata):
    """Require the clean E1 contract plus exact residual-only lineage."""

    base = base_probe.validate_loaded_seller(model, metadata, e0b_metadata)
    policy = model.policy
    expected = canonical_economic_architecture()
    recorded = getattr(model, trainer.ECONOMIC_ARCHITECTURE_ATTRIBUTE, None)
    training_code_revision = getattr(
        model, trainer.E1_TRAINING_CODE_REVISION_ATTRIBUTE, None
    )
    checks = dict(base["checks"])
    checks.update({
        "threshold_residual_enabled": bool(getattr(
            policy, "economic_threshold_residual", False
        )),
        "pure_64_input_base_head": (
            int(policy.economic_head[0].in_features) == 64
        ),
        "policy_economic_architecture_exact": (
            policy.economic_architecture_provenance() == expected
        ),
        "model_economic_architecture_provenance_exact": recorded == expected,
        "metadata_economic_architecture_exact": (
            metadata.get("economic_architecture") == expected
        ),
        "training_config_enables_threshold_residual": (
            metadata.get("training_config", {}).get(
                "economic_threshold_residual"
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
            "seller checkpoint failed threshold-residual probe checks: "
            + ", ".join(failed)
        )
    return {
        "passed": True,
        "checks": {name: bool(value) for name, value in checks.items()},
        "economic_architecture": expected,
        "e1_training_code_revision": training_code_revision,
    }


def economic_beta_means(model, values):
    """Return the learned base mean and the actual post-anchor Beta mean."""

    tensor_values, _ = model.policy.obs_to_tensor(values)
    with th.inference_mode():
        processed, _, state_features, _ = model.policy._actor_features(
            tensor_values
        )
        economic_features = model.policy._economic_state_features(
            processed, state_features
        )
        parameters = model.policy.economic_head(economic_features)
        alpha = (
            functional.softplus(parameters[:, 0]) + BETA_PARAMETER_EPSILON
        )
        beta = (
            functional.softplus(parameters[:, 1]) + BETA_PARAMETER_EPSILON
        )
        base_mean = (alpha / (alpha + beta)).detach().cpu().numpy().reshape(-1)
        final_mean = (
            model.policy.get_distribution(tensor_values)
            .economic_mean.detach().cpu().numpy().reshape(-1)
        )
    if base_mean.shape != (1,) or final_mean.shape != (1,):
        raise RuntimeError("seller probe expected exactly one Beta distribution")
    return float(base_mean[0]), float(final_mean[0])


def threshold_residual_warmup_gate(
        *, final_all_equal, final_coordinate_low, final_coordinate_high,
        base_all_equal, base_coordinate_low, base_coordinate_high,
):
    """Require numerical residual sanity and nonzero learned base response."""

    gate = base_probe.warmup_diagnostic_gate(
        final_all_equal, final_coordinate_low, final_coordinate_high
    )
    final_coordinate_response = np.asarray(
        final_coordinate_high, dtype=np.float64
    ) - np.asarray(final_coordinate_low, dtype=np.float64)
    base_endpoint_response = (
        np.asarray(base_all_equal, dtype=np.float64)[-1]
        - np.asarray(base_all_equal, dtype=np.float64)[0]
    )
    base_coordinate_response = np.asarray(
        base_coordinate_high, dtype=np.float64
    ) - np.asarray(base_coordinate_low, dtype=np.float64)
    coordinate_minimum = float(np.min(final_coordinate_response))
    base_minimum = float(np.min(base_endpoint_response))
    coordinate_passed = bool(
        np.all(np.isfinite(final_coordinate_response))
        and coordinate_minimum >= MINIMUM_CURRENT_COORDINATE_RESPONSE
    )
    base_passed = bool(
        np.all(np.isfinite(base_endpoint_response))
        and base_minimum >= MINIMUM_LEARNED_BASE_ENDPOINT_RESPONSE
    )
    base_coordinate_minimum = float(np.min(base_coordinate_response))
    base_coordinate_passed = bool(
        np.all(np.isfinite(base_coordinate_response))
        and base_coordinate_minimum >= MINIMUM_LEARNED_BASE_ENDPOINT_RESPONSE
    )
    checks = dict(gate["checks"])
    checks.update({
        "minimum_current_coordinate_final_beta_mean_response": {
            "actual": coordinate_minimum,
            "operator": ">=",
            "required": MINIMUM_CURRENT_COORDINATE_RESPONSE,
            "passed": coordinate_passed,
        },
        "minimum_learned_base_all_one_minus_all_zero_mean_response": {
            "actual": base_minimum,
            "operator": ">=",
            "required": MINIMUM_LEARNED_BASE_ENDPOINT_RESPONSE,
            "passed": base_passed,
        },
        "minimum_learned_base_current_coordinate_mean_response": {
            "actual": base_coordinate_minimum,
            "operator": ">=",
            "required": MINIMUM_LEARNED_BASE_ENDPOINT_RESPONSE,
            "passed": base_coordinate_passed,
        },
    })
    return {
        **gate,
        "name": PROBE_GATE_NAME,
        "passed": bool(all(item["passed"] for item in checks.values())),
        "checks": checks,
        "fixed_anchor_weight": 0.5,
        "fixed_anchor_is_inductive_bias": True,
        "learned_threshold_slope_claim": False,
        "per_event_current_coordinate_final_beta_mean_response": (
            final_coordinate_response.tolist()
        ),
        "per_event_learned_base_all_one_minus_all_zero_mean_response": (
            base_endpoint_response.tolist()
        ),
        "per_event_learned_base_current_coordinate_mean_response": (
            base_coordinate_response.tolist()
        ),
    }


def collect_conditioning_report(model):
    """Measure learned base and final residual means on canonical trades."""

    thresholds = base_probe.ALL_EQUAL_THRESHOLDS
    event_count = base_probe.NUM_TRADE_EVENTS
    base_all_equal = np.empty((len(thresholds), event_count), dtype=np.float64)
    final_all_equal = np.empty_like(base_all_equal)
    all_equal_rows = []
    for threshold_row, threshold in enumerate(thresholds):
        context = np.full(event_count, threshold, dtype=np.float32)
        events = []
        for event_index in range(event_count):
            values = base_probe.canonical_trade_observation(
                model.policy,
                event_index=event_index,
                thresholds=context,
            )
            base_mean, final_mean = economic_beta_means(model, values)
            base_all_equal[threshold_row, event_index] = base_mean
            final_all_equal[threshold_row, event_index] = final_mean
            events.append({
                "event_index": event_index,
                "actor_state": values[base_probe.ACTOR_STATE].tolist(),
                "learned_base_beta_mean_price": base_mean,
                "final_residual_beta_mean_price": final_mean,
            })
        all_equal_rows.append({
            "threshold": threshold,
            "opponent_commitment": context.tolist(),
            "event_learned_base_beta_mean_prices": (
                base_all_equal[threshold_row].tolist()
            ),
            "event_final_residual_beta_mean_prices": (
                final_all_equal[threshold_row].tolist()
            ),
            "events": events,
        })

    base_low = np.empty(event_count, dtype=np.float64)
    base_high = np.empty(event_count, dtype=np.float64)
    final_low = np.empty(event_count, dtype=np.float64)
    final_high = np.empty(event_count, dtype=np.float64)
    coordinate_rows = []
    for event_index in range(event_count):
        low_context = np.full(
            event_count,
            base_probe.COORDINATE_BASELINE_THRESHOLD,
            dtype=np.float32,
        )
        high_context = np.array(low_context, copy=True)
        low_context[event_index] = 0.0
        high_context[event_index] = 1.0
        low_values = base_probe.canonical_trade_observation(
            model.policy, event_index=event_index, thresholds=low_context
        )
        high_values = base_probe.canonical_trade_observation(
            model.policy, event_index=event_index, thresholds=high_context
        )
        base_low[event_index], final_low[event_index] = economic_beta_means(
            model, low_values
        )
        base_high[event_index], final_high[event_index] = economic_beta_means(
            model, high_values
        )
        coordinate_rows.append({
            "event_index": event_index,
            "other_coordinate_threshold": (
                base_probe.COORDINATE_BASELINE_THRESHOLD
            ),
            "low_context": low_context.tolist(),
            "high_context": high_context.tolist(),
            "low_learned_base_beta_mean_price": base_low[event_index],
            "high_learned_base_beta_mean_price": base_high[event_index],
            "learned_base_sensitivity": (
                base_high[event_index] - base_low[event_index]
            ),
            "low_final_residual_beta_mean_price": final_low[event_index],
            "high_final_residual_beta_mean_price": final_high[event_index],
            "final_residual_sensitivity": (
                final_high[event_index] - final_low[event_index]
            ),
        })

    base_endpoint = base_all_equal[-1] - base_all_equal[0]
    final_endpoint = final_all_equal[-1] - final_all_equal[0]
    return {
        "protocol": {
            "actor_state_dimension": base_probe.ACTOR_STATE_DIM,
            "actor_state_order": [
                "ammo_fraction",
                "projectile_active",
                "normalized_time",
                "trade_mode",
                "event_one_hot[0:5]",
                "opponent_threshold_commitment[0:5]",
            ],
            "gameplay_horizon": base_probe.CANONICAL_GAMEPLAY_HORIZON,
            "event_steps": list(base_probe.CANONICAL_EVENT_STEPS),
            "ammo_fraction": base_probe.CANONICAL_AMMO_FRACTION,
            "projectile_active": 0.0,
            "trade_mode": 1.0,
            "action_credit": [0.0, 1.0],
            "image": "zero dummy image in the checkpoint observation space",
            "action_mask": "all actions feasible",
            "critic_state": "zero; actor-invisible and unused by the probe",
            "economic_architecture": canonical_economic_architecture(),
            "base_mean_definition": (
                "softplus base-head parameters before threshold anchoring"
            ),
            "final_mean_definition": (
                "ordinary Beta mean after the fixed threshold-residual transform"
            ),
        },
        "all_equal_thresholds": all_equal_rows,
        "learned_base_low_to_high_response": {
            "definition": "all-one minus all-zero learned base Beta mean",
            "per_event": base_endpoint.tolist(),
            **base_probe._finite_summary(base_endpoint),
        },
        "final_residual_low_to_high_response": {
            "definition": "all-one minus all-zero final residual Beta mean",
            "per_event": final_endpoint.tolist(),
            **base_probe._finite_summary(final_endpoint),
        },
        "current_coordinate_only_sensitivity": {
            "definition": (
                "current threshold 1 minus 0 with all other thresholds at 0.5"
            ),
            "rows": coordinate_rows,
            "learned_base": base_probe._finite_summary(base_high - base_low),
            "final_residual": base_probe._finite_summary(final_high - final_low),
        },
        "warmup_gate": threshold_residual_warmup_gate(
            final_all_equal=final_all_equal,
            final_coordinate_low=final_low,
            final_coordinate_high=final_high,
            base_all_equal=base_all_equal,
            base_coordinate_low=base_low,
            base_coordinate_high=base_high,
        ),
    }


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
                "learned base Beta mean", "final residual Beta mean"
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
    report = run_probe_from_checkpoints(
        checkpoint=args.checkpoint,
        e0b_checkpoint=args.e0b_checkpoint,
        device=args.device,
    )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite threshold-residual probe: {output}"
            )
        write_json(output, report)
        print(json.dumps({
            "output": str(output),
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "warmup_gate_passed": report["warmup_gate"]["passed"],
        }, sort_keys=True, allow_nan=False), flush=True)
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if args.require_pass and not report["warmup_gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
