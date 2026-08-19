"""No-ALE learned-conditioning probe for the seller shared-context v5 policy.

The probe evaluates protocol-valid paused-trade observations directly through
the checkpoint policy.  It additionally performs reversible in-memory
ablations of the shared context branch and shared current-threshold skip.
No ALE environment is constructed and checkpoint bytes are never modified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import torch as th

from replication.atari import probe_atari_e1_seller_conditioning as base_probe
from replication.atari import train_atari_meta_response_sb3 as trainer
from stackelberg_pomdp.atari.training import write_json
from stackelberg_pomdp.atari.policies.composite import (
    SELLER_SHARED_CONTEXT_BETA_V5,
    seller_shared_context_architecture_provenance,
)


PROBE_NAME = "clean_atari_e1_seller_shared_context_probe_v5"
PROBE_GATE_NAME = "clean_e1_seller_shared_context_conditioning_gate_v5"
ARCHITECTURE_SCHEMA = "stackpomdp.atari.economic_actor_architecture.v5"
PARAMETERIZATION = SELLER_SHARED_CONTEXT_BETA_V5

# Preregistered before the 82,000-step diagnostic is run.  The small response
# floors test whether conditioning is being learned; the formal selector later
# applies the substantially stronger real-ALE economic gate.
MINIMUM_ENDPOINT_RESPONSE = 0.01
MINIMUM_CURRENT_COORDINATE_RESPONSE = 0.01
MAXIMUM_ADJACENT_REVERSAL = 0.15
MAXIMUM_COMBINED_ABLATION_RESPONSE = 1.0e-7

VARIANT_FULL = "full"
VARIANT_SKIP_ABLATED = "current_skip_ablated"
VARIANT_CONTEXT_ABLATED = "context_branch_ablated"
VARIANT_BOTH_ABLATED = "context_and_current_skip_ablated"
VARIANTS = (
    VARIANT_FULL,
    VARIANT_SKIP_ABLATED,
    VARIANT_CONTEXT_ABLATED,
    VARIANT_BOTH_ABLATED,
)


def canonical_economic_architecture():
    return seller_shared_context_architecture_provenance()


def canonical_initialization_provenance(policy):
    return trainer.shared_context_initialization_provenance(policy)


def validate_loaded_seller(model, metadata, e0b_metadata):
    """Require the exact v5 architecture and frozen E0b gameplay actor."""

    base = base_probe.validate_loaded_seller(model, metadata, e0b_metadata)
    policy = model.policy
    architecture = canonical_economic_architecture()
    initialization = canonical_initialization_provenance(policy)
    recorded_architecture = getattr(
        model, trainer.ECONOMIC_ARCHITECTURE_ATTRIBUTE, None
    )
    recorded_initialization = getattr(
        model, trainer.SHARED_CONTEXT_INITIALIZATION_ATTRIBUTE, None
    )
    revision = getattr(
        model, trainer.E1_TRAINING_CODE_REVISION_ATTRIBUTE, None
    )
    source = metadata.get("e0b_source_provenance", {})
    group_names = [
        group.get("group_name")
        for group in policy.optimizer.param_groups
    ]
    frozen_hashes = trainer.validate_frozen_gameplay_actor(model)
    checks = dict(base["checks"])
    checks.update({
        "v5_policy_parameterization": (
            getattr(policy, "economic_architecture", None)
            == PARAMETERIZATION
        ),
        "legacy_threshold_residual_disabled": not bool(getattr(
            policy, "economic_threshold_residual", False
        )),
        "legacy_direct_threshold_input_disabled": not bool(getattr(
            policy, "economic_threshold_residual_direct_input", False
        )),
        "policy_architecture_provenance_exact": (
            policy.economic_architecture_provenance() == architecture
        ),
        "model_architecture_provenance_exact": (
            recorded_architecture == architecture
        ),
        "metadata_architecture_provenance_exact": (
            metadata.get("economic_architecture") == architecture
        ),
        "policy_initialization_provenance_exact": (
            initialization == trainer.shared_context_initialization_contract()
        ),
        "model_initialization_provenance_exact": (
            recorded_initialization == initialization
        ),
        "metadata_initialization_provenance_exact": (
            metadata.get("shared_context_initialization") == initialization
        ),
        "training_config_enables_only_v5": (
            metadata.get("training_config", {}).get(
                "economic_architecture"
            ) == PARAMETERIZATION
            and "economic_threshold_residual" not in metadata.get(
                "training_config", {}
            )
            and "economic_threshold_residual_direct_input" not in metadata.get(
                "training_config", {}
            )
        ),
        "live_branch_shapes_exact": (
            policy.economic_live_encoder[0].in_features == 9
            and policy.economic_live_encoder[0].out_features == 32
            and policy.economic_live_output.in_features == 32
            and policy.economic_live_output.out_features == 2
        ),
        "context_branch_shapes_exact": (
            policy.economic_context_encoder[0].in_features == 10
            and policy.economic_context_encoder[0].out_features == 32
            and policy.economic_context_output.in_features == 32
            and policy.economic_context_output.out_features == 1
            and tuple(policy.economic_current_slope.shape) == ()
        ),
        "optimizer_groups_exact": group_names == [
            "seller_v5_live", "seller_v5_context", "seller_v5_critic"
        ],
        "optimizer_group_scales_exact": [
            group.get("lr_scale") for group in policy.optimizer.param_groups
        ] == [1.0, 4.0, 0.2],
        "optimizer_group_rates_exact": [
            group.get("lr") for group in policy.optimizer.param_groups
        ] == [5.0e-4, 2.0e-3, 1.0e-4],
        "independent_gradient_clip_norm_exact": (
            float(getattr(model, "max_grad_norm", float("nan"))) == 0.5
        ),
        "frozen_gameplay_hashes_match_source": (
            isinstance(frozen_hashes, dict)
            and frozen_hashes == source.get("frozen_gameplay_actor_sha256")
        ),
        "training_code_revision_is_full_sha": (
            isinstance(revision, str)
            and re.fullmatch(r"[0-9a-f]{40}", revision) is not None
        ),
        "metadata_training_code_revision_exact": (
            metadata.get("e1_training_code_revision") == revision
        ),
    })
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise ValueError(
            "seller checkpoint failed shared-context probe checks: "
            + ", ".join(failed)
        )
    return {
        "passed": True,
        "checks": {name: bool(value) for name, value in checks.items()},
        "economic_architecture": architecture,
        "shared_context_initialization": initialization,
        "frozen_gameplay_actor_sha256": frozen_hashes,
        "e1_training_code_revision": revision,
    }


def economic_beta_mean(model, values):
    tensor_values, _ = model.policy.obs_to_tensor(values)
    with th.inference_mode():
        means = (
            model.policy.get_distribution(tensor_values)
            .economic_mean.detach().cpu().numpy().reshape(-1)
        )
    if means.shape != (1,):
        raise RuntimeError("shared-context probe expected one Beta distribution")
    return float(means[0])


def _snapshot_ablatable_parameters(policy):
    return {
        "context_weight": policy.economic_context_output.weight.detach().clone(),
        "context_bias": policy.economic_context_output.bias.detach().clone(),
        "current_slope": policy.economic_current_slope.detach().clone(),
    }


def _snapshot_equal(policy, snapshot):
    return bool(
        th.equal(
            policy.economic_context_output.weight.detach(),
            snapshot["context_weight"],
        )
        and th.equal(
            policy.economic_context_output.bias.detach(),
            snapshot["context_bias"],
        )
        and th.equal(
            policy.economic_current_slope.detach(),
            snapshot["current_slope"],
        )
    )


@contextmanager
def _in_memory_ablation(policy, *, context_branch, current_skip):
    """Temporarily zero selected v5 paths and restore them bit-exactly."""

    snapshot = _snapshot_ablatable_parameters(policy)
    with th.no_grad():
        if context_branch:
            policy.economic_context_output.weight.zero_()
            policy.economic_context_output.bias.zero_()
        if current_skip:
            policy.economic_current_slope.zero_()
    try:
        yield
    finally:
        with th.no_grad():
            policy.economic_context_output.weight.copy_(
                snapshot["context_weight"]
            )
            policy.economic_context_output.bias.copy_(snapshot["context_bias"])
            policy.economic_current_slope.copy_(snapshot["current_slope"])
        if not _snapshot_equal(policy, snapshot):
            raise RuntimeError("in-memory shared-context ablation did not restore")


def _variant_ablation(variant):
    if variant == VARIANT_FULL:
        return False, False
    if variant == VARIANT_SKIP_ABLATED:
        return False, True
    if variant == VARIANT_CONTEXT_ABLATED:
        return True, False
    if variant == VARIANT_BOTH_ABLATED:
        return True, True
    raise ValueError(f"unknown shared-context probe variant: {variant}")


def _collect_variant(model, variant):
    context_ablation, skip_ablation = _variant_ablation(variant)
    event_count = base_probe.NUM_TRADE_EVENTS
    all_equal = np.empty(
        (len(base_probe.ALL_EQUAL_THRESHOLDS), event_count),
        dtype=np.float64,
    )
    current_low = np.empty(event_count, dtype=np.float64)
    current_high = np.empty(event_count, dtype=np.float64)
    noncurrent_low = np.empty(event_count, dtype=np.float64)
    noncurrent_high = np.empty(event_count, dtype=np.float64)
    before = _snapshot_ablatable_parameters(model.policy)
    with _in_memory_ablation(
            model.policy,
            context_branch=context_ablation,
            current_skip=skip_ablation,
    ):
        for threshold_index, threshold in enumerate(
                base_probe.ALL_EQUAL_THRESHOLDS
        ):
            context = np.full(event_count, threshold, dtype=np.float32)
            for event_index in range(event_count):
                values = base_probe.canonical_trade_observation(
                    model.policy,
                    event_index=event_index,
                    thresholds=context,
                )
                all_equal[threshold_index, event_index] = economic_beta_mean(
                    model, values
                )
        for event_index in range(event_count):
            low = np.full(
                event_count,
                base_probe.COORDINATE_BASELINE_THRESHOLD,
                dtype=np.float32,
            )
            high = np.array(low, copy=True)
            low[event_index] = 0.0
            high[event_index] = 1.0
            current_low[event_index] = economic_beta_mean(
                model,
                base_probe.canonical_trade_observation(
                    model.policy, event_index=event_index, thresholds=low
                ),
            )
            current_high[event_index] = economic_beta_mean(
                model,
                base_probe.canonical_trade_observation(
                    model.policy, event_index=event_index, thresholds=high
                ),
            )
            noncurrent_index = (event_index + 1) % event_count
            low = np.full(
                event_count,
                base_probe.COORDINATE_BASELINE_THRESHOLD,
                dtype=np.float32,
            )
            high = np.array(low, copy=True)
            low[noncurrent_index] = 0.0
            high[noncurrent_index] = 1.0
            noncurrent_low[event_index] = economic_beta_mean(
                model,
                base_probe.canonical_trade_observation(
                    model.policy, event_index=event_index, thresholds=low
                ),
            )
            noncurrent_high[event_index] = economic_beta_mean(
                model,
                base_probe.canonical_trade_observation(
                    model.policy, event_index=event_index, thresholds=high
                ),
            )
    return {
        "all_equal_beta_mean_prices": all_equal.tolist(),
        "current_coordinate_low": current_low.tolist(),
        "current_coordinate_high": current_high.tolist(),
        "noncurrent_coordinate_low": noncurrent_low.tolist(),
        "noncurrent_coordinate_high": noncurrent_high.tolist(),
        "parameter_restore_exact": _snapshot_equal(model.policy, before),
    }


def _variant_arrays(value):
    return {
        "all_equal": np.asarray(
            value["all_equal_beta_mean_prices"], dtype=np.float64
        ),
        "current_low": np.asarray(
            value["current_coordinate_low"], dtype=np.float64
        ),
        "current_high": np.asarray(
            value["current_coordinate_high"], dtype=np.float64
        ),
        "noncurrent_low": np.asarray(
            value["noncurrent_coordinate_low"], dtype=np.float64
        ),
        "noncurrent_high": np.asarray(
            value["noncurrent_coordinate_high"], dtype=np.float64
        ),
    }


def shared_context_conditioning_gate(variants, learning_evidence):
    """Apply immutable learned-response and reversible-ablation gates."""

    if set(variants) != set(VARIANTS):
        raise ValueError("shared-context probe variants changed")
    arrays = {name: _variant_arrays(variants[name]) for name in VARIANTS}
    expected_all_equal = (
        len(base_probe.ALL_EQUAL_THRESHOLDS), base_probe.NUM_TRADE_EVENTS
    )
    for name, value in arrays.items():
        if value["all_equal"].shape != expected_all_equal:
            raise ValueError(f"{name} all-equal matrix has the wrong shape")
        for key in (
                "current_low", "current_high",
                "noncurrent_low", "noncurrent_high",
        ):
            if value[key].shape != (base_probe.NUM_TRADE_EVENTS,):
                raise ValueError(f"{name} {key} has the wrong shape")

    full = arrays[VARIANT_FULL]
    no_skip = arrays[VARIANT_SKIP_ABLATED]
    both = arrays[VARIANT_BOTH_ABLATED]
    all_outputs = np.concatenate([
        value.reshape(-1)
        for variant in arrays.values()
        for value in variant.values()
    ])
    finite_unit = bool(
        np.all(np.isfinite(all_outputs))
        and np.all((all_outputs >= 0.0) & (all_outputs <= 1.0))
    )
    endpoint = full["all_equal"][-1] - full["all_equal"][0]
    current = full["current_high"] - full["current_low"]
    adjacent_reversal = float(max(
        0.0,
        np.max(full["all_equal"][:-1] - full["all_equal"][1:]),
    ))
    ablated_responses = np.concatenate([
        both["all_equal"][-1] - both["all_equal"][0],
        both["current_high"] - both["current_low"],
        both["noncurrent_high"] - both["noncurrent_low"],
    ])
    maximum_ablated_response = float(np.max(np.abs(ablated_responses)))
    no_skip_responses = np.concatenate([
        no_skip["all_equal"][-1] - no_skip["all_equal"][0],
        no_skip["current_high"] - no_skip["current_low"],
        no_skip["noncurrent_high"] - no_skip["noncurrent_low"],
    ])
    maximum_context_branch_response = float(
        np.max(np.abs(no_skip_responses))
    )
    conditioning_parameter_linf = float(
        learning_evidence["conditioning_parameter_linf"]
    )
    optimizer_steps = learning_evidence["optimizer_steps"]
    optimizer_steps_positive = bool(
        optimizer_steps
        and all(int(step) > 0 for step in optimizer_steps.values())
    )
    restore_exact = bool(all(
        variants[name].get("parameter_restore_exact") is True
        for name in VARIANTS
    ))
    checks = {
        "all_outputs_finite_and_in_unit_interval": {
            "actual": finite_unit,
            "required": True,
            "passed": finite_unit,
        },
        "minimum_all_one_minus_all_zero_beta_mean_price": {
            "actual": float(np.min(endpoint)),
            "operator": ">=",
            "required": MINIMUM_ENDPOINT_RESPONSE,
            "passed": bool(np.min(endpoint) >= MINIMUM_ENDPOINT_RESPONSE),
        },
        "minimum_current_coordinate_beta_mean_response": {
            "actual": float(np.min(current)),
            "operator": ">=",
            "required": MINIMUM_CURRENT_COORDINATE_RESPONSE,
            "passed": bool(
                np.min(current) >= MINIMUM_CURRENT_COORDINATE_RESPONSE
            ),
        },
        "largest_adjacent_threshold_price_reversal": {
            "actual": adjacent_reversal,
            "operator": "<=",
            "required": MAXIMUM_ADJACENT_REVERSAL,
            "passed": bool(
                adjacent_reversal <= MAXIMUM_ADJACENT_REVERSAL
            ),
        },
        "combined_context_and_skip_ablation_commitment_response": {
            "actual": maximum_ablated_response,
            "operator": "<=",
            "required": MAXIMUM_COMBINED_ABLATION_RESPONSE,
            "passed": bool(
                maximum_ablated_response
                <= MAXIMUM_COMBINED_ABLATION_RESPONSE
            ),
        },
        "conditioning_parameters_changed_from_exact_zero": {
            "actual": conditioning_parameter_linf,
            "operator": ">",
            "required": 0.0,
            "passed": bool(conditioning_parameter_linf > 0.0),
        },
        "conditioning_parameters_have_positive_optimizer_steps": {
            "actual": optimizer_steps,
            "required": "every recorded step > 0",
            "passed": optimizer_steps_positive,
        },
        "in_memory_ablation_restored_parameters_exactly": {
            "actual": restore_exact,
            "required": True,
            "passed": restore_exact,
        },
    }
    return {
        "name": PROBE_GATE_NAME,
        "predeclared": True,
        "passed": bool(all(check["passed"] for check in checks.values())),
        "checks": checks,
        "architecture_parameterization": PARAMETERIZATION,
        "fixed_anchor_is_inductive_bias": False,
        "learned_context_claim": True,
        "per_event_endpoint_response": endpoint.tolist(),
        "per_event_current_coordinate_response": current.tolist(),
        "maximum_absolute_noncurrent_response": float(np.max(np.abs(
            full["noncurrent_high"] - full["noncurrent_low"]
        ))),
        "diagnostics_not_gated": {
            "maximum_context_branch_response_with_skip_ablated": (
                maximum_context_branch_response
            ),
            "reason_not_gated": (
                "a rational policy may place its learned response in the "
                "shared current-threshold skip"
            ),
        },
    }


def _optimizer_step(policy, parameter):
    state = policy.optimizer.state.get(parameter, {})
    raw = state.get("step")
    if raw is None:
        return 0
    return int(raw.item() if hasattr(raw, "item") else raw)


def conditioning_learning_evidence(policy):
    """Report learned conditioning values and their Adam update clocks."""

    named = {
        "context_output_weight": policy.economic_context_output.weight,
        "context_output_bias": policy.economic_context_output.bias,
        "current_slope": policy.economic_current_slope,
    }
    linf = {
        name: float(parameter.detach().abs().max().cpu().item())
        for name, parameter in named.items()
    }
    return {
        "parameter_linf": linf,
        "conditioning_parameter_linf": max(linf.values()),
        "optimizer_steps": {
            name: _optimizer_step(policy, parameter)
            for name, parameter in named.items()
        },
        "initialization_reference": "all three parameter sets were exact zero",
    }


def collect_conditioning_report(model):
    """Collect full and causally ablated outputs without stepping ALE."""

    policy = model.policy
    baseline = _snapshot_ablatable_parameters(policy)
    variants = {
        variant: _collect_variant(model, variant)
        for variant in VARIANTS
    }
    if not _snapshot_equal(policy, baseline):
        raise RuntimeError("shared-context probe changed policy parameters")
    learning_evidence = conditioning_learning_evidence(policy)
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
            "all_equal_threshold_grid": list(
                base_probe.ALL_EQUAL_THRESHOLDS
            ),
            "coordinate_baseline_threshold": (
                base_probe.COORDINATE_BASELINE_THRESHOLD
            ),
            "economic_architecture": canonical_economic_architecture(),
            "shared_context_initialization": (
                canonical_initialization_provenance(policy)
            ),
            "ablation": (
                "reversible in-memory zeroing; checkpoint bytes and optimizer "
                "state are untouched"
            ),
        },
        "variants": variants,
        "conditioning_learning_evidence": learning_evidence,
        "warmup_gate": shared_context_conditioning_gate(
            variants, learning_evidence
        ),
    }


def run_probe_from_checkpoints(*, checkpoint, e0b_checkpoint, device="cpu"):
    """Reload and probe immutable checkpoint bytes without constructing ALE."""

    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    candidate = evaluator.checkpoint_path(checkpoint)
    before = evaluator.checkpoint_sha256(candidate)
    e0b = evaluator.validate_e0b(e0b_checkpoint, device=device)
    model, metadata = evaluator.load_candidate(
        candidate,
        role=base_probe.SELLER_ROLE,
        e0b_sha256=e0b["sha256"],
        device=device,
    )
    try:
        verification = validate_loaded_seller(model, metadata, e0b)
        measurements = collect_conditioning_report(model)
    finally:
        del model
    after = evaluator.checkpoint_sha256(candidate)
    if after != before or metadata["sha256"] != before:
        raise RuntimeError("seller checkpoint bytes changed during v5 probe")
    return {
        "probe": PROBE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
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
        output = Path(args.output).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite v5 probe: {output}")
        write_json(output, result)
        print(json.dumps({
            "output": str(output),
            "checkpoint_sha256": result["checkpoint"]["sha256"],
            "conditioning_gate_passed": result["warmup_gate"]["passed"],
        }, sort_keys=True, allow_nan=False), flush=True)
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.require_pass and not result["warmup_gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
