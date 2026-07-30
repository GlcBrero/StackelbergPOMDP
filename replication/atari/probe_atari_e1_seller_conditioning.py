"""Probe E1 seller threshold conditioning without constructing ALE.

The probe loads one clean E1 seller checkpoint through the same strict
metadata validator used by the official selector, then evaluates the
deterministic Beta mean on canonical, protocol-valid paused-trade
observations.  It never creates or steps an Atari environment.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch as th

from replication.atari.sb3_common import write_json
from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTION_MASK,
    ACTOR_STATE,
    ACTOR_STATE_DIM,
    EVENT_SLICE,
    FOLLOWER_TRADE,
    IMAGE,
    NUM_TRADE_EVENTS,
    OPPONENT_COMMITMENT_SLICE,
    TRADE_MODE_INDEX,
    actor_state,
    observation,
)
from stackelberg_pomdp.atari.stackpomdp_policy import StackPOMDPAtariPolicy


PROBE_NAME = "clean_atari_e1_seller_conditioning_probe_v1"
SELLER_ROLE = "seller"
CANONICAL_GAMEPLAY_HORIZON = 200
CANONICAL_EVENT_STEPS = (20, 50, 80, 110, 140)
CANONICAL_AMMO_FRACTION = 1.0 / NUM_TRADE_EVENTS
ALL_EQUAL_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0)
COORDINATE_BASELINE_THRESHOLD = 0.5
MINIMUM_ENDPOINT_RESPONSE = 0.25
MAXIMUM_ADJACENT_REVERSAL = 0.15


def _finite_unit(value):
    value = float(value)
    return bool(np.isfinite(value) and 0.0 <= value <= 1.0)


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _finite_summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return {"minimum": None, "mean": None, "maximum": None}
    return {
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def canonical_trade_observation(policy, *, event_index, thresholds):
    """Build one exact 14D live seller trade observation for the probe.

    The seller has the newly arrived event bullet, no projectile is active,
    and all previous inventory has been sold or fired.  These choices give a
    dynamically feasible state at every event while holding non-context state
    fixed across counterfactual threshold vectors.
    """

    event_index = int(event_index)
    if not 0 <= event_index < NUM_TRADE_EVENTS:
        raise ValueError("event_index must lie in [0, 4]")
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if thresholds.shape != (NUM_TRADE_EVENTS,):
        raise ValueError("thresholds must contain exactly five values")

    observation_space = policy.observation_space
    image_space = observation_space.spaces[IMAGE]
    image = np.zeros(image_space.shape, dtype=image_space.dtype)
    mask_space = observation_space.spaces[ACTION_MASK]
    action_mask = np.ones(mask_space.shape, dtype=np.float32)
    state = actor_state(
        ammo_fraction=CANONICAL_AMMO_FRACTION,
        projectile_active=0.0,
        normalized_time=(
            float(CANONICAL_EVENT_STEPS[event_index])
            / CANONICAL_GAMEPLAY_HORIZON
        ),
        trade_mode=1.0,
        event_index=event_index,
        opponent_commitment=thresholds,
    )
    result = observation(
        image=image,
        state=state,
        action_mask=action_mask,
        decision_kind=FOLLOWER_TRADE,
    )
    if not observation_space.contains(result):
        raise RuntimeError("canonical trade observation is outside checkpoint space")
    if not np.array_equal(result[ACTOR_STATE][EVENT_SLICE], np.eye(
            NUM_TRADE_EVENTS, dtype=np.float32
    )[event_index]):
        raise RuntimeError("canonical trade observation has an invalid event code")
    if not np.array_equal(
            result[ACTOR_STATE][OPPONENT_COMMITMENT_SLICE], thresholds
    ):
        raise RuntimeError("canonical trade observation changed the threshold vector")
    if not np.isclose(result[ACTOR_STATE][TRADE_MODE_INDEX], 1.0):
        raise RuntimeError("canonical trade observation is not in trade mode")
    if not np.array_equal(
            result[ACTION_CREDIT], np.array([0.0, 1.0], dtype=np.float32)
    ):
        raise RuntimeError("canonical trade observation has wrong action credit")
    return result


def economic_beta_mean(model, values):
    """Return the deterministic seller price statistic for one observation."""

    tensor_values, _ = model.policy.obs_to_tensor(values)
    with th.inference_mode():
        means = (
            model.policy.get_distribution(tensor_values)
            .economic_mean.detach().cpu().numpy().reshape(-1)
        )
    if means.shape != (1,):
        raise RuntimeError("seller probe expected exactly one Beta distribution")
    return float(means[0])


def validate_loaded_seller(model, metadata, e0b_metadata):
    """Fail closed unless the loaded artifact is the canonical clean E1 seller."""

    policy = model.policy
    provenance = metadata.get("e0b_source_provenance")
    training = metadata.get("training_config")
    sampler = metadata.get("atari_e1_sampler_provenance")
    history = metadata.get("atari_e1_sampler_history")
    checks = {
        "clean_policy_class": isinstance(policy, StackPOMDPAtariPolicy),
        "seller_role": getattr(policy, "economic_role", None) == SELLER_ROLE,
        "full_economic_input": (
            getattr(policy, "economic_input_mode", None) == "full"
        ),
        "metadata_seller_role": metadata.get("role") == SELLER_ROLE,
        "metadata_full_economic_input": (
            metadata.get("economic_input_mode") == "full"
        ),
        "checkpoint_sha256_present": (
            isinstance(metadata.get("sha256"), str)
            and len(metadata["sha256"]) == 64
        ),
        "checkpoint_timestep_matches_model": (
            type(metadata.get("training_timesteps")) is int
            and int(metadata["training_timesteps"])
            == int(getattr(model, "num_timesteps", -1))
        ),
        "ppo_training_metadata": (
            isinstance(training, dict)
            and training.get("algorithm") == "PPO"
        ),
        "canonical_205_step_return": (
            int(getattr(model, "n_steps", -1)) == 205
            and np.isclose(float(getattr(model, "gamma", np.nan)), 1.0)
            and np.isclose(float(getattr(model, "gae_lambda", np.nan)), 1.0)
        ),
        "pretrained_learning_rate_scale": np.isclose(
            float(getattr(policy, "pretrained_lr_scale", np.nan)),
            0.1,
            atol=0.0,
            rtol=0.0,
        ),
        "e0b_source_metadata_present": isinstance(provenance, dict),
        "e0b_source_bytes_match": (
            isinstance(provenance, dict)
            and provenance.get("sha256") == e0b_metadata.get("sha256")
        ),
        "clean_commitment_column_reset": (
            isinstance(provenance, dict)
            and provenance.get("zero_initialized_actor_state_indices")
            == [9, 10, 11, 12, 13]
        ),
        "sampler_provenance_present": isinstance(sampler, dict),
        "sampler_history_present": isinstance(history, list) and bool(history),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise ValueError(
            "seller checkpoint failed conditioning-probe metadata checks: "
            + ", ".join(failed)
        )
    return {"passed": True, "checks": checks}


def warmup_diagnostic_gate(
        all_equal_prices,
        coordinate_low_prices,
        coordinate_high_prices,
):
    """Apply the predeclared no-ALE warmup gate to Beta-mean prices."""

    all_equal = np.asarray(all_equal_prices, dtype=np.float64)
    low = np.asarray(coordinate_low_prices, dtype=np.float64).reshape(-1)
    high = np.asarray(coordinate_high_prices, dtype=np.float64).reshape(-1)
    expected_shape = (len(ALL_EQUAL_THRESHOLDS), NUM_TRADE_EVENTS)
    if all_equal.shape != expected_shape:
        raise ValueError(
            f"all_equal_prices must have shape {expected_shape}, "
            f"got {all_equal.shape}"
        )
    if low.shape != (NUM_TRADE_EVENTS,) or high.shape != (NUM_TRADE_EVENTS,):
        raise ValueError("coordinate-only price vectors must each have five values")

    all_outputs = np.concatenate([all_equal.reshape(-1), low, high])
    outputs_valid = bool(all(_finite_unit(value) for value in all_outputs))
    endpoints = all_equal[-1] - all_equal[0]
    endpoint_valid = bool(np.all(np.isfinite(endpoints)))
    minimum_endpoint = float(np.min(endpoints)) if endpoint_valid else None
    adjacent_drops = all_equal[:-1] - all_equal[1:]
    reversal_valid = bool(np.all(np.isfinite(adjacent_drops)))
    largest_reversal = (
        float(max(0.0, np.max(adjacent_drops)))
        if reversal_valid
        else None
    )
    endpoint_passed = bool(
        minimum_endpoint is not None
        and minimum_endpoint >= MINIMUM_ENDPOINT_RESPONSE
    )
    monotonic_passed = bool(
        largest_reversal is not None
        and largest_reversal <= MAXIMUM_ADJACENT_REVERSAL
    )
    checks = {
        "all_outputs_finite_and_in_unit_interval": {
            "actual": outputs_valid,
            "required": True,
            "passed": outputs_valid,
        },
        "minimum_all_one_minus_all_zero_beta_mean_price": {
            "actual": minimum_endpoint,
            "operator": ">=",
            "required": MINIMUM_ENDPOINT_RESPONSE,
            "passed": endpoint_passed,
        },
        "largest_adjacent_threshold_price_reversal": {
            "actual": largest_reversal,
            "operator": "<=",
            "required": MAXIMUM_ADJACENT_REVERSAL,
            "passed": monotonic_passed,
        },
    }
    return {
        "name": "clean_e1_seller_conditioning_warmup_gate_v1",
        "predeclared": True,
        "passed": bool(all(item["passed"] for item in checks.values())),
        "checks": checks,
        "per_event_all_one_minus_all_zero_beta_mean_price": [
            _json_number(value) for value in endpoints
        ],
        "per_event_largest_adjacent_reversal": [
            _json_number(max(0.0, float(np.max(adjacent_drops[:, event]))))
            if np.all(np.isfinite(adjacent_drops[:, event]))
            else None
            for event in range(NUM_TRADE_EVENTS)
        ],
    }


def collect_conditioning_report(model):
    """Evaluate all predeclared counterfactual trade observations."""

    all_equal = np.empty(
        (len(ALL_EQUAL_THRESHOLDS), NUM_TRADE_EVENTS), dtype=np.float64
    )
    all_equal_rows = []
    for threshold_row, threshold in enumerate(ALL_EQUAL_THRESHOLDS):
        context = np.full(NUM_TRADE_EVENTS, threshold, dtype=np.float32)
        event_rows = []
        for event_index in range(NUM_TRADE_EVENTS):
            values = canonical_trade_observation(
                model.policy,
                event_index=event_index,
                thresholds=context,
            )
            price = economic_beta_mean(model, values)
            all_equal[threshold_row, event_index] = price
            event_rows.append({
                "event_index": event_index,
                "actor_state": values[ACTOR_STATE].tolist(),
                "beta_mean_price": _json_number(price),
            })
        all_equal_rows.append({
            "threshold": threshold,
            "opponent_commitment": context.tolist(),
            "event_beta_mean_prices": [
                _json_number(value) for value in all_equal[threshold_row]
            ],
            "events": event_rows,
        })

    coordinate_low = np.empty(NUM_TRADE_EVENTS, dtype=np.float64)
    coordinate_high = np.empty(NUM_TRADE_EVENTS, dtype=np.float64)
    coordinate_rows = []
    for event_index in range(NUM_TRADE_EVENTS):
        low_context = np.full(
            NUM_TRADE_EVENTS,
            COORDINATE_BASELINE_THRESHOLD,
            dtype=np.float32,
        )
        high_context = np.array(low_context, copy=True)
        low_context[event_index] = 0.0
        high_context[event_index] = 1.0
        low_values = canonical_trade_observation(
            model.policy, event_index=event_index, thresholds=low_context
        )
        high_values = canonical_trade_observation(
            model.policy, event_index=event_index, thresholds=high_context
        )
        low_price = economic_beta_mean(model, low_values)
        high_price = economic_beta_mean(model, high_values)
        coordinate_low[event_index] = low_price
        coordinate_high[event_index] = high_price
        sensitivity = high_price - low_price
        coordinate_rows.append({
            "event_index": event_index,
            "other_coordinate_threshold": COORDINATE_BASELINE_THRESHOLD,
            "low_context": low_context.tolist(),
            "high_context": high_context.tolist(),
            "low_actor_state": low_values[ACTOR_STATE].tolist(),
            "high_actor_state": high_values[ACTOR_STATE].tolist(),
            "low_beta_mean_price": _json_number(low_price),
            "high_beta_mean_price": _json_number(high_price),
            "sensitivity": _json_number(sensitivity),
        })

    endpoint_response = all_equal[-1] - all_equal[0]
    coordinate_sensitivity = coordinate_high - coordinate_low
    return {
        "protocol": {
            "actor_state_dimension": ACTOR_STATE_DIM,
            "actor_state_order": [
                "ammo_fraction",
                "projectile_active",
                "normalized_time",
                "trade_mode",
                "event_one_hot[0:5]",
                "opponent_threshold_commitment[0:5]",
            ],
            "gameplay_horizon": CANONICAL_GAMEPLAY_HORIZON,
            "event_steps": list(CANONICAL_EVENT_STEPS),
            "ammo_fraction": CANONICAL_AMMO_FRACTION,
            "projectile_active": 0.0,
            "trade_mode": 1.0,
            "action_credit": [0.0, 1.0],
            "image": "zero dummy image in the checkpoint observation space",
            "action_mask": "all actions feasible",
            "critic_state": "zero; actor-invisible and unused by the probe",
        },
        "all_equal_thresholds": all_equal_rows,
        "low_to_high_response": {
            "definition": "all-one minus all-zero Beta-mean price",
            "per_event": [
                _json_number(value) for value in endpoint_response
            ],
            **_finite_summary(endpoint_response),
        },
        "current_coordinate_only_sensitivity": {
            "definition": (
                "current threshold 1 minus 0 with all other thresholds fixed "
                "at 0.5"
            ),
            "rows": coordinate_rows,
            **_finite_summary(coordinate_sensitivity),
        },
        "warmup_gate": warmup_diagnostic_gate(
            all_equal, coordinate_low, coordinate_high
        ),
    }


def run_probe_from_checkpoints(*, checkpoint, e0b_checkpoint, device="cpu"):
    """Load, validate, and probe immutable checkpoint bytes without ALE."""

    # Importing here keeps the pure probe helpers independent of the full E1
    # evaluator.  Its loaders inspect SB3 archives only; no environment is
    # constructed by this function.
    from replication.atari import evaluate_atari_meta_response_sb3 as evaluator

    candidate_path = evaluator.checkpoint_path(checkpoint)
    before = evaluator.checkpoint_sha256(candidate_path)
    e0b = evaluator.validate_e0b(e0b_checkpoint, device=device)
    model, metadata = evaluator.load_candidate(
        candidate_path,
        role=SELLER_ROLE,
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
            "deterministic_statistic": "Beta mean",
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
    parser.add_argument(
        "--output",
        help="optional immutable JSON report path; stdout is used otherwise",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="exit with status 1 after emitting JSON when the warmup gate fails",
    )
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
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite seller conditioning report: {output}"
            )
        write_json(output, report)
        print(json.dumps({
            "output": str(output),
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "warmup_gate_passed": report["warmup_gate"]["passed"],
        }, sort_keys=True, allow_nan=False))
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return int(bool(args.require_pass and not report["warmup_gate"]["passed"]))


if __name__ == "__main__":
    raise SystemExit(main())
