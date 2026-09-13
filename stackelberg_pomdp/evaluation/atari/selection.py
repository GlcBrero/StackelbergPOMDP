"""Economic hypothesis checks and deterministic screen/confirmation selection rules."""

import numpy as np

from stackelberg_pomdp.atari.training import STANDARD_ACTOR_LOSS_MODE
from stackelberg_pomdp.evaluation.atari.contracts import (
    E2_INIT_CONCENTRATION,
    E2_INIT_MEAN,
    ECONOMIC_CONTROL_COMMITMENTS,
    ECONOMIC_GATE_HYPOTHESIS,
    ECONOMIC_GATE_MIN_FACTUAL_PAYOFF,
    ECONOMIC_GATE_MIN_MEAN_ADVANTAGE,
    ECONOMIC_GATE_MIN_MEDIAN_ADVANTAGE,
    ECONOMIC_GATE_MIN_TOTAL_SHOTS,
    ECONOMIC_GATE_MIN_WIN_RATE,
    PROTOCOL_ATOL,
    SELECTION_RULE,
)


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
