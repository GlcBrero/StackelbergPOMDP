"""Aggregate Atari E2 uncertainty across independently trained policies.

The default contract is the paper's two-role, ten-seed cohort.  Explicit CLI
arguments also support the retained buyer-leader frozen-gameplay ablation
without duplicating the aggregation implementation.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics

from replication.atari.automation.compact_atari_e2_selection import (
    SCHEMA as COMPACT_SCHEMA,
    write_json_atomic,
)


ROLES = ("buyer", "seller")
EXPECTED_SEEDS = tuple(range(1, 11))
EXPECTED_SCREEN_SEEDS = (4_000_001, 4_000_020)
EXPECTED_CONFIRMATION_SEEDS = (5_000_001, 5_000_100)
EXPECTED_TIMESTEPS = 2_000_040


def _integer_range(value):
    """Parse either ``a-b`` (inclusive) or comma-separated integers."""

    value = str(value).strip()
    if "," not in value and "-" in value:
        start, end = (int(part) for part in value.split("-", 1))
        if end < start:
            raise argparse.ArgumentTypeError("range end must not precede start")
        return tuple(range(start, end + 1))
    result = tuple(int(part) for part in value.split(",") if part.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique integer values")
    return result


def read_compact(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema") != COMPACT_SCHEMA:
        raise ValueError(f"not an Atari E2 compact report: {path}")
    return value


def training_seed(report):
    confirmation = report.get("confirmation") or {}
    result = confirmation.get("result") or {}
    manifest = result.get("e2_provenance_manifest") or {}
    scientific = manifest.get("scientific_config") or {}
    return int(scientific.get("optimization", {}).get("seed"))


def scientific_identity_without_seed(report):
    confirmation = report["confirmation"]
    manifest = confirmation["result"]["e2_provenance_manifest"]
    identity = deepcopy(manifest["scientific_config"])
    identity["optimization"].pop("seed", None)
    identity["environment"].pop("seed", None)
    return {
        "scientific_config_without_seed": identity,
        "artifacts": manifest["artifacts"],
    }


def numeric_summary(rows, *, seeds=None):
    """Summarize one independent evaluation mean per trained policy."""

    seeds = EXPECTED_SEEDS if seeds is None else tuple(seeds)
    if len(rows) != len(seeds):
        raise ValueError("one metric row is required per training seed")
    common = set.intersection(*(
        {
            key for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value))
        }
        for row in rows
    ))
    result = {}
    for key in sorted(common):
        values = [float(row[key]) for row in rows]
        sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
        result[key] = {
            "n_policies": len(values),
            "mean": statistics.fmean(values),
            "sample_std_across_policies": sample_std,
            "sem_across_policies": sample_std / math.sqrt(len(values)),
            "values_by_training_seed": {
                str(seed): value for seed, value in zip(seeds, values)
            },
        }
    return result


def validate_role(
        role,
        reports_by_seed,
        *,
        expected_seeds=EXPECTED_SEEDS,
        expected_screen_seeds=EXPECTED_SCREEN_SEEDS,
        expected_confirmation_seeds=EXPECTED_CONFIRMATION_SEEDS,
        expected_timesteps=EXPECTED_TIMESTEPS,
        expected_candidates=6,
        expected_screen_episodes=20,
        expected_confirmation_episodes=100,
):
    expected_seeds = tuple(expected_seeds)
    if tuple(sorted(reports_by_seed)) != tuple(sorted(expected_seeds)):
        raise ValueError(
            f"{role} cohort must contain exactly seeds {expected_seeds}; got "
            f"{sorted(reports_by_seed)}"
        )
    identities = []
    summaries = []
    rows = []
    for seed in expected_seeds:
        report = reports_by_seed[seed]
        config = report.get("environment_config") or {}
        if config.get("leader_role") != role:
            raise ValueError(f"seed {seed} has wrong leader role")
        if (report["screen"]["seed_start"], report["screen"]["seed_end"]) != (
                tuple(expected_screen_seeds)
        ):
            raise ValueError(f"{role} seed {seed} has the wrong screen seeds")
        if report["screen"].get("episodes_per_checkpoint") != (
                expected_screen_episodes
        ):
            raise ValueError(f"{role} seed {seed} has the wrong screen size")
        if report["screen"].get("common_seed_schedule_check", {}).get(
                "passed"
        ) is not True:
            raise ValueError(
                f"{role} seed {seed} failed the common-screen-seed audit"
            )
        confirmation = report.get("confirmation")
        if confirmation is None:
            raise ValueError(f"{role} seed {seed} has no confirmed screen winner")
        if (confirmation["seed_start"], confirmation["seed_end"]) != (
                tuple(expected_confirmation_seeds)
        ):
            raise ValueError(
                f"{role} seed {seed} has the wrong confirmation seeds"
            )
        if confirmation.get("episodes") != expected_confirmation_episodes:
            raise ValueError(
                f"{role} seed {seed} has the wrong confirmation size"
            )
        if confirmation.get("disjoint_from_screen") is not True:
            raise ValueError(
                f"{role} seed {seed} reused screening seeds for confirmation"
            )
        result = confirmation["result"]
        if result.get("training_total_timesteps") != expected_timesteps:
            raise ValueError(f"{role} seed {seed} has wrong training length")
        if result.get("protocol", {}).get("passed") is not True:
            raise ValueError(f"{role} seed {seed} failed the E2 protocol audit")
        if len(report["screen"]["checkpoint_results"]) != expected_candidates:
            raise ValueError(
                f"{role} seed {seed} did not screen "
                f"{expected_candidates} candidates"
            )
        identities.append(scientific_identity_without_seed(report))
        summaries.append(result["summary"])
        rows.append({
            "training_seed": seed,
            "selector_passed": bool(report.get("passed")),
            "economic_gate_passed": bool(
                result.get("economic_gate", {}).get("passed")
            ),
            "selected_checkpoint_sha256": result.get("checkpoint_sha256"),
            "training_total_timesteps": result.get(
                "training_total_timesteps"
            ),
            "confirmation_summary": result.get("summary"),
            "compact_report": report["full_report"],
        })
    reference = identities[0]
    if any(identity != reference for identity in identities[1:]):
        raise ValueError(
            f"{role} policies do not share one scientific configuration and "
            "immutable E1 cohort after removing only the training seed"
        )
    return {
        "training_seeds": list(expected_seeds),
        "n_policies": len(expected_seeds),
        "selector_passes": sum(row["selector_passed"] for row in rows),
        "economic_gate_passes": sum(
            row["economic_gate_passed"] for row in rows
        ),
        "policy_level_rows": rows,
        "metrics": numeric_summary(summaries, seeds=expected_seeds),
        "scientific_identity_without_training_seed": reference,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--roles", type=lambda value: tuple(value.split(",")), default=ROLES
    )
    parser.add_argument("--seeds", type=_integer_range, default=EXPECTED_SEEDS)
    parser.add_argument(
        "--screen-seeds",
        type=_integer_range,
        default=tuple(range(EXPECTED_SCREEN_SEEDS[0], EXPECTED_SCREEN_SEEDS[1] + 1)),
    )
    parser.add_argument(
        "--confirmation-seeds",
        type=_integer_range,
        default=tuple(range(
            EXPECTED_CONFIRMATION_SEEDS[0], EXPECTED_CONFIRMATION_SEEDS[1] + 1
        )),
    )
    parser.add_argument(
        "--expected-timesteps", type=int, default=EXPECTED_TIMESTEPS
    )
    parser.add_argument("--expected-candidates", type=int, default=6)
    args = parser.parse_args(argv)
    if not args.roles or any(role not in ROLES for role in args.roles):
        parser.error("--roles must be a comma-separated subset of buyer,seller")
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = sorted(Path(args.input_dir).glob("**/*.compact.json"))
    reports = {role: {} for role in args.roles}
    for path in paths:
        report = read_compact(path)
        role = report.get("environment_config", {}).get("leader_role")
        if role not in reports:
            continue
        seed = training_seed(report)
        if seed in reports[role]:
            raise ValueError(f"duplicate {role} training seed {seed}")
        reports[role][seed] = report
    screen_range = (args.screen_seeds[0], args.screen_seeds[-1])
    confirmation_range = (
        args.confirmation_seeds[0], args.confirmation_seeds[-1]
    )
    result = {
        "schema": "stackelberg_pomdp.atari.e2_multiseed_aggregate.v1",
        "uncertainty_unit": (
            "one deterministic held-out mean per independently "
            "trained-and-screen-selected policy"
        ),
        "sem_definition": (
            "sample standard deviation of policy means divided by sqrt(n); "
            "evaluation episodes are not pooled"
        ),
        "common_random_numbers": {
            "screen_seed_range": list(screen_range),
            "confirmation_seed_range": list(confirmation_range),
        },
        "roles": {
            role: validate_role(
                role,
                reports[role],
                expected_seeds=args.seeds,
                expected_screen_seeds=screen_range,
                expected_confirmation_seeds=confirmation_range,
                expected_timesteps=args.expected_timesteps,
                expected_candidates=args.expected_candidates,
            )
            for role in args.roles
        },
    }
    write_json_atomic(args.output, result)
    print({
        role: {
            "n_policies": result["roles"][role]["n_policies"],
            "selector_passes": result["roles"][role]["selector_passes"],
            "leader_payoff": result["roles"][role]["metrics"].get(
                "mean_leader_payoff"
            ),
        }
        for role in args.roles
    }, flush=True)


if __name__ == "__main__":
    main()
