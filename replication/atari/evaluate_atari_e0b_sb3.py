"""Deterministically audit one or two clean PPO checkpoints in E0b.

The evaluator always runs the canonical five-transfer E0b environment.  A
comparison is paired: both checkpoints see the same ALE/no-op seeds and the
same event schedules.  Artifacts are written to a dedicated results directory
and never reuse the training ``*.evaluation.json`` filenames.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stackpomdp-matplotlib")
)

from stackelberg_pomdp.atari.training import (
    ScaledLearningRatePPO,
    evaluate_model,
    write_csv,
    write_json,
)
from stackelberg_pomdp.envs.atari.curriculum import (
    AtariCurriculumConfig,
    AtariCurriculumEnv,
)
from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS
from stackelberg_pomdp.atari.sampling import ExactFiveEventSchedule
from stackelberg_pomdp.policies.atari.composite import StackPOMDPAtariPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "replication/atari/results/e0b_evaluations"
)
PROTOCOL_ATOL = 1.0e-9


@dataclass(frozen=True)
class EvaluationScenario:
    """One random or fixed E0b event-schedule condition."""

    name: str
    fixed_event_steps: tuple = None


def _parse_event_steps(raw):
    values = tuple(int(value.strip()) for value in str(raw).split(","))
    if len(values) != NUM_TRADE_EVENTS:
        raise ValueError(
            "--fixed-event-steps requires five comma-separated steps"
        )
    return values


def _checkpoint_path(raw):
    path = Path(raw).expanduser()
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path


def _scenario_name(fixed_event_steps):
    if fixed_event_steps is None:
        return "random"
    return "fixed_" + "_".join(str(value) for value in fixed_event_steps)


def evaluation_scenarios(fixed_schedules=()):
    """Return the random condition followed by unique fixed schedules."""

    scenarios = [EvaluationScenario("random")]
    seen = set()
    for schedule in fixed_schedules:
        values = tuple(int(value) for value in schedule)
        if values in seen:
            raise ValueError(f"duplicate fixed event schedule: {values}")
        seen.add(values)
        scenarios.append(EvaluationScenario(_scenario_name(values), values))
    return tuple(scenarios)


def make_e0b_env(args, *, seed, fixed_event_steps=None):
    """Construct the canonical E0b environment for one evaluation seed."""

    return AtariCurriculumEnv(AtariCurriculumConfig(
        stage="e0b",
        seed=int(seed),
        gameplay_horizon=int(args.gameplay_horizon),
        event_tail_steps=int(args.event_tail_steps),
        noop_max=int(args.noop_max),
        frame_skip=4,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        max_frames=int(args.max_frames),
        rom_path=args.rom_path,
        fixed_event_steps=fixed_event_steps,
    ))


def load_clean_checkpoint(path, *, device="cpu"):
    """Load a clean composite PPO checkpoint without attaching a train env."""

    model = ScaledLearningRatePPO.load(path, device=device)
    if not isinstance(model.policy, StackPOMDPAtariPolicy):
        raise TypeError(
            f"{path} does not contain the clean composite Atari policy"
        )
    if model.policy.economic_role != "gameplay":
        raise ValueError(
            f"{path} is not an E0 gameplay checkpoint "
            f"(economic_role={model.policy.economic_role!r})"
        )
    if model.policy.economic_input_mode != "full":
        raise ValueError(
            f"{path} does not use the full E0 actor state "
            f"(economic_input_mode={model.policy.economic_input_mode!r})"
        )
    return model


def checkpoint_sha256(path):
    """Hash a checkpoint so every evaluation binds to exact model bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_number(value):
    if isinstance(value, (bool, int, float, np.number)):
        return float(value)
    return None


def outcome_summary(rows):
    """Summarize gameplay quality while retaining horizon-censored ammo."""

    rows = list(rows)
    rewards = [float(row["game_reward"]) for row in rows]
    shots = [float(row["shots_fired"]) for row in rows]
    ammo = [float(row["final_ammo"]) for row in rows]
    fifth_steps = [int(row["fifth_event_step"]) for row in rows]
    if not rows:
        return {
            "episodes": 0,
            "mean_game_reward": None,
            "mean_shots_fired": None,
            "mean_final_ammo": None,
        }
    return {
        "episodes": len(rows),
        "mean_game_reward": float(np.mean(rewards)),
        "std_game_reward": float(np.std(rewards)),
        "min_game_reward": float(np.min(rewards)),
        "max_game_reward": float(np.max(rewards)),
        "reward_five_rate": float(np.mean(np.asarray(rewards) >= 5.0)),
        "mean_shots_fired": float(np.mean(shots)),
        "min_shots_fired": float(np.min(shots)),
        "fired_all_rate": float(
            np.mean(np.asarray(shots) >= NUM_TRADE_EVENTS)
        ),
        "mean_final_ammo": float(np.mean(ammo)),
        "max_final_ammo": float(np.max(ammo)),
        "mean_fifth_event_step": float(np.mean(fifth_steps)),
        "min_fifth_event_step": int(np.min(fifth_steps)),
        "max_fifth_event_step": int(np.max(fifth_steps)),
    }


def _protocol_violation(row, field, expected, actual):
    return {
        "evaluation_episode": int(row.get("evaluation_episode", -1)),
        "evaluation_seed": int(row.get("evaluation_seed", -1)),
        "field": str(field),
        "expected": expected,
        "actual": actual,
    }


def audit_e0b_protocol(
        rows,
        *,
        required_episodes,
        gameplay_horizon,
        event_tail_steps=0,
        fixed_event_steps=None,
):
    """Check every E0b transition-count, transfer, and payoff identity."""

    rows = list(rows)
    horizon = int(gameplay_horizon)
    expected_outer = horizon + NUM_TRADE_EVENTS
    event_stop = horizon - int(event_tail_steps)
    fixed = (
        None
        if fixed_event_steps is None
        else tuple(int(value) for value in fixed_event_steps)
    )
    violations = []

    if len(rows) != int(required_episodes):
        violations.append({
            "evaluation_episode": -1,
            "evaluation_seed": -1,
            "field": "episode_count",
            "expected": int(required_episodes),
            "actual": len(rows),
        })

    exact_fields = {
        "stage": "e0b",
        "evaluation_steps": expected_outer,
        "gameplay_steps": horizon,
        "trade_transitions": NUM_TRADE_EVENTS,
        "outer_transition_count": expected_outer,
        "free_transfers": NUM_TRADE_EVENTS,
        "emulator_step_calls": horizon,
    }
    zero_fields = ("payments", "bullet_accounting_error")

    for row in rows:
        for field, expected in exact_fields.items():
            actual = row.get(field)
            if actual != expected:
                violations.append(
                    _protocol_violation(row, field, expected, actual)
                )
        for field in zero_fields:
            actual = _as_number(row.get(field))
            if actual is None or abs(actual) > PROTOCOL_ATOL:
                violations.append(
                    _protocol_violation(row, field, 0.0, row.get(field))
                )

        reward = _as_number(row.get("game_reward"))
        evaluation_return = _as_number(row.get("evaluation_return"))
        episode_reward = _as_number(row.get("episode_reward"))
        if (
                reward is None
                or evaluation_return is None
                or abs(evaluation_return - reward) > PROTOCOL_ATOL
        ):
            violations.append(_protocol_violation(
                row, "evaluation_return == game_reward", reward,
                evaluation_return,
            ))
        if (
                reward is None
                or episode_reward is None
                or abs(episode_reward - reward) > PROTOCOL_ATOL
        ):
            violations.append(_protocol_violation(
                row, "episode_reward == game_reward", reward, episode_reward,
            ))

        shots = _as_number(row.get("shots_fired"))
        ammo = _as_number(row.get("final_ammo"))
        if (
                shots is None
                or ammo is None
                or shots < 0.0
                or ammo < 0.0
                or abs(shots + ammo - NUM_TRADE_EVENTS) > PROTOCOL_ATOL
        ):
            violations.append(_protocol_violation(
                row,
                "shots_fired + final_ammo",
                NUM_TRADE_EVENTS,
                None if shots is None or ammo is None else shots + ammo,
            ))

        raw_steps = row.get("event_steps")
        try:
            event_steps = tuple(int(value) for value in raw_steps)
        except (TypeError, ValueError):
            event_steps = ()
        valid_schedule = bool(
            len(event_steps) == NUM_TRADE_EVENTS
            and tuple(sorted(set(event_steps))) == event_steps
            and event_steps[0] >= 0
            and event_steps[-1] < event_stop
        )
        if not valid_schedule:
            violations.append(_protocol_violation(
                row,
                "event_steps",
                f"five sorted distinct steps in [0, {event_stop})",
                raw_steps,
            ))
        elif fixed is not None and event_steps != fixed:
            violations.append(_protocol_violation(
                row, "event_steps", list(fixed), list(event_steps)
            ))

    return {
        "passed": not violations,
        "episodes": len(rows),
        "required_episodes": int(required_episodes),
        "expected_gameplay_steps": horizon,
        "expected_trade_transitions": NUM_TRADE_EVENTS,
        "expected_outer_transitions": expected_outer,
        "max_abs_bullet_accounting_error": max(
            (
                abs(float(row["bullet_accounting_error"]))
                for row in rows
                if _as_number(row.get("bullet_accounting_error")) is not None
            ),
            default=None,
        ),
        "max_abs_payment": max(
            (
                abs(float(row["payments"]))
                for row in rows
                if _as_number(row.get("payments")) is not None
            ),
            default=None,
        ),
        "max_abs_return_error": max(
            (
                abs(float(row["evaluation_return"]) - float(row["game_reward"]))
                for row in rows
                if _as_number(row.get("evaluation_return")) is not None
                and _as_number(row.get("game_reward")) is not None
            ),
            default=None,
        ),
        "violations": violations,
    }


def _timing_band(fifth_event_step, gameplay_horizon):
    fraction = float(fifth_event_step) / float(gameplay_horizon)
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def timing_strata(rows, *, gameplay_horizon):
    """Report outcomes for each exact fifth-event step and horizon third."""

    rows = list(rows)
    exact = defaultdict(list)
    bands = defaultdict(list)
    for row in rows:
        step = int(row["fifth_event_step"])
        exact[step].append(row)
        bands[_timing_band(step, gameplay_horizon)].append(row)

    by_step = []
    for step in sorted(exact):
        selected = exact[step]
        by_step.append({
            "fifth_event_step": int(step),
            "fifth_event_fraction": float(step) / float(gameplay_horizon),
            **outcome_summary(selected),
        })

    by_band = []
    for name in ("early", "middle", "late"):
        selected = bands.get(name, [])
        summary = outcome_summary(selected)
        by_band.append({
            "timing_band": name,
            "fifth_event_fraction_low": {
                "early": 0.0,
                "middle": 1.0 / 3.0,
                "late": 2.0 / 3.0,
            }[name],
            "fifth_event_fraction_high": {
                "early": 1.0 / 3.0,
                "middle": 2.0 / 3.0,
                "late": 1.0,
            }[name],
            **summary,
        })
    return {"by_fifth_event_step": by_step, "by_timing_band": by_band}


def _policy_metadata(model):
    policy = model.policy
    return {
        "policy_class": type(policy).__name__,
        "economic_role": getattr(policy, "economic_role", None),
        "economic_input_mode": getattr(policy, "economic_input_mode", None),
        "pretrained_lr_scale": getattr(policy, "pretrained_lr_scale", None),
        "training_total_timesteps": int(getattr(model, "num_timesteps", 0)),
    }


def evaluate_checkpoint(model, checkpoint, args, scenarios):
    """Evaluate one loaded model on all schedule scenarios."""

    checkpoint = Path(checkpoint).resolve()
    checkpoint_id = checkpoint.stem
    evaluations = []
    for scenario in scenarios:
        result = evaluate_model(
            model,
            lambda episode, scenario=scenario: make_e0b_env(
                args,
                seed=int(args.seed_start) + int(episode),
                fixed_event_steps=scenario.fixed_event_steps,
            ),
            episodes=args.episodes,
        )
        rows = []
        for source in result["episode_rows"]:
            row = dict(source)
            event_steps = tuple(int(value) for value in row.get("event_steps", ()))
            row.update({
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(checkpoint),
                "scenario": scenario.name,
                "evaluation_seed": (
                    int(args.seed_start) + int(row["evaluation_episode"])
                ),
                "event_steps": list(event_steps),
                "fifth_event_step": (
                    int(event_steps[-1]) if event_steps else -1
                ),
                "fifth_event_fraction": (
                    float(event_steps[-1]) / float(args.gameplay_horizon)
                    if event_steps else None
                ),
            })
            rows.append(row)
        protocol = audit_e0b_protocol(
            rows,
            required_episodes=args.episodes,
            gameplay_horizon=args.gameplay_horizon,
            event_tail_steps=args.event_tail_steps,
            fixed_event_steps=scenario.fixed_event_steps,
        )
        timing = timing_strata(rows, gameplay_horizon=args.gameplay_horizon)
        evaluations.append({
            "scenario": scenario.name,
            "fixed_event_steps": (
                None
                if scenario.fixed_event_steps is None
                else list(scenario.fixed_event_steps)
            ),
            "summary": outcome_summary(rows),
            "protocol": protocol,
            "timing": timing,
            "episode_rows": rows,
        })
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        **_policy_metadata(model),
        "scenarios": evaluations,
    }


def _paired_summary(rows):
    rows = list(rows)
    if not rows:
        return {"paired_episodes": 0}
    reward_deltas = np.asarray(
        [row["delta_game_reward_b_minus_a"] for row in rows], dtype=float
    )
    shot_deltas = np.asarray(
        [row["delta_shots_fired_b_minus_a"] for row in rows], dtype=float
    )
    ammo_deltas = np.asarray(
        [row["delta_final_ammo_b_minus_a"] for row in rows], dtype=float
    )
    return {
        "paired_episodes": len(rows),
        "mean_delta_game_reward_b_minus_a": float(np.mean(reward_deltas)),
        "mean_delta_shots_fired_b_minus_a": float(np.mean(shot_deltas)),
        "mean_delta_final_ammo_b_minus_a": float(np.mean(ammo_deltas)),
        "checkpoint_b_reward_win_rate": float(np.mean(reward_deltas > 0.0)),
        "checkpoint_a_reward_win_rate": float(np.mean(reward_deltas < 0.0)),
        "reward_tie_rate": float(np.mean(reward_deltas == 0.0)),
    }


def paired_comparison(left, right):
    """Compare two checkpoint results row-by-row on identical conditions."""

    def indexed(result):
        return {
            (scenario["scenario"], int(row["evaluation_seed"])): row
            for scenario in result["scenarios"]
            for row in scenario["episode_rows"]
        }

    left_rows = indexed(left)
    right_rows = indexed(right)
    if set(left_rows) != set(right_rows):
        raise RuntimeError("checkpoint evaluations do not share paired seed keys")

    pairs = []
    for key in sorted(left_rows):
        first = left_rows[key]
        second = right_rows[key]
        if first["event_steps"] != second["event_steps"]:
            raise RuntimeError(
                "paired checkpoints received different event schedules for "
                f"{key}: {first['event_steps']} != {second['event_steps']}"
            )
        pairs.append({
            "scenario": key[0],
            "evaluation_seed": key[1],
            "event_steps": list(first["event_steps"]),
            "fifth_event_step": int(first["fifth_event_step"]),
            "checkpoint_a": left["checkpoint_path"],
            "checkpoint_b": right["checkpoint_path"],
            "game_reward_a": float(first["game_reward"]),
            "game_reward_b": float(second["game_reward"]),
            "delta_game_reward_b_minus_a": (
                float(second["game_reward"]) - float(first["game_reward"])
            ),
            "shots_fired_a": float(first["shots_fired"]),
            "shots_fired_b": float(second["shots_fired"]),
            "delta_shots_fired_b_minus_a": (
                float(second["shots_fired"]) - float(first["shots_fired"])
            ),
            "final_ammo_a": float(first["final_ammo"]),
            "final_ammo_b": float(second["final_ammo"]),
            "delta_final_ammo_b_minus_a": (
                float(second["final_ammo"]) - float(first["final_ammo"])
            ),
        })

    by_scenario = []
    for scenario in sorted({row["scenario"] for row in pairs}):
        selected = [row for row in pairs if row["scenario"] == scenario]
        by_scenario.append({"scenario": scenario, **_paired_summary(selected)})
    return {
        "checkpoint_a": left["checkpoint_path"],
        "checkpoint_b": right["checkpoint_path"],
        "same_seed_and_schedule_pairs": True,
        "summary": _paired_summary(pairs),
        "summary_by_scenario": by_scenario,
        "episode_rows": pairs,
    }


def run_evaluations(args):
    """Load checkpoints and return the complete in-memory E0b audit."""

    checkpoints = tuple(_checkpoint_path(path) for path in args.checkpoint)
    scenarios = evaluation_scenarios(args.fixed_event_steps)
    results = []
    for checkpoint in checkpoints:
        model = load_clean_checkpoint(checkpoint, device=args.device)
        try:
            results.append(evaluate_checkpoint(
                model, checkpoint, args, scenarios
            ))
        finally:
            del model
    comparison = (
        paired_comparison(results[0], results[1])
        if len(results) == 2 else None
    )
    protocol_passed = all(
        scenario["protocol"]["passed"]
        for result in results
        for scenario in result["scenarios"]
    )
    return {
        "schema_version": 1,
        "evaluator": "clean_e0b_deterministic_v1",
        "deterministic_argmax": True,
        "config": {
            "episodes_per_scenario": int(args.episodes),
            "seed_start": int(args.seed_start),
            "seed_end": int(args.seed_start) + int(args.episodes) - 1,
            "gameplay_horizon": int(args.gameplay_horizon),
            "event_tail_steps": int(args.event_tail_steps),
            "noop_max": int(args.noop_max),
            "frame_skip": 4,
            "frame_stack": 4,
            "episodic_life": True,
            "clipped_game_rewards": True,
            "scenarios": [
                {
                    "name": scenario.name,
                    "fixed_event_steps": (
                        None
                        if scenario.fixed_event_steps is None
                        else list(scenario.fixed_event_steps)
                    ),
                }
                for scenario in scenarios
            ],
        },
        "all_protocol_checks_passed": bool(protocol_passed),
        "checkpoint_results": results,
        "paired_comparison": comparison,
    }


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")


def default_run_name(report):
    checkpoint_ids = [
        result["checkpoint_id"] for result in report["checkpoint_results"]
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = report["config"]
    return _slug(
        "e0b_audit_"
        + "_vs_".join(checkpoint_ids)
        + f"_seed{config['seed_start']}-{config['seed_end']}_{stamp}"
    )


def _csv_row(row):
    result = dict(row)
    if isinstance(result.get("event_steps"), (list, tuple)):
        result["event_steps"] = json.dumps(result["event_steps"], separators=(",", ":"))
    return result


def write_evaluation_artifacts(report, *, output_dir, run_name=None):
    """Write dedicated audit artifacts and refuse every filename collision."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(run_name or default_run_name(report))
    if not stem:
        raise ValueError("evaluation run name cannot be empty")
    paths = {
        "report_json": output_dir / f"{stem}.json",
        "episode_rows_csv": output_dir / f"{stem}.episodes.csv",
        "timing_rows_csv": output_dir / f"{stem}.timing.csv",
    }
    if report.get("paired_comparison") is not None:
        paths.update({
            "paired_rows_csv": output_dir / f"{stem}.paired.csv",
            "paired_summary_csv": output_dir / f"{stem}.paired_summary.csv",
        })
    training_evaluations = {
        Path(result["checkpoint_path"]).resolve().with_name(
            f"{Path(result['checkpoint_path']).stem}.evaluation.json"
        )
        for result in report["checkpoint_results"]
    }
    if paths["report_json"] in training_evaluations:
        raise ValueError(
            "the audit report path cannot reuse a training evaluation filename"
        )
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing evaluation artifacts: "
            + ", ".join(collisions)
        )

    episode_rows = []
    timing_rows = []
    for checkpoint in report["checkpoint_results"]:
        for scenario in checkpoint["scenarios"]:
            episode_rows.extend(_csv_row(row) for row in scenario["episode_rows"])
            for row in scenario["timing"]["by_fifth_event_step"]:
                timing_rows.append({
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_path": checkpoint["checkpoint_path"],
                    "scenario": scenario["scenario"],
                    "stratification": "exact_fifth_event_step",
                    **row,
                })
            for row in scenario["timing"]["by_timing_band"]:
                timing_rows.append({
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_path": checkpoint["checkpoint_path"],
                    "scenario": scenario["scenario"],
                    "stratification": "horizon_third",
                    **row,
                })

    artifact_paths = {key: str(path) for key, path in paths.items()}
    report = {**report, "artifacts": artifact_paths}
    write_csv(paths["episode_rows_csv"], episode_rows)
    write_csv(paths["timing_rows_csv"], timing_rows)
    if report.get("paired_comparison") is not None:
        comparison = report["paired_comparison"]
        write_csv(
            paths["paired_rows_csv"],
            [_csv_row(row) for row in comparison["episode_rows"]],
        )
        write_csv(
            paths["paired_summary_csv"],
            comparison["summary_by_scenario"],
        )
    write_json(paths["report_json"], report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="clean PPO checkpoint; repeat exactly once for a paired comparison",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=3_000_001)
    parser.add_argument("--gameplay-horizon", type=int, default=200)
    parser.add_argument("--event-tail-steps", type=int, default=0)
    parser.add_argument(
        "--fixed-event-steps",
        action="append",
        default=[],
        help=(
            "optional comma-separated five-event schedule; repeat for multiple "
            "fixed conditions"
        ),
    )
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=100_000)
    parser.add_argument("--rom-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)

    if not 1 <= len(args.checkpoint) <= 2:
        parser.error("provide one checkpoint, or two for a paired comparison")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.gameplay_horizon < NUM_TRADE_EVENTS:
        parser.error("--gameplay-horizon must be at least five")
    if not 0 <= args.event_tail_steps < args.gameplay_horizon:
        parser.error(
            "--event-tail-steps must lie in [0, gameplay_horizon)"
        )
    if args.gameplay_horizon - args.event_tail_steps < NUM_TRADE_EVENTS:
        parser.error("the E0b event window must contain at least five steps")
    if args.noop_max < 0:
        parser.error("--noop-max must be nonnegative")
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    try:
        parsed_schedules = tuple(
            _parse_event_steps(raw) for raw in args.fixed_event_steps
        )
        for schedule in parsed_schedules:
            ExactFiveEventSchedule(
                gameplay_horizon=args.gameplay_horizon,
                tail_steps=args.event_tail_steps,
                fixed_event_steps=schedule,
            )
        evaluation_scenarios(parsed_schedules)
    except ValueError as error:
        parser.error(str(error))
    args.fixed_event_steps = parsed_schedules
    return args


def main(argv=None):
    args = parse_args(argv)
    report = run_evaluations(args)
    report = write_evaluation_artifacts(
        report,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print({
        "all_protocol_checks_passed": report["all_protocol_checks_passed"],
        "report": report["artifacts"]["report_json"],
        "checkpoints": [
            {
                "checkpoint": result["checkpoint_path"],
                "scenarios": {
                    scenario["scenario"]: scenario["summary"]
                    for scenario in result["scenarios"]
                },
            }
            for result in report["checkpoint_results"]
        ],
    }, flush=True)
    return 0 if report["all_protocol_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
