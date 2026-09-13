"""Command-line entry point for screening and confirming Atari StackPOMDP leaders."""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS
from stackelberg_pomdp.atari.sampling import ExactFiveEventSchedule
from stackelberg_pomdp.checkpoints.files import rollback_new_selected_alias
from stackelberg_pomdp.envs.atari.bilateral import BUYER, SELLER
from stackelberg_pomdp.evaluation.atari.contracts import (
    CANONICAL_GAMEPLAY_HORIZON,
    CONFIRMATION_EPISODES,
    DEFAULT_OUTPUT_DIR,
    SCREEN_EPISODES,
    _ranges_overlap,
)
from stackelberg_pomdp.evaluation.atari.reporting import (
    _artifact_paths,
    _slug,
    write_selection_artifacts,
)
from stackelberg_pomdp.evaluation.atari.workflow import run_selection


def _parse_event_steps(raw):
    if raw is None:
        return None
    values = tuple(int(value.strip()) for value in str(raw).split(","))
    if len(values) != NUM_TRADE_EVENTS:
        raise ValueError("--fixed-event-steps requires five comma-separated steps")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-role", choices=(BUYER, SELLER), required=True)
    parser.add_argument("--response-checkpoint", required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="E2 step checkpoint; repeat for every candidate in the common screen",
    )
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--screen-episodes", type=int, default=SCREEN_EPISODES)
    parser.add_argument("--screen-seed-start", type=int, default=4_000_001)
    parser.add_argument(
        "--confirmation-episodes", type=int, default=CONFIRMATION_EPISODES
    )
    parser.add_argument("--confirmation-seed-start", type=int, default=5_000_001)
    parser.add_argument("--gameplay-horizon", type=int, default=200)
    parser.add_argument("--event-tail-steps", type=int, default=0)
    parser.add_argument("--fixed-event-steps")
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=100_000)
    parser.add_argument("--rom-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)

    if args.screen_episodes <= 0:
        parser.error("--screen-episodes must be positive")
    if args.confirmation_episodes <= 0:
        parser.error("--confirmation-episodes must be positive")
    if args.screen_episodes != SCREEN_EPISODES:
        parser.error(
            f"clean E2 selection requires exactly {SCREEN_EPISODES} "
            "common screening episodes"
        )
    if args.confirmation_episodes != CONFIRMATION_EPISODES:
        parser.error(
            f"clean E2 confirmation requires exactly {CONFIRMATION_EPISODES} "
            "episodes"
        )
    if args.screen_seed_start < 0 or args.confirmation_seed_start < 0:
        parser.error("evaluation seeds must be nonnegative")
    if _ranges_overlap(
            args.screen_seed_start, args.screen_episodes,
            args.confirmation_seed_start, args.confirmation_episodes,
    ):
        parser.error("screen and confirmation seed ranges must be disjoint")
    if args.gameplay_horizon < NUM_TRADE_EVENTS:
        parser.error("--gameplay-horizon must be at least five")
    if args.gameplay_horizon != CANONICAL_GAMEPLAY_HORIZON:
        parser.error(
            "clean E2 selection requires the canonical 200-step gameplay "
            "horizon (210 outer transitions)"
        )
    if not 0 <= args.event_tail_steps < args.gameplay_horizon:
        parser.error("--event-tail-steps must lie in [0, gameplay_horizon)")
    if args.gameplay_horizon - args.event_tail_steps < NUM_TRADE_EVENTS:
        parser.error("the E2 event window must contain at least five steps")
    if args.noop_max < 0:
        parser.error("--noop-max must be nonnegative")
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    try:
        args.fixed_event_steps = _parse_event_steps(args.fixed_event_steps)
        if args.fixed_event_steps is not None:
            ExactFiveEventSchedule(
                gameplay_horizon=args.gameplay_horizon,
                tail_steps=args.event_tail_steps,
                fixed_event_steps=args.fixed_event_steps,
            )
    except ValueError as error:
        parser.error(str(error))
    selected = Path(args.selected_checkpoint).expanduser()
    if selected.suffix != ".zip":
        selected = selected.with_suffix(".zip")
    args.selected_checkpoint = str(selected.parent.resolve() / selected.name)
    return args


def main(argv=None):
    args = parse_args(argv)
    if os.path.lexists(args.selected_checkpoint):
        raise FileExistsError(
            "refusing to overwrite selected checkpoint alias: "
            f"{args.selected_checkpoint}"
        )
    if args.run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_name = (
            f"e2_{args.leader_role}_selection_seed"
            f"{args.screen_seed_start}-"
            f"{args.screen_seed_start + args.screen_episodes - 1}_{stamp}"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    planned_stem = _slug(args.run_name)
    if not planned_stem:
        raise ValueError("selection run name cannot be empty")
    planned = _artifact_paths(
        output_dir, planned_stem, has_confirmation=True
    )
    collisions = [
        str(path) for path in planned.values() if os.path.lexists(path)
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing E2 selection artifacts: "
            + ", ".join(collisions)
        )
    report = run_selection(args)
    try:
        report = write_selection_artifacts(
            report, output_dir=args.output_dir, run_name=args.run_name
        )
    except BaseException:
        rollback_new_selected_alias(
            report, expected_path=args.selected_checkpoint
        )
        raise
    print({
        "passed": report["passed"],
        "selected_checkpoint": (
            None if report["selected_alias"] is None
            else report["selected_alias"]["selected_checkpoint_path"]
        ),
        "selected_checkpoint_sha256": report["selection"][
            "selected_checkpoint_sha256"
        ],
        "report": report["artifacts"]["report_json"],
    }, flush=True)
    return 0 if report["passed"] else 2


from stackelberg_pomdp.evaluation.atari.legacy import __getattr__


if __name__ == "__main__":
    raise SystemExit(main())
