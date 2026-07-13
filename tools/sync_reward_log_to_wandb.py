"""Backfill and follow reward-only StackelbergPOMDP metrics in W&B."""

import argparse
import os
import re
import time
from pathlib import Path

import wandb


REWARD_RE = re.compile(
    r"^\[reward\] steps=(?P<global_step>\d+)\s+.*?"
    r"reward_phase_avg=(?P<reward>[-+0-9.eE]+)(?:\s|$)"
)
EVAL_RE = re.compile(
    r"^\[eval\] steps=(?P<global_step>\d+)\s+"
    r"reward_phase_avg=(?P<reward>[-+0-9.eE]+)(?:\s|$)"
)


def parse_reward(line, source="reward"):
    pattern = REWARD_RE if source == "reward" else EVAL_RE
    match = pattern.match(line)
    if match is None:
        return None
    return int(match.group("global_step")), float(match.group("reward"))


def log_reward(global_step, reward):
    wandb.log({"global_step": global_step, "reward": reward})


def sync_existing(path, source):
    last_step = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = parse_reward(line, source)
            if parsed is None:
                continue
            global_step, reward = parsed
            log_reward(global_step, reward)
            last_step = global_step
    return last_step


def follow(path, poll_seconds, stop_at_step, source):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if not line:
                time.sleep(poll_seconds)
                continue
            parsed = parse_reward(line, source)
            if parsed is None:
                continue
            global_step, reward = parsed
            log_reward(global_step, reward)
            if stop_at_step is not None and global_step >= stop_at_step:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--project", default="StackPOMDP")
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", default=None)
    parser.add_argument("--id", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--source", choices=("reward", "eval"), default="reward")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--start-at-end", action="store_true")
    parser.add_argument("--stop-at-step", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    run = wandb.init(
        project=args.project,
        name=args.name,
        group=args.group,
        id=args.id,
        resume=args.resume,
        config={
            "source_log": str(args.log),
            "source": args.source,
            "reward_only_bridge": True,
        },
    )
    wandb.define_metric("global_step")
    wandb.define_metric("reward", step_metric="global_step")
    try:
        last_step = None
        if not args.start_at_end:
            last_step = sync_existing(args.log, args.source)
        if (
            args.follow
            and not (
                args.stop_at_step is not None
                and last_step is not None
                and last_step >= args.stop_at_step
            )
        ):
            follow(args.log, args.poll_seconds, args.stop_at_step, args.source)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
