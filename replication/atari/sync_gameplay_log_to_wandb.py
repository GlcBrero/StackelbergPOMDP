"""Backfill and follow a five-bullet gameplay training log in W&B.

This is used when an RL process must keep running but its original W&B client
can no longer publish.  It parses only the scientific time-series metrics and
deliberately excludes iteration and constant learning rate from the time
series.  Learning rate remains recorded once in the run configuration.
"""

import argparse
import os
import re
import time
from pathlib import Path


METRIC_LINE = re.compile(
    r"iter=\s*(?P<iteration>\d+)\s+"
    r"steps=\s*(?P<steps>\d+)\s+"
    r"reward=\s*(?P<reward>[-+\d.eE]+)\s+"
    r"length=\s*(?P<length>[-+\d.eE]+)\s+"
    r"shots=\s*(?P<shots>[-+\d.eE]+)\s+"
    r"ammo=\s*(?P<ammo>[-+\d.eE]+)\s+"
    r"reward_per_bullet=\s*(?P<reward_per_bullet>[-+\d.eE]+)"
)


def parse_metric(line):
    match = METRIC_LINE.search(line)
    if match is None:
        return None
    return int(match.group("steps")), {
        "episode_reward": float(match.group("reward")),
        "episode_length": float(match.group("length")),
        "shots_fired": float(match.group("shots")),
        "final_ammo": float(match.group("ammo")),
        "reward_per_bullet": float(match.group("reward_per_bullet")),
        "total_timesteps": int(match.group("steps")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--algorithm", required=True, choices=["PPO", "A3C"])
    parser.add_argument("--project", default="StackPOMDP")
    parser.add_argument("--entity", default="glcbrero")
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 2.5e-4 if args.algorithm == "PPO" else 1e-4

    import wandb

    os.environ.setdefault("WANDB_START_METHOD", "thread")
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        resume="allow",
        name=args.name,
        config={
            "seed": args.seed,
            "algorithm": args.algorithm,
            "learning_rate": args.learning_rate,
            "checkpoint_path": str(Path(args.checkpoint).expanduser().resolve()),
            "initial_bullets": 5,
            "no_replenishment": True,
            "clip_game_rewards": True,
            "evaluation_policy": "deterministic_argmax",
            "metric_source": "live_training_log_bridge",
        },
    )

    path = Path(args.log).expanduser().resolve()
    last_step = int(run.summary.get("total_timesteps", 0) or 0)
    with path.open("r", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                time.sleep(args.poll_seconds)
                continue
            parsed = parse_metric(line)
            if parsed is None:
                continue
            step, metrics = parsed
            if step <= last_step:
                continue
            run.log(metrics, step=step)
            last_step = step
            print(f"synced_step={step}", flush=True)


if __name__ == "__main__":
    main()
