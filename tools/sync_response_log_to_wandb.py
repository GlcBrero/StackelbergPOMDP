"""Sync StackelbergPOMDP response diagnostics from a training log to W&B."""

import argparse
import os
import re
import time
from pathlib import Path

import wandb


RESPONSE_RE = re.compile(
    r"^\[response\] steps=(?P<steps>\d+) episode=(?P<episode>\d+) "
    r"bcce_gap=(?P<bcce_gap>[-+0-9.eE]+) "
    r"(?P<reward_key>expected_leader_reward|bcce_leader_reward)="
    r"(?P<bcce_leader_reward>[-+0-9.eE]+) "
    r"snapshots=(?P<snapshots>\d+)"
    r"(?: stop_reason=(?P<stop_reason>\S+))?"
)
TRAIN_RE = re.compile(
    r"^\[train\] steps=(?P<steps>\d+)/(?P<total_steps>\d+) "
    r"\((?P<percent>[-+0-9.eE]+)%\) fps=(?P<fps>[-+0-9.eE]+)"
)
REWARD_RE = re.compile(
    r"^\[reward\] steps=(?P<steps>\d+) episode=(?P<episode>\d+) "
    r"reward_phase_avg=(?P<reward_phase_avg>[-+0-9.eE]+) "
    r"reward_phase_sum=(?P<reward_phase_sum>[-+0-9.eE]+) "
    r"(?P<count_key>reward_phase_steps|reward_generated_steps)="
    r"(?P<reward_generated_steps>\d+)"
)
RESPONSE_DETAIL_RE = re.compile(
    r"^\[response_detail\] steps=(?P<steps>\d+) episode=(?P<episode>\d+) "
    r"records=(?P<records>\d+) "
    r"type_profiles=(?P<type_profiles>\d+) "
    r"mean_dominant_action_share=(?P<mean_dominant_action_share>[-+0-9.eE]+) "
    r"min_dominant_action_share=(?P<min_dominant_action_share>[-+0-9.eE]+) "
    r"mean_normalized_entropy=(?P<mean_normalized_entropy>[-+0-9.eE]+)"
)
ZERO_WELFARE_RE = re.compile(
    r"^\[zero_welfare_equilibrium\] steps=(?P<steps>\d+) episode=(?P<episode>\d+) "
    r"bcce_gap=(?P<bcce_gap>[-+0-9.eE]+) "
    r"bcce_leader_reward=(?P<bcce_leader_reward>[-+0-9.eE]+) "
    r"snapshots=(?P<snapshots>\d+)"
)


def parse_line(line, state=None, minimal=False):
    state = state if state is not None else {}
    match = RESPONSE_RE.match(line)
    if match:
        stop_reason = match.group("stop_reason")
        stop_bcce_threshold = stop_reason == "bcce_threshold"
        state["latest_bcce_gap"] = float(match.group("bcce_gap"))
        state["latest_response_stop_reason"] = stop_reason
        state["latest_response_stop_bcce_threshold"] = stop_bcce_threshold
        state["latest_response_stop_max_budget"] = stop_reason == "max_response_episodes"
        if minimal:
            return None, None
        values = {
            "bcce_gap": state["latest_bcce_gap"],
        }
        if not minimal:
            values["bcce_leader_reward"] = float(match.group("bcce_leader_reward"))
            values["response_snapshots"] = int(match.group("snapshots"))
        if stop_reason:
            values["response_stop_max_budget"] = float(stop_reason == "max_response_episodes")
            if not minimal:
                values["response_stop_bcce_threshold"] = float(stop_bcce_threshold)
        return int(match.group("steps")), values

    match = TRAIN_RE.match(line)
    if match:
        values = {
            "fps": float(match.group("fps")),
        }
        return int(match.group("steps")), values

    match = REWARD_RE.match(line)
    if match:
        reward_phase_sum = float(match.group("reward_phase_sum"))
        best_reward = max(
            reward_phase_sum,
            state.get("best_clean_reward_phase_sum", float("-inf")),
        )
        state["best_clean_reward_phase_sum"] = best_reward
        is_bcce_certified = bool(state.get("latest_response_stop_bcce_threshold", False))
        best_bcce_reward = state.get("best_bcce_clean_reward_phase_sum", float("-inf"))
        if is_bcce_certified:
            best_bcce_reward = max(reward_phase_sum, best_bcce_reward)
            state["best_bcce_clean_reward_phase_sum"] = best_bcce_reward
        values = {
            "clean_reward_phase_sum": reward_phase_sum,
            "zero_welfare_loss_and_bcce": float(
                abs(reward_phase_sum) < 1e-9 and is_bcce_certified
            ),
        }
        if minimal and "latest_bcce_gap" in state:
            values["bcce_gap"] = state["latest_bcce_gap"]
            values["response_stop_max_budget"] = float(
                state.get("latest_response_stop_max_budget", False)
            )
        if not minimal:
            values.update({
                "clean_reward_phase_avg": float(match.group("reward_phase_avg")),
                "welfare_loss": -reward_phase_sum,
                "zero_welfare_loss": float(abs(reward_phase_sum) < 1e-9),
                "near_zero_welfare_loss": float(reward_phase_sum >= -0.01),
                "best_clean_reward_phase_sum": best_reward,
                "bcce_certified_reward_phase": float(is_bcce_certified),
            })
        if best_bcce_reward != float("-inf"):
            values["best_bcce_clean_reward_phase_sum"] = best_bcce_reward
            if not minimal:
                values["best_bcce_welfare_loss"] = -best_bcce_reward
        return int(match.group("steps")), values

    match = RESPONSE_DETAIL_RE.match(line)
    if match:
        if minimal:
            return None, None
        values = {
            "response_records": int(match.group("records")),
            "response_type_profiles": int(match.group("type_profiles")),
            "mean_dominant_action_share": float(match.group("mean_dominant_action_share")),
            "min_dominant_action_share": float(match.group("min_dominant_action_share")),
            "mean_normalized_entropy": float(match.group("mean_normalized_entropy")),
        }
        return int(match.group("steps")), values

    match = ZERO_WELFARE_RE.match(line)
    if match:
        values = {
            "zero_welfare_equilibrium_found": 1.0,
            "zero_welfare_bcce_gap": float(match.group("bcce_gap")),
        }
        if not minimal:
            values.update({
                "zero_welfare_bcce_leader_reward": float(match.group("bcce_leader_reward")),
                "zero_welfare_snapshots": int(match.group("snapshots")),
                "zero_welfare_episode": int(match.group("episode")),
            })
        return int(match.group("steps")), values

    return None, None


def sync_existing(path, state, minimal):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            step, values = parse_line(line.rstrip(), state, minimal)
            if values is not None:
                wandb.log(values, step=step)


def follow(path, poll_seconds, state, minimal):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if not line:
                time.sleep(poll_seconds)
                continue
            step, values = parse_line(line.rstrip(), state, minimal)
            if values is not None:
                wandb.log(values, step=step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--project", default="StackPOMDP")
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", default=None)
    parser.add_argument("--id", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--start_at_end", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--poll_seconds", type=float, default=5.0)
    args = parser.parse_args()

    run = wandb.init(
        project=args.project,
        name=args.name,
        group=args.group,
        id=args.id,
        resume=args.resume,
        config={"source_log": str(args.log)},
    )
    try:
        state = {}
        if not args.start_at_end:
            sync_existing(args.log, state, args.minimal)
        if args.follow:
            follow(args.log, args.poll_seconds, state, args.minimal)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
