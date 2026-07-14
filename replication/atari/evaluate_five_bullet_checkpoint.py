"""Evaluate an exported five-bullet Atari checkpoint without starting Ray.

The evaluator uses the same scarce-ammo environment and fixed fresh-seed suite
as the training entrypoint.  Actions are deterministic argmax actions, making
the resulting JSON suitable for selecting a frozen buyer gameplay policy.
"""

import argparse
from collections import Counter
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from stackelberg_pomdp.atari_models import AmmoAwareNatureCNNTorch
from stackelberg_pomdp.gym_envs.envs.atari_envs import (
    AmmoAwareSinglePlayerAtariBulletEnv,
    SinglePlayerAtariBulletEnv,
)
from stackerlberg.train.atari_models import NatureCNNTorch


INITIAL_BULLETS = 5


def _env_config(seed, args):
    return {
        "seed": int(seed),
        "initial_bullets": INITIAL_BULLETS,
        "max_steps": args.max_steps,
        "noop_max": args.noop_max,
        "frame_skip": args.frame_skip,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
    }


def _load_model(checkpoint, args):
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    source = (
        payload.get("agent_1")
        or payload.get("agent_0")
        or payload.get("default_policy")
    )
    if source is None:
        raise ValueError(f"{checkpoint} has no usable policy weights: {list(payload)}")
    metadata = payload.get("metadata", {})
    ammo_aware = bool(metadata.get("ammo_aware", False))
    environment_class = (
        AmmoAwareSinglePlayerAtariBulletEnv
        if ammo_aware
        else SinglePlayerAtariBulletEnv
    )
    model_class = AmmoAwareNatureCNNTorch if ammo_aware else NatureCNNTorch

    probe = environment_class(_env_config(args.eval_seed, args))
    try:
        model = model_class(
            probe.observation_space,
            probe.action_space,
            probe.action_space.n,
            {
                "vf_share_layers": True,
                "custom_model_config": {
                    "ammo_hidden": int(metadata.get("ammo_hidden") or 32),
                },
            },
            "five_bullet_checkpoint_evaluation",
        )
    finally:
        probe.close()

    current = model.state_dict()
    patched = {}
    copied = 0
    for key, value in current.items():
        if key in source and tuple(source[key].shape) == tuple(value.shape):
            patched[key] = torch.as_tensor(source[key])
            copied += 1
        else:
            patched[key] = value
    if copied == 0:
        raise ValueError(f"no compatible model tensors found in {checkpoint}")
    model.load_state_dict(patched)
    model.eval()
    return model, metadata, copied


def evaluate(checkpoint, args):
    model, metadata, copied_tensors = _load_model(checkpoint, args)
    environment_class = (
        AmmoAwareSinglePlayerAtariBulletEnv
        if metadata.get("ammo_aware", False)
        else SinglePlayerAtariBulletEnv
    )
    episodes = []
    action_counts = Counter()
    for episode_idx in range(args.episodes):
        seed = args.eval_seed + episode_idx * 1009
        env = environment_class(_env_config(seed, args))
        observation = env.reset()
        done = False
        reward = 0.0
        length = 0
        last_info = {}
        while not done:
            if isinstance(observation, dict):
                model_observation = {
                    key: torch.as_tensor(value[None, ...])
                    for key, value in observation.items()
                }
            else:
                model_observation = torch.as_tensor(observation[None, ...])
            with torch.no_grad():
                logits, _ = model(
                    {"obs": model_observation}, [], None
                )
            action = int(torch.argmax(logits[0]).item())
            action_counts[action] += 1
            observation, step_reward, done, last_info = env.step(action)
            reward += float(step_reward)
            length += 1
        env.close()
        episodes.append(
            {
                "episode": episode_idx,
                "seed": seed,
                "reward": reward,
                "length": length,
                "shots_fired": int(last_info.get("shots_fired", 0)),
                "final_ammo": float(last_info.get("final_ammo", 0.0)),
                "reward_per_bullet": float(last_info.get("reward_per_bullet", 0.0)),
            }
        )
        print(
            f"episode={episode_idx:>2} reward={reward:.1f} length={length} "
            f"shots={episodes[-1]['shots_fired']} ammo={episodes[-1]['final_ammo']:.0f}",
            flush=True,
        )

    keys = ("reward", "length", "shots_fired", "final_ammo", "reward_per_bullet")
    summary = {
        "episodes": args.episodes,
        "evaluator": "deterministic_argmax",
        "initial_bullets": INITIAL_BULLETS,
        "no_replenishment": True,
        "ammo_aware": bool(metadata.get("ammo_aware", False)),
        "projectile_fire_mask": bool(metadata.get("projectile_fire_mask", False)),
        **{
            f"mean_{key}": float(np.mean([episode[key] for episode in episodes]))
            for key in keys
        },
        "positive_reward_episodes": sum(episode["reward"] > 0 for episode in episodes),
        "used_all_bullets_episodes": sum(
            episode["final_ammo"] == 0 for episode in episodes
        ),
    }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "copied_tensors": copied_tensors,
        "summary": summary,
        "action_counts": {str(key): value for key, value in sorted(action_counts.items())},
        "episode_results": episodes,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=100_001)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint.with_suffix(".evaluation.json")
    )
    result = evaluate(checkpoint, args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"evaluation={output}", flush=True)


if __name__ == "__main__":
    main()
