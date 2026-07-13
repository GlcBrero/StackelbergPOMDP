"""Evaluate a saved MSPM PI policy with deterministic leader actions."""

import argparse
from collections import Counter
import json

from stackelberg_pomdp.baselines_utils import CustomPPO
from stackelberg_pomdp.env_setups import get_mspm_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--response-episodes", type=int, default=300)
    args = parser.parse_args()

    config = {
        "logger": None,
        "seed": args.seed,
        "setting": "PI",
        "num_types": 2,
        "num_messages": 2,
        "followers_algorithm": "MW",
        "tot_num_response_episodes": args.response_episodes,
        "tot_num_reward_episodes": 4,
        "critic_obs": "full",
        "pomdp_mode": "stackelberg",
    }
    env = get_mspm_env(config)
    model = CustomPPO.load(args.checkpoint)
    model.policy.fix_policy_actions()

    rewards = []
    response_updates = []
    for _ in range(args.episodes):
        model.policy.clear_obs_action_map()
        observation = env.reset()
        done = False
        exact_reward = 0.0
        while not done:
            policy_observation = {
                key: observation[key]
                for key in model.observation_space.spaces
            }
            action, _ = model.predict(policy_observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            if info.get("is_reward_phase", False):
                exact_reward += float(reward)
            if info.get("response_phase_done", False):
                response_updates.append(info.get("response_updates"))
        rewards.append(exact_reward)

    rounded_rewards = [round(reward, 8) for reward in rewards]
    payload = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "seed": args.seed,
        "response_episodes": args.response_episodes,
        "mean_expected_reward": sum(rewards) / len(rewards),
        "zero_reward_fraction": sum(abs(reward) <= 1e-8 for reward in rewards) / len(rewards),
        "reward_counts": dict(sorted(Counter(rounded_rewards).items())),
        "response_update_counts": dict(sorted(Counter(response_updates).items())),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
