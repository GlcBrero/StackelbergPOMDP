import argparse

from stackelberg_pomdp.experiments.common import (
    add_common_training_args,
    add_response_phase_args,
    finalized_config,
    run_training,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Simple allocation StackPOMDP experiment.")
    parser.add_argument("--num_messages", type=int, default=3)
    add_response_phase_args(parser, default_response_episodes=100, default_reward_episodes=30)
    add_common_training_args(parser, default_algorithm="PPO", default_max_steps=100000)
    return parser


def main():
    args = build_parser().parse_args()
    config = finalized_config(args, f"simple_allocation:{args.num_messages}")
    run_training(config)


if __name__ == "__main__":
    main()
