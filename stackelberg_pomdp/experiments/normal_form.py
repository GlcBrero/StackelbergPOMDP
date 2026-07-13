import argparse

from stackelberg_pomdp.experiments.common import (
    add_common_training_args,
    add_response_phase_args,
    finalized_config,
    run_training,
    str_to_bool,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Normal-form StackPOMDP experiment.")
    parser.add_argument("--game_name", type=str, default="game_2")
    parser.add_argument("--randomized", type=str_to_bool, default=True)
    add_response_phase_args(parser, default_response_episodes=100, default_reward_episodes=1)
    add_common_training_args(parser, default_algorithm="PPO")
    return parser


def main():
    args = build_parser().parse_args()
    config = finalized_config(args, f"normal_form:{args.game_name}:{args.randomized}")
    run_training(config)


if __name__ == "__main__":
    main()
