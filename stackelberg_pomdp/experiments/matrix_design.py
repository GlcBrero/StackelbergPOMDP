import argparse

from stackelberg_pomdp.experiments.common import (
    add_common_training_args,
    add_response_phase_args,
    finalized_config,
    run_training,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Matrix-design StackPOMDP experiment.")
    add_response_phase_args(parser, default_response_episodes=4, default_reward_episodes=1)
    add_common_training_args(parser, default_algorithm="PPO")
    return parser


def main():
    args = build_parser().parse_args()
    config = finalized_config(args, "matrix_design")
    run_training(config)


if __name__ == "__main__":
    main()
