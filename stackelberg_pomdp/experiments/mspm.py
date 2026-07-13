import argparse

from stackelberg_pomdp.experiments.common import (
    add_common_training_args,
    add_response_phase_args,
    finalized_config,
    run_training,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Sequential posted-price mechanism experiment.")
    parser.add_argument("--setting", type=str, default="MSGSpace", choices=("PI", "MSGSpace"))
    parser.add_argument("--num_types", type=int, required=True)
    parser.add_argument("--num_messages", type=int, required=True)
    add_response_phase_args(parser, default_response_episodes=300, default_reward_episodes=100)
    add_common_training_args(parser, default_algorithm="PPO")
    parser.set_defaults(
        learning_method="RL:Standard",
        # Keep the economically motivated undiscounted objective (gamma=1 in
        # the trainer), but otherwise use SB3's PPO defaults. One rollout is
        # exactly one complete StackPOMDP episode.
        learning_rate=3e-4,
        ent_coef=0.0,
        ppo_episodes_per_batch=1,
        ppo_batch_size=64,
        ppo_n_epochs=10,
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = finalized_config(args, f"mspm:{args.setting}:{args.num_types}:{args.num_messages}")
    run_training(config)


if __name__ == "__main__":
    main()
