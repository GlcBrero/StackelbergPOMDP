import argparse

from stackelberg_pomdp.experiments.common import (
    add_common_training_args,
    add_response_phase_args,
    finalized_config,
    run_training,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Price-collusion intervention StackPOMDP experiment.")
    parser.add_argument("--platform_intervention", type=str, default="learn_threshold")
    parser.add_argument(
        "--platform_observation_space",
        type=str,
        default="price_profile",
        choices=("price_profile", "no_observation"),
    )
    parser.add_argument("--price_grid_length", type=int, default=4)
    parser.add_argument("--price_min", type=float, default=1.05)
    parser.add_argument("--price_max", type=float, default=1.7)
    parser.add_argument("--marginal_cost", type=float, default=1.0)
    parser.add_argument("--num_pricing_agents", type=int, default=2)
    add_response_phase_args(parser, default_response_episodes=50000, default_reward_episodes=30)
    parser.add_argument("--follower_alpha", type=float, default=0.25)
    parser.add_argument("--follower_beta", type=float, default=1e-4)
    parser.add_argument("--q_tables_path", type=str, default=None)
    parser.add_argument("--intervention_lambda", type=float, default=0.0)
    parser.add_argument("--leader_k", type=int, default=1)
    parser.add_argument("--sort_obs", action="store_true", default=False)
    parser.add_argument("--warm_start_q", action="store_true", default=False)
    add_common_training_args(parser, default_algorithm="A2C", default_max_steps=50000000)
    parser.set_defaults(followers_algorithm="Qlearning")
    return parser


def main():
    args = build_parser().parse_args()
    config = finalized_config(args, "bertrand")
    run_training(config)


if __name__ == "__main__":
    main()
