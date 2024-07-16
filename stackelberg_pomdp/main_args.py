import argparse
from run_setups import train_run

def run(config_dict):
    train_run(config_dict)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="StackPOMDP")

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random number generator seed",
    )

    parser.add_argument(
        "--training_seed",
        type=int,
        default=None,
        help="The training seed, defaults to the value of --seed if not specified",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000000,
        help="Number of training steps"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        help="DeepRL algorithm used for leader. Options: 'A2C', 'PPO'"
    )

    parser.add_argument(
        "--tot_num_reward_episodes",
        type=int,
        default=1,
        help="Number of reward sub_episodes in Stackelberg POMDP"
    )

    parser.add_argument(
        "--tot_num_eq_episodes",
        type=int,
        default=100,
        help="Number of equilibrium sub_episodes in Stackelberg POMDP"
    )

    parser.add_argument(
        "--experiment_type",
        type=str,
        default="normal_form:game_1:False",
        help="Type of experiment to run. "
             "For normal_form, format is 'normal_form:game_name:randomized' where game_name is either 'game_1' or 'game_2' and randomized is either 'True' or 'False'. "
             "For simple_allocation, format is 'simple_allocation:num_followers_messages' where num_followers_messages is an integer. "
             "For mspm, format is 'mspm:mspm_setting:num_followers_types:num_followers_messages' where mspm_setting is either 'PI' or 'MSGSpace', and num_followers_types and num_followers_messages are integers. "
             "Other options are 'matrix_design'"
    )

    parser.add_argument(
        "--critic_obs",
        type=str,
        default="full",
        help="Determines which additional info is given to critic network. Options: none (critic has same observation as actor), full (critic observes POMDP states)"
    )

    parser.add_argument(
        "--fix_episode_actions",
        type=str,
        default="True",
        help="If true, keep observation-action mapping so that leader policy behaves deterministically during each StackPOMDP episode"
    )

    parser.add_argument(
        "--followers_algorithm",
        type=str,
        default="MW",
        help="Determines the followers' learning algorithm. Options: MW, Qlearning"
    )

    parser.add_argument(
        "--response_phase_prob",
        type=float,
        default=1.0,
        help="Probability that each step in response phase is included in the RL buffer"
    )

    parser.add_argument(
        "--learning_method",
        type=str,
        default='RL:Standard',
        help="Options: RL:Standard, RL:StopOnThreshold, PolicyEnumeration:n"
    )

    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="Use Weights & Biases for logging. Set to True to turn on."
    )

    args = parser.parse_args()

    if args.training_seed is None:
        args.training_seed = args.seed

    config_dict = vars(args)

    run(config_dict)