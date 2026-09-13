"""Run one appendix diagnostic treatment or train its contextual follower."""

import argparse

from stackelberg_pomdp.matrix_ablations.profiles import PROFILES
from stackelberg_pomdp.matrix_ablations.presets import FIGURE_EXPERIMENTS
from stackelberg_pomdp.matrix_ablations.artifacts import (
    canonical_json,
    DEFAULT_RESULTS_ROOT,
)
from stackelberg_pomdp.matrix_ablations.meta_training import (
    train_meta_follower,
    evaluate_meta_follower,
    _meta_model,
    validate_meta_args,
    meta_config,
)
from stackelberg_pomdp.matrix_ablations.leader_training import (
    train_leader,
    EXPERIMENT_CONDITIONS,
)
from stackelberg_pomdp.matrix_ablations.evaluation import (
    evaluate_leader_policy,
)


def positive_int_tuple(value):
    if isinstance(value, str) and value.strip().lower() == "linear":
        return ()
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(
            "network widths must be positive integers"
        )
    return values


def add_logging_args(parser):
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--attempt", type=int, default=0,
        help="Immutable retry number; zero is the first attempt.",
    )
    parser.add_argument("--sweep-id")
    parser.add_argument("--record-key")
    parser.add_argument("--sweep-plan")
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--wandb-project", default="StackPOMDP")
    parser.add_argument("--wandb-group", default="matrix_qualitative_replications")
    parser.add_argument("--wandb-name")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    meta = subparsers.add_parser(
        "meta-follower", description="Train one frozen E1 matrix response."
    )
    meta.add_argument(
        "--matrix", choices=("prisoners_dilemma",),
        default="prisoners_dilemma",
    )
    meta.add_argument("--memory-mode", choices=("joint", "opponent"))
    meta.add_argument("--profile-id", choices=tuple(sorted(PROFILES)))
    meta.add_argument(
        "--algorithm", choices=("PPO", "A2C", "DQN", "REINFORCE"),
        default="PPO"
    )
    meta.add_argument("--seed", type=int, default=1)
    meta.add_argument("--timesteps", type=int)
    meta.add_argument("--learning-rate", type=float)
    meta.add_argument("--ent-coef", type=float)
    meta.add_argument("--episodes-per-batch", type=int)
    meta.add_argument("--n-steps", type=int)
    meta.add_argument("--batch-size", type=int)
    meta.add_argument("--n-epochs", type=int)
    meta.add_argument(
        "--net-arch", type=positive_int_tuple,
        help=("Comma-separated hidden-layer widths, or 'linear'. Algorithm-"
              "specific defaults use SB3 or the appendix's linear REINFORCE."),
    )
    dqn = meta.add_argument_group("DQN replay and exploration")
    dqn.add_argument(
        "--dqn-learning-starts", type=int,
        help="Replay transitions collected before learning starts.",
    )
    dqn.add_argument(
        "--dqn-target-update-interval", type=int,
        help="Environment steps between target-network updates.",
    )
    dqn.add_argument(
        "--dqn-exploration-steps", type=int,
        help="Steps over which epsilon decays from one to its final value.",
    )
    dqn.add_argument(
        "--dqn-final-epsilon", type=float,
        help="Final epsilon for DQN exploration.",
    )
    meta.add_argument("--device", default="cpu")
    add_logging_args(meta)

    leader = subparsers.add_parser(
        "leader", description="Train one condition from a matrix diagnostic."
    )
    selection = leader.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--experiment", choices=tuple(EXPERIMENT_CONDITIONS)
    )
    selection.add_argument("--figure", choices=tuple(FIGURE_EXPERIMENTS))
    leader.add_argument("--condition", required=True)
    leader.add_argument("--matrix", choices=tuple(sorted(MATRIX for MATRIX in (
        "prisoners_dilemma", "battle_of_the_sexes",
        "coordination_zero_miscoordination",
        "coordination_penalized_miscoordination",
    ))))
    leader.add_argument("--memory-mode", choices=("joint", "opponent"))
    leader.add_argument(
        "--algorithm", choices=("PG", "SIMPLEQ", "PPO", "A2C"),
        help="Default: PG for commitment/reset; SIMPLEQ for reward timing.",
    )
    leader.add_argument("--seed", type=int, default=1)
    leader.add_argument(
        "--timesteps", type=int,
        help=(
            "Requested environment transitions."
        ),
    )
    leader.add_argument("--learning-rate", type=float)
    leader.add_argument(
        "--ent-coef", type=float,
        help="Entropy coefficient (PG/SIMPLEQ: 0; A2C/PPO default: 0.01).",
    )
    leader.add_argument("--response-checkpoint")
    leader.add_argument("--response-contract-path")
    leader.add_argument(
        "--response-algorithm",
        choices=("PPO", "A2C", "DQN", "REINFORCE"), default="REINFORCE"
    )
    leader.add_argument("--max-response-regret", type=float, default=0.25)
    leader.add_argument("--allow-uncertified-response", action="store_true")
    leader.add_argument("--response-episodes", type=int, default=10)
    leader.add_argument("--q-alpha", type=float)
    leader.add_argument("--q-epsilon", type=float)
    leader.add_argument(
        "--q-exploration", choices=("epsilon_greedy", "parameter_noise")
    )
    leader.add_argument("--q-init", choices=("small_normal", "zero"))
    leader.add_argument("--q-init-std", type=float, default=0.01)
    leader.add_argument("--ppo-episodes-per-batch", type=int)
    leader.add_argument("--ppo-batch-size", type=int)
    leader.add_argument("--ppo-n-epochs", type=int)
    leader.add_argument("--eval-freq", type=int, default=2_000)
    leader.add_argument("--eval-episodes", type=int, default=20)
    leader.add_argument("--eval-warmup", type=int)
    leader.add_argument("--eval-seed-start", type=int, default=2_000_001)
    leader.add_argument("--final-eval-episodes", type=int, default=100)
    leader.add_argument("--final-eval-seed-start", type=int, default=3_000_001)
    leader.add_argument("--device", default="cpu")
    add_logging_args(leader)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "meta-follower":
        run_dir = train_meta_follower(args)
    else:
        run_dir = train_leader(args)
    print(canonical_json({"completed": True, "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
