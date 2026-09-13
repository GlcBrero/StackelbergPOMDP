"""Configure and train one of the three appendix diagnostic treatments."""

from pathlib import Path
import time
import numpy as np

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stackelberg_pomdp.callbacks import FixPolicyActionsCallback
from stackelberg_pomdp.envs.matrix import (
    get_matrix_game,
    load_response_model,
    make_meta_leader_env,
    make_tabular_q_leader_env,
    validate_response_checkpoint_contract,
)
from stackelberg_pomdp.matrix_ablations.profiles import profile_snapshot_for_spec
from stackelberg_pomdp.matrix_ablations.reinforce import Reinforce
from stackelberg_pomdp.matrix_ablations.presets import (
    FIGURE_EXPERIMENTS,
    LEADER_DEFAULTS,
    leader_protocol,
)
from stackelberg_pomdp.matrix_ablations.simple_q import SimpleQ
from stackelberg_pomdp.rl_trainer_setup import get_custom_training_algorithm
from stackelberg_pomdp.training_defaults import algorithm_defaults
from stackelberg_pomdp.matrix_ablations.artifacts import (
    _checkpoint_path,
    _maybe_wandb,
    _run_dir,
    append_jsonl,
    file_sha256,
    finish_manifest,
    start_manifest,
    validate_logging_args,
    write_json,
    SCHEMA_VERSION,
)
from stackelberg_pomdp.matrix_ablations.evaluation import (
    JsonlEvaluationCallback,
    evaluate_leader_policy,
)


EXPERIMENT_CONDITIONS = {
    "phase_observability": ("visible", "hidden"),
    "q_reset": ("reset", "ongoing"),
    "response_reward": ("excluded", "included"),
}
DEFAULT_MATRICES = {
    "phase_observability": "prisoners_dilemma",
    "q_reset": "battle_of_the_sexes",
    "response_reward": "coordination_zero_miscoordination",
}


def _resolved_q_protocol(args):
    if args.experiment == "q_reset":
        defaults = (0.1, 0.1, "parameter_noise", "small_normal")
    elif args.experiment == "response_reward":
        defaults = (0.2, 0.1, "parameter_noise", "zero")
    else:
        return None
    alpha, epsilon, exploration, initialization = defaults
    return {
        "alpha": alpha if args.q_alpha is None else args.q_alpha,
        "epsilon": epsilon if args.q_epsilon is None else args.q_epsilon,
        "exploration": args.q_exploration or exploration,
        "initialization": args.q_init or initialization,
        "initialization_std": args.q_init_std,
    }


def validate_leader_args(args):
    validate_logging_args(args)
    if args.figure:
        args.experiment = FIGURE_EXPERIMENTS[args.figure]
    defaults = LEADER_DEFAULTS[args.experiment]
    args.algorithm = args.algorithm or defaults["algorithm"]
    args.timesteps = defaults["timesteps"] if args.timesteps is None else args.timesteps
    if args.learning_rate is None:
        args.learning_rate = (defaults["learning_rate"] if args.algorithm in ("PG", "SIMPLEQ")
                              else algorithm_defaults(args.algorithm)["learning_rate"])
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if args.ent_coef is None:
        args.ent_coef = 0.0 if args.algorithm in ("PG", "SIMPLEQ") else 0.01
    if args.algorithm == "PPO":
        for key, parameter in (("ppo_batch_size", "batch_size"), ("ppo_n_epochs", "n_epochs")):
            if getattr(args, key) is None:
                setattr(args, key, algorithm_defaults("PPO")[parameter])
    if args.algorithm in ("PG", "SIMPLEQ") and args.ent_coef != 0:
        raise ValueError("PG/SIMPLEQ appendix recipes do not use entropy regularization")
    if args.algorithm == "SIMPLEQ" and args.experiment != "response_reward":
        raise ValueError("SIMPLEQ is only used for response_reward")
    args.matrix = args.matrix or DEFAULT_MATRICES[args.experiment]
    if args.experiment == "phase_observability" and args.memory_mode is None:
        args.memory_mode = "joint"
    if args.condition not in EXPERIMENT_CONDITIONS[args.experiment]:
        raise ValueError(
            "condition {!r} is invalid for {}; choose {}".format(
                args.condition,
                args.experiment,
                ", ".join(EXPERIMENT_CONDITIONS[args.experiment]),
            )
        )
    if args.eval_freq < 1:
        raise ValueError("eval_freq must be positive")
    if args.timesteps < 1:
        raise ValueError("timesteps must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if args.eval_episodes < 1 or args.final_eval_episodes < 1:
        raise ValueError("evaluation episode counts must be positive")
    if args.response_episodes < 1:
        raise ValueError("response_episodes must be positive")
    if args.experiment == "phase_observability":
        if not args.response_checkpoint:
            raise ValueError("meta-response experiments require --response-checkpoint")
        args.response_checkpoint = str(_checkpoint_path(args.response_checkpoint))
    if args.experiment == "response_reward" and args.matrix not in (
        "coordination_zero_miscoordination",
        "coordination_penalized_miscoordination",
    ):
        raise ValueError("response_reward requires a coordination diagnostic matrix")
    if args.experiment == "q_reset" and args.matrix != "battle_of_the_sexes":
        raise ValueError("q_reset requires battle_of_the_sexes")
    if (
        args.experiment == "phase_observability"
        and args.matrix != "prisoners_dilemma"
    ):
        raise ValueError("phase_observability requires prisoners_dilemma")
    if args.eval_warmup is None:
        args.eval_warmup = 0
    if args.experiment == "q_reset" and args.condition == "ongoing" and args.eval_warmup:
        raise ValueError("carried-state snapshot evaluation requires --eval-warmup 0")
    args.q_protocol = _resolved_q_protocol(args)


def leader_config(args, spec):
    sweep_plan = Path(args.sweep_plan).resolve() if args.sweep_plan else None
    response_hash = (
        file_sha256(args.response_checkpoint) if args.response_checkpoint else None
    )
    profile = profile_snapshot_for_spec(spec)
    config = {
        "schema_version": SCHEMA_VERSION,
        "stage": "leader",
        "profile_id": profile["profile_id"],
        "profile": profile,
        "experiment": args.experiment,
        "condition": args.condition,
        "matrix": spec.name,
        "payoffs": spec.payoffs.tolist(),
        "reward_offset": spec.reward_offset,
        "episode_length": spec.episode_length,
        "memory_mode": spec.memory_mode,
        "query_states": list(range(spec.num_leader_states)),
        "algorithm": args.algorithm,
        "figure_or_table": next(name for name, experiment in FIGURE_EXPERIMENTS.items()
                                if experiment == args.experiment),
        "leader_protocol": leader_protocol(
            args, spec.num_leader_states + spec.episode_length
            if args.experiment == "phase_observability" else args.response_episodes + 1,
        ),
        "seed": args.seed,
        "timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "response_episodes": args.response_episodes,
        "response_checkpoint": args.response_checkpoint,
        "response_checkpoint_sha256": response_hash,
        "response_contract_path": args.response_contract_path,
        "response_algorithm": args.response_algorithm,
        "max_response_regret": args.max_response_regret,
        "allow_uncertified_response": args.allow_uncertified_response,
        "q_protocol": args.q_protocol,
        "ppo_episodes_per_batch": args.ppo_episodes_per_batch,
        "ppo_batch_size": args.ppo_batch_size,
        "ppo_n_epochs": args.ppo_n_epochs,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "eval_warmup": args.eval_warmup,
        "evaluation_response_state": (
            "training_snapshot" if args.experiment == "q_reset" and args.condition == "ongoing"
            else "fresh_initialization"
        ),
        "eval_seed_start": args.eval_seed_start,
        "final_eval_episodes": args.final_eval_episodes,
        "final_eval_seed_start": args.final_eval_seed_start,
        "device": args.device,
        "checkpoint_every": args.checkpoint_every,
        "attempt": args.attempt,
        "sweep_id": args.sweep_id,
        "record_key": args.record_key,
        "sweep_plan": str(sweep_plan) if sweep_plan else None,
        "sweep_plan_sha256": file_sha256(sweep_plan) if sweep_plan else None,
    }
    return config


def _validate_response_quality(args, spec):
    if not args.response_checkpoint:
        return None
    contract = validate_response_checkpoint_contract(
        spec,
        args.response_checkpoint,
        algorithm=args.response_algorithm,
        contract_path=args.response_contract_path,
    )
    evaluation = (contract.get("metadata") or {}).get("evaluation") or {}
    max_regret = evaluation.get("max_regret")
    if not args.allow_uncertified_response:
        if max_regret is None:
            raise ValueError(
                "response contract has no evaluation certificate; pass "
                "--allow-uncertified-response only for diagnostics"
            )
        if float(max_regret) > args.max_response_regret:
            raise ValueError(
                "response max regret {} exceeds threshold {}".format(
                    max_regret, args.max_response_regret
                )
            )
    return contract


def leader_env_factory(args, spec, evaluation=False):
    shared_response_model = None
    if args.experiment == "phase_observability":
        shared_response_model = load_response_model(
            args.response_checkpoint, args.response_algorithm, device=args.device
        )

    def factory(seed):
        if args.experiment == "phase_observability":
            return make_meta_leader_env(
                spec,
                response_checkpoint=args.response_checkpoint,
                response_algorithm=args.response_algorithm,
                response_model=shared_response_model,
                query_visibility="observed",
                phase_observable=(args.condition == "visible"),
                seed=seed,
                device=args.device,
            )
        protocol = args.q_protocol
        return make_tabular_q_leader_env(
            spec,
            response_episodes=args.response_episodes,
            warm_start_q=(
                args.experiment == "q_reset" and args.condition == "ongoing"
            ),
            include_response_reward=(
                args.experiment == "response_reward"
                and args.condition == "included"
                and not evaluation
            ),
            q_alpha=protocol["alpha"],
            q_epsilon=protocol["epsilon"],
            exploration=protocol["exploration"],
            q_init=protocol["initialization"],
            q_init_std=protocol["initialization_std"],
            seed=seed,
        )

    return factory


def _progress_metadata(args):
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment": args.experiment,
        "condition": args.condition,
        "matrix": args.matrix,
        "algorithm": args.algorithm,
        "seed": args.seed,
    }
    metadata["learning_rate"] = args.learning_rate
    return metadata


def train_sb3_leader(args, spec, run_dir, config, wandb_run=None):
    training_factory = leader_env_factory(args, spec, evaluation=False)
    evaluation_factory = leader_env_factory(args, spec, evaluation=True)
    env = Monitor(training_factory(args.seed), filename=str(run_dir / "train_monitor"))
    algorithm_config = {
        "algorithm": args.algorithm,
        "training_seed": args.seed,
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "ppo_episodes_per_batch": args.ppo_episodes_per_batch,
        "ppo_batch_size": args.ppo_batch_size,
        "ppo_n_epochs": args.ppo_n_epochs,
        "tot_num_reward_episodes": spec.episode_length,
    }
    if args.algorithm == "PG":
        model = Reinforce(
            env=env, learning_rate=args.learning_rate,
            n_steps=config["leader_protocol"]["batch_steps"],
            policy_kwargs={"cache_actions": True}, seed=args.seed, device=args.device,
        )
    elif args.algorithm == "SIMPLEQ":
        model = SimpleQ(env=env, learning_rate=args.learning_rate,
                        seed=args.seed, device=args.device)
    else:
        model = get_custom_training_algorithm(algorithm_config, env)
    metadata = _progress_metadata(args)
    initial = evaluate_leader_policy(
        model,
        evaluation_factory,
        args.eval_episodes,
        args.eval_warmup,
        args.eval_seed_start,
    )
    initial_summary = initial["per_stage_summary"]
    append_jsonl(run_dir / "progress.jsonl", {
        **metadata,
        "evaluation_target_step": 0,
        "global_step": 0,
        "wall_time": time.time(),
        "evaluation_mean": initial_summary["mean"],
        "evaluation_std": initial_summary["std"],
        "evaluation_sem": initial_summary["sem"],
        "evaluation_episodes": initial_summary["n"],
        "dependent_response_stream": initial["dependent_response_stream"],
    })
    evaluation_callback = JsonlEvaluationCallback(
            evaluation_factory,
            run_dir / "progress.jsonl",
            args.eval_freq,
            args.eval_episodes,
            args.eval_warmup,
            args.eval_seed_start,
            metadata,
            wandb_run,
        )
    callbacks = [FixPolicyActionsCallback()]
    post_update_evaluation = args.algorithm in ("PG", "SIMPLEQ")
    if not post_update_evaluation:
        callbacks.append(evaluation_callback)
    if args.checkpoint_every > 0:
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_every,
            save_path=str(run_dir),
            name_prefix="leader_step",
        ))
    try:
        if post_update_evaluation:
            target = min(args.eval_freq, args.timesteps)
            while model.num_timesteps < args.timesteps:
                model.learn(
                    total_timesteps=target - model.num_timesteps,
                    reset_num_timesteps=(model.num_timesteps == 0),
                    callback=CallbackList(callbacks),
                )
                evaluation_callback.record(model, target)
                target = min(
                    (model.num_timesteps // args.eval_freq + 1) * args.eval_freq,
                    args.timesteps,
                )
        else:
            model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks))
        model.save(str(run_dir / "model"))
        evaluation = evaluate_leader_policy(
            model,
            evaluation_factory,
            args.final_eval_episodes,
            args.eval_warmup,
            args.final_eval_seed_start,
        )
        evaluation.update({"schema_version": SCHEMA_VERSION, "config": config})
        write_json(run_dir / "evaluation.json", evaluation)
    finally:
        env.close()
    return {
        "model": run_dir / "model.zip",
        "evaluation": run_dir / "evaluation.json",
        "progress": run_dir / "progress.jsonl",
    }


def train_leader(args):
    validate_leader_args(args)
    spec = get_matrix_game(
        args.matrix,
        memory_mode=args.memory_mode if args.experiment == "phase_observability" else "none",
    )
    _validate_response_quality(args, spec)
    config = leader_config(args, spec)
    run_dir = _run_dir(args.output_root, config)
    config["run_dir"] = str(run_dir)
    write_json(run_dir / "config.json", config)
    manifest = start_manifest(run_dir, config)
    run = _maybe_wandb(args, config, run_dir.name)
    try:
        artifacts = train_sb3_leader(args, spec, run_dir, config, run)
        finish_manifest(
            run_dir,
            manifest,
            "completed",
            artifacts=artifacts,
        )
    except Exception as exc:
        finish_manifest(run_dir, manifest, "failed", error=exc)
        raise
    finally:
        if run is not None:
            run.finish()
    return run_dir
