"""Train and certify the frozen contextual follower for commitment consistency."""

import hashlib
import math
import numpy as np

from itertools import product
from pathlib import Path
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stackelberg_pomdp.training_defaults import algorithm_defaults, complete_episode_count
from stackelberg_pomdp.envs.matrix import (
    MatrixFixedCommitmentResponseEnv,
    RepeatedMatrixGame,
    get_matrix_game,
    write_response_checkpoint_contract,
)
from stackelberg_pomdp.matrix_ablations.profiles import profile_for_spec
from stackelberg_pomdp.matrix_ablations.reinforce import (
    LEGACY_REINFORCE_LEARNING_RATE,
    LEGACY_REINFORCE_PRETRAIN_ITERATIONS,
    Reinforce,
    legacy_pg_rollout_geometry,
    reinforce_checkpoint_geometry,
    save_evaluated_response_checkpoint,
)
from stackelberg_pomdp.matrix_ablations.artifacts import (
    _maybe_wandb,
    _run_dir,
    append_jsonl,
    canonical_json,
    file_sha256,
    finish_manifest,
    mean_summary,
    start_manifest,
    validate_logging_args,
    write_json,
    SCHEMA_VERSION,
)


def _best_stationary_follower_return(spec, commitment):
    best_return = -np.inf
    best_policy = None
    for policy in product(
            range(spec.num_follower_actions), repeat=spec.num_follower_states
    ):
        game = RepeatedMatrixGame(spec)
        game.reset()
        total = 0.0
        done = False
        while not done:
            leader_action = commitment[game.leader_state]
            follower_action = policy[game.follower_state]
            _, rewards, done, _ = game.step(leader_action, follower_action)
            total += rewards["follower_0"]
        if total > best_return:
            best_return = float(total)
            best_policy = tuple(int(value) for value in policy)
    return best_return, best_policy


def evaluate_meta_follower(model, spec):
    rows = []
    for commitment in MatrixFixedCommitmentResponseEnv.commitments(spec):
        env = MatrixFixedCommitmentResponseEnv(
            spec, seed=0, fixed_commitment=commitment
        )
        observation = env.reset()
        total = 0.0
        actions = []
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            actions.append(action)
            observation, reward, done, _ = env.step(action)
            total += float(reward)
        optimal, best_policy = _best_stationary_follower_return(spec, commitment)
        rows.append({
            "leader_commitment": list(commitment),
            "follower_actions": actions,
            "follower_return": total,
            "best_stationary_return": optimal,
            "best_stationary_policy": list(best_policy),
            "regret": float(optimal - total),
        })
    regrets = [row["regret"] for row in rows]
    returns = [row["follower_return"] for row in rows]
    return {
        "summary": {
            **mean_summary(returns),
            "commitments": len(rows),
            "mean_regret": float(np.mean(regrets)),
            "max_regret": float(np.max(regrets)),
            "optimal_commitments": int(sum(value <= 1e-9 for value in regrets)),
        },
        "commitment_rows": rows,
    }


def _resolve_meta_defaults(args):
    """Use SB3 defaults, or the separately defined appendix REINFORCE recipe."""

    reinforce = args.algorithm == "REINFORCE"
    geometry = None
    if reinforce:
        spec = _meta_spec(args)
        geometry = legacy_pg_rollout_geometry(
            spec.episode_length, spec.num_leader_states
        )
    defaults = {
        "timesteps": (
            geometry["follower_gradient_samples_per_update"]
            * LEGACY_REINFORCE_PRETRAIN_ITERATIONS
            if reinforce else 200_000
        ),
        "learning_rate": (
            LEGACY_REINFORCE_LEARNING_RATE if reinforce
            else algorithm_defaults(args.algorithm)["learning_rate"]
        ),
        "ent_coef": 0.0,
        "episodes_per_batch": (
            geometry["episodes_per_update"] if reinforce
            else (complete_episode_count(args.algorithm, _meta_spec(args).episode_length)
                  if args.algorithm != "DQN" else 1)
        ),
        "net_arch": () if reinforce else (64, 64),
        "n_epochs": algorithm_defaults("PPO")["n_epochs"],
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if reinforce and args.n_steps is None:
        args.n_steps = geometry["follower_gradient_samples_per_update"]
    elif args.algorithm in ("PPO", "A2C") and args.n_steps is None:
        args.n_steps = args.episodes_per_batch * _meta_spec(args).episode_length
    if args.algorithm in ("PPO", "DQN") and args.batch_size is None:
        args.batch_size = algorithm_defaults(args.algorithm)["batch_size"]
    if args.algorithm == "DQN":
        dqn = algorithm_defaults("DQN")
        for name, value in {
            "dqn_learning_starts": dqn["learning_starts"],
            "dqn_target_update_interval": dqn["target_update_interval"],
            "dqn_exploration_steps": max(1, int(dqn["exploration_fraction"] * args.timesteps)),
            "dqn_final_epsilon": dqn["exploration_final_eps"],
        }.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    return args


def _meta_model(args, env, episode_length):
    _resolve_meta_defaults(args)
    if args.algorithm == "REINFORCE":
        n_steps = int(args.n_steps)
        if n_steps % int(episode_length):
            raise ValueError(
                "REINFORCE n_steps must contain complete matrix episodes"
            )
        return Reinforce(
            env=env,
            learning_rate=args.learning_rate,
            n_steps=n_steps,
            seed=args.seed,
            device=args.device,
            verbose=0,
        )

    if args.algorithm == "DQN":
        protocol = _resolved_dqn_protocol(args)
        return DQN(
            policy="MlpPolicy",
            env=env,
            gamma=1.0,
            learning_rate=args.learning_rate,
            buffer_size=protocol["buffer_size"],
            learning_starts=protocol["learning_starts"],
            batch_size=protocol["batch_size"],
            train_freq=protocol["train_freq"],
            gradient_steps=protocol["gradient_steps"],
            target_update_interval=protocol["target_update_interval"],
            exploration_fraction=protocol["exploration_fraction"],
            exploration_initial_eps=protocol["initial_epsilon"],
            exploration_final_eps=protocol["final_epsilon"],
            seed=args.seed,
            device=args.device,
            verbose=0,
            policy_kwargs={"net_arch": list(args.net_arch)},
        )

    n_steps = int(args.n_steps or args.episodes_per_batch * episode_length)
    common = dict(
        policy="MlpPolicy",
        env=env,
        gamma=1.0,
        gae_lambda=algorithm_defaults(args.algorithm)["gae_lambda"],
        learning_rate=args.learning_rate,
        n_steps=n_steps,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        verbose=1,
        policy_kwargs={"net_arch": list(args.net_arch)},
    )
    if args.algorithm == "PPO":
        return PPO(
            **common,
            batch_size=int(args.batch_size),
            n_epochs=args.n_epochs,
        )
    return A2C(**common)


def _meta_spec(args):
    return get_matrix_game(
        args.matrix,
        memory_mode=args.memory_mode,
        profile_id=args.profile_id,
    )


def _resolved_dqn_protocol(args):
    """Resolve SB3 replay defaults plus explicitly requested DQN overrides."""

    _resolve_meta_defaults(args)
    batch_size = int(args.batch_size)
    buffer_size = algorithm_defaults("DQN")["buffer_size"]
    return {
        "buffer_size": buffer_size,
        "learning_starts": int(args.dqn_learning_starts),
        "batch_size": batch_size,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": int(args.dqn_target_update_interval),
        "exploration_steps": int(args.dqn_exploration_steps),
        "exploration_fraction": min(
            1.0, float(args.dqn_exploration_steps) / float(args.timesteps)
        ),
        "initial_epsilon": 1.0,
        "final_epsilon": float(args.dqn_final_epsilon),
    }


def validate_meta_args(args):
    _resolve_meta_defaults(args)
    validate_logging_args(args)
    if args.timesteps < 1:
        raise ValueError("timesteps must be positive")
    if args.algorithm == "REINFORCE":
        spec = _meta_spec(args)
        geometry = legacy_pg_rollout_geometry(
            spec.episode_length, spec.num_leader_states
        )
        expected_n_steps = geometry["follower_gradient_samples_per_update"]
        if args.n_steps != expected_n_steps:
            raise ValueError(
                "legacy-compatible REINFORCE requires n_steps={} for {} "
                "query states".format(expected_n_steps, spec.num_leader_states)
            )
        if args.episodes_per_batch != geometry["episodes_per_update"]:
            raise ValueError(
                "legacy-compatible REINFORCE requires episodes_per_batch={}"
                .format(geometry["episodes_per_update"])
            )
        if args.learning_rate <= 0:
            raise ValueError("REINFORCE learning_rate must be positive")
        if args.ent_coef != 0.0:
            raise ValueError("legacy REINFORCE has no entropy bonus")
        if tuple(args.net_arch):
            raise ValueError("legacy REINFORCE requires --net-arch linear")
        if args.timesteps % expected_n_steps:
            raise ValueError(
                "REINFORCE timesteps must contain complete optimizer updates"
            )
        if args.checkpoint_every:
            raise ValueError(
                "REINFORCE uses post-update checkpoints automatically; "
                "do not set --checkpoint-every"
            )
        return
    if args.algorithm != "DQN":
        return
    if args.dqn_learning_starts < 0:
        raise ValueError("dqn_learning_starts must be nonnegative")
    if args.dqn_learning_starts >= args.timesteps:
        raise ValueError("dqn_learning_starts must be smaller than timesteps")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.dqn_target_update_interval < 1:
        raise ValueError("dqn_target_update_interval must be positive")
    if args.dqn_exploration_steps < 1:
        raise ValueError("dqn_exploration_steps must be positive")
    if not 0.0 <= args.dqn_final_epsilon <= 1.0:
        raise ValueError("dqn_final_epsilon must be between zero and one")


def meta_config(args, spec):
    _resolve_meta_defaults(args)
    sweep_plan = Path(args.sweep_plan).resolve() if args.sweep_plan else None
    profile = profile_for_spec(spec)
    config = {
        "schema_version": SCHEMA_VERSION,
        "stage": "meta_follower",
        "profile_id": profile.profile_id,
        "profile": profile.to_dict(),
        "matrix": spec.name,
        "payoffs": spec.payoffs.tolist(),
        "reward_offset": spec.reward_offset,
        "reward_profile": profile.profile_id,
        "episode_length": spec.episode_length,
        "memory_mode": spec.memory_mode,
        "query_states": list(range(spec.num_leader_states)),
        "algorithm": args.algorithm,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "episodes_per_batch": args.episodes_per_batch,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "net_arch": list(args.net_arch),
        "device": args.device,
        "checkpoint_every": args.checkpoint_every,
        "attempt": args.attempt,
        "sweep_id": args.sweep_id,
        "record_key": args.record_key,
        "sweep_plan": str(sweep_plan) if sweep_plan else None,
        "sweep_plan_sha256": file_sha256(sweep_plan) if sweep_plan else None,
    }
    if args.algorithm == "DQN":
        config["dqn_protocol"] = _resolved_dqn_protocol(args)
    elif args.algorithm == "REINFORCE":
        rollout_geometry = legacy_pg_rollout_geometry(
            spec.episode_length, spec.num_leader_states
        )
        checkpoint_geometry = reinforce_checkpoint_geometry(
            spec.episode_length, spec.num_leader_states
        )
        config["reinforce_protocol"] = {
            "loss": "negative_mean_logp_times_undiscounted_reward_to_go",
            "gamma": 1.0,
            "baseline": "none",
            "advantage_normalization": False,
            "entropy_coefficient": 0.0,
            "gradient_clipping": False,
            "optimizer": "Adam",
            "optimizer_epsilon": 1e-8,
            "learning_rate": float(args.learning_rate),
            "batch_steps": int(args.n_steps),
            "batch_mode": "complete_episodes",
            "rollout_geometry": rollout_geometry,
            "policy": "bias_free_linear_categorical",
            "policy_initialization": "normc_0.01",
            "default_pretraining_iterations": (
                LEGACY_REINFORCE_PRETRAIN_ITERATIONS
            ),
            "checkpoint_geometry": checkpoint_geometry,
            "checkpoint_semantics": "post_optimizer_update",
            "commitment_distribution": "uniform_deterministic_tables",
            "target": profile.interpretation,
            "state_count": int(spec.num_follower_states),
            "training_reward_offset": float(spec.reward_offset),
            "canonical_legacy_learning_rate": (
                LEGACY_REINFORCE_LEARNING_RATE
            ),
            "uses_canonical_legacy_learning_rate": math.isclose(
                args.learning_rate, LEGACY_REINFORCE_LEARNING_RATE
            ),
            "legacy_final_run_state_count": 3,
            "legacy_training_reward_offset": -2.5,
            "finite_sample_reward_protocol_differs_from_legacy": (
                profile.profile_id != "legacy_opponent_v1"
            ),
            "provenance": [
                "stackerlberg/train/experiments/configurations.py:"
                "smipd_leadermemory_pg_pg",
                "stackerlberg/models/linear_torch_model.py:LinearTorchModel",
                "stackerlberg/wrappers/dict_to_discrete_obs_wrapper.py:"
                "DictToDiscreteObsWrapper",
                "ray/rllib/algorithms/pg/pg_torch_policy.py:pg_torch_loss",
                "ray/rllib/algorithms/pg/utils.py:post_process_advantages",
            ],
        }
    return config


def _train_reinforce_with_post_update_checkpoints(
        model, args, spec, config, run_dir, run=None
):
    """Train in bounded chunks and certify only fully applied updates."""

    rollout = legacy_pg_rollout_geometry(
        spec.episode_length, spec.num_leader_states
    )
    cadence = reinforce_checkpoint_geometry(
        spec.episode_length, spec.num_leader_states
    )
    interval = cadence["follower_gradient_samples_interval"]
    config_digest = hashlib.sha256(
        canonical_json(config).encode("utf-8")
    ).hexdigest()
    checkpoint_log = run_dir / "checkpoint_evaluations.jsonl"
    artifacts = {"checkpoint_evaluations": checkpoint_log}
    final_evaluation = None
    first_chunk = True

    while int(model.num_timesteps) < int(args.timesteps):
        current = int(model.num_timesteps)
        next_regular = ((current // interval) + 1) * interval
        target = min(next_regular, int(args.timesteps))
        chunk = target - current
        if chunk < 1 or chunk % int(args.n_steps):
            raise RuntimeError(
                "post-update checkpoint chunk does not contain whole updates"
            )
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=first_chunk,
        )
        first_chunk = False

        follower_steps = int(model.num_timesteps)
        completed_updates = int(model._n_updates)
        expected_updates = follower_steps // int(args.n_steps)
        if completed_updates != expected_updates:
            raise RuntimeError(
                "REINFORCE update accounting mismatch: observed {}, expected {}"
                .format(completed_updates, expected_updates)
            )
        executed_steps = (
            completed_updates * rollout["executed_env_steps_per_update"]
        )
        is_final = follower_steps == int(args.timesteps)
        checkpoint = (
            run_dir / "model.zip" if is_final else
            run_dir / "meta_follower_update{:04d}_follower{}.zip".format(
                completed_updates, follower_steps
            )
        )
        evaluation, record = save_evaluated_response_checkpoint(
            model,
            checkpoint,
            lambda current_model: evaluate_meta_follower(current_model, spec),
        )
        record.update({
            "schema_version": SCHEMA_VERSION,
            "profile_id": config["profile_id"],
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "completed_updates": completed_updates,
            "follower_gradient_samples": follower_steps,
            "executed_equivalent_steps": int(executed_steps),
            "training_config_sha256": config_digest,
            "checkpoint_semantics": "post_optimizer_update",
        })
        contract_path = (
            run_dir / "response_contract.json" if is_final else
            checkpoint.with_suffix(".response_contract.json")
        )
        write_response_checkpoint_contract(
            spec,
            checkpoint,
            "REINFORCE",
            path=contract_path,
            metadata={
                "evaluation": evaluation["summary"],
                "seed": int(args.seed),
                "training_config_sha256": config_digest,
                "response_training": config["reinforce_protocol"],
                "completed_updates": completed_updates,
                "follower_gradient_samples": follower_steps,
                "executed_equivalent_steps": int(executed_steps),
            },
        )
        record.update({
            "contract_filename": contract_path.name,
            "contract_sha256": file_sha256(contract_path),
        })
        append_jsonl(checkpoint_log, record)

        suffix = "final" if is_final else "update{:04d}".format(
            completed_updates
        )
        artifacts["checkpoint_{}".format(suffix)] = checkpoint
        artifacts["checkpoint_contract_{}".format(suffix)] = contract_path
        if is_final:
            artifacts["model"] = checkpoint
            artifacts["response_contract"] = contract_path
            final_evaluation = evaluation
        if run is not None:
            run.log({
                "checkpoint_eval/mean_regret": record["mean_regret"],
                "checkpoint_eval/max_regret": record["max_regret"],
                "checkpoint_eval/optimal_commitments": (
                    record["optimal_commitments"]
                ),
                "checkpoint_eval/completed_updates": completed_updates,
                "checkpoint_eval/follower_gradient_samples": follower_steps,
                "checkpoint_eval/executed_equivalent_steps": executed_steps,
            }, step=executed_steps)

    if final_evaluation is None:
        raise RuntimeError("REINFORCE training produced no final checkpoint")
    return final_evaluation, artifacts


def train_meta_follower(args):
    validate_meta_args(args)
    spec = _meta_spec(args)
    config = meta_config(args, spec)
    run_dir = _run_dir(args.output_root, config)
    config["run_dir"] = str(run_dir)
    write_json(run_dir / "config.json", config)
    manifest = start_manifest(run_dir, config)
    run = _maybe_wandb(args, config, run_dir.name)
    vec_env = DummyVecEnv([
        lambda: Monitor(
            MatrixFixedCommitmentResponseEnv(spec, seed=args.seed),
            filename=str(run_dir / "train_monitor"),
        )
    ])
    try:
        model = _meta_model(args, vec_env, spec.episode_length)
        if args.algorithm == "REINFORCE":
            evaluation, artifacts = _train_reinforce_with_post_update_checkpoints(
                model, args, spec, config, run_dir, run=run
            )
            contract_path = artifacts["response_contract"]
        else:
            callbacks = []
            if args.checkpoint_every > 0:
                callbacks.append(CheckpointCallback(
                    save_freq=args.checkpoint_every,
                    save_path=str(run_dir),
                    name_prefix="meta_follower_step",
                ))
            model.learn(
                total_timesteps=args.timesteps,
                callback=CallbackList(callbacks) if callbacks else None,
            )
            model.save(str(run_dir / "model"))
            evaluation = evaluate_meta_follower(model, spec)
            contract_path = write_response_checkpoint_contract(
                spec,
                run_dir / "model.zip",
                args.algorithm,
                metadata={
                    "evaluation": evaluation["summary"],
                    "seed": args.seed,
                    "training_config_sha256": manifest["config_sha256"],
                },
            )
            artifacts = {
                "model": run_dir / "model.zip",
                "response_contract": contract_path,
            }
        evaluation.update({"schema_version": SCHEMA_VERSION, "config": config})
        write_json(run_dir / "evaluation.json", evaluation)
        artifacts["evaluation"] = run_dir / "evaluation.json"
        finish_manifest(
            run_dir,
            manifest,
            "completed",
            artifacts=artifacts,
        )
        if run is not None:
            run.log({
                "evaluation/mean_regret": evaluation["summary"]["mean_regret"],
                "evaluation/max_regret": evaluation["summary"]["max_regret"],
            })
    except Exception as exc:
        finish_manifest(run_dir, manifest, "failed", error=exc)
        raise
    finally:
        vec_env.close()
        if run is not None:
            run.finish()
    return run_dir
