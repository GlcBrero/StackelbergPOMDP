"""Run one qualitative matrix-game paper experiment.

This module is deliberately a single-run interface.  It never launches a
sweep; see ``replication/matrix_ablations/sweep.py`` for explicit planning.
"""

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import gym
import stable_baselines3
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from stackelberg_pomdp.callbacks import FixPolicyActionsCallback
from stackelberg_pomdp.envs.matrix import (
    MatrixFixedCommitmentResponseEnv,
    RepeatedMatrixGame,
    get_matrix_game,
    load_response_model,
    make_meta_leader_env,
    make_tabular_q_leader_env,
    validate_response_checkpoint_contract,
    write_response_checkpoint_contract,
)
from stackelberg_pomdp.matrix_ablations.profiles import (
    PROFILES,
    get_profile,
    profile_for_spec,
    profile_snapshot_for_spec,
)
from stackelberg_pomdp.matrix_ablations.rllib_es import (
    RllibESSettings,
    train_rllib_es,
)
from stackelberg_pomdp.matrix_ablations.reinforce import (
    LEGACY_REINFORCE_LEARNING_RATE,
    LEGACY_REINFORCE_PRETRAIN_ITERATIONS,
    REINFORCE_CHECKPOINT_INTERVAL_UPDATES,
    Reinforce,
    legacy_pg_rollout_geometry,
    reinforce_checkpoint_geometry,
    save_evaluated_response_checkpoint,
)
from stackelberg_pomdp.matrix_ablations.pg import (
    LEGACY_LEADER_PG_LEARNING_RATE,
    LEGACY_LEADER_PG_OUTER_UPDATES,
    LeaderPolicyGradient,
    leader_pg_rollout_geometry,
)
from stackelberg_pomdp.rl_trainer_setup import get_custom_training_algorithm


SCHEMA_VERSION = 2
DEFAULT_RESULTS_ROOT = Path("replication/matrix_ablations/results/single_runs")
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONDITIONS = {
    "hidden_queries": ("observed", "hidden"),
    "phase_observability": ("visible", "hidden"),
    "q_reset": ("reset", "ongoing"),
    "response_reward": ("excluded", "included"),
}
DEFAULT_MATRICES = {
    "hidden_queries": "modified_pd",
    "phase_observability": "prisoners_dilemma",
    "q_reset": "battle_of_the_sexes",
    "response_reward": "coordination_zero_miscoordination",
}
LEADER_PROFILE_MATRICES = {
    "paper_joint_v1": "modified_pd",
    "paper_opponent_sensitivity_v1": "modified_pd",
    "legacy_opponent_v1": "prisoners_dilemma",
}


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


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("cannot serialize {!r}".format(type(value)))


def canonical_json(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload))
        handle.write("\n")


def file_sha256(path):
    path = Path(path)
    if path.is_dir():
        digest = hashlib.sha256(b"tree-sha256-v1\0")
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(file_sha256(child)))
        return digest.hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "sem": (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else 0.0
        ),
        "minimum": float(np.min(values)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
    }


def _git_value(*args):
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance():
    source_paths = [
        Path(__file__),
        REPO_ROOT / "stackelberg_pomdp/envs/matrix.py",
        REPO_ROOT / "stackelberg_pomdp/matrix_ablations/rllib_es.py",
        REPO_ROOT / "stackelberg_pomdp/matrix_ablations/profiles.py",
        REPO_ROOT / "stackelberg_pomdp/matrix_ablations/pg.py",
        REPO_ROOT / "stackelberg_pomdp/matrix_ablations/reinforce.py",
        REPO_ROOT / "stackelberg_pomdp/algorithms/on_policy.py",
        REPO_ROOT / "stackelberg_pomdp/policies/generic.py",
        REPO_ROOT / "stackelberg_pomdp/callbacks.py",
        REPO_ROOT / "stackelberg_pomdp/rl_trainer_setup.py",
        REPO_ROOT / "environment.yml",
    ]
    status = _git_value("status", "--porcelain")
    return {
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "hostname": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "gym_version": gym.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): file_sha256(path)
            for path in source_paths if path.exists()
        },
    }


def start_manifest(run_dir, config):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_unix": time.time(),
        "config_sha256": hashlib.sha256(
            canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "provenance": provenance(),
    }
    write_json(run_dir / "run_manifest.json", payload)
    return payload


def finish_manifest(
        run_dir, manifest, status, artifacts=None, error=None, metadata=None
):
    payload = dict(manifest)
    payload.update({"status": status, "finished_unix": time.time()})
    if metadata:
        payload.update(metadata)
    if artifacts:
        payload["artifacts"] = {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in artifacts.items() if Path(path).exists()
        }
    if error is not None:
        payload["error"] = str(error)
        payload["traceback"] = traceback.format_exc()
    write_json(run_dir / "run_manifest.json", payload)


def _checkpoint_path(path):
    path = Path(path)
    if path.exists():
        return path.resolve()
    zipped = Path(str(path) + ".zip")
    if zipped.exists():
        return zipped.resolve()
    raise FileNotFoundError("checkpoint does not exist: {}".format(path))


def _run_id(config):
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:12]


def _run_dir(output_root, config):
    if config["stage"] == "meta_follower":
        parts = (
            "meta_follower",
            config["memory_mode"],
            config["algorithm"].lower(),
            "seed{}-{}".format(config["seed"], _run_id(config)),
        )
    else:
        parts = (
            config["experiment"],
            config["matrix"],
            config["algorithm"].lower(),
            config["condition"],
            "seed{}-{}".format(config["seed"], _run_id(config)),
        )
    path = Path(output_root).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    protected = (
        "config.json", "run_manifest.json", "progress.jsonl",
        "evaluation.json", "model.zip",
        "response_lookup.json", "ray_checkpoint",
    )
    existing = [name for name in protected if (path / name).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite run artifacts {} in {}".format(
                ", ".join(existing), path
            )
        )
    return path


def _maybe_wandb(args, config, run_name):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("--wandb requested but wandb is unavailable") from exc
    return wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name or run_name,
        config=config,
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
    """Apply algorithm-specific defaults without changing existing PPO paths."""

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
            LEGACY_REINFORCE_LEARNING_RATE if reinforce else 2e-3
        ),
        "ent_coef": 0.0 if reinforce else 0.01,
        "episodes_per_batch": (
            geometry["episodes_per_update"] if reinforce else 32
        ),
        "net_arch": () if reinforce else (32, 32),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if reinforce and args.n_steps is None:
        args.n_steps = geometry["follower_gradient_samples_per_update"]
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
        gae_lambda=1.0,
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
            batch_size=int(args.batch_size or n_steps),
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
    """Resolve the compact replay protocol recorded with each DQN response."""

    _resolve_meta_defaults(args)
    batch_size = int(args.batch_size or 32)
    buffer_size = max(
        int(args.timesteps), int(args.dqn_learning_starts) + 1, batch_size
    )
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


def validate_logging_args(args):
    if args.attempt < 0:
        raise ValueError("attempt must be nonnegative")
    if bool(args.sweep_id) != bool(args.record_key):
        raise ValueError("sweep_id and record_key must be supplied together")
    if args.sweep_plan:
        sweep_plan = Path(args.sweep_plan)
        if not sweep_plan.is_file():
            raise FileNotFoundError("sweep plan does not exist: {}".format(sweep_plan))
        if not args.sweep_id:
            raise ValueError("sweep_plan requires sweep_id and record_key")


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
                "smipd_hiddenqueries_pg_pg_new",
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


def _resolved_q_protocol(args):
    if args.experiment == "q_reset":
        defaults = (0.1, 0.1, "epsilon_greedy", "small_normal")
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


def _leader_pg_geometry(args, spec):
    """Resolve the complete-episode PG batch for one leader condition."""

    if args.experiment != "hidden_queries":
        raise ValueError(
            "legacy-faithful PG is currently defined only for hidden_queries"
        )
    return leader_pg_rollout_geometry(
        episode_length=spec.episode_length,
        num_query_states=spec.num_leader_states,
        hidden_queries=(args.condition == "hidden"),
    )


def validate_leader_args(args):
    validate_logging_args(args)
    if args.ent_coef is None:
        args.ent_coef = 0.0 if args.algorithm in ("PG", "ES") else 0.01
    if args.profile_id is not None:
        if args.experiment != "hidden_queries":
            raise ValueError("named leader profiles are defined only for hidden_queries")
        profile = get_profile(args.profile_id)
        expected_matrix = LEADER_PROFILE_MATRICES[profile.profile_id]
        if args.matrix is not None and args.matrix != expected_matrix:
            raise ValueError(
                "profile {!r} requires matrix {!r}".format(
                    profile.profile_id, expected_matrix
                )
            )
        if (
                args.memory_mode is not None
                and args.memory_mode != profile.memory_mode
        ):
            raise ValueError(
                "profile {!r} requires memory_mode {!r}".format(
                    profile.profile_id, profile.memory_mode
                )
            )
        args.matrix = expected_matrix
        args.memory_mode = profile.memory_mode
    else:
        args.matrix = args.matrix or DEFAULT_MATRICES[args.experiment]
        if (
                args.experiment in ("hidden_queries", "phase_observability")
                and args.memory_mode is None
        ):
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
    if args.algorithm != "ES" and args.timesteps < 1:
        raise ValueError("timesteps must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if args.eval_episodes < 1 or args.final_eval_episodes < 1:
        raise ValueError("evaluation episode counts must be positive")
    if args.response_episodes < 1:
        raise ValueError("response_episodes must be positive")
    if args.algorithm == "ES":
        if args.experiment != "hidden_queries":
            raise ValueError("ES is defined only for hidden_queries")
        if args.ent_coef != 0.0:
            raise ValueError("ES uses parameter noise, not an entropy bonus")
        if args.es_sigma <= 0 or args.es_stepsize <= 0:
            raise ValueError("ES sigma and stepsize must be positive")
        if args.es_l2_coeff < 0:
            raise ValueError("ES l2_coeff must be nonnegative")
        if args.es_num_workers < 1:
            raise ValueError("ES requires at least one rollout worker")
        if args.es_episodes_per_batch < 1 or args.es_train_batch_size < 1:
            raise ValueError("ES batch minima must be positive")
        if args.es_noise_size < 1 or args.es_report_length < 1:
            raise ValueError("ES noise_size and report_length must be positive")
        if not 0.0 <= args.es_eval_prob <= 1.0:
            raise ValueError("ES eval_prob must lie in [0, 1]")
        if args.checkpoint_every:
            raise ValueError(
                "ES always retains its terminal native RLlib checkpoint; "
                "--checkpoint-every is unsupported"
            )
    if args.algorithm == "PG":
        if args.experiment != "hidden_queries":
            raise ValueError(
                "legacy-faithful PG is currently defined only for "
                "hidden_queries; other diagnostics require a separate "
                "protocol audit"
            )
        if args.learning_rate <= 0:
            raise ValueError("PG learning_rate must be positive")
        if args.ent_coef != 0.0:
            raise ValueError("legacy PG has no entropy bonus")
    if args.es_eval_every is not None and args.es_eval_every < 1:
        raise ValueError("es_eval_every must be positive")
    if args.es_iterations is not None and args.es_iterations < 1:
        raise ValueError("es_iterations must be positive when specified")
    if args.experiment in ("hidden_queries", "phase_observability"):
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
    if args.profile_id == "legacy_opponent_v1":
        raise ValueError(
            "legacy_opponent_v1 is retained as provenance only, not an active "
            "leader treatment"
        )
    if args.experiment == "hidden_queries" and args.matrix != "modified_pd":
        raise ValueError("hidden_queries requires modified_pd")
    if args.algorithm == "ES" and args.memory_mode != "joint":
        raise ValueError("native RLlib ES requires joint five-state memory")
    if (
        args.experiment == "phase_observability"
        and args.matrix != "prisoners_dilemma"
    ):
        raise ValueError("phase_observability requires prisoners_dilemma")
    if args.eval_warmup is None:
        args.eval_warmup = 100 if (
            args.experiment == "q_reset" and args.condition == "ongoing"
        ) else 0
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
        "seed": args.seed,
        # Native RLlib ES is controlled by es_iterations.  Its time-based
        # batches overshoot both batch minima, so no requested timestep count
        # is meaningful; measured cumulative timesteps remain in progress.
        "timesteps": None if args.algorithm == "ES" else args.timesteps,
        "learning_rate": (
            None if args.algorithm == "ES" else args.learning_rate
        ),
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
    if args.algorithm == "ES":
        settings = RllibESSettings(
            iterations=args.es_iterations,
            num_workers=args.es_num_workers,
            episodes_per_batch=args.es_episodes_per_batch,
            train_batch_size=args.es_train_batch_size,
            noise_stdev=args.es_sigma,
            stepsize=args.es_stepsize,
            l2_coeff=args.es_l2_coeff,
            eval_prob=args.es_eval_prob,
            report_length=args.es_report_length,
            noise_size=args.es_noise_size,
            explicit_eval_every=args.es_eval_every,
            explicit_eval_episodes=args.eval_episodes,
            final_eval_episodes=args.final_eval_episodes,
        )
        config["es_settings"] = asdict(settings)
        config["es_protocol"] = {
            **settings.protocol(spec),
            "validation_role": "paper_profile_qualitative_revalidation",
            "matrix": spec.name,
            "memory_mode": spec.memory_mode,
            "leader_states": int(spec.num_leader_states),
            "response_checkpoint_sha256": response_hash,
            "historical_effective_runtime": {
                "ray_version": "2.0.1",
                "policy": "default_fully_connected_2x256_tanh",
                "observation_filter": "MeanStdFilter",
                "noise_stdev": 0.02,
                "stepsize": 0.01,
                "l2_coeff": 0.005,
                "episodes_per_batch": 1000,
                "train_batch_size": 1000,
                "eval_prob": 0.03,
                "report_length": 10,
            },
            "provenance": [
                "stackerlberg/train/experiments/configurations.py:"
                "smipd_es_pg_new",
                "stackerlberg/trainers/stackerlberg_trainable_es.py:"
                "leader_config.pop_multiagent",
                "ray/rllib/algorithms/es/es.py@2.0.1",
                "ray/rllib/algorithms/es/es_torch_policy.py@2.0.1",
                "ray/rllib/algorithms/es/optimizers.py@2.0.1",
            ],
        }
    elif args.algorithm == "PG":
        geometry = _leader_pg_geometry(args, spec)
        planned_updates = int(math.ceil(
            args.timesteps / geometry["executed_env_steps_per_update"]
        ))
        config["pg_protocol"] = {
            "loss": "negative_mean_logp_times_undiscounted_reward_to_go",
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "critic": "none_frozen_zero_compatibility_head",
            "baseline": "none",
            "advantage_normalization": False,
            "entropy_coefficient": 0.0,
            "gradient_clipping": False,
            "optimizer": "Adam",
            "optimizer_epsilon": 1e-8,
            "learning_rate": float(args.learning_rate),
            "policy": "bias_free_linear_categorical",
            "policy_initialization": "row_normc_0.01",
            "training_action_sampling": "independent_per_state_visit",
            "stationarity_constraint": (
                "one_shared_policy_distribution_across_query_and_reward_phases"
            ),
            "evaluation_action_selection": "deterministic_argmax",
            "batch_mode": "complete_episodes",
            "hidden_query_transitions": (
                "executed_but_excluded_from_rollout"
                if geometry["hidden_queries"] else "executed_and_stored"
            ),
            "rollout_geometry": geometry,
            "requested_executed_env_steps": int(args.timesteps),
            "planned_updates": planned_updates,
            "planned_executed_env_steps": int(
                planned_updates * geometry["executed_env_steps_per_update"]
            ),
            "planned_stored_gradient_samples": int(
                planned_updates
                * geometry["stored_gradient_samples_per_update"]
            ),
            "canonical_legacy_learning_rate": (
                LEGACY_LEADER_PG_LEARNING_RATE
            ),
            "canonical_legacy_outer_updates": (
                LEGACY_LEADER_PG_OUTER_UPDATES
            ),
            "provenance": [
                "stackerlberg/train/experiments/configurations.py:"
                "smipd_hiddenqueries_pg_pg_new",
                "stackerlberg/models/linear_torch_model.py:LinearTorchModel",
                "stackerlberg/trainers/callbacks.py:"
                "DeleteHiddenQueriesCallback",
                "ray/rllib/algorithms/pg/pg_torch_policy.py:pg_torch_loss",
                "ray/rllib/algorithms/pg/utils.py:post_process_advantages",
            ],
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
    if args.experiment in ("hidden_queries", "phase_observability"):
        shared_response_model = load_response_model(
            args.response_checkpoint, args.response_algorithm, device=args.device
        )

    def factory(seed):
        if args.experiment == "hidden_queries":
            return make_meta_leader_env(
                spec,
                response_checkpoint=args.response_checkpoint,
                response_algorithm=args.response_algorithm,
                response_model=shared_response_model,
                query_visibility=args.condition,
                phase_observable=None,
                seed=seed,
                device=args.device,
            )
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


def evaluate_leader_policy(model, env_factory, episodes, warmup, seed_start):
    env = env_factory(seed_start)
    rows = []
    try:
        if hasattr(model.policy, "fix_policy_actions"):
            model.policy.fix_policy_actions()
        dependent_stream = bool(getattr(env.unwrapped, "warm_start_q", False))
        for sequence_index in range(int(warmup) + int(episodes)):
            if not dependent_stream and hasattr(env.unwrapped, "seed"):
                env.unwrapped.seed(int(seed_start) + sequence_index)
            if hasattr(model.policy, "clear_obs_action_map"):
                model.policy.clear_obs_action_map()
            observation = env.reset()
            total = 0.0
            reward_phase_total = 0.0
            reward_phase_steps = 0
            executed_steps = 0
            commitment_consistent = True
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, done, info = env.step(action)
                total += float(reward)
                executed_steps += 1
                if info.get("is_reward_phase", False):
                    reward_phase_total += float(reward)
                    reward_phase_steps += 1
                    commitment_consistent = (
                        commitment_consistent
                        and info.get("commitment_consistent", True)
                    )
            if sequence_index >= warmup:
                rows.append({
                    "episode": int(sequence_index - warmup),
                    "seed": (
                        None if dependent_stream
                        else int(seed_start) + sequence_index
                    ),
                    "rng_stream_seed": int(seed_start),
                    "stream_sequence_index": int(sequence_index),
                    "dependent_response_stream": dependent_stream,
                    "outer_return": total,
                    "reward_phase_return": reward_phase_total,
                    "reward_phase_steps": reward_phase_steps,
                    "leader_reward_per_stage": (
                        reward_phase_total / reward_phase_steps
                    ),
                    "executed_steps": executed_steps,
                    "commitment_consistent": bool(commitment_consistent),
                })
    finally:
        env.close()
        if hasattr(model.policy, "clear_obs_action_map"):
            model.policy.clear_obs_action_map()
    return {
        "per_stage_summary": mean_summary([
            row["leader_reward_per_stage"] for row in rows
        ]),
        "return_summary": mean_summary([
            row["reward_phase_return"] for row in rows
        ]),
        "dependent_response_stream": bool(
            rows and rows[0]["dependent_response_stream"]
        ),
        "episode_rows": rows,
    }


class JsonlEvaluationCallback(BaseCallback):
    """Evaluate at completed outer episodes on a fixed target-step grid."""

    def __init__(
            self,
            eval_factory,
            output_path,
            eval_freq,
            eval_episodes,
            eval_warmup,
            eval_seed_start,
            metadata,
            wandb_run=None,
    ):
        super().__init__()
        self.eval_factory = eval_factory
        self.output_path = Path(output_path)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.eval_warmup = int(eval_warmup)
        self.eval_seed_start = int(eval_seed_start)
        self.metadata = dict(metadata)
        self.wandb_run = wandb_run
        self.next_target = self.eval_freq

    def _on_step(self):
        dones = np.asarray(self.locals.get("dones", [False]), dtype=bool)
        if not np.any(dones) or self.num_timesteps < self.next_target:
            return True
        target = self.next_target
        result = evaluate_leader_policy(
            self.model,
            self.eval_factory,
            self.eval_episodes,
            self.eval_warmup,
            self.eval_seed_start,
        )
        summary = result["per_stage_summary"]
        row = {
            **self.metadata,
            "evaluation_target_step": int(target),
            "global_step": int(self.num_timesteps),
            "wall_time": time.time(),
            "evaluation_mean": summary["mean"],
            "evaluation_std": summary["std"],
            "evaluation_sem": summary["sem"],
            "evaluation_episodes": summary["n"],
            "dependent_response_stream": result["dependent_response_stream"],
        }
        append_jsonl(self.output_path, row)
        if self.wandb_run is not None:
            self.wandb_run.log({
                "global_step": row["global_step"],
                "evaluation/leader_reward_mean": row["evaluation_mean"],
                "evaluation/leader_reward_sem": row["evaluation_sem"],
            })
        if hasattr(self.model.policy, "clear_obs_action_map"):
            self.model.policy.clear_obs_action_map()
        while self.next_target <= self.num_timesteps:
            self.next_target += self.eval_freq
        return True


def _progress_metadata(args):
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment": args.experiment,
        "condition": args.condition,
        "matrix": args.matrix,
        "algorithm": args.algorithm,
        "seed": args.seed,
    }
    if args.algorithm == "ES":
        metadata.update({
            "es_implementation": "ray_rllib_es_2_0_1",
            "es_stepsize": args.es_stepsize,
            "es_sigma": args.es_sigma,
            "es_l2_coeff": args.es_l2_coeff,
        })
    else:
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
    callbacks = [
        FixPolicyActionsCallback(),
        JsonlEvaluationCallback(
            evaluation_factory,
            run_dir / "progress.jsonl",
            args.eval_freq,
            args.eval_episodes,
            args.eval_warmup,
            args.eval_seed_start,
            metadata,
            wandb_run,
        ),
    ]
    if args.checkpoint_every > 0:
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_every,
            save_path=str(run_dir),
            name_prefix="leader_step",
        ))
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
        )
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


def train_pg_leader(args, spec, run_dir, config, wandb_run=None):
    """Train the legacy matrix PG leader with post-update accounting."""

    training_factory = leader_env_factory(args, spec, evaluation=False)
    evaluation_factory = leader_env_factory(args, spec, evaluation=True)
    env = Monitor(
        training_factory(args.seed), filename=str(run_dir / "train_monitor")
    )
    protocol = config["pg_protocol"]
    geometry = protocol["rollout_geometry"]
    model = LeaderPolicyGradient(
        env=env,
        learning_rate=args.learning_rate,
        n_steps=geometry["stored_gradient_samples_per_update"],
        seed=args.seed,
        device=args.device,
        verbose=0,
    )
    model.min_completed_episodes_per_rollout = geometry["episodes_per_update"]
    metadata = _progress_metadata(args)
    counter_settings = {
        "collection_target_env_steps": geometry[
            "collection_target_env_steps"
        ],
        "episodes_per_update": geometry["episodes_per_update"],
        "stored_gradient_samples_per_update": geometry[
            "stored_gradient_samples_per_update"
        ],
        "executed_env_steps_per_update": geometry[
            "executed_env_steps_per_update"
        ],
        "hidden_query_steps_excluded_per_update": geometry[
            "hidden_query_steps_excluded_per_update"
        ],
    }

    def accounting(current_model):
        updates = int(current_model._n_updates)
        stored = (
            updates * geometry["stored_gradient_samples_per_update"]
        )
        executed = int(current_model.num_timesteps)
        expected_executed = (
            updates * geometry["executed_env_steps_per_update"]
        )
        if stored != updates * int(current_model.n_steps):
            raise RuntimeError("PG stored-sample accounting mismatch")
        if executed != expected_executed:
            raise RuntimeError(
                "PG executed-step accounting mismatch: observed {}, expected {}"
                .format(executed, expected_executed)
            )
        if updates and (
            int(current_model.last_rollout_stored_steps)
            != geometry["stored_gradient_samples_per_update"]
            or int(current_model.last_rollout_executed_steps)
            != geometry["executed_env_steps_per_update"]
            or int(current_model.last_rollout_completed_episodes)
            != geometry["episodes_per_update"]
        ):
            raise RuntimeError("PG last-rollout geometry mismatch")
        return {
            "completed_updates": updates,
            "stored_gradient_samples": int(stored),
            "executed_env_steps": executed,
            **counter_settings,
        }

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
        "row_type": "evaluation",
        "evaluation_target_step": 0,
        "global_step": 0,
        "wall_time": time.time(),
        "evaluation_mean": initial_summary["mean"],
        "evaluation_std": initial_summary["std"],
        "evaluation_sem": initial_summary["sem"],
        "evaluation_episodes": initial_summary["n"],
        "dependent_response_stream": initial["dependent_response_stream"],
        **accounting(model),
    })

    next_evaluation = [int(args.eval_freq)]
    next_checkpoint = [
        int(args.checkpoint_every) if args.checkpoint_every else None
    ]
    checkpoint_artifacts = {}

    def post_update(current_model):
        counters = accounting(current_model)
        append_jsonl(run_dir / "progress.jsonl", {
            **metadata,
            "row_type": "training_update",
            "global_step": counters["executed_env_steps"],
            "wall_time": time.time(),
            "policy_loss": current_model.last_policy_loss,
            **counters,
        })

        if counters["executed_env_steps"] >= next_evaluation[0]:
            target = next_evaluation[0]
            result = evaluate_leader_policy(
                current_model,
                evaluation_factory,
                args.eval_episodes,
                args.eval_warmup,
                args.eval_seed_start,
            )
            summary = result["per_stage_summary"]
            row = {
                **metadata,
                "row_type": "evaluation",
                "evaluation_target_step": int(target),
                "global_step": counters["executed_env_steps"],
                "wall_time": time.time(),
                "evaluation_mean": summary["mean"],
                "evaluation_std": summary["std"],
                "evaluation_sem": summary["sem"],
                "evaluation_episodes": summary["n"],
                "dependent_response_stream": result[
                    "dependent_response_stream"
                ],
                **counters,
            }
            append_jsonl(run_dir / "progress.jsonl", row)
            if wandb_run is not None:
                wandb_run.log({
                    "evaluation/leader_reward_mean": row["evaluation_mean"],
                    "evaluation/leader_reward_sem": row["evaluation_sem"],
                    "training/completed_updates": counters[
                        "completed_updates"
                    ],
                    "training/stored_gradient_samples": counters[
                        "stored_gradient_samples"
                    ],
                    "training/executed_env_steps": counters[
                        "executed_env_steps"
                    ],
                }, step=counters["executed_env_steps"])
            while next_evaluation[0] <= counters["executed_env_steps"]:
                next_evaluation[0] += int(args.eval_freq)

        checkpoint_target = next_checkpoint[0]
        if (
            checkpoint_target is not None
            and counters["executed_env_steps"] >= checkpoint_target
        ):
            checkpoint = run_dir / (
                "leader_update{:04d}_executed{}.zip".format(
                    counters["completed_updates"],
                    counters["executed_env_steps"],
                )
            )
            current_model.save(str(checkpoint.with_suffix("")))
            checkpoint_artifacts[
                "checkpoint_update{:04d}".format(
                    counters["completed_updates"]
                )
            ] = checkpoint
            while next_checkpoint[0] <= counters["executed_env_steps"]:
                next_checkpoint[0] += int(args.checkpoint_every)

    model.post_update_hook = post_update
    try:
        model.learn(total_timesteps=args.timesteps)
        final_accounting = accounting(model)
        if final_accounting["completed_updates"] != protocol["planned_updates"]:
            raise RuntimeError(
                "PG completed-update count differs from the immutable plan"
            )
        model.post_update_hook = None
        model.save(str(run_dir / "model"))
        evaluation = evaluate_leader_policy(
            model,
            evaluation_factory,
            args.final_eval_episodes,
            args.eval_warmup,
            args.final_eval_seed_start,
        )
        evaluation.update({
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "training_accounting": final_accounting,
        })
        write_json(run_dir / "evaluation.json", evaluation)
    finally:
        model.post_update_hook = None
        env.close()

    artifacts = {
        "model": run_dir / "model.zip",
        "evaluation": run_dir / "evaluation.json",
        "progress": run_dir / "progress.jsonl",
        **checkpoint_artifacts,
    }
    manifest_metadata = {
        "training_accounting": final_accounting,
        "training_protocol": protocol,
    }
    return artifacts, manifest_metadata


def train_rllib_es_leader(args, spec, run_dir, config, wandb_run=None):
    """Train the sole maintained ES treatment with native RLlib 2.0.1."""

    settings = RllibESSettings(**config["es_settings"])
    metadata = _progress_metadata(args)

    def log_progress(row):
        payload = {
            **metadata,
            **row,
            "global_step": int(row["measured_timesteps_total"]),
            "evaluation_target_step": int(row["measured_timesteps_total"]),
            "evaluation_mean": row["native_leader_reward_per_stage"],
            "wall_time": time.time(),
        }
        append_jsonl(run_dir / "progress.jsonl", payload)
        if wandb_run is not None and payload["evaluation_mean"] is not None:
            wandb_run.log({
                "global_step": payload["global_step"],
                "iteration": payload["iteration"],
                "native/episode_reward_mean": payload[
                    "native_episode_reward_mean"
                ],
                "evaluation/leader_reward_mean": payload["evaluation_mean"],
            }, step=payload["global_step"])

    result = train_rllib_es(
        spec=spec,
        condition=args.condition,
        seed=args.seed,
        response_checkpoint=args.response_checkpoint,
        response_algorithm=args.response_algorithm,
        run_dir=run_dir,
        settings=settings,
        progress_callback=log_progress,
        explicit_eval_seed_start=args.eval_seed_start,
        final_eval_seed_start=args.final_eval_seed_start,
    )
    final = result["evaluation"]["stochastic_evaluation"]
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        **result["evaluation"],
        "per_stage_summary": final["per_stage_summary"],
        "return_summary": final["return_summary"],
        "episode_rows": final["episodes"],
        "last_training_row": result["progress"][-1],
    }
    write_json(run_dir / "evaluation.json", evaluation)
    artifacts = {
        **result["artifacts"],
        "evaluation": run_dir / "evaluation.json",
        "progress": run_dir / "progress.jsonl",
    }
    return artifacts, result["metadata"]


def train_leader(args):
    validate_leader_args(args)
    if args.profile_id is not None:
        spec = get_matrix_game(args.matrix, profile_id=args.profile_id)
    else:
        spec = get_matrix_game(args.matrix, memory_mode=(
            args.memory_mode if args.experiment in (
                "hidden_queries", "phase_observability"
            ) else "none"
        ))
    if args.algorithm == "ES":
        if args.es_iterations is None:
            args.es_iterations = 300
    _validate_response_quality(args, spec)
    config = leader_config(args, spec)
    run_dir = _run_dir(args.output_root, config)
    config["run_dir"] = str(run_dir)
    write_json(run_dir / "config.json", config)
    manifest = start_manifest(run_dir, config)
    run = _maybe_wandb(args, config, run_dir.name)
    try:
        manifest_metadata = None
        if args.algorithm == "ES":
            artifacts, manifest_metadata = train_rllib_es_leader(
                args, spec, run_dir, config, run
            )
        elif args.algorithm == "PG":
            artifacts, manifest_metadata = train_pg_leader(
                args, spec, run_dir, config, run
            )
        else:
            artifacts = train_sb3_leader(args, spec, run_dir, config, run)
        finish_manifest(
            run_dir,
            manifest,
            "completed",
            artifacts=artifacts,
            metadata=manifest_metadata,
        )
    except Exception as exc:
        finish_manifest(run_dir, manifest, "failed", error=exc)
        raise
    finally:
        if run is not None:
            run.finish()
    return run_dir


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
        "--matrix", choices=("modified_pd", "prisoners_dilemma"),
        default="modified_pd",
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
    meta.add_argument("--n-epochs", type=int, default=4)
    meta.add_argument(
        "--net-arch", type=positive_int_tuple,
        help=("Comma-separated hidden-layer widths, or 'linear'. Algorithm-"
              "specific defaults preserve PPO/A2C and legacy REINFORCE."),
    )
    dqn = meta.add_argument_group("DQN replay and exploration")
    dqn.add_argument(
        "--dqn-learning-starts", type=int, default=0,
        help="Replay transitions collected before learning starts.",
    )
    dqn.add_argument(
        "--dqn-target-update-interval", type=int, default=1_000,
        help="Environment steps between target-network updates.",
    )
    dqn.add_argument(
        "--dqn-exploration-steps", type=int, default=4_000,
        help="Steps over which epsilon decays from one to its final value.",
    )
    dqn.add_argument(
        "--dqn-final-epsilon", type=float, default=0.1,
        help="Final epsilon for DQN exploration.",
    )
    meta.add_argument("--device", default="cpu")
    add_logging_args(meta)

    leader = subparsers.add_parser(
        "leader", description="Train one condition from a matrix diagnostic."
    )
    leader.add_argument(
        "--experiment", choices=tuple(EXPERIMENT_CONDITIONS), required=True
    )
    leader.add_argument("--condition", required=True)
    leader.add_argument("--matrix", choices=tuple(sorted(MATRIX for MATRIX in (
        "modified_pd", "prisoners_dilemma", "battle_of_the_sexes",
        "coordination_zero_miscoordination",
        "coordination_penalized_miscoordination",
    ))))
    leader.add_argument("--memory-mode", choices=("joint", "opponent"))
    leader.add_argument(
        "--profile-id", choices=tuple(sorted(PROFILES)),
        help=(
            "Explicit repeated-game profile. legacy_opponent_v1 is provenance "
            "only and cannot launch a maintained leader treatment."
        ),
    )
    leader.add_argument(
        "--algorithm", choices=("PG", "PPO", "A2C", "ES"),
        default="A2C",
    )
    leader.add_argument("--seed", type=int, default=1)
    leader.add_argument(
        "--timesteps", type=int, default=200_000,
        help=(
            "Requested environment steps for PG/A2C/PPO; ignored for native "
            "RLlib ES, which is controlled by --es-iterations."
        ),
    )
    leader.add_argument("--learning-rate", type=float, default=8e-3)
    leader.add_argument(
        "--ent-coef", type=float,
        help="Defaults to 0 for PG/ES and 0.01 for A2C/PPO.",
    )
    leader.add_argument("--response-checkpoint")
    leader.add_argument("--response-contract-path")
    leader.add_argument(
        "--response-algorithm",
        choices=("PPO", "A2C", "DQN", "REINFORCE"), default="PPO"
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
    leader.add_argument("--ppo-episodes-per-batch", type=int, default=16)
    leader.add_argument("--ppo-batch-size", type=int)
    leader.add_argument("--ppo-n-epochs", type=int, default=4)
    leader.add_argument("--eval-freq", type=int, default=2_000)
    leader.add_argument("--eval-episodes", type=int, default=20)
    leader.add_argument("--eval-warmup", type=int)
    leader.add_argument("--eval-seed-start", type=int, default=2_000_001)
    leader.add_argument("--final-eval-episodes", type=int, default=100)
    leader.add_argument("--final-eval-seed-start", type=int, default=3_000_001)
    leader.add_argument("--device", default="cpu")
    leader.add_argument(
        "--es-iterations", type=int, default=300,
        help="Native RLlib optimizer iterations (paper cohort: 300).",
    )
    leader.add_argument(
        "--es-num-workers", type=int, default=1,
        help="Native RLlib rollout workers (historical treatment: 1).",
    )
    leader.add_argument("--es-sigma", type=float, default=0.02)
    leader.add_argument("--es-stepsize", type=float, default=0.01)
    leader.add_argument("--es-l2-coeff", type=float, default=0.005)
    leader.add_argument(
        "--es-episodes-per-batch", type=int, default=1_000,
        help="Native RLlib minimum episodes per update.",
    )
    leader.add_argument(
        "--es-train-batch-size", type=int, default=1_000,
        help="Native RLlib minimum sampled timesteps per update.",
    )
    leader.add_argument(
        "--es-eval-prob", type=float, default=0.03,
        help="Native RLlib probability of an unperturbed evaluation rollout.",
    )
    leader.add_argument(
        "--es-report-length", type=int, default=10,
        help="Native episode_reward_mean rolling-window length.",
    )
    leader.add_argument(
        "--es-noise-size", type=int, default=250_000_000,
        help="Shared RLlib Gaussian-noise table length (historical default).",
    )
    leader.add_argument(
        "--es-eval-every", type=int, default=10,
        help="Iterations between explicit stochastic evaluations.",
    )
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
