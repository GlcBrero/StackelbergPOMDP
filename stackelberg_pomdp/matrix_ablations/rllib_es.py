"""Native RLlib 2.0.1 evolution strategies for the hidden-query diagnostic.

Ray is an intentionally optional, experiment-local dependency.  Every Ray
import is contained in :func:`require_rllib_es`, so importing the maintained
StackelbergPOMDP package does not import or require Ray.  The worker processes
receive the certified finite follower response as plain lists in ``env_config``
and therefore never load Stable-Baselines3 checkpoints.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import gym
import numpy as np
import torch

from .envs import (
    decode_meta_follower_observation,
    encode_meta_follower_observation,
    get_matrix_game,
    load_response_model,
    make_meta_leader_env,
)


EXPECTED_RAY_VERSION = "2.0.1"
RLLIB_ES_IMPLEMENTATION = "ray_rllib_es_2_0_1"
RLLIB_ENV_NAME = "stackpomdp_matrix_hidden_queries_rllib_es_v1"
LOOKUP_SCHEMA_VERSION = 1


def require_rllib_es():
    """Lazily import and version-check the sole Ray dependency surface."""

    try:
        import ray
        from ray.rllib.algorithms.es import ES, ESConfig
        from ray.tune.registry import register_env
    except ImportError as exc:
        raise ImportError(
            "native matrix ES requires the dedicated environment from "
            "environment-ray-es.yml"
        ) from exc
    if ray.__version__ != EXPECTED_RAY_VERSION:
        raise RuntimeError(
            "native matrix ES requires ray=={}; found {}".format(
                EXPECTED_RAY_VERSION, ray.__version__
            )
        )
    return SimpleNamespace(
        ray=ray,
        ES=ES,
        ESConfig=ESConfig,
        register_env=register_env,
    )


@dataclass(frozen=True)
class RllibESSettings:
    """Immutable settings matching the effective historical RLlib treatment."""

    iterations: int = 300
    num_workers: int = 1
    episodes_per_batch: int = 1_000
    train_batch_size: int = 1_000
    noise_stdev: float = 0.02
    stepsize: float = 0.01
    l2_coeff: float = 0.005
    eval_prob: float = 0.03
    report_length: int = 10
    noise_size: int = 250_000_000
    explicit_eval_every: int = 10
    explicit_eval_episodes: int = 20
    final_eval_episodes: int = 100

    def __post_init__(self):
        positive_integers = (
            "iterations",
            "num_workers",
            "episodes_per_batch",
            "train_batch_size",
            "report_length",
            "noise_size",
            "explicit_eval_every",
            "explicit_eval_episodes",
            "final_eval_episodes",
        )
        for name in positive_integers:
            if int(getattr(self, name)) < 1:
                raise ValueError("{} must be positive".format(name))
        if self.noise_stdev <= 0 or self.stepsize <= 0:
            raise ValueError("noise_stdev and stepsize must be positive")
        if self.l2_coeff < 0:
            raise ValueError("l2_coeff must be nonnegative")
        if not 0.0 <= self.eval_prob <= 1.0:
            raise ValueError("eval_prob must lie in [0, 1]")

    def protocol(self, spec):
        return {
            "implementation": RLLIB_ES_IMPLEMENTATION,
            "optimizer_class": "ray.rllib.algorithms.es.ES",
            "ray_version": EXPECTED_RAY_VERSION,
            "framework": "torch",
            "num_rollout_workers": int(self.num_workers),
            "policy": "RLlib default FullyConnectedNetwork",
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
                "vf_share_layers": True,
            },
            "observation": "Discrete({}) one-hot preprocessor".format(
                spec.num_leader_states
            ),
            "observation_filter": "MeanStdFilter",
            "action_sampling": "stochastic categorical on every state visit",
            "parameter_perturbations": "mirrored shared-noise-table Gaussian",
            "noise_stdev": float(self.noise_stdev),
            "fitness_transform": "RLlib centered ranks",
            "optimizer": "RLlib Adam",
            "stepsize": float(self.stepsize),
            "l2_coeff": float(self.l2_coeff),
            "episodes_per_batch": int(self.episodes_per_batch),
            "train_batch_size": int(self.train_batch_size),
            "eval_prob": float(self.eval_prob),
            "report_length": int(self.report_length),
            "noise_size": int(self.noise_size),
            "iterations": int(self.iterations),
            "native_metric": "episode_reward_mean",
            "explicit_evaluation": "stochastic current policy",
            "explicit_eval_every": int(self.explicit_eval_every),
            "explicit_eval_episodes": int(self.explicit_eval_episodes),
            "final_eval_episodes": int(self.final_eval_episodes),
            "query_visibility_enters_objective": False,
            "follower_response": (
                "certified deterministic finite lookup passed in env_config"
            ),
            "checkpoint": "native RLlib checkpoint containing weights and filter",
            "step_accounting": "measured RLlib timesteps; batch overshoot retained",
        }


def _canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _mean_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "sem": (
            float(values.std(ddof=1) / np.sqrt(values.size))
            if values.size > 1 else 0.0
        ),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _commitment_index(commitment, num_actions):
    index = 0
    for action in commitment:
        index = index * int(num_actions) + int(action)
    return int(index)


class FiniteLookupResponse:
    """Serializable deterministic response used by local and Ray worker envs."""

    def __init__(self, spec, actions):
        self.spec = spec
        self.actions = np.asarray(actions, dtype=np.int64)
        expected = (
            spec.num_leader_actions ** spec.num_leader_states,
            spec.num_follower_states,
        )
        if self.actions.shape != expected:
            raise ValueError(
                "response lookup has shape {}; expected {}".format(
                    self.actions.shape, expected
                )
            )
        if np.any(self.actions < 0) or np.any(
            self.actions >= spec.num_follower_actions
        ):
            raise ValueError("response lookup contains an invalid action")

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("the certified response lookup is deterministic")
        follower_state, commitment = decode_meta_follower_observation(
            self.spec, int(np.asarray(observation).reshape(-1)[0])
        )
        index = _commitment_index(commitment, self.spec.num_leader_actions)
        return np.asarray(self.actions[index, follower_state]), None


class DiscreteLeaderObservation(gym.Wrapper):
    """Expose only the existing five-state actor observation to RLlib ES."""

    def __init__(self, env):
        super().__init__(env)
        base_space = env.observation_space.spaces.get("base_environment")
        if not isinstance(base_space, gym.spaces.Discrete):
            raise TypeError("matrix ES requires a discrete base_environment")
        self.observation_space = base_space

    @staticmethod
    def _actor_observation(observation):
        return int(np.asarray(observation["base_environment"]).reshape(-1)[0])

    def reset(self):
        return self._actor_observation(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._actor_observation(observation), reward, done, info


def _ray_env_creator(env_config):
    """Construct the five-state environment from only plain env_config data."""

    spec = get_matrix_game(
        str(env_config["matrix"]), profile_id=str(env_config["profile_id"])
    )
    response = FiniteLookupResponse(spec, env_config["response_actions"])
    env = make_meta_leader_env(
        spec,
        response_model=response,
        query_visibility=str(env_config["condition"]),
        phase_observable=None,
        seed=int(env_config.get("seed", 0)),
        device="cpu",
    )
    return DiscreteLeaderObservation(env)


def build_certified_response_lookup(spec, response_checkpoint, response_algorithm):
    """Evaluate the certified follower on every commitment/state combination."""

    model = load_response_model(
        response_checkpoint, response_algorithm, device="cpu"
    )
    commitments = tuple(product(
        range(spec.num_leader_actions), repeat=spec.num_leader_states
    ))
    actions = np.empty(
        (len(commitments), spec.num_follower_states), dtype=np.int64
    )
    for commitment_index, commitment in enumerate(commitments):
        for follower_state in range(spec.num_follower_states):
            observation = encode_meta_follower_observation(
                spec, follower_state, commitment
            )
            action, _ = model.predict(observation, deterministic=True)
            actions[commitment_index, follower_state] = int(
                np.asarray(action).reshape(-1)[0]
            )
    lookup = FiniteLookupResponse(spec, actions)
    payload = {
        "schema_version": LOOKUP_SCHEMA_VERSION,
        "kind": "certified_deterministic_matrix_response_lookup_v1",
        "ordered_commitments": [list(values) for values in commitments],
        "actions": actions.tolist(),
        "shape": list(actions.shape),
    }
    payload["lookup_sha256"] = _sha256_bytes(
        _canonical_json(payload).encode("utf-8")
    )
    return model, lookup, payload


def validate_lookup_rollouts(spec, response_model, response_lookup):
    """Exhaustively compare direct-checkpoint and finite-lookup environments."""

    commitments = tuple(product(
        range(spec.num_leader_actions), repeat=spec.num_leader_states
    ))
    for index, commitment in enumerate(commitments):
        records = []
        for model in (response_model, response_lookup):
            env = make_meta_leader_env(
                spec,
                response_model=model,
                query_visibility="observed",
                seed=8_000_001 + index,
            )
            observation = env.reset()
            total_return = 0.0
            follower_actions = []
            transitions = 0
            done = False
            try:
                while not done:
                    state = int(observation["base_environment"])
                    observation, reward, done, info = env.step(commitment[state])
                    total_return += float(reward)
                    transitions += 1
                    if info.get("is_reward_phase"):
                        follower_actions.append(int(info["meta_follower_action"]))
            finally:
                env.close()
            records.append((total_return, follower_actions, transitions))
        if records[0] != records[1]:
            raise AssertionError(
                "finite response lookup differs on commitment {}".format(commitment)
            )
    return {
        "validation": "exhaustive_direct_checkpoint_vs_lookup_rollouts",
        "commitments_checked": len(commitments),
        "transitions_per_commitment": int(
            spec.num_leader_states + spec.episode_length
        ),
        "exact_match": True,
    }


@contextmanager
def _evaluation_seed(seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def evaluate_stochastic_policy(
    algorithm, env_payload, *, episodes, seed_coordinate
):
    """Explicitly evaluate RLlib's current stochastic categorical policy."""

    env = _ray_env_creator(env_payload)
    rows = []
    try:
        with _evaluation_seed(seed_coordinate):
            for episode in range(int(episodes)):
                observation = env.reset()
                reward_phase_return = 0.0
                executed_steps = 0
                query_actions = []
                reward_actions = []
                follower_actions = []
                done = False
                while not done:
                    action = int(algorithm.compute_single_action(observation))
                    observation, reward, done, info = env.step(action)
                    executed_steps += 1
                    if info.get("is_query"):
                        query_actions.append(action)
                    if info.get("is_reward_phase"):
                        reward_phase_return += float(reward)
                        reward_actions.append(action)
                        follower_actions.append(int(info["meta_follower_action"]))
                rows.append({
                    "episode": int(episode),
                    "seed_coordinate": int(seed_coordinate),
                    "reward_phase_return": float(reward_phase_return),
                    "leader_reward_per_stage": float(
                        reward_phase_return / env.unwrapped.game_spec.episode_length
                    ),
                    "executed_env_steps": int(executed_steps),
                    "query_actions": query_actions,
                    "reward_actions": reward_actions,
                    "follower_actions": follower_actions,
                })
    finally:
        env.close()
    per_stage = [row["leader_reward_per_stage"] for row in rows]
    returns = [row["reward_phase_return"] for row in rows]
    return {
        "stochastic": True,
        "sampling": "RLlib ESTorchPolicy categorical sample",
        "episodes": rows,
        "per_stage_summary": _mean_summary(per_stage),
        "return_summary": _mean_summary(returns),
    }


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _checkpoint_files(checkpoint_result):
    checkpoint = Path(str(checkpoint_result))
    artifacts = {"ray_checkpoint": checkpoint}
    state_files = sorted(checkpoint.glob("checkpoint-*")) if checkpoint.is_dir() else []
    if len(state_files) == 1:
        artifacts["ray_checkpoint_state"] = state_files[0]
    elif checkpoint.is_dir():
        raise RuntimeError("native RLlib checkpoint has no unique state file")
    metadata = (
        checkpoint / ".tune_metadata"
        if checkpoint.is_dir()
        else Path(str(checkpoint) + ".tune_metadata")
    )
    if metadata.exists():
        artifacts["ray_checkpoint_metadata"] = metadata
    return artifacts


def train_rllib_es(
    *,
    spec,
    condition,
    seed,
    response_checkpoint,
    response_algorithm,
    run_dir,
    settings=None,
    progress_callback=None,
    explicit_eval_seed_start=2_000_001,
    final_eval_seed_start=3_000_001,
):
    """Train one native RLlib ES condition and return audited artifacts."""

    if spec.num_leader_states != 5 or spec.memory_mode != "joint":
        raise ValueError("native paper ES requires the joint five-state profile")
    if condition not in ("observed", "hidden"):
        raise ValueError("condition must be observed or hidden")
    settings = settings or RllibESSettings()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    response_model, response_lookup, lookup_payload = (
        build_certified_response_lookup(
            spec, response_checkpoint, response_algorithm
        )
    )
    lookup_validation = validate_lookup_rollouts(
        spec, response_model, response_lookup
    )
    lookup_payload.update({
        "source_checkpoint": str(response_checkpoint),
        "source_checkpoint_sha256": _file_sha256(response_checkpoint),
        "validation": lookup_validation,
    })
    lookup_path = run_dir / "response_lookup.json"
    _atomic_json(lookup_path, lookup_payload)

    env_payload = {
        "matrix": spec.name,
        "profile_id": "paper_joint_v1",
        "condition": condition,
        "seed": int(seed),
        "response_actions": response_lookup.actions.tolist(),
        "response_lookup_sha256": lookup_payload["lookup_sha256"],
    }
    runtime = require_rllib_es()
    runtime.register_env(RLLIB_ENV_NAME, _ray_env_creator)
    started_ray = not runtime.ray.is_initialized()
    if started_ray:
        runtime.ray.init(
            num_cpus=int(settings.num_workers) + 1,
            include_dashboard=False,
            log_to_driver=False,
        )

    config = runtime.ESConfig()
    config.environment(
        env=RLLIB_ENV_NAME,
        env_config=env_payload,
        disable_env_checking=False,
    )
    config.framework("torch")
    config.rollouts(
        num_rollout_workers=int(settings.num_workers),
        observation_filter="MeanStdFilter",
    )
    config.resources(num_gpus=0, num_cpus_per_worker=1)
    config.debugging(seed=int(seed), log_level="WARN", log_sys_usage=False)
    config.training(
        action_noise_std=0.01,
        l2_coeff=float(settings.l2_coeff),
        noise_stdev=float(settings.noise_stdev),
        episodes_per_batch=int(settings.episodes_per_batch),
        eval_prob=float(settings.eval_prob),
        stepsize=float(settings.stepsize),
        noise_size=int(settings.noise_size),
        report_length=int(settings.report_length),
        train_batch_size=int(settings.train_batch_size),
    )
    # Keep the historically effective actor architecture explicit.
    config.model.update({
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "vf_share_layers": True,
    })

    algorithm = None
    progress = []
    artifacts = {"response_lookup": lookup_path}
    try:
        algorithm = config.build()
        for iteration in range(1, int(settings.iterations) + 1):
            result = algorithm.train()
            measured_timesteps = int(result.get(
                "timesteps_total",
                result.get("num_env_steps_sampled", 0),
            ))
            native_return = _finite_or_none(result.get("episode_reward_mean"))
            row = {
                "row_type": "training_iteration",
                "iteration": int(iteration),
                "measured_timesteps_total": measured_timesteps,
                "native_episode_reward_mean": native_return,
                "native_leader_reward_per_stage": (
                    None if native_return is None else float(
                        native_return / spec.episode_length
                    )
                ),
                "episodes_this_iteration": int(
                    (result.get("info") or {}).get("episodes_this_iter", 0)
                ),
                "episodes_so_far": int(
                    (result.get("info") or {}).get("episodes_so_far", 0)
                ),
                "timesteps_this_iteration": int(
                    result.get("timesteps_this_iter", 0)
                ),
            }
            if (
                iteration == 1
                or iteration % int(settings.explicit_eval_every) == 0
                or iteration == int(settings.iterations)
            ):
                evaluation = evaluate_stochastic_policy(
                    algorithm,
                    env_payload,
                    episodes=int(settings.explicit_eval_episodes),
                    seed_coordinate=(
                        int(explicit_eval_seed_start) + int(iteration)
                    ),
                )
                row.update({
                    "explicit_evaluation": evaluation["per_stage_summary"],
                    "explicit_evaluation_seed_coordinate": (
                        int(explicit_eval_seed_start) + int(iteration)
                    ),
                })
            progress.append(row)
            if progress_callback is not None:
                progress_callback(dict(row))

        checkpoint_parent = run_dir / "ray_checkpoint"
        checkpoint_parent.mkdir(parents=True, exist_ok=False)
        checkpoint_result = algorithm.save(str(checkpoint_parent))
        artifacts.update(_checkpoint_files(checkpoint_result))
        final_evaluation = evaluate_stochastic_policy(
            algorithm,
            env_payload,
            episodes=int(settings.final_eval_episodes),
            seed_coordinate=int(final_eval_seed_start),
        )
        evaluation_payload = {
            "implementation": RLLIB_ES_IMPLEMENTATION,
            "ray_version": EXPECTED_RAY_VERSION,
            "condition": condition,
            "seed": int(seed),
            "checkpoint_selection": "terminal_native_RLlib_ES_state",
            "native_checkpoint_contains": ["weights", "MeanStdFilter"],
            "stochastic_evaluation": final_evaluation,
            "response_lookup": {
                "path": str(lookup_path),
                "sha256": lookup_payload["lookup_sha256"],
                "validation": lookup_validation,
            },
            "training_accounting": {
                "completed_iterations": int(settings.iterations),
                "measured_timesteps_total": int(
                    progress[-1]["measured_timesteps_total"]
                ),
                "requested_minimum_episodes_per_iteration": int(
                    settings.episodes_per_batch
                ),
                "requested_minimum_timesteps_per_iteration": int(
                    settings.train_batch_size
                ),
                "native_overshoot_retained": True,
            },
        }
    finally:
        if algorithm is not None:
            algorithm.stop()
        if started_ray and runtime.ray.is_initialized():
            runtime.ray.shutdown()

    return {
        "artifacts": artifacts,
        "evaluation": evaluation_payload,
        "progress": progress,
        "metadata": {
            "training_accounting": evaluation_payload["training_accounting"],
            "training_protocol": settings.protocol(spec),
            "response_lookup_validation": lookup_validation,
        },
    }


__all__ = [
    "EXPECTED_RAY_VERSION",
    "RLLIB_ES_IMPLEMENTATION",
    "RllibESSettings",
    "DiscreteLeaderObservation",
    "FiniteLookupResponse",
    "build_certified_response_lookup",
    "evaluate_stochastic_policy",
    "require_rllib_es",
    "train_rllib_es",
    "validate_lookup_rollouts",
]
