import json
import os
import threading
import time
from itertools import product
from typing import Union

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

try:
    import wandb
except ImportError:
    wandb = None

from stackelberg_pomdp.baselines_utils import CustomPolicy
from stackelberg_pomdp.gym_envs.envs.wrappers import StackPOMDPWrapper
from stackelberg_pomdp.leader_policies import BaselinePolicyWrapper
from stackelberg_pomdp.utils import (
    check_empirical_bcce_gap,
    compute_empirical_welfare,
)
try:
    from .utils import get_all_wrappers
except ImportError:
    from utils import get_all_wrappers


def _wandb_log(metrics, step):
    if wandb is not None and wandb.run is not None:
        metrics = dict(metrics)
        metrics["global_step"] = step
        wandb.log(metrics)


class FixPolicyActionsCallback(BaseCallback):

    def __init__(self):
        super(FixPolicyActionsCallback, self).__init__()

    def _on_step(self) -> bool:
        dones = np.asarray(
            self.locals.get("dones", [False]), dtype=bool
        ).reshape(-1)
        completed_rows = np.flatnonzero(dones)
        if completed_rows.size:
            clear = self.model.policy.clear_obs_action_map
            try:
                # Composite Atari policies keep independent caches for each
                # vector-environment row.  Finishing row 1 must not erase the
                # still-active commitment sampled for row 0.
                clear(rows=completed_rows.tolist())
            except TypeError:
                # Older game-agnostic policies expose only all-or-nothing
                # clearing.  Preserve their established single-env behavior.
                clear()
        return True

    def _init_callback(self):
        self.model.policy.fix_policy_actions()
        print("[policy_cache] enabled=true reset_on_episode_done=true", flush=True)


class TrainingProgressCallback(BaseCallback):
    """Print lightweight training progress for long paper runs."""

    def __init__(self, total_timesteps, print_freq=10000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = max(1, int(print_freq))
        self._last_print_timestep = 0
        self._start_time = None

    def _init_callback(self) -> None:
        self._start_time = time.time()
        self.effective_total_timesteps = max(
            self.total_timesteps,
            getattr(self.model, "n_steps", self.total_timesteps),
        )
        print(
            f"[train] start total_timesteps={self.total_timesteps} "
            f"effective_total={self.effective_total_timesteps} "
            f"progress_freq={self.print_freq}",
            flush=True,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print_timestep < self.print_freq:
            return True

        elapsed = max(time.time() - self._start_time, 1e-9)
        fps = self.num_timesteps / elapsed
        pct = 100 * self.num_timesteps / max(self.effective_total_timesteps, 1)

        print(
            f"[train] steps={self.num_timesteps}/{self.effective_total_timesteps} "
            f"({pct:.1f}%) fps={fps:.1f}",
            flush=True,
        )
        self._last_print_timestep = self.num_timesteps
        return True


class TrainingRewardCallback(BaseCallback):
    """Print the reward-phase return actually observed during training."""

    def __init__(self, print_freq=1, zero_welfare_tol=None):
        super().__init__()
        self.print_freq = max(1, int(print_freq))
        self.zero_welfare_tol = None if zero_welfare_tol is None else float(zero_welfare_tol)
        self.episode_count = 0
        self.reward_phase_sum = 0.0
        self.reward_generated_steps = 0
        self.efficiency_sum = 0.0
        self.efficiency_weight_sum = 0.0
        self.exact_reward_phase = False
        self.last_bcce_gap = None
        self.best_bcce_clean_reward_phase_sum = None
        self.best_bcce_allocative_efficiency = None

    def _init_callback(self) -> None:
        self.stack_env = None
        for env in get_all_wrappers(self.training_env):
            if type(env) == StackPOMDPWrapper:
                self.stack_env = env
                break

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        if not infos:
            return True

        info = infos[0]
        if info.get("is_reward_phase", False):
            self.reward_phase_sum += float(rewards[0])
            if info.get("reward_generated", False):
                self.reward_generated_steps += 1
                if "exact_profile_weight" in info:
                    self.exact_reward_phase = True
                if "weighted_efficiency" in info:
                    self.efficiency_sum += float(info["weighted_efficiency"])
                    self.efficiency_weight_sum += float(info.get("exact_profile_weight", 1.0))
                elif "efficiency" in info:
                    self.efficiency_sum += float(info["efficiency"])
                    self.efficiency_weight_sum += 1.0

        if bool(dones[0]):
            self.episode_count += 1
            if self.episode_count % self.print_freq == 0:
                reward_phase_avg = (
                    self.reward_phase_sum
                    if self.exact_reward_phase
                    else (
                        self.reward_phase_sum / self.reward_generated_steps
                        if self.reward_generated_steps
                        else 0.0
                    )
                )
                allocative_efficiency = (
                    self.efficiency_sum / self.efficiency_weight_sum
                    if self.efficiency_weight_sum
                    else None
                )
                self.logger.record("clean_reward_phase_sum", self.reward_phase_sum)
                self.logger.record("clean_reward_phase_avg", reward_phase_avg)
                self.logger.record("reward_generated_steps", self.reward_generated_steps)
                wandb_metrics = {}
                if allocative_efficiency is not None:
                    self.logger.record("clean_allocative_efficiency", allocative_efficiency)
                    # Common paper-facing metric shared with the direct SPM baseline.
                    self.logger.record("paper_allocative_efficiency", allocative_efficiency)
                self.logger.record("paper_expected_reward", reward_phase_avg)
                self.logger.record("reward", reward_phase_avg)
                wandb_metrics["reward"] = reward_phase_avg
                bcce_gap = getattr(self.stack_env, "last_response_bcce_gap", None)
                bcce_certified = False
                if bcce_gap is not None:
                    self.last_bcce_gap = bcce_gap
                    if bcce_gap <= getattr(self.stack_env, "response_bcce_threshold", float("inf")):
                        bcce_certified = True
                        if (
                                self.best_bcce_clean_reward_phase_sum is None
                                or self.reward_phase_sum > self.best_bcce_clean_reward_phase_sum
                        ):
                            self.best_bcce_clean_reward_phase_sum = self.reward_phase_sum
                        if allocative_efficiency is not None and (
                                self.best_bcce_allocative_efficiency is None
                                or allocative_efficiency > self.best_bcce_allocative_efficiency
                        ):
                            self.best_bcce_allocative_efficiency = allocative_efficiency
                    self.logger.record("bcce_gap", bcce_gap)
                if bcce_certified and allocative_efficiency is not None:
                    self.logger.record(
                        "bcce_certified_allocative_efficiency",
                        allocative_efficiency,
                    )
                if self.best_bcce_clean_reward_phase_sum is not None:
                    self.logger.record(
                        "best_bcce_clean_reward_phase_sum",
                        self.best_bcce_clean_reward_phase_sum,
                    )
                if self.best_bcce_allocative_efficiency is not None:
                    self.logger.record(
                        "best_bcce_allocative_efficiency",
                        self.best_bcce_allocative_efficiency,
                    )
                _wandb_log(wandb_metrics, step=self.num_timesteps)
                efficiency_text = (
                    f"allocative_efficiency={allocative_efficiency:.6g} "
                    if allocative_efficiency is not None
                    else ""
                )
                print(
                    f"[reward] steps={self.num_timesteps} "
                    f"episode={self.episode_count} "
                    f"reward_phase_avg={reward_phase_avg:.6g} "
                    f"reward_phase_sum={self.reward_phase_sum:.6g} "
                    f"{efficiency_text}"
                    f"reward_generated_steps={self.reward_generated_steps}",
                    flush=True,
                )
            if (
                self.zero_welfare_tol is not None
                and
                self.reward_generated_steps
                and abs(self.reward_phase_sum) <= self.zero_welfare_tol
            ):
                self._print_zero_welfare_response()
            self.reward_phase_sum = 0.0
            self.reward_generated_steps = 0
            self.efficiency_sum = 0.0
            self.efficiency_weight_sum = 0.0
            self.exact_reward_phase = False

        return True

    def _print_zero_welfare_response(self):
        if self.stack_env is None:
            return

        policy = getattr(self.stack_env, "response_leader_policy", None)
        if policy is None:
            policy = BaselinePolicyWrapper(self.model.policy, self.stack_env, deterministic=False)
        response_strategy = getattr(self.stack_env, "last_response_strategy", None)
        if response_strategy is None:
            if not hasattr(self.stack_env, "response_strategy"):
                return
            response_strategy = self.stack_env.response_strategy()
        if not response_strategy:
            return

        bcce_gap = check_empirical_bcce_gap(self.stack_env, policy, response_strategy)
        leader_reward = compute_empirical_welfare(self.stack_env, policy, response_strategy)

        print(
            f"[zero_welfare_equilibrium] steps={self.num_timesteps} "
            f"episode={self.episode_count} "
            f"bcce_gap={bcce_gap:.6g} "
            f"bcce_leader_reward={leader_reward:.6g} "
            f"snapshots={len(response_strategy)}",
            flush=True,
        )
        self.logger.record("zero_welfare_bcce_gap", bcce_gap)
        self.logger.record("zero_welfare_leader_reward", leader_reward)
        _wandb_log(
            {
                "zero_welfare_bcce_gap": bcce_gap,
                "zero_welfare_leader_reward": leader_reward,
            },
            step=self.num_timesteps,
        )


class RewardEpisodeTraceCallback(BaseCallback):
    """Print the first complete reward episode matching each requested target."""

    def __init__(self, targets, tolerance=1e-6):
        super().__init__()
        self.targets = tuple(float(target) for target in targets)
        self.tolerance = float(tolerance)
        self.pending_targets = set(self.targets)
        self.reward_rows = []
        self.reward_sum = 0.0
        self.reward_games = 0
        self.exact_reward_phase = False
        self.episode_start_step = 0

    def _init_callback(self) -> None:
        self.stack_env = None
        for env in get_all_wrappers(self.training_env):
            if type(env) == StackPOMDPWrapper:
                self.stack_env = env
                break

    @staticmethod
    def _jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): RewardEpisodeTraceCallback._jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [RewardEpisodeTraceCallback._jsonable(item) for item in value]
        return value

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        if not infos:
            return True

        info = infos[0]
        if info.get("is_reward_phase", False):
            self.reward_sum += float(rewards[0])
            if info.get("reward_generated", False):
                self.reward_games += 1
                if "exact_profile_weight" in info:
                    self.exact_reward_phase = True
                self.reward_rows.append({
                    "type_profile": info.get("type_profile"),
                    "profile_weight": info.get("exact_profile_weight"),
                    "follower_messages": info.get("followers_actions"),
                    "leader_action_at_completion": info.get("leader_action"),
                    "mechanism_outcome": info.get("mechanism_outcome"),
                    "raw_reward": info.get("unweighted_reward", info.get("surplus")),
                    "scaled_reward": float(rewards[0]),
                })

        if bool(dones[0]):
            reward_avg = (
                self.reward_sum
                if self.exact_reward_phase
                else (self.reward_sum / self.reward_games if self.reward_games else 0.0)
            )
            match = next(
                (
                    target
                    for target in sorted(self.pending_targets)
                    if abs(reward_avg - target) <= self.tolerance
                ),
                None,
            )
            if match is not None:
                payload = {
                    "target": match,
                    "steps": self.num_timesteps,
                    "episode_transitions": self.num_timesteps - self.episode_start_step,
                    "reward_phase_avg": reward_avg,
                    "cached_actor_observations": len(
                        getattr(self.model.policy, "obs_action_map", {})
                    ),
                    "response_updates": getattr(self.stack_env, "last_response_updates", None),
                    "response_type_profile_counts": getattr(
                        self.stack_env,
                        "last_response_type_profile_counts",
                        None,
                    ),
                    "final_weights": getattr(self.stack_env, "last_response_weights", None),
                    "reward_profiles": self.reward_rows,
                }
                print(
                    "[reward_trace] "
                    + json.dumps(self._jsonable(payload), sort_keys=True),
                    flush=True,
                )
                self.pending_targets.remove(match)

            self.reward_rows = []
            self.reward_sum = 0.0
            self.reward_games = 0
            self.exact_reward_phase = False
            self.episode_start_step = self.num_timesteps

        return True


class ExactSPMEvaluationCallback(BaseCallback):
    """Evaluate the direct SPM baseline exactly over all type profiles."""

    def __init__(self, eval_env, print_freq=10000, deterministic=True, action_samples=1):
        super().__init__()
        self.eval_env = eval_env
        self.print_freq = max(1, int(print_freq))
        self.deterministic = deterministic
        self.action_samples = max(1, int(action_samples))
        self._last_eval_timestep = 0

    def _init_callback(self) -> None:
        self._log_evaluation(step=0)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep < self.print_freq:
            return True

        self._log_evaluation(step=self.num_timesteps)
        self._last_eval_timestep = self.num_timesteps
        return True

    def _log_evaluation(self, step):
        metrics = self._evaluate()
        self.logger.record("spm_exact_reward", metrics["spm_exact_reward"])
        self.logger.record("spm_exact_allocated", metrics["spm_exact_allocated"])
        self.logger.record("spm_exact_efficient", metrics["spm_exact_efficient"])
        self.logger.record("spm_exact_allocative_efficiency", metrics["spm_exact_allocative_efficiency"])
        self.logger.record("exact_expected_reward", metrics["spm_exact_reward"])
        self.logger.record("paper_allocative_efficiency", metrics["spm_exact_allocative_efficiency"])
        metrics["exact_expected_reward"] = metrics["spm_exact_reward"]
        metrics["paper_allocative_efficiency"] = metrics["spm_exact_allocative_efficiency"]
        print(
            f"[spm_exact] steps={step} "
            f"reward={metrics['spm_exact_reward']:.6g} "
            f"allocated={metrics['spm_exact_allocated']:.6g} "
            f"efficient={metrics['spm_exact_efficient']:.6g} "
            f"alloc_eff={metrics['spm_exact_allocative_efficiency']:.6g} "
            f"deterministic={self.deterministic}",
            flush=True,
        )

    def _evaluate(self):
        base_env = self.eval_env.env
        game = base_env.game
        total_reward = 0.0
        total_allocated = 0.0
        total_efficient = 0.0
        total_allocative_efficiency = 0.0
        total_weight = 0.0
        type_profiles = list(product(range(game.num_types), repeat=len(game.followers_list)))

        previous_freeze = base_env.freeze_types
        previous_types = getattr(base_env, "types", None)
        base_env.freeze_types = True
        try:
            for type_values in type_profiles:
                base_env.types = {
                    follower: type_value
                    for follower, type_value in zip(game.followers_list, type_values)
                }
                profile_weight = game.type_profile_probability(base_env.types)
                sample_weight = profile_weight / self.action_samples
                for _ in range(self.action_samples):
                    obs = self.eval_env.reset()
                    done = False
                    reward_sum = 0.0
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=self.deterministic)
                        obs, reward, done, _ = self.eval_env.step(action)
                        reward_sum += float(np.asarray(reward).item())
                    allocated = base_env.mechanism_episode.allocated_value
                    efficient = game.efficient_welfare(base_env.mechanism_episode.valuations)
                    # The PI plot reports average realized allocative efficiency,
                    # not the ratio of ex ante expected allocated to efficient value.
                    allocative_efficiency = 1.0 if efficient == 0 else allocated / efficient

                    total_reward += sample_weight * reward_sum
                    total_allocated += sample_weight * allocated
                    total_efficient += sample_weight * efficient
                    total_allocative_efficiency += sample_weight * allocative_efficiency
                    total_weight += sample_weight
        finally:
            base_env.freeze_types = previous_freeze
            if previous_types is not None:
                base_env.types = previous_types

        return {
            "spm_exact_reward": total_reward / total_weight,
            "spm_exact_allocated": total_allocated / total_weight,
            "spm_exact_efficient": total_efficient / total_weight,
            "spm_exact_allocative_efficiency": total_allocative_efficiency / total_weight,
        }


class ResponsePhaseDiagnosticsCallback(BaseCallback):
    """Report passive response-strategy diagnostics at response-phase boundaries."""

    def __init__(self, print_freq=1):
        super().__init__()
        self.print_freq = max(1, int(print_freq))
        self.response_phase_count = 0

    def _init_callback(self) -> None:
        self.stack_env = None
        for env in get_all_wrappers(self.training_env):
            if type(env) == StackPOMDPWrapper:
                self.stack_env = env
                break
        if self.stack_env is None:
            raise ValueError("ResponsePhaseDiagnosticsCallback requires a StackPOMDPWrapper.")

    def _on_step(self) -> bool:
        if not any(info.get("response_phase_done", False) for info in self.locals.get("infos", [])):
            return True

        response_info = next(
            info for info in self.locals.get("infos", [])
            if info.get("response_phase_done", False)
        )
        self.response_phase_count += 1
        if self.response_phase_count % self.print_freq != 0:
            return True
        if not hasattr(self.stack_env, "response_strategy"):
            return True

        policy = getattr(self.stack_env, "response_leader_policy", None)
        if policy is None:
            policy = BaselinePolicyWrapper(self.model.policy, self.stack_env, deterministic=False)
        response_strategy = self.stack_env.response_strategy()
        bcce_gap = check_empirical_bcce_gap(self.stack_env, policy, response_strategy)
        bcce_leader_reward = compute_empirical_welfare(self.stack_env, policy, response_strategy)

        self.stack_env.last_response_bcce_gap = bcce_gap
        print(
            f"[response] steps={self.num_timesteps} "
            f"episode={self.response_phase_count} "
            f"bcce_gap={bcce_gap:.6g} "
            f"bcce_leader_reward={bcce_leader_reward:.6g} "
            f"snapshots={len(response_strategy)} "
            f"stop_reason={response_info.get('response_phase_stop_reason')}",
            flush=True,
        )
        self.logger.record("bcce_gap", bcce_gap)
        self.logger.record("bcce_leader_reward", bcce_leader_reward)
        self.logger.record("response_snapshots", len(response_strategy))
        _wandb_log(
            {
                "bcce_gap": bcce_gap,
                "bcce_leader_reward": bcce_leader_reward,
                "response_snapshots": len(response_strategy),
            },
            step=self.num_timesteps,
        )
        return True


class ResponsePhasePolicyCallback(BaseCallback):
    """Attach the current fixed episode leader policy for response checks.

    StackPOMDP training samples one action per actor observation and caches it
    for the whole episode. Response checks must use that same sampled/cached
    policy, not the deterministic mean action for unseen counterfactual
    observations.
    """

    def _init_callback(self) -> None:
        self.stack_env = None
        for env in get_all_wrappers(self.training_env):
            if type(env) == StackPOMDPWrapper:
                self.stack_env = env
                break
        if self.stack_env is None:
            raise ValueError("ResponsePhasePolicyCallback requires a StackPOMDPWrapper.")

        self.stack_env.set_response_leader_policy(
            BaselinePolicyWrapper(self.model.policy, self.stack_env, deterministic=False)
        )

    def _on_step(self) -> bool:
        return True


class ClearCacheOnResetWrapper(gym.Wrapper):
    """Clears the eval policy's action cache on episode reset."""

    def __init__(self, env, policy):
        super().__init__(env)
        self._policy = policy

    def reset(self):
        self._policy.clear_obs_action_map()
        return self.env.reset()


class BackgroundEvalCallback(BaseCallback):
    """Run paper-style deterministic StackPOMDP evaluation and print rewards.

    Creates a separate CustomPolicy instance (never trained) and periodically
    copies weights from the training policy. Eval runs in a daemon thread so
    training is never blocked.
    """

    def __init__(self, eval_env, eval_freq, n_eval_episodes=1, reward_steps=1):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.reward_steps = max(1, int(reward_steps))
        self._eval_thread = None

    def _init_callback(self) -> None:
        policy = self.model.policy
        self._eval_policy = CustomPolicy(
            observation_space=self.model.observation_space,
            action_space=self.model.action_space,
            lr_schedule=lambda _: 0.0,
            cutoff_entry=policy.mlp_extractor.cutoff_entry,
            actor_obs_keys=policy.actor_obs_keys,
        )
        self._eval_policy.fix_policy_actions()

        self.eval_env = ClearCacheOnResetWrapper(self.eval_env, self._eval_policy)

        for env in get_all_wrappers(self.eval_env):
            if type(env) == StackPOMDPWrapper:
                env.model = self._eval_policy

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self._eval_thread is not None and self._eval_thread.is_alive():
                return True

            state_dict = {k: v.clone() for k, v in self.model.policy.state_dict().items()}
            self._eval_policy.load_state_dict(state_dict)

            for env in get_all_wrappers(self.eval_env):
                if type(env) == StackPOMDPWrapper:
                    env.tot_num_steps = self.n_calls

            self._eval_thread = threading.Thread(target=self._run_eval, daemon=True)
            self._eval_thread.start()
        return True

    def _run_eval(self):
        try:
            reward_phase_totals = []
            episode_lengths = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                reward_phase_total = 0.0
                exact_reward_phase = False
                episode_length = 0
                while not done:
                    action, _ = self._eval_policy.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    if info.get("is_reward_phase", False):
                        reward_phase_total += reward
                        if "exact_profile_weight" in info:
                            exact_reward_phase = True
                    episode_length += 1
                reward_phase_totals.append(
                    reward_phase_total
                    if exact_reward_phase
                    else reward_phase_total / self.reward_steps
                )
                episode_lengths.append(episode_length)

            mean_reward_phase_avg = (
                sum(reward_phase_totals) / len(reward_phase_totals)
            )
            print(
                f"[eval] steps={self.num_timesteps} "
                f"reward_phase_avg={mean_reward_phase_avg:.4g} "
                f"episodes={len(reward_phase_totals)} "
                f"length={sum(episode_lengths) / len(episode_lengths):.1f}",
                flush=True,
            )
        except Exception as e:
            print(f"Eval error: {e}")

    def _on_training_end(self) -> None:
        if self._eval_thread is not None and self._eval_thread.is_alive():
            self._eval_thread.join(timeout=60)



class CustomCheckpointCallback(CheckpointCallback):

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(CustomCheckpointCallback, self).__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, verbose=verbose)

    def _on_training_end(self) -> None:
        path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
        self.model.save(path)
        if self.verbose > 1:
            print(f"Saving model checkpoint to {path}")
