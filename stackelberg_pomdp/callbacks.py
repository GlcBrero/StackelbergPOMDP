import os
import threading
from typing import Union

import gym
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from stackelberg_pomdp.baselines_utils import CustomPolicy
from stackelberg_pomdp.gym_envs.envs.custom_envs import StackPOMDPWrapper
from utils import compute_welfare_loss, get_all_wrappers

class GiveModelToEnvCallback(BaseCallback):

    def __init__(self):
        super(GiveModelToEnvCallback, self).__init__()

    def _on_step(self) -> None:
        super(GiveModelToEnvCallback, self)._on_step()

    def _init_callback(self):
        for current_env in get_all_wrappers(self.training_env):
            if type(current_env) == StackPOMDPWrapper:
                current_env.model = self.model



class FixPolicyActionsCallback(BaseCallback):

    def __init__(self):
        super(FixPolicyActionsCallback, self).__init__()

    def _on_step(self) -> None:
        if self.locals.get('dones', [False])[0]:
            self.model.policy.clear_obs_action_map()

    def _init_callback(self):
        self.model.policy.fix_policy_actions()



class ClearCacheOnResetWrapper(gym.Wrapper):
    """Clears the eval policy's action cache on episode reset."""

    def __init__(self, env, policy):
        super().__init__(env)
        self._policy = policy

    def reset(self):
        self._policy.clear_obs_action_map()
        return self.env.reset()


class BackgroundEvalCallback(BaseCallback):
    """Run deterministic evaluation in a background thread with a frozen policy snapshot.

    Creates a separate CustomPolicy instance (never trained) and periodically
    copies weights from the training policy. Eval runs in a daemon thread so
    training is never blocked. The eval env has its own logger writing to a
    separate CSV.
    """

    def __init__(self, eval_env, eval_freq, n_eval_episodes=1):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._eval_thread = None

    def _init_callback(self) -> None:
        policy = self.model.policy
        self._eval_policy = CustomPolicy(
            observation_space=self.model.observation_space,
            action_space=self.model.action_space,
            lr_schedule=lambda _: 0.0,
            cutoff_entry=policy.mlp_extractor.cutoff_entry,
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
            evaluate_policy(
                self._eval_policy, self.eval_env,
                n_eval_episodes=self.n_eval_episodes, deterministic=True,
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