import os
from typing import Union

import gym
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

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
        super(FixPolicyActionsCallback, self)._on_step()
        if 'done' in self.locals and self.locals['done']:
            self.model.policy.clear_obs_action_map()

    def _init_callback(self):
        self.model.policy.fix_policy_actions()



class CustomEvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """
    def __init__(self,
                 eval_env: Union[gym.Env, VecEnv],
                 n_eval_episodes: int = 1,
                 eval_freq: int = 1000,
                 ):

        super(CustomEvalCallback, self).__init__()
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env


    def _init_callback(self) -> None:

        # First, we need to give model to evaluation environment too
        for current_env in get_all_wrappers(self.eval_env):
            if type(current_env) == StackPOMDPWrapper:
                current_env.model = self.model

        super(CustomEvalCallback, self)._init_callback()


    def _on_step(self) -> None:

        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            for current_env in get_all_wrappers(self.eval_env):
                if type(current_env) == StackPOMDPWrapper:
                    current_env.tot_num_steps = self.n_calls

            _, _ = evaluate_policy(
                                    self.model,
                                    self.eval_env,
                                    n_eval_episodes=self.n_eval_episodes,
                                    )



class CustomCheckpointCallback(CheckpointCallback):

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(CustomCheckpointCallback, self).__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, verbose=verbose)

    def _on_training_end(self) -> None:
        path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
        self.model.save(path)
        if self.verbose > 1:
            print(f"Saving model checkpoint to {path}")