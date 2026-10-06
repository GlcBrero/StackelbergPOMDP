"""PPO collection at real decisions, with complete outer episodes.

EpisodePPO inherits Stable-Baselines3's PPO.train unchanged. This adapter
extends collection to episode boundaries and keeps the corresponding buffer.
"""
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.ppo import PPO
from .rollout_buffer import EpisodeDictRolloutBuffer, EpisodeRolloutBuffer


class EpisodeOnPolicyAlgorithm(OnPolicyAlgorithm):
    def _setup_model(self):
        super()._setup_model()
        buffer_cls = EpisodeDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else EpisodeRolloutBuffer
        self.rollout_buffer = buffer_cls(self.n_steps, self.observation_space,
            self.action_space, device=self.device, gamma=self.gamma,
            gae_lambda=self.gae_lambda, n_envs=self.n_envs)
        self.num_episodes = getattr(self, "num_episodes", 0)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()
        steps, completed = 0, 0
        dones = np.array(self._last_episode_starts, dtype=bool)
        try:
            phase_execution = all(env.get_attr("phase_execution"))
        except AttributeError:
            phase_execution = False
        episodes_target = getattr(self, "episodes_per_rollout", None)
        complete_episodes = phase_execution or episodes_target is not None
        def enough():
            budget_met = completed >= episodes_target if episodes_target is not None else steps >= n_rollout_steps
            return budget_met and (not complete_episodes or np.all(dones))
        while not enough():
            if complete_episodes and env.num_envs > 1 and steps >= n_rollout_steps and not np.all(dones):
                raise ValueError("Variable-length phase episodes require one environment; vector episodes must finish together")
            if self.use_sde and self.sde_sample_freq > 0 and steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actor_credit = np.ones(env.num_envs, dtype=np.float32)
            if isinstance(obs_tensor, dict) and "critic:action_credit" in obs_tensor:
                actor_credit *= (obs_tensor["critic:action_credit"].sum(dim=-1) > 0).cpu().numpy()
            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            discounts = np.array([info.get("continuation_discount", self.gamma) for info in infos])
            durations = np.array([info.get("elapsed_steps", 1) for info in infos])
            completed += int(np.sum(dones))
            self.num_episodes += int(np.sum(dones))
            self.num_timesteps += env.num_envs
            steps += 1
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
            self._update_info_buffer(infos, dones)
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)
            for index, done in enumerate(dones):
                if done and infos[index].get("TimeLimit.truncated", False):
                    terminal = self.policy.obs_to_tensor(infos[index]["terminal_observation"])[0]
                    with th.no_grad():
                        rewards[index] += discounts[index] * self.policy.predict_values(terminal)[0].item()
            rollout_buffer.add(self._last_obs, actions, rewards,
                self._last_episode_starts, values, log_probs, actor_credit=actor_credit,
                continuation_discount=discounts, elapsed_steps=durations)
            self._last_obs, self._last_episode_starts = new_obs, dones
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.finish()
        rollout_buffer.compute_returns_and_advantage(values, dones)
        self.last_rollout_completed_episodes = completed
        self.logger.record("rollout/completed_episodes", completed)
        self.logger.record("rollout/credited_draws", float(rollout_buffer.actor_credit.sum()))
        required = getattr(self, "min_completed_episodes_per_rollout", 0)
        if completed < required:
            raise RuntimeError(f"Rollout completed {completed} episodes; expected at least {required}")
        callback.on_rollout_end()
        return True


class EpisodePPO(PPO, EpisodeOnPolicyAlgorithm):
    """Library PPO updates, with complete-episode collection when requested.

    The experiment chooses its policy and ordinary PPO hyperparameters.
    No policy-loss weighting or copied PPO optimizer lives in this class.
    """

    def __init__(self, *args, episodes_per_rollout=None, **kwargs):
        self.episodes_per_rollout = episodes_per_rollout
        super().__init__(*args, **kwargs)


# Retain import paths used by earlier checkpoints and scripts.
CustomOnPolicyAlgorithm = EpisodeOnPolicyAlgorithm
CustomPPO = EpisodePPO
