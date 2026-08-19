"""SB3 on-policy algorithms with StackPOMDP rollout filtering."""

import gym
import numpy as np
import torch as th

from stable_baselines3.a2c import A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO


class CustomOnPolicyAlgorithm(OnPolicyAlgorithm):
    """Exclude marked response transitions without skipping environment steps."""

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect one rollout while honoring ``exclude_from_buffer``."""

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        completed_episodes = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # Hidden response queries execute before the first stored transition.
        # Keep the reset boundary until a transition reaches the buffer so GAE
        # cannot connect consecutive outer episodes.
        pending_buffer_episode_starts = np.array(
            self._last_episode_starts, copy=True
        )
        while n_steps < n_rollout_steps:
            if (
                    self.use_sde
                    and self.sde_sample_freq > 0
                    and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            completed_episodes += int(np.sum(dones))

            # Count every executed step, including hidden queries omitted from
            # the PPO buffer.
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)

            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            terminal_obs
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            exclude_from_buffer = infos[0].get("exclude_from_buffer", False)
            if not exclude_from_buffer:
                n_steps += 1
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    pending_buffer_episode_starts,
                    values,
                    log_probs,
                )
                pending_buffer_episode_starts = np.array(dones, copy=True)
            else:
                pending_buffer_episode_starts = np.logical_or(
                    pending_buffer_episode_starts, dones
                )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device)
            )

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )

        self.last_rollout_completed_episodes = completed_episodes
        required_completions = getattr(
            self, "min_completed_episodes_per_rollout", 0
        )
        if completed_episodes < required_completions:
            raise RuntimeError(
                "PPO rollout did not contain the required completed episodes: "
                f"observed={completed_episodes}, "
                f"required={required_completions}, stored_steps={n_steps}."
            )
        if not getattr(self, "_reported_rollout_coverage", False):
            print(
                f"[rollout_coverage] stored_steps={n_steps} "
                f"completed_episodes={completed_episodes} "
                f"required>={required_completions}",
                flush=True,
            )
            self._reported_rollout_coverage = True

        callback.on_rollout_end()
        return True


class CustomA2C(A2C, CustomOnPolicyAlgorithm):
    """A2C with StackPOMDP rollout filtering."""


class CustomPPO(PPO, CustomOnPolicyAlgorithm):
    """PPO with StackPOMDP rollout filtering."""


__all__ = ["CustomA2C", "CustomOnPolicyAlgorithm", "CustomPPO"]
