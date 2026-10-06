"""Simple Q-learning for the appendix reward-timing diagnostic.

SB3 supplies uniform replay, the terminal-masked one-step target, Huber loss,
Adam, and target-network copies. The small specialization here supplies the
historical bias-free linear model and episode-level Gaussian parameter noise.
Training never perturbs the clean network. This is a maintained implementation,
not an exact replay of RLlib's worker scheduling or random-number stream.
"""

import numpy as np
import torch
from torch import nn
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy, QNetwork

from experiments.ablations._support.reinforce import _normc_


class SimpleQPolicy(MultiInputPolicy):
    """One linear Q head, with a fresh noisy commitment each outer episode."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("net_arch", [])
        super().__init__(*args, **kwargs)
        self.register_buffer("parameter_noise_std", torch.tensor(1.0))
        self._noise = None
        self._episode_features = None

    def make_q_net(self):
        kwargs = self._update_features_extractor(self.net_args, features_extractor=None)
        network = QNetwork(**kwargs).to(self.device)
        network.q_net = nn.Linear(network.features_dim, self.action_space.n, bias=False).to(self.device)
        _normc_(network.q_net.weight)
        return network

    def _predict(self, observation, deterministic=True):
        if deterministic:
            return self.q_net(observation).argmax(dim=1)
        features = self.q_net.extract_features(observation, self.q_net.features_extractor)
        if features.shape[0] != 1:
            raise ValueError("appendix SimpleQ supports one environment")
        if self._noise is None:
            self._noise = torch.randn_like(self.q_net.q_net.weight) * self.parameter_noise_std
        logits = nn.functional.linear(features, self.q_net.q_net.weight + self._noise)
        self._episode_features = features.detach()
        return logits.argmax(dim=1)

    def reset_episode(self):
        # These one-shot games have one leader observation. Thus this is also
        # the episode-average KL(clean || noisy) used by RLlib ParameterNoise.
        if self._noise is not None and self._episode_features is not None:
            with torch.no_grad():
                clean = self.q_net.q_net(self._episode_features).log_softmax(dim=1)
                noisy = nn.functional.linear(
                    self._episode_features, self.q_net.q_net.weight + self._noise,
                ).log_softmax(dim=1)
                distance = (clean.exp() * (clean - noisy)).sum().clamp_min(0)
                # Historical sub-exploration epsilon was zero, so target KL=0.
                self.parameter_noise_std.mul_(1.01 if distance <= 0 else 1 / 1.01)
        self._noise = None
        self._episode_features = None


class SimpleQ(DQN):
    """Linear replay Q-learning with complete-episode collection and no epsilon."""

    policy_aliases = {"MultiInputPolicy": SimpleQPolicy}

    def __init__(self, policy=SimpleQPolicy, env=None, learning_rate=0.1,
                 seed=None, device="cpu", _init_setup_model=True):
        super().__init__(
            policy, env, learning_rate=learning_rate,
            buffer_size=50_000, learning_starts=100, batch_size=1024,
            gamma=1.0, train_freq=(1, "episode"), gradient_steps=1,
            target_update_interval=500, max_grad_norm=40,
            exploration_initial_eps=0.0, exploration_final_eps=0.0,
            seed=seed, device=device, _init_setup_model=_init_setup_model,
        )

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # SB3's default warm-up samples a new random action every step. Here
        # parameter noise defines one commitment even before replay learning.
        if n_envs != 1:
            raise ValueError("appendix SimpleQ supports one environment")
        action, _ = self.predict(self._last_obs, deterministic=False)
        return np.asarray(action), np.asarray(action)

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        # Keep the paper's physical-transition replay/target-update schedule
        # even though the executor asks the actor only for new commitments.
        rows = infos[0].get("replay_transitions")
        if rows is None:
            self._physical_steps = 1
            super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        else:
            if not bool(dones[0]):
                raise RuntimeError("A one-state commitment should finish the Q-response episode")
            for obs, action, value, following, terminal in rows:
                replay_buffer.add({key: np.asarray([val]) for key, val in obs.items()},
                    {key: np.asarray([val]) for key, val in following.items()},
                    np.asarray([[action]]), np.asarray([value]), np.asarray([terminal]), [{}])
            self._last_obs = new_obs
            self._physical_steps = len(rows)
            self.num_timesteps += len(rows) - 1
        if np.any(dones):
            self.policy.reset_episode()

    def _on_step(self):
        for _ in range(getattr(self, "_physical_steps", 1)):
            super()._on_step()
