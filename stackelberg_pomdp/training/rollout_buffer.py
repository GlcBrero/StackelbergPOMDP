"""Episode storage and returns for the shared on-policy collector.

The buffer grows to finish complete episodes and discounts across internally
executed game steps. Optional actor_credit is stored as per-row metadata;
only experiment-specific learners decide how it affects their losses.
"""

from collections import namedtuple

import numpy as np
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer


EpisodeRolloutSamples = namedtuple("EpisodeRolloutSamples", (
    "observations", "actions", "old_values", "old_log_prob", "advantages",
    "returns", "actor_credit",
))


class _EpisodeBufferMixin:
    def __init__(self, buffer_size, *args, **kwargs):
        self.target_size = buffer_size
        super().__init__(buffer_size, *args, **kwargs)

    def reset(self):
        self.buffer_size = self.target_size
        super().reset()
        self.actor_credit = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        self.continuation_discounts = np.full_like(self.actor_credit, self.gamma)
        self.trace_discounts = np.full_like(self.actor_credit, self.gamma * self.gae_lambda)

    def _resize(self, size):
        def resize(array):
            result = np.zeros((size, *array.shape[1:]), dtype=array.dtype)
            count = min(size, len(array))
            result[:count] = array[:count]
            return result
        if isinstance(self.observations, dict):
            self.observations = {key: resize(value) for key, value in self.observations.items()}
        else:
            self.observations = resize(self.observations)
        for name in ("actions", "rewards", "episode_starts", "values", "log_probs",
                     "advantages", "returns", "actor_credit", "continuation_discounts", "trace_discounts"):
            setattr(self, name, resize(getattr(self, name)))
        self.buffer_size = size

    def add(self, *args, actor_credit=None, continuation_discount=None, elapsed_steps=None, **kwargs):
        if self.pos == self.buffer_size:
            self._resize(self.buffer_size + max(64, self.target_size, self.buffer_size))
            self.full = False
        credit = np.ones(self.n_envs) if actor_credit is None else np.asarray(actor_credit)
        if credit.shape != (self.n_envs,) or not np.all((credit == 0) | (credit == 1)):
            raise ValueError("actor credit must contain one binary value per environment")
        self.actor_credit[self.pos] = credit
        discounts = self.gamma if continuation_discount is None else np.asarray(continuation_discount)
        durations = 1 if elapsed_steps is None else np.asarray(elapsed_steps)
        if np.any(np.asarray(discounts) < 0) or np.any(np.asarray(discounts) > 1) or np.any(np.asarray(durations) < 0):
            raise ValueError("Invalid transition discount or duration")
        self.continuation_discounts[self.pos] = discounts
        self.trace_discounts[self.pos] = discounts * self.gae_lambda ** durations
        super().add(*args, **kwargs)

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.clone().cpu().numpy().flatten()
        gae = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                nonterminal, next_values = 1.0 - dones, last_values
            else:
                nonterminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.continuation_discounts[step] * next_values * nonterminal - self.values[step]
            gae = delta + self.trace_discounts[step] * nonterminal * gae
            self.advantages[step] = gae
        self.returns = self.advantages + self.values

    def finish(self):
        """Seal an optionally extended rollout at its actual episode boundary."""
        self._resize(self.pos)
        self.full = True

    def get(self, batch_size=None):
        if not self.generator_ready:
            self.actor_credit = self.swap_and_flatten(self.actor_credit)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        data = super()._get_samples(batch_inds, env)
        return EpisodeRolloutSamples(*data, self.to_torch(self.actor_credit[batch_inds].flatten()))


class EpisodeRolloutBuffer(_EpisodeBufferMixin, RolloutBuffer):
    pass


class EpisodeDictRolloutBuffer(_EpisodeBufferMixin, DictRolloutBuffer):
    pass
