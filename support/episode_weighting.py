"""Optional paper loss normalization shared by MSPM and platform.

The generic trainer is EpisodePPO. These experiments additionally scale the
actor and entropy terms by decisions per episode / an experiment denominator.
Scaling normalized advantages and the entropy coefficient lets the library
perform the PPO update, including clipping, value loss and optimization.
"""
import numpy as np
from stackelberg_pomdp.training import EpisodePPO


class _WeightedAdvantages:
    """Temporary view of a rollout with its actor advantages rescaled."""

    def __init__(self, buffer, scale, normalize):
        self.buffer, self.scale, self.normalize = buffer, scale, normalize

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def get(self, batch_size=None):
        for batch in self.buffer.get(batch_size):
            advantages = batch.advantages
            if self.normalize and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            yield batch._replace(advantages=self.scale * advantages)


class EpisodeWeightedPPO(EpisodePPO):
    """Library PPO with the paper experiments' explicit actor normalization."""

    def __init__(self, *args, actor_episode_denominator=1., **kwargs):
        if not np.isfinite(actor_episode_denominator) or actor_episode_denominator <= 0:
            raise ValueError("The actor episode denominator must be positive")
        self.actor_episode_denominator = float(actor_episode_denominator)
        super().__init__(*args, **kwargs)

    def train(self):
        buffer = self.rollout_buffer
        completed = self.last_rollout_completed_episodes
        if completed < 1 or not np.all(buffer.actor_credit == 1):
            raise ValueError("Episode weighting requires complete episodes of genuine decisions")
        scale = buffer.buffer_size * buffer.n_envs / completed / self.actor_episode_denominator
        normalize, entropy = self.normalize_advantage, self.ent_coef
        self.rollout_buffer = _WeightedAdvantages(buffer, scale, normalize)
        # Normalize before scaling; library PPO must not normalize it away.
        self.normalize_advantage = False
        self.ent_coef = entropy * scale
        try:
            super().train()
        finally:
            self.rollout_buffer = buffer
            self.normalize_advantage = normalize
            self.ent_coef = entropy
        self.logger.record("train/actor_episode_scale", scale)
