"""Atari's active-row PPO loss for pretraining and standard leader runs.

Some pretraining transitions have no action from this actor. Their value
updates remain active, but their policy/entropy/KL terms must be masked.
The generic EpisodePPO uses Stable-Baselines3's unmodified training method.
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from gym import spaces
from stable_baselines3.common.utils import explained_variance
from stackelberg_pomdp.training import EpisodePPO


def masked_mean(values, mask):
    """Mean over active actor rows; an empty selection has zero gradient."""
    mask = mask.to(device=values.device, dtype=th.bool)
    return th.where(mask, values, th.zeros_like(values)).sum() / mask.sum().clamp_min(1)


def credit_mean(values, credit):
    """Mask actor terms without changing the rollout's normalization.

    In particular, an action-dependent number of visited keys must not divide
    the trajectory's likelihood-ratio score by its number of new draws.
    Atari's explicit per-head balancing remains a separate optimizer choice.
    """
    return th.where(credit.bool(), values, th.zeros_like(values)).mean()


def normalize_advantages(advantages, credit):
    active = advantages[credit.bool()]
    if active.numel() <= 1:
        return advantages
    return (advantages - active.mean()) / (active.std() + 1e-8)


class AtariPPO(EpisodePPO):
    def __init__(self, *args, deterministic_leader=None, **kwargs):
        # Historical checkpoint metadata; commitments now belong to the env.
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                credit = rollout_data.actor_credit
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = normalize_advantages(advantages, credit)

                # ratio between old and new policy, should be one at the first iteration
                log_ratio = th.where(credit.bool(), log_prob - rollout_data.old_log_prob, th.zeros_like(log_prob))
                ratio = th.exp(log_ratio)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -credit_mean(th.min(policy_loss_1, policy_loss_2), credit)

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = masked_mean((th.abs(ratio - 1) > clip_range).float(), credit.bool()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = credit_mean(log_prob, credit)
                else:
                    entropy_loss = -credit_mean(entropy, credit)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    approx_kl_div = masked_mean((th.exp(log_ratio) - 1) - log_ratio, credit.bool()).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

