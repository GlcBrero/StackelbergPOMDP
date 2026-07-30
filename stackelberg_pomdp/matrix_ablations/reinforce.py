"""Legacy-compatible REINFORCE for the matrix meta-follower.

This module ports the policy-gradient mechanics used by the historical
StackeRLberg matrix experiments while retaining Stable-Baselines3's rollout,
callback, and checkpoint interfaces.  The reference implementation is:

* ``stackerlberg/train/experiments/configurations.py``
  (``smipd_hiddenqueries_pg_pg_new``),
* RLlib's ``PGTorchPolicy`` and ``post_process_advantages``, and
* ``stackerlberg/models/linear_torch_model.py``.

The loss is exactly ``-mean(log pi(a|s) * reward_to_go)``.  There is no
critic, baseline, advantage normalization, entropy bonus, or gradient
clipping.  The categorical logits are produced by one bias-free linear layer
with RLlib's row-wise normc initialization at scale 0.01, and Adam performs one
update over each complete follower rollout.  The rollout length is derived from
RLlib's 100 *environment*-step collection target, which also counted query
turns on which the follower had no sample.
"""

import hashlib
from pathlib import Path
from typing import Callable, Optional, Type, Union

from gym import spaces
import numpy as np
import torch as th
from torch import nn

from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule


LEGACY_PG_COLLECTION_TARGET_ENV_STEPS = 100
LEGACY_REINFORCE_LEARNING_RATE = 0.02
LEGACY_REINFORCE_PRETRAIN_ITERATIONS = 500
REINFORCE_CHECKPOINT_INTERVAL_UPDATES = 100


def legacy_pg_rollout_geometry(episode_length, num_query_states):
    """Translate RLlib env-step collection into follower gradient samples."""

    episode_length = int(episode_length)
    num_query_states = int(num_query_states)
    executed_per_episode = episode_length + num_query_states
    episodes = int(np.ceil(
        LEGACY_PG_COLLECTION_TARGET_ENV_STEPS / executed_per_episode
    ))
    return {
        "collection_target_env_steps": LEGACY_PG_COLLECTION_TARGET_ENV_STEPS,
        "episodes_per_update": episodes,
        "executed_env_steps_per_update": episodes * executed_per_episode,
        "query_env_steps_per_update": episodes * num_query_states,
        "follower_gradient_samples_per_update": episodes * episode_length,
    }


def reinforce_checkpoint_geometry(
        episode_length,
        num_query_states,
        update_interval=REINFORCE_CHECKPOINT_INTERVAL_UPDATES,
):
    """Return post-update checkpoint cadence in both measured step units."""

    update_interval = int(update_interval)
    if update_interval < 1:
        raise ValueError("update_interval must be positive")
    rollout = legacy_pg_rollout_geometry(episode_length, num_query_states)
    return {
        "update_interval": update_interval,
        "follower_gradient_samples_interval": (
            update_interval * rollout["follower_gradient_samples_per_update"]
        ),
        "executed_env_steps_interval": (
            update_interval * rollout["executed_env_steps_per_update"]
        ),
    }


def _normc_(weight, scale=0.01):
    """Apply RLlib's row-wise normc initializer in place."""

    with th.no_grad():
        weight.normal_(0.0, 1.0)
        row_norm = th.sqrt(th.sum(th.square(weight), dim=1, keepdim=True))
        weight.mul_(float(scale) / row_norm)
    return weight


class ReinforcePolicy(ActorCriticPolicy):
    """Bias-free linear categorical policy used by legacy RLlib PG.

    SB3's rollout buffer always asks an actor-critic policy for values.  The
    frozen zero value head below is only a compatibility shim: with gamma and
    GAE lambda equal to one it makes SB3's advantages undiscounted, per-episode
    reward-to-go, matching RLlib PG's no-critic postprocessor.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule: Schedule,
            net_arch=None,
            **kwargs,
    ):
        if not isinstance(observation_space, spaces.Discrete):
            raise TypeError("ReinforcePolicy requires a Discrete observation space")
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("ReinforcePolicy requires a Discrete action space")
        net_arch = [] if net_arch is None else net_arch
        if net_arch not in ([], (), {"pi": [], "vf": []}):
            raise ValueError(
                "legacy REINFORCE uses a linear policy; net_arch must be empty"
            )
        optimizer_kwargs = dict(kwargs.pop("optimizer_kwargs", {}) or {})
        # torch.optim.Adam (and the legacy RLlib TorchPolicy) defaults to 1e-8;
        # ActorCriticPolicy otherwise changes Adam epsilon to 1e-5.
        optimizer_kwargs.setdefault("eps", 1e-8)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            ortho_init=False,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        self.action_net = nn.Linear(
            self.mlp_extractor.latent_dim_pi,
            self.action_space.n,
            bias=False,
        ).to(self.device)
        _normc_(self.action_net.weight, scale=0.01)

        self.value_net = nn.Linear(
            self.mlp_extractor.latent_dim_vf, 1, bias=False
        ).to(self.device)
        nn.init.zeros_(self.value_net.weight)
        self.value_net.weight.requires_grad_(False)

        trainable = (parameter for parameter in self.parameters()
                     if parameter.requires_grad)
        self.optimizer = self.optimizer_class(
            trainable,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )


class Reinforce(A2C):
    """One-batch Monte Carlo policy gradient with an SB3-compatible API."""

    policy_aliases = {"MlpPolicy": ReinforcePolicy}

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]] = ReinforcePolicy,
            env: Optional[GymEnv] = None,
            learning_rate=LEGACY_REINFORCE_LEARNING_RATE,
            n_steps=50,
            seed=None,
            device="auto",
            verbose=0,
            policy_kwargs=None,
            _init_setup_model=True,
    ):
        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs.setdefault("net_arch", [])
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=1.0,
            gae_lambda=1.0,
            ent_coef=0.0,
            vf_coef=0.0,
            max_grad_norm=np.inf,
            use_rms_prop=False,
            normalize_advantage=False,
            seed=seed,
            device=device,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            _init_setup_model=_init_setup_model,
        )

    def train(self) -> None:
        """Perform the legacy RLlib PG loss over the complete rollout."""

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions.long().flatten()
            _, log_prob, _ = self.policy.evaluate_actions(
                rollout_data.observations, actions
            )
            reward_to_go = rollout_data.advantages
            policy_loss = reinforce_policy_loss(log_prob, reward_to_go)

            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record(
            "train/reward_to_go_mean", reward_to_go.mean().item()
        )


def reinforce_policy_loss(log_prob, reward_to_go):
    """The legacy vanilla policy-gradient objective as a testable primitive."""

    if log_prob.shape != reward_to_go.shape:
        raise ValueError("log_prob and reward_to_go must have identical shapes")
    return -(reward_to_go * log_prob).mean()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_evaluated_response_checkpoint(model, checkpoint, evaluate_fn: Callable):
    """Save and exhaustively evaluate a policy after a completed update.

    This deliberately is not an SB3 callback.  ``BaseCallback._on_step`` runs
    before the rollout's optimizer update, which made nominal checkpoint steps
    off by one update.  The caller invokes this function only after a bounded
    ``learn`` chunk has returned.
    """

    checkpoint = Path(checkpoint)
    if checkpoint.suffix != ".zip":
        raise ValueError("response checkpoint must use a .zip suffix")
    if checkpoint.exists():
        raise FileExistsError(
            "refusing to overwrite response checkpoint: {}".format(checkpoint)
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint.with_suffix("")))
    evaluation = evaluate_fn(model)
    summary = evaluation["summary"]
    record = {
        "mean_regret": float(summary["mean_regret"]),
        "max_regret": float(summary["max_regret"]),
        "optimal_commitments": int(summary["optimal_commitments"]),
        "commitments": int(summary["commitments"]),
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
    }
    return evaluation, record
