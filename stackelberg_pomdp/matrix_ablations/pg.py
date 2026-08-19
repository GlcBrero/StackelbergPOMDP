"""Legacy-faithful vanilla policy gradient for matrix-game leaders.

The historical hidden-query treatment used RLlib's ``PG`` algorithm with
``batch_mode=complete_episodes`` and a minimum collection target of 100
executed environment transitions.  Query transitions still executed when
hidden, but the legacy callback removed them from the leader's training batch.

This module implements those mechanics without Ray.  It deliberately remains
separate from the maintained A2C/PPO implementation: the only trainable tensor
is a bias-free linear categorical policy, and the loss is exactly
``-mean(log pi(a|s) * undiscounted_reward_to_go)``.
"""

from typing import Optional, Type, Union

from gym import spaces
import numpy as np
import torch as th
from torch import nn

from stable_baselines3 import A2C
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from stackelberg_pomdp.algorithms.on_policy import CustomOnPolicyAlgorithm


LEGACY_PG_COLLECTION_TARGET_ENV_STEPS = 100
LEGACY_LEADER_PG_LEARNING_RATE = 0.008
LEGACY_LEADER_PG_OUTER_UPDATES = 2_000


def leader_pg_rollout_geometry(
        episode_length,
        num_query_states,
        hidden_queries,
        collection_target=LEGACY_PG_COLLECTION_TARGET_ENV_STEPS,
):
    """Return one complete-episode leader-PG collection geometry.

    ``collection_target`` applies to executed environment transitions, as it
    did in RLlib.  Complete-episode collection may exceed that target.  Hidden
    query transitions count as executed but not as stored gradient samples.
    """

    episode_length = int(episode_length)
    num_query_states = int(num_query_states)
    collection_target = int(collection_target)
    if episode_length < 1:
        raise ValueError("episode_length must be positive")
    if num_query_states < 0:
        raise ValueError("num_query_states must be nonnegative")
    if collection_target < 1:
        raise ValueError("collection_target must be positive")

    executed_per_episode = episode_length + num_query_states
    stored_per_episode = (
        episode_length if bool(hidden_queries) else executed_per_episode
    )
    episodes_per_update = int(np.ceil(
        collection_target / executed_per_episode
    ))
    return {
        "collection_target_env_steps": collection_target,
        "hidden_queries": bool(hidden_queries),
        "episodes_per_update": episodes_per_update,
        "executed_env_steps_per_episode": executed_per_episode,
        "stored_gradient_samples_per_episode": stored_per_episode,
        "query_env_steps_per_update": (
            episodes_per_update * num_query_states
        ),
        "hidden_query_steps_excluded_per_update": (
            episodes_per_update * num_query_states
            if bool(hidden_queries) else 0
        ),
        "executed_env_steps_per_update": (
            episodes_per_update * executed_per_episode
        ),
        "stored_gradient_samples_per_update": (
            episodes_per_update * stored_per_episode
        ),
    }


def _normc_(weight, scale=0.01):
    """Apply RLlib's row-wise normc initialization in place."""

    with th.no_grad():
        weight.normal_(0.0, 1.0)
        row_norm = th.sqrt(th.sum(th.square(weight), dim=1, keepdim=True))
        weight.mul_(float(scale) / row_norm)
    return weight


class LeaderPGPolicy(MultiInputActorCriticPolicy):
    """Bias-free linear categorical policy used by the matrix leader.

    The value head is a frozen zero-valued compatibility shim.  With
    ``gamma=gae_lambda=1`` it makes the SB3 rollout buffer produce the same
    undiscounted reward-to-go used by RLlib PG, without introducing a critic or
    baseline into either the loss or optimizer.

    There is deliberately no action cache here.  Historical matrix PG sampled
    independently at every visit during leader training.  Query and reward
    calls therefore share one stationary policy distribution, not one sampled
    action table.  Deterministic evaluation calls the same policy with
    ``deterministic=True`` and consequently uses a consistent argmax action.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule: Schedule,
            net_arch=None,
            **kwargs,
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("LeaderPGPolicy requires a Dict observation space")
        if not observation_space.spaces or any(
                not isinstance(space, spaces.Discrete)
                for space in observation_space.spaces.values()
        ):
            raise TypeError(
                "LeaderPGPolicy requires exclusively Discrete Dict entries"
            )
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("LeaderPGPolicy requires a Discrete action space")
        net_arch = [] if net_arch is None else net_arch
        if net_arch not in ([], (), {"pi": [], "vf": []}):
            raise ValueError("legacy leader PG uses a linear policy")

        optimizer_kwargs = dict(kwargs.pop("optimizer_kwargs", {}) or {})
        # SB3's actor-critic policy otherwise changes Adam epsilon to 1e-5.
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

        trainable = (
            parameter for parameter in self.parameters()
            if parameter.requires_grad
        )
        self.optimizer = self.optimizer_class(
            trainable,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )


def leader_pg_policy_loss(log_prob, reward_to_go):
    """The historical vanilla policy-gradient objective."""

    if log_prob.shape != reward_to_go.shape:
        raise ValueError("log_prob and reward_to_go must have identical shapes")
    return -(log_prob * reward_to_go).mean()


class LeaderPolicyGradient(A2C, CustomOnPolicyAlgorithm):
    """Ray-free legacy leader PG with hidden-query buffer exclusion."""

    policy_aliases = {"MultiInputPolicy": LeaderPGPolicy}

    def __init__(
            self,
            policy: Union[str, Type[MultiInputActorCriticPolicy]] = LeaderPGPolicy,
            env: Optional[GymEnv] = None,
            learning_rate=LEGACY_LEADER_PG_LEARNING_RATE,
            n_steps=100,
            seed=None,
            device="auto",
            verbose=0,
            policy_kwargs=None,
            _init_setup_model=True,
    ):
        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs.setdefault("net_arch", [])
        self.post_update_hook = None
        self.last_rollout_stored_steps = 0
        self.last_rollout_executed_steps = 0
        self.last_policy_loss = None
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=int(n_steps),
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

    def collect_rollouts(
            self, env, callback, rollout_buffer, n_rollout_steps
    ):
        executed_before = int(self.num_timesteps)
        completed = CustomOnPolicyAlgorithm.collect_rollouts(
            self,
            env,
            callback,
            rollout_buffer,
            n_rollout_steps,
        )
        self.last_rollout_stored_steps = int(n_rollout_steps)
        self.last_rollout_executed_steps = (
            int(self.num_timesteps) - executed_before
        )
        return completed

    def train(self) -> None:
        """Apply one exact Monte Carlo policy-gradient update."""

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        batch_count = 0
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions.long().flatten()
            _, log_prob, _ = self.policy.evaluate_actions(
                rollout_data.observations, actions
            )
            reward_to_go = rollout_data.advantages
            policy_loss = leader_pg_policy_loss(log_prob, reward_to_go)

            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()
            batch_count += 1

        if batch_count != 1:
            raise RuntimeError(
                "leader PG requires exactly one complete rollout batch"
            )
        self._n_updates += 1
        self.last_policy_loss = float(policy_loss.detach().cpu().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_loss", self.last_policy_loss)
        self.logger.record(
            "train/reward_to_go_mean",
            float(reward_to_go.detach().mean().cpu().item()),
        )
        if self.post_update_hook is not None:
            self.post_update_hook(self)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["post_update_hook"]
