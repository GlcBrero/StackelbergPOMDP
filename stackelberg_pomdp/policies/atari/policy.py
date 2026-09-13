"""One SB3 actor--critic for the complete Atari curriculum.

The deployed actor sees only a frame stack, the canonical 14-dimensional
state, and a deterministic Atari action mask.  Critic-prefixed fields carry
stage-specific value information and action-credit bookkeeping, but never
enter either actor head or the observation--action cache key.
"""

from functools import partial
import hashlib
import math

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs

from stackelberg_pomdp.policies.cache import FixedActionPolicyMixin
from stackelberg_pomdp.policies.atari.components import (
    CompositeAtariFeaturesExtractor,
    GatedCompositeAtariDistribution,
    initialize_economic_head,
    inverse_softplus,
    make_economic_head,
    optimizer_parameter_groups as generic_optimizer_parameter_groups,
    validate_composite_atari_spaces,
)
from stackelberg_pomdp.policies.atari.meta_seller import (
    BETA_PARAMETER_EPSILON,
    attach_meta_seller_modules,
    consume_removed_seller_checkpoint_kwargs,
    initialize_meta_seller,
    is_meta_seller_architecture,
    meta_seller_beta_parameters,
    meta_seller_modules,
    meta_seller_optimizer_parameter_groups,
    seller_shared_context_architecture_provenance,
    validate_meta_seller_configuration,
)
from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTION_MASK,
    ACTOR_OBSERVATION_FIELDS,
    ACTOR_STATE,
    CRITIC_STATE,
    CRITIC_STATE_DIM,
    EVENT_SLICE,
)


_VALID_ECONOMIC_ROLES = frozenset({"buyer", "seller", "gameplay"})
_VALID_ECONOMIC_INPUT_MODES = frozenset({"full", "event_only"})
# Stable scientific identifier retained across the package reorganization.
# Released checkpoints record this historical import path, which remains
# loadable through ``atari.stackpomdp_policy``.
ATARI_POLICY_PROVENANCE_ID = (
    "stackelberg_pomdp.atari.stackpomdp_policy.StackPOMDPAtariPolicy"
)
ATARI_POLICY_PROVENANCE_ALIASES = frozenset({
    ATARI_POLICY_PROVENANCE_ID,
    "stackelberg_pomdp.atari.policies.composite.StackPOMDPAtariPolicy",
    "stackelberg_pomdp.policies.atari.composite.StackPOMDPAtariPolicy",
    "stackelberg_pomdp.policies.atari.policy.StackPOMDPAtariPolicy",
})


def canonical_atari_policy_provenance_id(value):
    """Normalize pre/post-reorganization import paths for one architecture."""

    value = str(value)
    if value in ATARI_POLICY_PROVENANCE_ALIASES:
        return ATARI_POLICY_PROVENANCE_ID
    return value


class StackPOMDPAtariPolicy(FixedActionPolicyMixin, ActorCriticPolicy):
    """Composite Atari policy used by every retained curriculum stage.

    ``economic_input_mode='full'`` is used for meta-followers, whose
    price/threshold may depend on live state and the opponent commitment.
    ``'event_only'`` is used for StackPOMDP leaders: before the economic head,
    every state coordinate except the five-entry event identity is set to zero.
    The retained meta-seller uses the shared-context v5 economic head.

    The observation--action map caches complete two-coordinate actions for
    every repeated actor observation.  Critic-only policy-credit fields are
    excluded, so a canonical query and its reward-trade replay are guaranteed
    to share a cache key while receiving different PPO credit.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *,
            economic_role="gameplay",
            economic_input_mode="full",
            visual_features=512,
            state_features=64,
            economic_hidden=64,
            critic_hidden=256,
            pretrained_lr_scale=1.0,
            gameplay_actor_frozen=False,
            economic_architecture=None,
            **kwargs,
    ):
        if economic_role not in _VALID_ECONOMIC_ROLES:
            raise ValueError(f"unknown economic role: {economic_role!r}")
        if economic_input_mode not in _VALID_ECONOMIC_INPUT_MODES:
            raise ValueError(
                "economic_input_mode must be one of "
                f"{sorted(_VALID_ECONOMIC_INPUT_MODES)}"
            )
        if not isinstance(gameplay_actor_frozen, (bool, np.bool_)):
            raise TypeError("gameplay_actor_frozen must be Boolean")
        consume_removed_seller_checkpoint_kwargs(kwargs)
        economic_architecture = validate_meta_seller_configuration(
            economic_architecture,
            role=economic_role,
            input_mode=economic_input_mode,
        )
        self.economic_role = str(economic_role)
        self.economic_input_mode = str(economic_input_mode)
        self.visual_features = int(visual_features)
        self.state_features = int(state_features)
        self.economic_hidden = int(economic_hidden)
        self.critic_hidden = int(critic_hidden)
        self.pretrained_lr_scale = float(pretrained_lr_scale)
        self.gameplay_actor_frozen = bool(gameplay_actor_frozen)
        self.economic_architecture = economic_architecture
        if min(
                self.visual_features,
                self.state_features,
                self.economic_hidden,
                self.critic_hidden,
        ) <= 0:
            raise ValueError("all policy dimensions must be positive")
        if not 0.0 < self.pretrained_lr_scale <= 1.0:
            raise ValueError(
                "pretrained learning-rate scale must lie in (0, 1]"
            )

        self.game_action_count = validate_composite_atari_spaces(
            observation_space,
            action_space,
        )
        kwargs.update(
            net_arch=[],
            activation_fn=nn.ReLU,
            ortho_init=True,
            features_extractor_class=CompositeAtariFeaturesExtractor,
            features_extractor_kwargs={
                "visual_features": self.visual_features,
                "state_features": self.state_features,
            },
        )
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

        # Runtime episode state is intentionally absent from checkpoints.
        self._initialize_fixed_action_cache()
        self.v5_context_ablation = False

    def _build(self, lr_schedule):
        actor_feature_dim = self.visual_features + self.state_features
        self.game_action_net = nn.Linear(
            actor_feature_dim, self.game_action_count
        )
        if is_meta_seller_architecture(self.economic_architecture):
            attach_meta_seller_modules(self)
        else:
            self.economic_head = make_economic_head(
                self.state_features,
                self.economic_hidden,
            )
        self.value_net = nn.Sequential(
            nn.Linear(actor_feature_dim + CRITIC_STATE_DIM, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, 1),
        )

        self.features_extractor.apply(
            partial(self.init_weights, gain=math.sqrt(2.0))
        )
        self.game_action_net.apply(partial(self.init_weights, gain=0.01))
        self.value_net.apply(partial(self.init_weights, gain=1.0))
        self._initialize_economic_head(mean=0.5, concentration=2.0)
        if (
                self.gameplay_actor_frozen
                or is_meta_seller_architecture(self.economic_architecture)
        ):
            self.freeze_gameplay_actor()
        # Keep this two-group layout in the policy constructor.  PyTorch can
        # restore optimizer state only when the saved and reconstructed group
        # layouts match, so regrouping only after construction makes otherwise
        # valid SB3 checkpoints unloadable.
        self.optimizer = self.optimizer_class(
            self.optimizer_parameter_groups(
                base_rate=lr_schedule(1),
                pretrained_scale=self.pretrained_lr_scale,
            ),
            **self.optimizer_kwargs,
        )

    def gameplay_actor_modules(self):
        """Return the exact transferred E0 actor modules."""

        return (
            self.features_extractor.visual,
            self.features_extractor.state_encoder,
            self.game_action_net,
        )

    def freeze_gameplay_actor(self):
        """Freeze every parameter that can change the Atari game action."""
        for module in self.gameplay_actor_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def optimizer_parameter_groups(self, *, base_rate, pretrained_scale):
        """Return the stable visual/game and newly trained parameter groups."""

        if is_meta_seller_architecture(self.economic_architecture):
            return meta_seller_optimizer_parameter_groups(
                self,
                self.value_net,
                base_rate=base_rate,
            )
        return generic_optimizer_parameter_groups(
            self.parameters(),
            pretrained_modules=(
                self.features_extractor.visual,
                self.game_action_net,
            ),
            base_rate=base_rate,
            pretrained_scale=pretrained_scale,
        )

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update({
            "economic_role": self.economic_role,
            "economic_input_mode": self.economic_input_mode,
            "visual_features": self.visual_features,
            "state_features": self.state_features,
            "economic_hidden": self.economic_hidden,
            "critic_hidden": self.critic_hidden,
            "pretrained_lr_scale": self.pretrained_lr_scale,
            "gameplay_actor_frozen": self.gameplay_actor_frozen,
        })
        if self.economic_architecture is not None:
            data["economic_architecture"] = self.economic_architecture
        return data

    def economic_architecture_provenance(self):
        """Return an exact, JSON-safe description of the economic actor path."""

        if is_meta_seller_architecture(self.economic_architecture):
            return seller_shared_context_architecture_provenance()
        return None

    def _initialize_economic_head(self, *, mean, concentration):
        if is_meta_seller_architecture(self.economic_architecture):
            initialize_meta_seller(
                self,
                mean=mean,
                concentration=concentration,
                inverse_softplus=inverse_softplus,
            )
            return
        initialize_economic_head(
            self.economic_head,
            init_weights=self.init_weights,
            mean=mean,
            concentration=concentration,
        )

    def reset_economic_head(self, *, mean=0.5, concentration=2.0):
        """Reinitialize the economic actor without touching transferred play."""

        self._initialize_economic_head(mean=mean, concentration=concentration)

    def load_actor_checkpoint(
            self,
            checkpoint,
            *,
            include_economic=True,
            device="cpu",
    ):
        """Transfer actor modules only; the stage-specific critic stays fresh."""

        from stackelberg_pomdp.checkpoints.atari import transfer_atari_actor

        return transfer_atari_actor(
            self,
            checkpoint,
            include_economic=include_economic,
            device=device,
        )

    def _processed(self, observations):
        return preprocess_obs(
            observations,
            self.observation_space,
            normalize_images=self.normalize_images,
        )

    def _actor_features(self, observations):
        processed = self._processed(observations)
        visual, state = self.features_extractor.forward_parts(processed)
        return processed, visual, state, th.cat([visual, state], dim=1)

    def _economic_state_features(self, processed, ordinary_state_features):
        if self.economic_input_mode == "full":
            return ordinary_state_features
        masked_state = th.zeros_like(processed[ACTOR_STATE])
        masked_state[:, EVENT_SLICE] = processed[ACTOR_STATE][:, EVENT_SLICE]
        return self.features_extractor.encode_state(masked_state)

    def set_v5_context_ablation(self, enabled):
        """Toggle an inference-only ablation of both shared context paths."""

        if not is_meta_seller_architecture(self.economic_architecture):
            raise ValueError("v5 context ablation requires seller v5")
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError("v5 context ablation flag must be Boolean")
        self.v5_context_ablation = bool(enabled)
        self.clear_obs_action_map()

    def _distribution(self, observations, actor_parts=None):
        if actor_parts is None:
            actor_parts = self._actor_features(observations)
        processed, _, state_features, actor_features = actor_parts
        logits = self.game_action_net(actor_features)
        action_mask = processed[ACTION_MASK].float().reshape(
            logits.shape[0], self.game_action_count
        )
        logits = th.where(
            action_mask > 0.0,
            logits,
            th.full_like(logits, -1.0e9),
        )
        if is_meta_seller_architecture(self.economic_architecture):
            alpha, beta = meta_seller_beta_parameters(
                processed[ACTOR_STATE],
                meta_seller_modules(self),
                context_ablation=self.v5_context_ablation,
            )
        else:
            economic_features = self._economic_state_features(
                processed, state_features
            )
            parameters = self.economic_head(economic_features)
            alpha = (
                nn.functional.softplus(parameters[:, 0])
                + BETA_PARAMETER_EPSILON
            )
            beta = (
                nn.functional.softplus(parameters[:, 1])
                + BETA_PARAMETER_EPSILON
            )
        return GatedCompositeAtariDistribution(
            game_logits=logits,
            economic_alpha=alpha,
            economic_beta=beta,
            action_credit=processed[ACTION_CREDIT],
            force_game_mode=(
                is_meta_seller_architecture(self.economic_architecture)
            ),
        )

    def _values(self, observations, actor_parts=None):
        if actor_parts is None:
            actor_parts = self._actor_features(observations)
        processed, _, _, actor_features = actor_parts
        critic_state = processed[CRITIC_STATE].float().reshape(
            actor_features.shape[0], CRITIC_STATE_DIM
        )
        return self.value_net(th.cat([actor_features, critic_state], dim=1))

    def _actor_cache_key(self, observations, row):
        """Hash one actor observation; omit all ``critic:*`` bookkeeping."""

        digest = hashlib.sha256()
        for name in ACTOR_OBSERVATION_FIELDS:
            values = observations[name][row].detach().cpu()
            array = values.contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return int(row), digest.digest()

    def _action_cache_row(self, key):
        """Return the vector-environment row encoded in an Atari cache key."""

        return int(key[0])

    def _actions(self, observations, distribution, *, deterministic):
        actions = distribution.get_actions(deterministic=deterministic)
        if not self.fix_actions:
            return actions
        for row in range(actions.shape[0]):
            key = self._actor_cache_key(observations, row)
            if key in self.obs_action_map:
                actions[row] = self.obs_action_map[key].to(actions.device)
            else:
                self.obs_action_map[key] = actions[row].detach().cpu().clone()
        return actions

    def forward(self, obs, deterministic=False):
        actor_parts = self._actor_features(obs)
        distribution = self._distribution(obs, actor_parts)
        actions = self._actions(obs, distribution, deterministic=deterministic)
        values = self._values(obs, actor_parts)
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        actor_parts = self._actor_features(obs)
        distribution = self._distribution(obs, actor_parts)
        return (
            self._values(obs, actor_parts),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def _predict(self, observation, deterministic=False):
        distribution = self._distribution(observation)
        return self._actions(
            observation,
            distribution,
            deterministic=deterministic,
        )

    def get_distribution(self, obs):
        return self._distribution(obs)

    def predict_values(self, obs):
        return self._values(obs)


__all__ = [
    "ATARI_POLICY_PROVENANCE_ALIASES",
    "ATARI_POLICY_PROVENANCE_ID",
    "StackPOMDPAtariPolicy",
    "canonical_atari_policy_provenance_id",
]
