"""One SB3 actor--critic for the clean Atari E0a--E2 curriculum.

The deployed actor sees only a frame stack, the canonical 14-dimensional
state, and a deterministic Atari action mask.  Critic-prefixed fields carry
stage-specific value information and action-credit bookkeeping, but never
enter either actor head or the observation--action cache key.
"""

from functools import partial
import hashlib
import math
from pathlib import Path

import gym
import numpy as np
import torch as th
from torch import nn
from torch.distributions import Beta, Categorical

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTION_MASK,
    ACTOR_OBSERVATION_FIELDS,
    ACTOR_STATE,
    ACTOR_STATE_DIM,
    CRITIC_STATE,
    CRITIC_STATE_DIM,
    CRITIC_PREFIX,
    EVENT_SLICE,
    FULL_ACTION_DIM,
    IMAGE,
    OPPONENT_COMMITMENT_SLICE,
)


ECONOMIC_ROLES = {"buyer", "seller", "gameplay"}
ECONOMIC_INPUT_MODES = {"full", "event_only"}
BETA_PARAMETER_EPSILON = 1.0e-4
THRESHOLD_RESIDUAL_BLEND_WEIGHT = 0.5
THRESHOLD_RESIDUAL_EPSILON = 1.0e-4


def current_event_threshold(actor_state_values):
    """Select the current threshold from the canonical actor state.

    This is deliberately a deterministic view, not a new observation: the
    five-entry event one-hot selects the matching entry of the five-entry
    opponent commitment already present in ``actor_state``.
    """

    if not th.is_tensor(actor_state_values):
        raise TypeError("current threshold selector requires a torch Tensor")
    if (
            actor_state_values.ndim < 1
            or actor_state_values.shape[-1] != ACTOR_STATE_DIM
    ):
        raise ValueError(
            "current threshold selector requires actor_state with final "
            f"dimension {ACTOR_STATE_DIM}"
        )
    event = actor_state_values[..., EVENT_SLICE]
    commitment = actor_state_values[..., OPPONENT_COMMITMENT_SLICE]
    return th.sum(event * commitment, dim=-1, keepdim=True)


def threshold_residual_architecture_provenance(*, state_features):
    """Return the exact pure-64 seller residual architecture contract."""

    return {
        "schema": "stackpomdp.atari.economic_actor_architecture.v2",
        "economic_role": "seller",
        "economic_input_mode": "full",
        "parameterization": "seller_threshold_residual_beta_v1",
        "base_head_input_features": int(state_features),
        "direct_extra_input_features": 0,
        "base_head_outputs": [
            "raw_alpha_parameter", "raw_beta_parameter"
        ],
        "base_positivity_transform": {
            "formula": "softplus(raw_parameter) + epsilon",
            "epsilon": BETA_PARAMETER_EPSILON,
        },
        "current_threshold": {
            "definition": (
                "dot(actor_state[event_one_hot], "
                "actor_state[opponent_commitment])"
            ),
            "deterministic_from_existing_actor_state": True,
            "new_observation_fields": [],
        },
        "mean_transform": {
            "formula": "mu = (1 - w) * mu_base + w * t_current",
            "threshold_weight": THRESHOLD_RESIDUAL_BLEND_WEIGHT,
            "base_weight": 1.0 - THRESHOLD_RESIDUAL_BLEND_WEIGHT,
            "interpretation": (
                "network learns a state/context-dependent markup or "
                "discount around a fixed threshold anchor; the anchor "
                "slope is an inductive bias, not a learned quantity"
            ),
            "unit_interval_clamp_epsilon": THRESHOLD_RESIDUAL_EPSILON,
        },
        "concentration_transform": (
            "alpha + beta is preserved from the base head"
        ),
        "distribution": (
            "Beta(mu * concentration, (1-mu) * concentration)"
        ),
        "state_encoder_uses_current_threshold_outside_existing_state": False,
        "game_head_uses_transform": False,
        "critic_uses_transform": False,
    }


def direct_threshold_residual_architecture_provenance(*, state_features):
    """Return the exact v3 direct-threshold residual architecture contract."""

    provenance = threshold_residual_architecture_provenance(
        state_features=state_features
    )
    provenance.update({
        "schema": "stackpomdp.atari.economic_actor_architecture.v3",
        "parameterization": "seller_direct_threshold_residual_beta_v3",
        "base_head_input_features": int(state_features) + 1,
        "ordinary_state_embedding_features": int(state_features),
        "direct_extra_input_features": 1,
        "base_head_input_order": [
            f"shared_state_embedding[0:{int(state_features)}]",
            "t_current",
        ],
        "direct_current_threshold_input": {
            "definition": (
                "dot(actor_state[event_one_hot], "
                "actor_state[opponent_commitment])"
            ),
            "deterministic_from_existing_actor_state": True,
            "new_observation_fields": [],
            "destination": "economic_base_head_only",
            "first_linear_column_index": int(state_features),
            "first_linear_column_width": 1,
            "initialization": "exact_zero",
            "initial_learned_base_threshold_slope": 0.0,
        },
        "initialization_invariance": {
            "canonical_64_input_head_prefix_copied_exactly": True,
            "only_new_direct_input_column_zero_initialized": True,
            "canonical_rng_stream_preserved": True,
            "non_economic_weights_same_seed_invariant": True,
        },
        "state_encoder_uses_current_threshold_outside_existing_state": False,
        "game_head_uses_direct_input": False,
        "critic_uses_direct_input": False,
    })
    return provenance


class CompositeAtariFeaturesExtractor(BaseFeaturesExtractor):
    """Nature CNN plus the shared 64-unit low-dimensional state encoder."""

    def __init__(
            self,
            observation_space,
            *,
            visual_features=512,
            state_features=64,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("composite Atari policy requires Dict observations")
        required = set(ACTOR_OBSERVATION_FIELDS) | {
            CRITIC_STATE,
            ACTION_CREDIT,
        }
        missing = required - set(observation_space.spaces)
        if missing:
            raise ValueError(f"Atari observation missing keys: {sorted(missing)}")
        state_dim = int(np.prod(observation_space.spaces[ACTOR_STATE].shape))
        if state_dim != ACTOR_STATE_DIM:
            raise ValueError(
                f"{ACTOR_STATE} must have {ACTOR_STATE_DIM} entries, got {state_dim}"
            )

        self.visual_features = int(visual_features)
        self.state_features = int(state_features)
        if min(self.visual_features, self.state_features) <= 0:
            raise ValueError("feature dimensions must be positive")
        super().__init__(
            observation_space,
            features_dim=self.visual_features + self.state_features,
        )

        image_space = observation_space.spaces[IMAGE]
        image_shape = tuple(image_space.shape)
        self.image_is_channel_last = bool(
            len(image_shape) == 3
            and image_shape[-1] in (1, 3, 4)
            and image_shape[0] not in (1, 3, 4)
        )
        if self.image_is_channel_last:
            cnn_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(image_shape[-1], image_shape[0], image_shape[1]),
                dtype=image_space.dtype,
            )
        else:
            cnn_space = image_space
        self.visual = NatureCNN(cnn_space, features_dim=self.visual_features)
        self.state_encoder = nn.Sequential(
            nn.Linear(ACTOR_STATE_DIM, self.state_features),
            nn.ReLU(),
        )

    def encode_visual(self, observations):
        image = observations[IMAGE].float()
        if self.image_is_channel_last:
            image = image.permute(0, 3, 1, 2)
        return self.visual(image)

    def encode_state(self, state):
        return self.state_encoder(
            state.float().reshape(state.shape[0], ACTOR_STATE_DIM)
        )

    def forward_parts(self, observations):
        return (
            self.encode_visual(observations),
            self.encode_state(observations[ACTOR_STATE]),
        )

    def forward(self, observations):
        visual, state = self.forward_parts(observations)
        return th.cat([visual, state], dim=1)


class GatedCompositeAtariDistribution:
    """Categorical game action plus Beta economic action with separate credit.

    Sampling and PPO credit use the same two gates.  Thus inactive coordinates
    are deterministic means/modes, and their log probabilities and entropies
    are exactly zero.  Cached trade replays use ``[0, 0]``: they remain in the
    rollout for reward propagation and value learning but are not new policy
    decisions.
    """

    def __init__(
            self,
            *,
            game_logits,
            economic_alpha,
            economic_beta,
            action_credit,
    ):
        self.game = Categorical(logits=game_logits)
        self.economic = Beta(economic_alpha, economic_beta)
        self.action_credit = action_credit.float().reshape(-1, FULL_ACTION_DIM)

    @property
    def game_gate(self):
        return self.action_credit[:, 0]

    @property
    def economic_gate(self):
        return self.action_credit[:, 1]

    @property
    def economic_mean(self):
        return self.economic.mean

    def mode(self):
        return th.stack(
            [th.argmax(self.game.logits, dim=1).float(), self.economic.mean],
            dim=1,
        )

    def sample(self):
        game_mode = th.argmax(self.game.logits, dim=1)
        game_action = th.where(
            self.game_gate > 0.5,
            self.game.sample(),
            game_mode,
        ).float()
        economic_action = th.where(
            self.economic_gate > 0.5,
            self.economic.sample(),
            self.economic.mean,
        )
        return th.stack([game_action, economic_action], dim=1)

    def get_actions(self, deterministic=False):
        return self.mode() if deterministic else self.sample()

    def log_prob(self, actions):
        values = actions.reshape(-1, FULL_ACTION_DIM)
        game_actions = values[:, 0].round().long()
        economic_actions = values[:, 1].clamp(1.0e-6, 1.0 - 1.0e-6)
        return (
            self.game_gate * self.game.log_prob(game_actions)
            + self.economic_gate * self.economic.log_prob(economic_actions)
        )

    def entropy(self):
        return (
            self.game_gate * self.game.entropy()
            + self.economic_gate * self.economic.entropy()
        )


class StackPOMDPAtariPolicy(ActorCriticPolicy):
    """Composite policy with a stable actor interface and private critic.

    ``economic_input_mode='full'`` is used for E1 meta-followers, whose
    price/threshold may depend on live state and the opponent commitment.
    A full-input E1 seller may optionally retain the ordinary 64-input Beta
    head but blend its mean with the current event's opponent threshold after
    the head.  A separately versioned seller-only variant also appends that
    deterministic scalar to the base head.  Both defaults are disabled for
    checkpoint compatibility.
    ``'event_only'`` is used for E2 leaders: before the economic head, every
    state coordinate except the five-entry event identity is set to zero.

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
            economic_threshold_residual=False,
            economic_threshold_residual_direct_input=False,
            **kwargs,
    ):
        if economic_role not in ECONOMIC_ROLES:
            raise ValueError(f"unknown economic role: {economic_role!r}")
        if economic_input_mode not in ECONOMIC_INPUT_MODES:
            raise ValueError(
                f"economic_input_mode must be one of {sorted(ECONOMIC_INPUT_MODES)}"
            )
        if not isinstance(economic_threshold_residual, (bool, np.bool_)):
            raise TypeError("economic_threshold_residual must be Boolean")
        if not isinstance(
                economic_threshold_residual_direct_input, (bool, np.bool_)
        ):
            raise TypeError(
                "economic_threshold_residual_direct_input must be Boolean"
            )
        if (
                economic_threshold_residual_direct_input
                and not economic_threshold_residual
        ):
            raise ValueError(
                "the direct threshold input requires the threshold-residual "
                "parameterization"
            )
        if economic_threshold_residual and (
                economic_role != "seller" or economic_input_mode != "full"
        ):
            raise ValueError(
                "the threshold-residual parameterization is reserved for "
                "full-input E1 seller response policies"
            )
        if economic_threshold_residual and int(state_features) != 64:
            raise ValueError(
                "the threshold-residual v1 contract requires exactly 64 "
                "shared state features"
            )
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("StackPOMDPAtariPolicy requires Dict observations")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("StackPOMDPAtariPolicy requires a Box action space")
        if action_space.shape != (FULL_ACTION_DIM,):
            raise ValueError("full Atari action must be [game_action, economic]")

        self.economic_role = str(economic_role)
        self.economic_input_mode = str(economic_input_mode)
        self.visual_features = int(visual_features)
        self.state_features = int(state_features)
        self.economic_hidden = int(economic_hidden)
        self.critic_hidden = int(critic_hidden)
        self.pretrained_lr_scale = float(pretrained_lr_scale)
        self.economic_threshold_residual = bool(economic_threshold_residual)
        self.economic_threshold_residual_direct_input = bool(
            economic_threshold_residual_direct_input
        )
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

        self._validate_spaces(observation_space, action_space)
        self.game_action_count = int(round(float(action_space.high[0]))) + 1

        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.pop("net_arch", None)
        kwargs.pop("activation_fn", None)
        kwargs.pop("ortho_init", None)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=nn.ReLU,
            ortho_init=True,
            features_extractor_class=CompositeAtariFeaturesExtractor,
            features_extractor_kwargs={
                "visual_features": self.visual_features,
                "state_features": self.state_features,
            },
            **kwargs,
        )

        # Runtime episode state is intentionally absent from checkpoints.
        self.fix_actions = False
        self.obs_action_map = {}

    @staticmethod
    def _validate_spaces(observation_space, action_space):
        required = set(ACTOR_OBSERVATION_FIELDS) | {
            CRITIC_STATE,
            ACTION_CREDIT,
        }
        missing = required - set(observation_space.spaces)
        if missing:
            raise ValueError(f"Atari observation missing keys: {sorted(missing)}")
        unexpected_actor_fields = {
            name
            for name in observation_space.spaces
            if name not in ACTOR_OBSERVATION_FIELDS
            and not name.startswith(CRITIC_PREFIX)
        }
        if unexpected_actor_fields:
            raise ValueError(
                "Atari observation has undeclared actor-visible keys: "
                f"{sorted(unexpected_actor_fields)}"
            )
        expected = {
            ACTOR_STATE: ACTOR_STATE_DIM,
            CRITIC_STATE: CRITIC_STATE_DIM,
            ACTION_CREDIT: FULL_ACTION_DIM,
        }
        for key, size in expected.items():
            actual = int(np.prod(observation_space.spaces[key].shape))
            if actual != size:
                raise ValueError(f"{key} must have {size} entries, got {actual}")

        action_low = np.asarray(action_space.low, dtype=np.float64)
        action_high = np.asarray(action_space.high, dtype=np.float64)
        if not np.allclose(action_low, np.array([0.0, 0.0])):
            raise ValueError("full Atari action lower bound must be [0, 0]")
        if not np.isclose(action_high[1], 1.0):
            raise ValueError("economic action must be normalized to [0, 1]")
        game_high = float(action_high[0])
        if game_high < 0 or not np.isclose(game_high, round(game_high)):
            raise ValueError("game-action upper bound must be an integer")
        mask_size = int(np.prod(observation_space.spaces[ACTION_MASK].shape))
        if mask_size != int(round(game_high)) + 1:
            raise ValueError("action-mask size must equal the Atari action count")

    def _build(self, lr_schedule):
        actor_feature_dim = self.visual_features + self.state_features
        self.game_action_net = nn.Linear(
            actor_feature_dim, self.game_action_count
        )
        if self.economic_threshold_residual_direct_input:
            # Constructing a 65-input layer would otherwise advance Torch's
            # global RNG farther than the canonical 64-input head.  Isolate
            # the wider construction, then consume exactly the canonical
            # construction stream.  The initialization below likewise copies
            # a canonical head and initializes only the new column to zero.
            with th.random.fork_rng(devices=[]):
                self.economic_head = self._make_economic_head(
                    self.state_features + 1
                )
            self._make_economic_head(self.state_features)
        else:
            self.economic_head = self._make_economic_head(self.state_features)
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

    def _make_economic_head(self, input_features):
        return nn.Sequential(
            nn.Linear(int(input_features), self.economic_hidden),
            nn.Tanh(),
            nn.Linear(self.economic_hidden, self.economic_hidden),
            nn.Tanh(),
            nn.Linear(self.economic_hidden, 2),
        )

    def optimizer_parameter_groups(self, *, base_rate, pretrained_scale):
        """Return the stable visual/game and newly trained parameter groups."""

        scale = float(pretrained_scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError("pretrained learning-rate scale must lie in (0, 1]")
        protected = {
            id(parameter)
            for module in (
                self.features_extractor.visual,
                self.game_action_net,
            )
            for parameter in module.parameters()
        }
        pretrained = [
            parameter
            for parameter in self.parameters()
            if id(parameter) in protected
        ]
        ordinary = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in protected
        ]
        return [
            {
                "params": pretrained,
                "lr": float(base_rate) * scale,
                "lr_scale": scale,
            },
            {
                "params": ordinary,
                "lr": float(base_rate),
                "lr_scale": 1.0,
            },
        ]

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
            "economic_threshold_residual": self.economic_threshold_residual,
            "economic_threshold_residual_direct_input": (
                self.economic_threshold_residual_direct_input
            ),
        })
        return data

    def economic_architecture_provenance(self):
        """Return an exact, JSON-safe description of the economic actor path."""

        if not self.economic_threshold_residual:
            return None
        if self.economic_threshold_residual_direct_input:
            return direct_threshold_residual_architecture_provenance(
                state_features=self.state_features
            )
        return threshold_residual_architecture_provenance(
            state_features=self.state_features
        )

    @staticmethod
    def _inverse_softplus(value):
        value = max(float(value), 1.0e-6)
        return math.log(math.expm1(value))

    def _initialize_economic_head_module(self, head, *, mean, concentration):
        mean = float(mean)
        concentration = float(concentration)
        if not 0.0 < mean < 1.0:
            raise ValueError("Beta initialization mean must lie in (0, 1)")
        if concentration <= 0.0:
            raise ValueError("Beta initialization concentration must be positive")
        for module in head:
            if isinstance(module, nn.Linear):
                self.init_weights(module, gain=1.0)
        final = head[-1]
        nn.init.zeros_(final.weight)
        alpha = max(mean * concentration, 1.0e-3)
        beta = max((1.0 - mean) * concentration, 1.0e-3)
        with th.no_grad():
            final.bias.copy_(th.tensor([
                self._inverse_softplus(alpha),
                self._inverse_softplus(beta),
            ], dtype=final.bias.dtype, device=final.bias.device))

    def _initialize_economic_head(self, *, mean, concentration):
        if not self.economic_threshold_residual_direct_input:
            self._initialize_economic_head_module(
                self.economic_head,
                mean=mean,
                concentration=concentration,
            )
            return

        # Build the temporary module without consuming global randomness, then
        # initialize it on the global stream exactly as the canonical head.
        # Copy every canonical parameter and set only the appended first-layer
        # input column to exact zero.
        with th.random.fork_rng(devices=[]):
            canonical = self._make_economic_head(self.state_features)
        self._initialize_economic_head_module(
            canonical,
            mean=mean,
            concentration=concentration,
        )
        with th.no_grad():
            self.economic_head[0].weight[:, :self.state_features].copy_(
                canonical[0].weight
            )
            self.economic_head[0].weight[:, self.state_features].zero_()
            self.economic_head[0].bias.copy_(canonical[0].bias)
            for index in (2, 4):
                self.economic_head[index].weight.copy_(canonical[index].weight)
                self.economic_head[index].bias.copy_(canonical[index].bias)

    def reset_economic_head(self, *, mean=0.5, concentration=2.0):
        """Reinitialize the economic actor without touching transferred play."""

        self._initialize_economic_head(mean=mean, concentration=concentration)

    @staticmethod
    def _checkpoint_file(checkpoint):
        path = Path(checkpoint).expanduser()
        if path.is_file():
            return path
        zip_path = Path(f"{path}.zip")
        if zip_path.is_file():
            return zip_path
        raise FileNotFoundError(f"Atari checkpoint does not exist: {path}")

    def load_actor_checkpoint(
            self,
            checkpoint,
            *,
            include_economic=True,
            device="cpu",
    ):
        """Transfer actor modules only; the stage-specific critic stays fresh."""

        from stable_baselines3 import PPO

        path = self._checkpoint_file(checkpoint)
        source_model = PPO.load(str(path), device=device)
        source = source_model.policy
        if not isinstance(source, StackPOMDPAtariPolicy):
            raise TypeError(
                "clean stages require a StackPOMDPAtariPolicy checkpoint; "
                "legacy Atari checkpoints are intentionally incompatible"
            )
        modules = ["features_extractor", "game_action_net"]
        if include_economic:
            if (
                    source.economic_threshold_residual
                    != self.economic_threshold_residual
                    or source.economic_threshold_residual_direct_input
                    != self.economic_threshold_residual_direct_input
            ):
                raise ValueError(
                    "economic-head transfer requires matching seller "
                    "economic parameterizations"
                )
            modules.append("economic_head")
        for name in modules:
            getattr(self, name).load_state_dict(
                getattr(source, name).state_dict(), strict=True
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_economic_role = source.economic_role
        source_economic_input_mode = source.economic_input_mode
        source_economic_threshold_residual = (
            source.economic_threshold_residual
        )
        source_economic_threshold_residual_direct_input = bool(getattr(
            source, "economic_threshold_residual_direct_input", False
        ))
        source_pretrained_lr_scale = source.pretrained_lr_scale
        del source_model
        return {
            "checkpoint": str(path.resolve()),
            "sha256": digest,
            "modules": tuple(modules),
            "critic_transferred": False,
            "source_economic_role": source_economic_role,
            "source_economic_input_mode": source_economic_input_mode,
            "source_economic_threshold_residual": (
                source_economic_threshold_residual
            ),
            "source_economic_threshold_residual_direct_input": (
                source_economic_threshold_residual_direct_input
            ),
            "source_pretrained_lr_scale": source_pretrained_lr_scale,
        }

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
            if not self.economic_threshold_residual_direct_input:
                return ordinary_state_features
            return th.cat([
                ordinary_state_features,
                current_event_threshold(processed[ACTOR_STATE]),
            ], dim=1)
        masked_state = th.zeros_like(processed[ACTOR_STATE])
        masked_state[:, EVENT_SLICE] = processed[ACTOR_STATE][:, EVENT_SLICE]
        return self.features_extractor.encode_state(masked_state)

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
        economic_features = self._economic_state_features(
            processed, state_features
        )
        parameters = self.economic_head(economic_features)
        alpha = (
            nn.functional.softplus(parameters[:, 0]) + BETA_PARAMETER_EPSILON
        )
        beta = (
            nn.functional.softplus(parameters[:, 1]) + BETA_PARAMETER_EPSILON
        )
        if self.economic_threshold_residual:
            concentration = alpha + beta
            base_mean = alpha / concentration
            current_threshold = current_event_threshold(
                processed[ACTOR_STATE]
            ).reshape(-1)
            mean = (
                (1.0 - THRESHOLD_RESIDUAL_BLEND_WEIGHT) * base_mean
                + THRESHOLD_RESIDUAL_BLEND_WEIGHT * current_threshold
            ).clamp(
                THRESHOLD_RESIDUAL_EPSILON,
                1.0 - THRESHOLD_RESIDUAL_EPSILON,
            )
            alpha = mean * concentration
            beta = (1.0 - mean) * concentration
        return GatedCompositeAtariDistribution(
            game_logits=logits,
            economic_alpha=alpha,
            economic_beta=beta,
            action_credit=processed[ACTION_CREDIT],
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

    def clear_obs_action_map(self, rows=None):
        """Clear every episode cache, or only completed vector-env rows."""

        if rows is None:
            self.obs_action_map = {}
            return
        completed = {int(row) for row in rows}
        self.obs_action_map = {
            key: value
            for key, value in self.obs_action_map.items()
            if key[0] not in completed
        }

    def fix_policy_actions(self):
        self.fix_actions = True

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
    "BETA_PARAMETER_EPSILON",
    "THRESHOLD_RESIDUAL_BLEND_WEIGHT",
    "THRESHOLD_RESIDUAL_EPSILON",
    "CompositeAtariFeaturesExtractor",
    "ECONOMIC_INPUT_MODES",
    "ECONOMIC_ROLES",
    "GatedCompositeAtariDistribution",
    "StackPOMDPAtariPolicy",
    "current_event_threshold",
    "direct_threshold_residual_architecture_provenance",
    "threshold_residual_architecture_provenance",
]
