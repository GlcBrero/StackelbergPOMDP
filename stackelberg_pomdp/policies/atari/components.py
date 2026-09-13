"""Reusable neural components for the composite Atari policy."""

import gym
import numpy as np
import torch as th
from torch import nn
from torch.distributions import Beta, Categorical

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
    FULL_ACTION_DIM,
    IMAGE,
)


def validate_composite_atari_spaces(observation_space, action_space):
    """Validate the public Atari interface and return its game-action count."""

    if not isinstance(observation_space, gym.spaces.Dict):
        raise TypeError("StackPOMDPAtariPolicy requires Dict observations")
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError("StackPOMDPAtariPolicy requires a Box action space")
    if action_space.shape != (FULL_ACTION_DIM,):
        raise ValueError("full Atari action must be [game_action, economic]")

    required = set(ACTOR_OBSERVATION_FIELDS) | {CRITIC_STATE, ACTION_CREDIT}
    missing = required - set(observation_space.spaces)
    if missing:
        raise ValueError(f"Atari observation missing keys: {sorted(missing)}")
    undeclared = {
        name
        for name in observation_space.spaces
        if name not in ACTOR_OBSERVATION_FIELDS
        and not name.startswith(CRITIC_PREFIX)
    }
    if undeclared:
        raise ValueError(
            "Atari observation has undeclared actor-visible keys: "
            f"{sorted(undeclared)}"
        )

    dimensions = {
        ACTOR_STATE: ACTOR_STATE_DIM,
        CRITIC_STATE: CRITIC_STATE_DIM,
        ACTION_CREDIT: FULL_ACTION_DIM,
    }
    for name, expected in dimensions.items():
        actual = int(np.prod(observation_space.spaces[name].shape))
        if actual != expected:
            raise ValueError(f"{name} must have {expected} entries, got {actual}")

    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    if not np.allclose(low, np.array([0.0, 0.0])):
        raise ValueError("full Atari action lower bound must be [0, 0]")
    if not np.isclose(high[1], 1.0):
        raise ValueError("economic action must be normalized to [0, 1]")
    game_high = float(high[0])
    if game_high < 0 or not np.isclose(game_high, round(game_high)):
        raise ValueError("game-action upper bound must be an integer")
    game_action_count = int(round(game_high)) + 1
    mask_size = int(np.prod(observation_space.spaces[ACTION_MASK].shape))
    if mask_size != game_action_count:
        raise ValueError("action-mask size must equal the Atari action count")
    return game_action_count


def make_economic_head(input_features, hidden_features):
    """Build the generic two-layer Beta-parameter head."""

    return nn.Sequential(
        nn.Linear(int(input_features), int(hidden_features)),
        nn.Tanh(),
        nn.Linear(int(hidden_features), int(hidden_features)),
        nn.Tanh(),
        nn.Linear(int(hidden_features), 2),
    )


def inverse_softplus(value):
    """Return the scalar inverse of PyTorch's softplus transform."""

    value = max(float(value), 1.0e-6)
    return float(np.log(np.expm1(value)))


def initialize_economic_head(
        head,
        *,
        init_weights,
        mean,
        concentration,
):
    """Initialize a generic Beta head to a state-independent distribution."""

    mean = float(mean)
    concentration = float(concentration)
    if not 0.0 < mean < 1.0:
        raise ValueError("Beta initialization mean must lie in (0, 1)")
    if concentration <= 0.0:
        raise ValueError("Beta initialization concentration must be positive")
    for module in head:
        if isinstance(module, nn.Linear):
            init_weights(module, gain=1.0)
    final = head[-1]
    nn.init.zeros_(final.weight)
    alpha = max(mean * concentration, 1.0e-3)
    beta = max((1.0 - mean) * concentration, 1.0e-3)
    with th.no_grad():
        final.bias.copy_(th.tensor(
            [inverse_softplus(alpha), inverse_softplus(beta)],
            dtype=final.bias.dtype,
            device=final.bias.device,
        ))


def optimizer_parameter_groups(
        parameters,
        *,
        pretrained_modules,
        base_rate,
        pretrained_scale,
):
    """Partition generic-policy parameters into pretrained and new groups."""

    scale = float(pretrained_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("pretrained learning-rate scale must lie in (0, 1]")
    parameters = list(parameters)
    protected = {
        id(parameter)
        for module in pretrained_modules
        for parameter in module.parameters()
    }
    pretrained = [
        parameter
        for parameter in parameters
        if parameter.requires_grad and id(parameter) in protected
    ]
    ordinary = [
        parameter
        for parameter in parameters
        if parameter.requires_grad and id(parameter) not in protected
    ]
    partition = pretrained + ordinary
    trainable = {id(parameter) for parameter in parameters if parameter.requires_grad}
    grouped = {id(parameter) for parameter in partition}
    if grouped != trainable or len(grouped) != len(partition):
        raise RuntimeError(
            "generic optimizer groups do not partition trainable parameters exactly"
        )
    rate = float(base_rate)
    return [
        {"params": pretrained, "lr": rate * scale, "lr_scale": scale},
        {"params": ordinary, "lr": rate, "lr_scale": 1.0},
    ]


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

    Sampling and PPO credit use the same two gates. Thus inactive coordinates
    are deterministic means/modes, and their log probabilities and entropies
    are exactly zero. Seller-specific heads may force the game coordinate to
    its masked categorical mode while retaining ordinary Beta sampling for
    credited economic decisions. Cached trade replays use ``[0, 0]``: they
    remain in the rollout for reward propagation but are not new decisions.
    """

    def __init__(
            self,
            *,
            game_logits,
            economic_alpha,
            economic_beta,
            action_credit,
            force_game_mode=False,
    ):
        self.game = Categorical(logits=game_logits)
        self.economic = Beta(economic_alpha, economic_beta)
        self.action_credit = action_credit.float().reshape(-1, FULL_ACTION_DIM)
        self.force_game_mode = bool(force_game_mode)

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
        if self.force_game_mode:
            game_action = game_mode.float()
        else:
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


__all__ = [
    "CompositeAtariFeaturesExtractor",
    "GatedCompositeAtariDistribution",
    "initialize_economic_head",
    "inverse_softplus",
    "make_economic_head",
    "optimizer_parameter_groups",
    "validate_composite_atari_spaces",
]
