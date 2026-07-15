"""Atari models owned by the StackelbergPOMDP replication codebase."""

import gym
import numpy as np
import torch
from ray.rllib.models.torch.misc import SlimConv2d, SlimFC, same_padding
from ray.rllib.models.torch.misc import normc_initializer as normc_initializer_torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch


_, nn = try_import_torch()


class AmmoAwareNatureCNNTorch(TorchModelV2, nn.Module):
    """Nature CNN with a learned projection of normalized remaining ammo.

    The visual trunk is the standard 512-feature Nature CNN.  The scalar
    ``ammo_fraction`` (remaining bullets divided by the five-bullet capacity)
    is projected into a small learned feature vector before concatenation.  An
    observation-provided action mask makes FIRE, LEFTFIRE, and RIGHTFIRE logits
    unavailable while a projectile is active or the pool is empty.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        original_space = getattr(obs_space, "original_space", obs_space)
        if not isinstance(original_space, gym.spaces.Dict):
            raise TypeError("AmmoAwareNatureCNNTorch requires a Dict observation")
        required = {"image", "ammo_fraction", "action_mask"}
        missing = required - set(original_space.spaces)
        if missing:
            raise ValueError(f"ammo-aware observation missing keys: {sorted(missing)}")

        h, w, in_channels = original_space.spaces["image"].shape
        activation = nn.ReLU
        conv_layers = []
        padding, out_size = same_padding([h, w], [8, 8], (4, 4))
        conv_layers.append(
            SlimConv2d(
                in_channels,
                32,
                [8, 8],
                4,
                padding,
                activation_fn=activation,
            )
        )
        padding, out_size = same_padding(out_size, [4, 4], (2, 2))
        conv_layers.append(
            SlimConv2d(32, 64, [4, 4], 2, padding, activation_fn=activation)
        )
        conv_layers.append(
            SlimConv2d(64, 64, [3, 3], 1, None, activation_fn=activation)
        )
        conv_layers.append(nn.Flatten())
        flat_size = _conv_flat_size(h, w, in_channels, conv_layers[:-1])
        conv_layers.append(
            SlimFC(
                in_size=flat_size,
                out_size=512,
                activation_fn=activation,
                initializer=normc_initializer_torch(1.0),
            )
        )
        self._convs = nn.Sequential(*conv_layers)

        custom_config = (model_config or {}).get("custom_model_config", {})
        self.ammo_hidden = int(custom_config.get("ammo_hidden", 32))
        if self.ammo_hidden <= 0:
            raise ValueError("ammo_hidden must be positive")
        self._ammo_encoder = SlimFC(
            in_size=1,
            out_size=self.ammo_hidden,
            activation_fn=activation,
            initializer=normc_initializer_torch(1.0),
        )
        head_size = 512 + self.ammo_hidden
        self.action = SlimFC(
            in_size=head_size,
            out_size=num_outputs,
            activation_fn=None,
            initializer=normc_initializer_torch(0.01),
        )
        self.value = SlimFC(
            in_size=head_size,
            out_size=1,
            activation_fn=None,
            initializer=normc_initializer_torch(0.01),
        )
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        observation = input_dict["obs"]
        image = observation["image"].float()
        if image.dim() == 2:
            image = image.reshape(image.shape[0], 84, 84, 4)
        if image.shape[1] != 4:
            image = image.permute(0, 3, 1, 2)
        visual_features = self._convs(image)

        ammo = observation["ammo_fraction"].float().reshape(image.shape[0], 1)
        ammo_features = self._ammo_encoder(ammo)
        self._features = torch.cat([visual_features, ammo_features], dim=1)
        logits = self.action(self._features)

        action_mask = observation["action_mask"].float().reshape(
            image.shape[0], logits.shape[1]
        )
        invalid = torch.full_like(logits, -1.0e9)
        masked_logits = torch.where(action_mask > 0.0, logits, invalid)
        return masked_logits, state

    def value_function(self):
        return self.value(self._features).squeeze(1)


class BuyerThresholdHeadTorch(TorchModelV2, nn.Module):
    """Small actor-critic head for the E1 buyer's scalar threshold.

    The frozen Atari gameplay network lives in the environment and is never a
    submodule of this model.  Consequently every parameter here belongs to the
    economic head.  The actor mean is constrained to the valid willingness-to-
    pay interval; PPO's exploratory Gaussian samples are clipped again by the
    environment before the trade rule is applied.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        original_space = getattr(obs_space, "original_space", obs_space)
        if not isinstance(original_space, gym.spaces.Box):
            raise TypeError("BuyerThresholdHeadTorch requires a Box observation")
        input_size = int(torch.tensor(original_space.shape).prod().item())

        custom_config = (model_config or {}).get("custom_model_config", {})
        hidden_size = int(custom_config.get("hidden_size", 64))
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.price_max = float(custom_config.get("price_max", 1.0))
        if self.price_max <= 0.0:
            raise ValueError("price_max must be positive")

        def make_hidden():
            return nn.Sequential(
                SlimFC(
                    in_size=input_size,
                    out_size=hidden_size,
                    activation_fn=nn.Tanh,
                    initializer=normc_initializer_torch(1.0),
                ),
                SlimFC(
                    in_size=hidden_size,
                    out_size=hidden_size,
                    activation_fn=nn.Tanh,
                    initializer=normc_initializer_torch(1.0),
                ),
            )

        self.hide_price_from_actor = bool(
            custom_config.get("hide_price_from_actor", False)
        )
        self.hidden = make_hidden()
        self.critic_hidden = make_hidden() if self.hide_price_from_actor else None
        self.threshold_raw_mean = SlimFC(
            in_size=hidden_size,
            out_size=1,
            activation_fn=None,
            initializer=normc_initializer_torch(0.01),
        )
        initial_threshold = float(
            custom_config.get("initial_threshold", 0.5 * self.price_max)
        )
        initial_fraction = min(
            max(initial_threshold / self.price_max, 1.0e-4), 1.0 - 1.0e-4
        )
        initial_raw_mean = np.log(initial_fraction / (1.0 - initial_fraction))
        with torch.no_grad():
            self.threshold_raw_mean._model[-1].bias.fill_(initial_raw_mean)

        initial_log_std = float(custom_config.get("initial_log_std", -0.7))
        self.threshold_log_std = nn.Parameter(
            torch.tensor([initial_log_std], dtype=torch.float32)
        )
        self.value = SlimFC(
            in_size=hidden_size,
            out_size=1,
            activation_fn=None,
            initializer=normc_initializer_torch(0.01),
        )
        self._features = None
        self._value_features = None

    def forward(self, input_dict, state, seq_lens):
        observation = input_dict["obs"].float()
        observation = observation.reshape(observation.shape[0], -1)
        actor_observation = observation
        if self.hide_price_from_actor:
            actor_observation = observation.clone()
            actor_observation[:, 0] = 0.0
        self._features = self.hidden(actor_observation)
        self._value_features = (
            self.critic_hidden(observation)
            if self.critic_hidden is not None
            else self._features
        )
        raw_mean = self.threshold_raw_mean(self._features)
        mean = self.price_max * torch.sigmoid(raw_mean)
        log_std = torch.clamp(
            self.threshold_log_std, -4.0, 1.0
        ).expand_as(mean)
        return torch.cat([mean, log_std], dim=1), state

    def value_function(self):
        return self.value(self._value_features).squeeze(1)


def _conv_flat_size(h, w, in_channels, layers):
    with torch.no_grad():
        probe = torch.zeros(1, in_channels, h, w)
        for layer in layers:
            probe = layer(probe)
        return int(torch.prod(torch.tensor(probe.shape[1:])))
