"""Atari models owned by the StackelbergPOMDP replication codebase."""

import gym
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

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
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


def _conv_flat_size(h, w, in_channels, layers):
    with torch.no_grad():
        probe = torch.zeros(1, in_channels, h, w)
        for layer in layers:
            probe = layer(probe)
        return int(torch.prod(torch.tensor(probe.shape[1:])))
