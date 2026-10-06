"""Default MLP: public observations feed the actor, all state feeds the critic.

Commitments belong to the environment. This policy only samples actions.
"""
import torch
from torch import nn
from gym import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


def observation_split(observation_space):
    if not isinstance(observation_space, spaces.Dict):
        raise TypeError("The default policy expects a Dict observation space")
    actor_keys, critic_dim, seen_critic = [], 0, False
    for key, space in observation_space.spaces.items():
        if key.startswith("critic:"):
            seen_critic = True
            critic_dim += get_flattened_obs_dim(space)
        else:
            if seen_critic:
                raise ValueError("Actor observations must precede critic-only observations")
            actor_keys.append(key)
    return critic_dim, tuple(actor_keys)


class SplitMLP(nn.Module):
    def __init__(self, feature_dim, critic_dim, net_arch, activation_fn):
        super().__init__()
        self._split_idx = feature_dim - critic_dim
        self.shared_net = nn.Identity()
        def network(input_dim, layers):
            modules = []
            for width in layers:
                modules.extend((nn.Linear(input_dim, width), activation_fn()))
                input_dim = width
            return nn.Sequential(*modules), input_dim
        self.policy_net, self.latent_dim_pi = network(self._split_idx, net_arch["pi"])
        self.value_net, self.latent_dim_vf = network(feature_dim, net_arch["vf"])

    def forward_actor(self, features):
        return self.policy_net(features[:, :self._split_idx])

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class DefaultPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # Accept the saved feature-split metadata in retained SB3 checkpoints.
        kwargs.pop("cutoff_entry", None)
        kwargs.pop("actor_obs_keys", None)
        kwargs.setdefault("net_arch", dict(pi=[64, 64], vf=[64, 64]))
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.cutoff_entry, self.actor_obs_keys = observation_split(self.observation_space)
        self.mlp_extractor = SplitMLP(self.features_dim, self.cutoff_entry,
                                     self.net_arch, self.activation_fn).to(self.device)

CustomPolicy = DefaultPolicy  # retained checkpoint class name
