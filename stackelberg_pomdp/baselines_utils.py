import gym
import numpy as np
import torch as th
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from itertools import zip_longest

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

class CustomPolicy(MultiInputActorCriticPolicy):
    """
    Modified MultiInputActorCriticPolicy to focus on deterministic policies.

    This policy uses an observation-action map to store and reuse actions for specific observations,
    effectively creating a deterministic behavior. It also uses epsilon-greedy exploration during
    training to help discover the optimal action mapping.
    """
    def __init__(self, *args, **kwargs):

        self.cutoff_entry = kwargs.pop("cutoff_entry", 0)
        self.actor_obs_keys = kwargs.pop("actor_obs_keys", ("base_environment",))

        super(CustomPolicy, self).__init__(
            *args,
            **kwargs,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )

        self.fix_actions = False
        self.obs_action_map = {}

    def _cache_key(self, obs):
        values = []
        for key in self.actor_obs_keys:
            values.extend(obs[key].reshape(-1).tolist())
        return tuple(values)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Validate that actor-visible entries are before critic-only entries.
        # The cutoff is computed from the final wrapped observation space.
        seen_critic = False
        for key in self.observation_space.spaces.keys():
            is_critic = key.split(":")[0] == "critic"
            if seen_critic and not is_critic:
                raise IOError("Critic-only entries need to be at the end of the observation!")
            if not seen_critic and is_critic:
                seen_critic = True

        self.mlp_extractor = CustomMLPExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            cutoff_entry=self.cutoff_entry,
        )

    def clear_obs_action_map(self):
        self.obs_action_map = {}


    def fix_policy_actions(self):
        self.fix_actions = True

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        obs_key = self._cache_key(obs)
        if self.fix_actions and obs_key in self.obs_action_map:
            actions = self.obs_action_map[obs_key]
        else:
            actions = distribution.get_actions(deterministic=deterministic)
            if self.fix_actions: self.obs_action_map[obs_key] = actions
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:


        processed_observation, vectorized_env = self.obs_to_tensor(observation)
        obs_key = self._cache_key(processed_observation)

        if self.fix_actions and obs_key in self.obs_action_map:
            actions = self.obs_action_map[obs_key]
            # Convert to numpy
            actions = actions.cpu().numpy()

            if isinstance(self.action_space, gym.spaces.Box):
                if self.squash_output:
                    # Rescale to proper domain when using squashing
                    actions = self.unscale_action(actions)
                else:
                    # Actions could be on arbitrary scale, so clip the actions to avoid
                    # out of bound error (e.g. if sampling from a Gaussian distribution)
                    actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Remove batch dimension if needed
            if not vectorized_env:
                actions = actions[0]
        else:
            actions, state = super(CustomPolicy, self).predict(observation, deterministic = deterministic)
        return actions, state


class CustomMLPExtractor(nn.Module):
    """
    Custom MLP feature extractor for policies with multiple inputs.

    It supports:
    - Different network architectures for actor and critic.
    - Splitting the features based on a 'cutoff_entry'.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        cutoff_entry: int = 0,
    ):
        super(CustomMLPExtractor, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # The last cutoff_entry features will only be used by the value network.
        # Keep this separate from SB3's net_arch because recent SB3 versions
        # normalize net_arch and drop custom metadata keys.
        self.cutoff_entry = cutoff_entry

        # Iterate through the shared layers and build the shared parts of the network
        if isinstance(net_arch, dict):
            policy_only_layers = net_arch.get("pi", [])
            value_only_layers = net_arch.get("vf", [])
        else:
            for layer in net_arch:
                if isinstance(layer, int):  # Check that this is a shared layer
                    # TODO: give layer a meaningful name
                    shared_net.append(nn.Linear(last_layer_dim_shared, layer))  # add linear of size layer
                    shared_net.append(activation_fn())
                    last_layer_dim_shared = layer
                else:
                    assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                    if "pi" in layer:
                        assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                        policy_only_layers = layer["pi"]

                    if "vf" in layer:
                        assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                        value_only_layers = layer["vf"]
                    break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared - self.cutoff_entry
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip_longest(policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self._split_idx = feature_dim - self.cutoff_entry

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        # Isolate features that are only fed into the value network (slicing = zero-copy view)
        shared_features = features[:, :self._split_idx]
        value_only_features = features[:, self._split_idx:]

        shared_latent = self.shared_net(shared_features)

        # Add those features to value network only
        shared_latent_value = th.cat([shared_latent, value_only_features], dim=1)

        return self.policy_net(shared_latent), self.value_net(shared_latent_value)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(self.shared_net(features[:, :self._split_idx]))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        shared_latent = self.shared_net(features[:, :self._split_idx])
        return self.value_net(th.cat([shared_latent, features[:, self._split_idx:]], dim=1))


class CustomOnPolicyAlgorithm(OnPolicyAlgorithm):
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collects rollouts using the current policy and fills the replay buffer.

        This method is modified to allow filtering and exclusion of specific transitions
        from the rollout buffer before they are added.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Count every executed environment step against max_steps, including
            # hidden-query response steps that are omitted from the PPO buffer.
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            exclude_from_buffer = infos[0].get("exclude_from_buffer", False)

            # Store only the unexcluded transitions
            if not exclude_from_buffer:
                n_steps += 1
                rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True


class CustomA2C(A2C, CustomOnPolicyAlgorithm):
    pass


class CustomPPO(PPO, CustomOnPolicyAlgorithm):
    pass
