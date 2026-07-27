"""Native SB3 policy for the Atari Stackelberg economic experiments.

The policy deliberately factors one full action into two independent parts:

``[deterministic_game_action, economic_scalar]``.

The gameplay part is copied from an E0 ``PriceAwareAtariPolicy`` checkpoint and
is then immutable.  The economic part is a Beta policy conditioned only on the
agent's ammunition, the trade-event identity, and the five queried actions of
the opponent.  In particular, critic-only phase state cannot leak into either
actor branch.

The policy also owns the per-episode full-action cache required by the
StackPOMDP construction.  Call ``fix_policy_actions()`` once before training and
``clear_obs_action_map()`` only at the outer episode boundary (the existing
``FixPolicyActionsCallback`` implements exactly that protocol).  Critic-only
fields are excluded from cache keys, so matching query and deployment
observations reuse the same ``[game action, economic action]`` pair.
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

from stackelberg_pomdp.atari.policy import AtariBuyerFeaturesExtractor


ECONOMIC_ROLES = {"buyer", "seller"}
DEFAULT_TRADE_EVENTS = 5


class FrozenGameplayEconomicDistribution:
    """Deterministic categorical gameplay plus a gated Beta economic action.

    Gameplay never contributes policy loss or entropy.  This is stronger than
    merely setting its optimizer learning rate to zero: no gameplay log
    probability is present in the PPO objective at all.
    """

    def __init__(
            self,
            *,
            game_logits,
            economic_alpha,
            economic_beta,
            event_gate,
    ):
        self.game = Categorical(logits=game_logits)
        self.economic = Beta(economic_alpha, economic_beta)
        self.event_gate = event_gate.reshape(-1)

    @property
    def economic_mean(self):
        return self.economic.mean

    def mode(self):
        game_action = th.argmax(self.game.logits, dim=1).float()
        return th.stack([game_action, self.economic.mean], dim=1)

    def sample(self):
        game_action = th.argmax(self.game.logits, dim=1).float()
        sampled = self.economic.sample()
        economic_action = th.where(
            self.event_gate > 0.5,
            sampled,
            self.economic.mean,
        )
        return th.stack([game_action, economic_action], dim=1)

    def get_actions(self, deterministic=False):
        return self.mode() if deterministic else self.sample()

    def log_prob(self, actions):
        actions = actions.reshape(-1, 2)
        economic_actions = actions[:, 1].clamp(1.0e-6, 1.0 - 1.0e-6)
        return self.event_gate * self.economic.log_prob(economic_actions)

    def entropy(self):
        return self.event_gate * self.economic.entropy()


class StackPOMDPAtariEconomicPolicy(ActorCriticPolicy):
    """Composite Atari policy with frozen E0 play and a trainable price head.

    Required actor-visible observation entries are:

    - the seven legacy E0 entries consumed by ``AtariBuyerFeaturesExtractor``;
    - ``event_active`` (used only as a loss/action-sampling gate);
    - ``event_one_hot`` (five-dimensional trade-event identity);
    - ``opponent_context`` (the opponent's five queried economic actions).

    ``critic:state`` is an arbitrary flat critic-only state vector.  It may
    include the response/reward phase and other privileged information; it is
    routed only to the value network.  It need not make the Atari POMDP fully
    observable.  The economic actor observes only
    ``ammo_fraction``, ``event_one_hot``, and ``opponent_context``.  The frozen
    gameplay actor observes exactly the inputs its E0 architecture used.

    Construct the surrounding PPO model first, then call
    ``load_frozen_gameplay_checkpoint`` before the first prediction or rollout.
    The copied tensors and a readiness marker are part of the normal PyTorch
    state dict, so subsequent PPO and direct-policy save/reload are standalone.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *,
            economic_role,
            trade_events=DEFAULT_TRADE_EVENTS,
            visual_features=512,
            ammo_features=32,
            market_features=16,
            economic_hidden=64,
            critic_hidden=256,
            **kwargs,
    ):
        if economic_role not in ECONOMIC_ROLES:
            raise ValueError(
                f"economic_role must be one of {sorted(ECONOMIC_ROLES)}"
            )
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError(
                "StackPOMDPAtariEconomicPolicy requires Dict observations"
            )
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                "StackPOMDPAtariEconomicPolicy requires a Box action space"
            )
        if action_space.shape != (2,):
            raise ValueError("full Atari action must be [game_action, economic]")

        self.economic_role = str(economic_role)
        self.trade_events = int(trade_events)
        self.visual_features = int(visual_features)
        self.ammo_features = int(ammo_features)
        self.market_features = int(market_features)
        self.economic_hidden = int(economic_hidden)
        self.critic_hidden = int(critic_hidden)
        if self.trade_events <= 0:
            raise ValueError("trade_events must be positive")
        if min(
                self.visual_features,
                self.ammo_features,
                self.market_features,
                self.economic_hidden,
                self.critic_hidden,
        ) <= 0:
            raise ValueError("all policy feature dimensions must be positive")

        self._validate_spaces(observation_space, action_space)
        self.game_action_count = int(round(float(action_space.high[0]))) + 1
        self.critic_state_dim = int(
            np.prod(observation_space.spaces["critic:state"].shape)
        )

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
            features_extractor_class=AtariBuyerFeaturesExtractor,
            features_extractor_kwargs={
                "visual_features": self.visual_features,
                "ammo_features": self.ammo_features,
                "market_features": self.market_features,
            },
            **kwargs,
        )

        # These are runtime episode state, intentionally absent from state_dict.
        self.fix_actions = False
        self.obs_action_map = {}

    def _validate_spaces(self, observation_space, action_space):
        required = {
            "image",
            "ammo_fraction",
            "projectile_active",
            "action_mask",
            "offer_active",
            "opportunities_remaining",
            "critic:price",
            "event_active",
            "event_one_hot",
            "opponent_context",
            "critic:state",
        }
        missing = required - set(observation_space.spaces)
        if missing:
            raise ValueError(
                "StackPOMDP Atari observation missing keys: "
                f"{sorted(missing)}"
            )
        expected_vectors = {
            "ammo_fraction": 1,
            "event_active": 1,
            "event_one_hot": self.trade_events,
            "opponent_context": self.trade_events,
        }
        for key, expected_size in expected_vectors.items():
            actual_size = int(np.prod(observation_space.spaces[key].shape))
            if actual_size != expected_size:
                raise ValueError(
                    f"{key} must have {expected_size} entries, got {actual_size}"
                )
        critic_size = int(
            np.prod(observation_space.spaces["critic:state"].shape)
        )
        if critic_size <= 0:
            raise ValueError("critic:state must contain at least one entry")

        action_low = np.asarray(action_space.low, dtype=np.float64)
        action_high = np.asarray(action_space.high, dtype=np.float64)
        if not np.allclose(action_low, np.array([0.0, 0.0])):
            raise ValueError("full Atari action lower bound must be [0, 0]")
        if not np.isclose(action_high[1], 1.0):
            raise ValueError("economic action must be normalized to [0, 1]")
        game_high = float(action_high[0])
        if game_high < 0 or not np.isclose(game_high, round(game_high)):
            raise ValueError("game-action upper bound must be a nonnegative integer")
        action_mask_size = int(
            np.prod(observation_space.spaces["action_mask"].shape)
        )
        if action_mask_size != int(round(game_high)) + 1:
            raise ValueError(
                "action_mask size must match the number of Atari actions"
            )

    def _build(self, lr_schedule):
        game_feature_dim = self.features_extractor.features_dim
        economic_input_dim = 1 + 2 * self.trade_events
        critic_input_dim = (
            game_feature_dim
            + 1
            + 2 * self.trade_events
            + self.critic_state_dim
        )

        self.game_action_net = nn.Linear(
            game_feature_dim, self.game_action_count
        )
        self.economic_head = nn.Sequential(
            nn.Linear(economic_input_dim, self.economic_hidden),
            nn.Tanh(),
            nn.Linear(self.economic_hidden, self.economic_hidden),
            nn.Tanh(),
            nn.Linear(self.economic_hidden, 2),
        )
        self.value_net = nn.Sequential(
            nn.Linear(critic_input_dim, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, 1),
        )

        # Match the E0 module initialization only to keep construction
        # deterministic; these tensors must still be replaced before use.
        self.features_extractor.apply(
            partial(self.init_weights, gain=math.sqrt(2.0))
        )
        self.game_action_net.apply(partial(self.init_weights, gain=0.01))
        self.value_net.apply(partial(self.init_weights, gain=1.0))
        for module in self.economic_head:
            if isinstance(module, nn.Linear):
                self.init_weights(module, gain=1.0)
        final_economic_layer = self.economic_head[-1]
        nn.init.zeros_(final_economic_layer.weight)
        uniform_beta_raw = math.log(math.exp(1.0) - 1.0)
        nn.init.constant_(final_economic_layer.bias, uniform_beta_raw)

        for module in (self.features_extractor, self.game_action_net):
            for parameter in module.parameters():
                parameter.requires_grad = False
            module.eval()

        # Persistent provenance prevents an accidentally random "frozen E0"
        # controller while remaining self-contained after save/reload.
        self.register_buffer(
            "gameplay_ready",
            th.tensor(False, dtype=th.bool),
        )
        self.register_buffer(
            "gameplay_checkpoint_sha256",
            th.zeros(32, dtype=th.uint8),
        )

        trainable_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = self.optimizer_class(
            trainable_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update({
            "economic_role": self.economic_role,
            "trade_events": self.trade_events,
            "visual_features": self.visual_features,
            "ammo_features": self.ammo_features,
            "market_features": self.market_features,
            "economic_hidden": self.economic_hidden,
            "critic_hidden": self.critic_hidden,
        })
        return data

    @staticmethod
    def _checkpoint_file(checkpoint):
        path = Path(checkpoint).expanduser()
        if path.is_file():
            return path
        zip_path = Path(f"{path}.zip")
        if zip_path.is_file():
            return zip_path
        raise FileNotFoundError(f"E0 checkpoint does not exist: {path}")

    def load_frozen_gameplay_checkpoint(self, checkpoint, *, device="cpu"):
        """Copy and verify the protected E0 computation from an SB3 checkpoint."""

        from stable_baselines3 import PPO

        checkpoint_path = self._checkpoint_file(checkpoint)
        source_model = PPO.load(str(checkpoint_path), device=device)
        source_policy = source_model.policy
        required_attributes = ("features_extractor", "game_action_net")
        missing = [
            name for name in required_attributes
            if not hasattr(source_policy, name)
        ]
        if missing:
            raise TypeError(
                "E0 checkpoint is not a compatible Atari gameplay policy; "
                f"missing {missing}"
            )
        source_game_actions = int(source_policy.game_action_net.out_features)
        if source_game_actions != self.game_action_count:
            raise ValueError(
                "E0 checkpoint game action count does not match target: "
                f"{source_game_actions} != {self.game_action_count}"
            )

        source_feature_state = source_policy.features_extractor.state_dict()
        source_game_state = source_policy.game_action_net.state_dict()
        self.features_extractor.load_state_dict(
            source_feature_state, strict=True
        )
        self.game_action_net.load_state_dict(source_game_state, strict=True)

        for module in (self.features_extractor, self.game_action_net):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        copied_feature_state = self.features_extractor.state_dict()
        copied_game_state = self.game_action_net.state_dict()
        mismatched = [
            f"features_extractor.{name}"
            for name, tensor in source_feature_state.items()
            if not th.equal(copied_feature_state[name], tensor)
        ]
        mismatched.extend(
            f"game_action_net.{name}"
            for name, tensor in source_game_state.items()
            if not th.equal(copied_game_state[name], tensor)
        )
        if mismatched:
            raise RuntimeError(
                "E0 gameplay copy verification failed: "
                f"{mismatched[:3]}"
            )

        digest = hashlib.sha256(checkpoint_path.read_bytes()).digest()
        digest_tensor = th.tensor(
            tuple(digest),
            dtype=th.uint8,
            device=self.gameplay_checkpoint_sha256.device,
        )
        self.gameplay_checkpoint_sha256.copy_(digest_tensor)
        self.gameplay_ready.fill_(True)
        del source_model
        return {
            "checkpoint": str(checkpoint_path.resolve()),
            "sha256": digest.hex(),
            "feature_tensors": len(source_feature_state),
            "game_action_tensors": len(source_game_state),
        }

    @property
    def gameplay_fingerprint(self):
        if not bool(self.gameplay_ready.item()):
            return None
        return bytes(self.gameplay_checkpoint_sha256.tolist()).hex()

    def _require_gameplay(self):
        if not bool(self.gameplay_ready.item()):
            raise RuntimeError(
                "frozen gameplay is uninitialized; call "
                "load_frozen_gameplay_checkpoint() before using the policy"
            )

    def set_training_mode(self, mode):
        super().set_training_mode(mode)
        # NatureCNN currently has no dropout or batch normalization, but keep
        # the protected controller explicitly in inference mode nonetheless.
        self.features_extractor.eval()
        self.game_action_net.eval()

    def clear_obs_action_map(self):
        self.obs_action_map = {}

    def fix_policy_actions(self):
        self.fix_actions = True

    def _game_features(self, observations):
        self._require_gameplay()
        processed = preprocess_obs(
            observations,
            self.observation_space,
            normalize_images=self.normalize_images,
        )
        with th.no_grad():
            return self.features_extractor(processed).detach()

    def _economic_inputs(self, observations):
        batch_size = observations["ammo_fraction"].shape[0]
        return th.cat(
            [
                observations["ammo_fraction"].float().reshape(batch_size, 1),
                observations["event_one_hot"].float().reshape(
                    batch_size, self.trade_events
                ),
                observations["opponent_context"].float().reshape(
                    batch_size, self.trade_events
                ),
            ],
            dim=1,
        )

    def _game_logits(self, observations, game_features):
        with th.no_grad():
            logits = self.game_action_net(game_features).detach()
        action_mask = observations["action_mask"].float().reshape(
            logits.shape[0], self.game_action_count
        )
        return th.where(
            action_mask > 0.0,
            logits,
            th.full_like(logits, -1.0e9),
        )

    def _distribution(self, observations, game_features=None):
        if game_features is None:
            game_features = self._game_features(observations)
        game_logits = self._game_logits(observations, game_features)
        parameters = self.economic_head(self._economic_inputs(observations))
        alpha = nn.functional.softplus(parameters[:, 0]) + 1.0e-4
        beta = nn.functional.softplus(parameters[:, 1]) + 1.0e-4
        return FrozenGameplayEconomicDistribution(
            game_logits=game_logits,
            economic_alpha=alpha,
            economic_beta=beta,
            event_gate=observations["event_active"].float(),
        )

    def _values(self, observations, game_features=None):
        if game_features is None:
            game_features = self._game_features(observations)
        batch_size = game_features.shape[0]
        critic_inputs = th.cat(
            [
                game_features,
                observations["event_active"].float().reshape(batch_size, 1),
                observations["event_one_hot"].float().reshape(
                    batch_size, self.trade_events
                ),
                observations["opponent_context"].float().reshape(
                    batch_size, self.trade_events
                ),
                observations["critic:state"].float().reshape(
                    batch_size, self.critic_state_dim
                ),
            ],
            dim=1,
        )
        return self.value_net(critic_inputs)

    def _actor_cache_key(self, observations, row):
        """Hash one complete actor observation, excluding privileged fields.

        The leading row index separates simultaneously collected vector-env
        episodes.  Continuous values are rounded before hashing so
        semantically identical query/deployment observations survive harmless
        float32 representation differences.
        """

        digest = hashlib.sha256()
        for name in self.observation_space.spaces:
            if name.startswith("critic:"):
                continue
            values = observations[name][row].detach().cpu()
            if values.is_floating_point():
                values = th.round(values * 1.0e6) / 1.0e6
            array = values.contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return int(row), digest.digest()

    def _actions(self, observations, distribution, *, deterministic):
        actions = distribution.get_actions(deterministic=deterministic)
        if not self.fix_actions:
            return actions

        active = distribution.event_gate > 0.5
        for row in range(actions.shape[0]):
            if not bool(active[row].item()):
                continue
            key = self._actor_cache_key(observations, row)
            if key in self.obs_action_map:
                actions[row] = self.obs_action_map[key].to(actions.device)
            else:
                self.obs_action_map[key] = actions[row].detach().cpu().clone()
        return actions

    def forward(self, obs, deterministic=False):
        game_features = self._game_features(obs)
        distribution = self._distribution(obs, game_features)
        actions = self._actions(
            obs, distribution, deterministic=deterministic
        )
        values = self._values(obs, game_features)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        game_features = self._game_features(obs)
        distribution = self._distribution(obs, game_features)
        return (
            self._values(obs, game_features),
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
