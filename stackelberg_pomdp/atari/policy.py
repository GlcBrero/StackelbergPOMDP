"""Native SB3 hybrid Atari-gameplay and willingness-to-pay policy."""

from functools import partial
import math

import gym
import numpy as np
import torch as th
from torch import nn
from torch.distributions import Beta, Categorical

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN


POLICY_STAGES = {"gameplay", "free_trade", "priced", "joint"}


class AtariBuyerFeaturesExtractor(BaseFeaturesExtractor):
    """Nature CNN plus actor-visible ammo and market state.

    ``critic:price`` and ``action_mask`` are deliberately excluded.  Price is
    appended only by the value branch, while the mask acts directly on logits.
    """

    def __init__(
            self,
            observation_space,
            *,
            visual_features=512,
            ammo_features=32,
            market_features=16,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("AtariBuyerFeaturesExtractor requires Dict observations")
        required = {
            "image",
            "ammo_fraction",
            "projectile_active",
            "action_mask",
            "offer_active",
            "opportunities_remaining",
            "critic:price",
        }
        missing = required - set(observation_space.spaces)
        if missing:
            raise ValueError(f"Atari observation missing keys: {sorted(missing)}")

        features_dim = int(visual_features + ammo_features + market_features)
        super().__init__(observation_space, features_dim=features_dim)
        image_space = observation_space.spaces["image"]
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
        self.visual = NatureCNN(
            cnn_space,
            features_dim=int(visual_features),
        )
        self.ammo = nn.Sequential(
            nn.Linear(1, int(ammo_features)),
            nn.ReLU(),
        )
        self.market = nn.Sequential(
            nn.Linear(3, int(market_features)),
            nn.ReLU(),
        )

    def forward(self, observations):
        image = observations["image"].float()
        if self.image_is_channel_last:
            image = image.permute(0, 3, 1, 2)
        visual = self.visual(image)
        ammo = self.ammo(
            observations["ammo_fraction"].float().reshape(image.shape[0], 1)
        )
        market_state = th.cat(
            [
                observations["projectile_active"].float().reshape(
                    image.shape[0], 1
                ),
                observations["offer_active"].float().reshape(
                    image.shape[0], 1
                ),
                observations["opportunities_remaining"].float().reshape(
                    image.shape[0], 1
                ),
            ],
            dim=1,
        )
        market = self.market(market_state)
        return th.cat([visual, ammo, market], dim=1)


class HybridAtariBuyerDistribution:
    """Categorical gameplay plus a Beta threshold with stage/opportunity gates."""

    def __init__(
            self,
            *,
            game_logits,
            threshold_alpha,
            threshold_beta,
            offer_gate,
            train_gameplay,
            train_threshold,
    ):
        self.game = Categorical(logits=game_logits)
        self.threshold = Beta(threshold_alpha, threshold_beta)
        self.offer_gate = offer_gate.reshape(-1)
        self.train_gameplay = bool(train_gameplay)
        self.train_threshold = bool(train_threshold)

    def mode(self):
        game_action = th.argmax(self.game.logits, dim=1).float()
        threshold = self.threshold.mean
        return th.stack([game_action, threshold], dim=1)

    def sample(self):
        game_action = (
            self.game.sample()
            if self.train_gameplay
            else th.argmax(self.game.logits, dim=1)
        ).float()
        threshold_mean = self.threshold.mean
        if self.train_threshold:
            sampled_threshold = self.threshold.sample()
            threshold = th.where(
                self.offer_gate > 0.5,
                sampled_threshold,
                threshold_mean,
            )
        else:
            threshold = threshold_mean
        return th.stack([game_action, threshold], dim=1)

    def get_actions(self, deterministic=False):
        return self.mode() if deterministic else self.sample()

    def log_prob(self, actions):
        actions = actions.reshape(-1, 2)
        result = th.zeros(
            actions.shape[0], dtype=actions.dtype, device=actions.device
        )
        if self.train_gameplay:
            game_actions = actions[:, 0].round().long()
            result = result + self.game.log_prob(game_actions)
        if self.train_threshold:
            thresholds = actions[:, 1].clamp(1.0e-6, 1.0 - 1.0e-6)
            result = result + self.offer_gate * self.threshold.log_prob(thresholds)
        return result

    def entropy(self):
        result = th.zeros_like(self.offer_gate)
        if self.train_gameplay:
            result = result + self.game.entropy()
        if self.train_threshold:
            result = result + self.offer_gate * self.threshold.entropy()
        return result


class PriceAwareAtariPolicy(ActorCriticPolicy):
    """Single SB3 policy used unchanged from E0 through joint fine-tuning."""

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *,
            stage="gameplay",
            visual_features=512,
            ammo_features=32,
            market_features=16,
            threshold_hidden=64,
            actor_economic_context=False,
            **kwargs,
    ):
        if stage not in POLICY_STAGES:
            raise ValueError(f"unknown policy stage: {stage}")
        if not isinstance(action_space, gym.spaces.Box) or action_space.shape != (2,):
            raise TypeError("PriceAwareAtariPolicy requires the two-component Box action")
        self.stage = stage
        self.visual_features = int(visual_features)
        self.ammo_features = int(ammo_features)
        self.market_features = int(market_features)
        self.threshold_hidden = int(threshold_hidden)
        self.actor_economic_context = bool(actor_economic_context)
        if self.actor_economic_context:
            required_context = {
                "price", "normalized_timestep", "time_remaining"
            }
            missing_context = required_context - set(observation_space.spaces)
            if missing_context:
                raise ValueError(
                    "actor economic context missing observation keys: "
                    f"{sorted(missing_context)}"
                )
        self.game_action_count = int(round(float(action_space.high[0]))) + 1
        self.train_gameplay = stage in {"gameplay", "free_trade", "joint"}
        self.train_threshold = stage in {"priced", "joint"}

        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
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
        self.set_stage(stage, rebuild_optimizer=False)

    def _build(self, lr_schedule):
        feature_dim = self.features_extractor.features_dim
        threshold_input_dim = feature_dim + (
            2 if self.actor_economic_context else 0
        )
        self.game_action_net = nn.Linear(feature_dim, self.game_action_count)
        self.threshold_net = nn.Sequential(
            nn.Linear(threshold_input_dim, self.threshold_hidden),
            nn.Tanh(),
            nn.Linear(self.threshold_hidden, self.threshold_hidden),
            nn.Tanh(),
            nn.Linear(self.threshold_hidden, 2),
        )
        self.value_net = nn.Sequential(
            nn.Linear(
                feature_dim + (2 if self.actor_economic_context else 1),
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.features_extractor.apply(
            partial(self.init_weights, gain=math.sqrt(2.0))
        )
        self.game_action_net.apply(partial(self.init_weights, gain=0.01))
        self.value_net.apply(partial(self.init_weights, gain=1.0))
        for module in self.threshold_net:
            if isinstance(module, nn.Linear):
                self.init_weights(module, gain=1.0)
        final_threshold_layer = self.threshold_net[-1]
        nn.init.zeros_(final_threshold_layer.weight)
        initial_beta_raw = math.log(math.exp(1.0) - 1.0)
        nn.init.constant_(final_threshold_layer.bias, initial_beta_raw)

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update({
            "stage": self.stage,
            "visual_features": self.visual_features,
            "ammo_features": self.ammo_features,
            "market_features": self.market_features,
            "threshold_hidden": self.threshold_hidden,
            "actor_economic_context": self.actor_economic_context,
        })
        return data

    def set_stage(self, stage, *, rebuild_optimizer=False):
        if stage not in POLICY_STAGES:
            raise ValueError(f"unknown policy stage: {stage}")
        self.stage = stage
        self.train_gameplay = stage in {"gameplay", "free_trade", "joint"}
        self.train_threshold = stage in {"priced", "joint"}

        gameplay_modules = (self.features_extractor, self.game_action_net)
        for module in gameplay_modules:
            for parameter in module.parameters():
                parameter.requires_grad = self.train_gameplay
        for parameter in self.threshold_net.parameters():
            parameter.requires_grad = self.train_threshold
        for parameter in self.value_net.parameters():
            parameter.requires_grad = True

        if rebuild_optimizer:
            learning_rate = self.optimizer.param_groups[0]["lr"]
            self.optimizer = self.optimizer_class(
                # Keep one stable parameter group across curriculum stages so
                # SB3 checkpoints can reconstruct and reload optimizer state.
                # Frozen tensors have no gradients and therefore remain
                # unchanged even though they are present in the optimizer.
                self.parameters(),
                lr=learning_rate,
                **self.optimizer_kwargs,
            )

    def _features(self, observations):
        processed = preprocess_obs(
            observations,
            self.observation_space,
            normalize_images=self.normalize_images,
        )
        return self.features_extractor(processed)

    def _distribution(self, observations, features=None):
        if features is None:
            features = self._features(observations)
        logits = self.game_action_net(features)
        action_mask = observations["action_mask"].float().reshape(
            logits.shape[0], self.game_action_count
        )
        logits = th.where(
            action_mask > 0.0,
            logits,
            th.full_like(logits, -1.0e9),
        )
        threshold_features = features
        if self.actor_economic_context:
            actor_context = th.cat(
                [
                    observations["price"].float().reshape(
                        features.shape[0], 1
                    ),
                    observations["normalized_timestep"].float().reshape(
                        features.shape[0], 1
                    ),
                ],
                dim=1,
            )
            threshold_features = th.cat(
                [features, actor_context], dim=1
            )
        threshold_parameters = self.threshold_net(threshold_features)
        alpha = nn.functional.softplus(threshold_parameters[:, 0]) + 1.0e-4
        beta = nn.functional.softplus(threshold_parameters[:, 1]) + 1.0e-4
        return HybridAtariBuyerDistribution(
            game_logits=logits,
            threshold_alpha=alpha,
            threshold_beta=beta,
            offer_gate=observations["offer_active"].float(),
            train_gameplay=self.train_gameplay,
            train_threshold=self.train_threshold,
        )

    def _values(self, observations, features=None):
        if features is None:
            features = self._features(observations)
        price = observations["critic:price"].float().reshape(
            features.shape[0], 1
        )
        value_context = [features, price]
        if self.actor_economic_context:
            value_context.append(
                observations["normalized_timestep"].float().reshape(
                    features.shape[0], 1
                )
            )
        return self.value_net(th.cat(value_context, dim=1))

    def forward(self, obs, deterministic=False):
        features = self._features(obs)
        distribution = self._distribution(obs, features)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self._values(obs, features)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self._features(obs)
        distribution = self._distribution(obs, features)
        return (
            self._values(obs, features),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def _predict(self, observation, deterministic=False):
        return self._distribution(observation).get_actions(
            deterministic=deterministic
        )

    def get_distribution(self, obs):
        return self._distribution(obs)

    def predict_values(self, obs):
        return self._values(obs)
