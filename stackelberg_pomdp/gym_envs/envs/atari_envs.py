from collections import OrderedDict

import gym
from gym.spaces import Box, Dict
import numpy as np

from stackelberg_pomdp.gym_envs.envs.base_envs import BaseEnv


def make_atari_buyer_env(
        *,
        seed,
        initial_bullets,
        noop_max,
        frame_skip,
        frame_stack,
        episodic_life,
        clip_game_rewards,
):
    try:
        from stackerlberg.envs.atari_space_invaders import SpaceInvadersGame
        from stackerlberg.wrappers.atari_preprocessing import (
            ClipRewardAgent,
            EpisodicLifeAgent,
            FrameStack,
            MaxAndSkipAgent,
            NoopResetAgent,
            WarpFrame,
        )
        from stackerlberg.wrappers.space_invaders_bullets import (
            ResourcePool,
            SpaceInvadersBulletsResource,
        )
    except ImportError as exc:
        raise ImportError(
            "Atari bullet-pricing environments require the StackeRLberg Atari "
            "environment package on PYTHONPATH."
        ) from exc

    env = SpaceInvadersGame(num_players=1, seed=seed)
    env = NoopResetAgent(env, noop_max=noop_max, seed=seed)
    if episodic_life:
        env = EpisodicLifeAgent(env)
    if clip_game_rewards:
        env = ClipRewardAgent(env)
    agent_id = env.list_of_agents[0]
    pool = ResourcePool(initial_resources=initial_bullets)
    env = SpaceInvadersBulletsResource(env, shots_pool={agent_id: pool})
    env = MaxAndSkipAgent(env, skip=frame_skip)
    env = WarpFrame(env)
    env = FrameStack(env, k=frame_stack)
    return env, agent_id, pool


class SinglePlayerAtariBulletEnv(gym.Env):
    """Single-player view of the scarce-bullet Space Invaders environment.

    This adapter deliberately reuses :func:`make_atari_buyer_env`, which is the
    builder used by the pricing environments below.  It therefore has identical
    bullet accounting and Atari preprocessing, but exposes only the gameplay
    image and game action.  There is no seller, price, trade, or threshold.

    The old Gym API is used because the replication environment is pinned to
    Gym 0.21 and RLlib 2.0.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config=None, **kwargs):
        super().__init__()
        worker_index = int(getattr(config, "worker_index", 0))
        vector_index = int(getattr(config, "vector_index", 0))
        values = dict(config or {})
        values.update(kwargs)

        base_seed = int(values.pop("seed", 1))
        # RLlib passes EnvContext metadata outside the dict payload. Give each
        # sampler an independent but reproducible ALE/noop-reset stream.
        self.seed_value = base_seed + 10_000 * worker_index + vector_index
        self.initial_bullets = int(values.pop("initial_bullets", 5))
        self.max_steps = values.pop("max_steps", None)
        noop_max = int(values.pop("noop_max", 30))
        frame_skip = int(values.pop("frame_skip", 4))
        frame_stack = int(values.pop("frame_stack", 4))
        episodic_life = bool(values.pop("episodic_life", True))
        clip_game_rewards = bool(values.pop("clip_game_rewards", True))
        if values:
            raise TypeError(f"unexpected environment options: {sorted(values)}")
        if self.initial_bullets < 0:
            raise ValueError("initial_bullets must be nonnegative")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
            if self.max_steps <= 0:
                raise ValueError("max_steps must be positive")

        self.env, self.agent_id, self.bullet_pool = make_atari_buyer_env(
            seed=self.seed_value,
            initial_bullets=self.initial_bullets,
            noop_max=noop_max,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            episodic_life=episodic_life,
            clip_game_rewards=clip_game_rewards,
        )

        inner_action_space = self.env.action_space[self.agent_id]
        if isinstance(inner_action_space, Dict):
            self._action_key = (
                "action"
                if "action" in inner_action_space.spaces
                else next(iter(inner_action_space.spaces))
            )
            self.action_space = inner_action_space[self._action_key]
        else:
            self._action_key = None
            self.action_space = inner_action_space

        inner_observation_space = self.env.observation_space[self.agent_id]
        if isinstance(inner_observation_space, Dict):
            self._image_key = (
                "image"
                if "image" in inner_observation_space.spaces
                else next(iter(inner_observation_space.spaces))
            )
            self.observation_space = inner_observation_space[self._image_key]
        else:
            self._image_key = None
            self.observation_space = inner_observation_space

        self._observation = None
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.shots_fired = 0

    def _image(self, observation):
        agent_observation = observation[self.agent_id]
        if self._image_key is None:
            return agent_observation
        return agent_observation[self._image_key]

    def _inner_action(self, action):
        action = int(np.asarray(action).reshape(-1)[0])
        if self._action_key is None:
            return {self.agent_id: action}
        return {self.agent_id: {self._action_key: action}}

    def reset(self):
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.shots_fired = 0
        self._observation = self.env.reset()
        return self._image(self._observation)

    def step(self, action):
        observation, rewards, dones, infos = self.env.step(self._inner_action(action))
        self._observation = observation
        reward = float(rewards.get(self.agent_id, 0.0))
        inner_info = dict(infos.get(self.agent_id, {}))
        shots_this_step = int(
            inner_info.get(
                "shots_fired_this_step",
                int(bool(inner_info.get("shot_did_fire", False))),
            )
        )

        self.episode_reward += reward
        self.episode_steps += 1
        self.shots_fired += shots_this_step
        done = bool(dones.get("__all__", False))
        if self.max_steps is not None and self.episode_steps >= self.max_steps:
            done = True

        final_ammo = float(self.bullet_pool.val)
        reward_per_bullet = (
            self.episode_reward / self.shots_fired if self.shots_fired else 0.0
        )
        info = {
            **inner_info,
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_steps,
            "shots_fired_this_step": shots_this_step,
            "shots_fired": self.shots_fired,
            "final_ammo": final_ammo,
            "reward_per_bullet": reward_per_bullet,
        }
        return self._image(observation), reward, done, info

    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


class AmmoAwareSinglePlayerAtariBulletEnv(SinglePlayerAtariBulletEnv):
    """Five-bullet gameplay view with ammo state and safe FIRE masking.

    The image preprocessing and bullet accounting are identical to
    :class:`SinglePlayerAtariBulletEnv`.  The policy additionally observes the
    remaining bullet fraction and an action mask.  FIRE-containing actions are
    unavailable while an ALE player projectile is active (and once ammo is
    exhausted).  When a projectile is active at decision time, the selected
    non-FIRE action remains non-FIRE throughout the outer four-frame repeat,
    even if that projectile disappears before the repeat ends.
    """

    AMMO_KEY = "ammo_fraction"
    ACTION_MASK_KEY = "action_mask"
    IMAGE_KEY = "image"
    PROJECTILE_RAM_SLOTS = (0x55, 0x56)
    INACTIVE_PROJECTILE_RAM_VALUE = 0xF6

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._image_observation_space = self.observation_space
        meanings = self.env.unwrapped.get_action_meanings()
        self.fire_action_indices = tuple(
            idx for idx, meaning in enumerate(meanings) if "FIRE" in meaning
        )
        self.observation_space = Dict(OrderedDict([
            (self.IMAGE_KEY, self._image_observation_space),
            (
                self.AMMO_KEY,
                Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            ),
            (
                self.ACTION_MASK_KEY,
                Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.action_space.n,),
                    dtype=np.float32,
                ),
            ),
        ]))

    def _projectile_active(self):
        ram = self.env.unwrapped.ale.getRAM()
        return any(
            ram[slot] != self.INACTIVE_PROJECTILE_RAM_VALUE
            for slot in self.PROJECTILE_RAM_SLOTS
        )

    def _action_mask(self):
        mask = np.ones(self.action_space.n, dtype=np.float32)
        if self._projectile_active() or not self.bullet_pool.available_bool(1):
            mask[list(self.fire_action_indices)] = 0.0
        return mask

    def _ammo_observation(self, image):
        denominator = max(float(self.initial_bullets), 1.0)
        ammo_fraction = np.array(
            [np.clip(float(self.bullet_pool.val) / denominator, 0.0, 1.0)],
            dtype=np.float32,
        )
        return OrderedDict([
            (self.IMAGE_KEY, image),
            (self.AMMO_KEY, ammo_fraction),
            (self.ACTION_MASK_KEY, self._action_mask()),
        ])

    def reset(self):
        return self._ammo_observation(super().reset())

    def step(self, action):
        image, reward, done, info = super().step(action)
        info = dict(info)
        info["projectile_active"] = bool(self._projectile_active())
        info["fire_actions_available"] = bool(
            self._action_mask()[list(self.fire_action_indices)].any()
        )
        return self._ammo_observation(image), reward, done, info


class BaseAtariBulletPricingEnv(BaseEnv):
    """One-ALE Atari trade game with seller pricing only.

    The seller has no Atari state/action branch. It posts a scalar bullet price
    and earns payment revenue. The buyer is the only Atari player: it plays its
    own Space Invaders ALE instance, starts with zero bullets by default, and
    chooses both an Atari game action and an acceptance threshold. At each of at
    most ``offer_chances`` trade opportunities, one accepted offer transfers one
    bullet to the buyer.

    This class is intentionally a base environment. It exposes the economic and
    Atari spaces needed by neural/frozen buyer policies; the existing tabular
    Q-learning follower wrappers are not suitable for its buyer action space.
    """

    SELLER = "agent_0"
    BUYER = "agent_1"

    def __init__(
            self,
            *,
            price_max=1.0,
            offer_chances=5,
            buyer_initial_bullets=0,
            max_replenish=5,
            buyer_game_reward_scale=1.0,
            payment_penalty_lambda=1.0,
            max_steps=None,
            noop_max=30,
            frame_skip=4,
            frame_stack=4,
            episodic_life=True,
            clip_game_rewards=True,
            seed=None,
            logger=None,
    ):
        super().__init__(
            leader=self.SELLER,
            followers_list=[self.BUYER],
            logger=logger,
            seed=seed,
        )

        if price_max <= 0:
            raise ValueError("price_max must be positive")
        if offer_chances < 0:
            raise ValueError("offer_chances must be nonnegative")
        if max_replenish < 0:
            raise ValueError("max_replenish must be nonnegative")

        self.price_max = float(price_max)
        self.offer_chances = int(offer_chances)
        self.max_replenish = int(max_replenish)
        self.buyer_game_reward_scale = float(buyer_game_reward_scale)
        self.payment_penalty_lambda = float(payment_penalty_lambda)
        self.max_steps = max_steps

        self.buyer_env, self._buyer_inner, self.buyer_pool = make_atari_buyer_env(
            seed=1 if seed is None else int(seed),
            initial_bullets=int(buyer_initial_bullets),
            noop_max=int(noop_max),
            frame_skip=int(frame_skip),
            frame_stack=int(frame_stack),
            episodic_life=bool(episodic_life),
            clip_game_rewards=bool(clip_game_rewards),
        )

        buyer_inner_action_space = self.buyer_env.action_space[self._buyer_inner]
        if isinstance(buyer_inner_action_space, Dict):
            if "action" in buyer_inner_action_space.spaces:
                self._buyer_game_action_key = "action"
            else:
                self._buyer_game_action_key = next(iter(buyer_inner_action_space.spaces))
            buyer_game_space = buyer_inner_action_space[self._buyer_game_action_key]
        else:
            self._buyer_game_action_key = None
            buyer_game_space = buyer_inner_action_space
        self.buyer_game_action_space = buyer_game_space
        buyer_inner_observation_space = self.buyer_env.observation_space[self._buyer_inner]
        if isinstance(buyer_inner_observation_space, Dict):
            if "image" in buyer_inner_observation_space.spaces:
                self._buyer_image_key = "image"
            else:
                self._buyer_image_key = next(iter(buyer_inner_observation_space.spaces))
            buyer_image_space = buyer_inner_observation_space[self._buyer_image_key]
        else:
            self._buyer_image_key = None
            buyer_image_space = buyer_inner_observation_space
        self.buyer_image_space = buyer_image_space

        self.observation_space = Dict({
            "base_environment": Box(0.0, 1.0, shape=(4,), dtype=np.float32),
        })
        self.action_space = Box(0.0, self.price_max, shape=(1,), dtype=np.float32)
        self.followers_observation_space = {
            self.BUYER: Dict({
                "image": buyer_image_space,
                "econ": Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            })
        }
        self.followers_action_space = {
            self.BUYER: Dict({
                "game_action": buyer_game_space,
                "threshold": Box(0.0, self.price_max, shape=(1,), dtype=np.float32),
            })
        }

        self._buyer_done = False
        self._buyer_obs = None
        self.step_count = 0
        self.current_price = 0.0
        self._trade_now = False
        self._offer_events_used = 0
        self._accepted_trades = 0
        self.cumulative_revenue = 0.0

    def _buyer_econ_obs(self):
        remaining = max(self.offer_chances - self._offer_events_used, 0)
        denom = max(float(self.offer_chances), 1.0)
        capacity = max(float(self.max_replenish), 1.0)
        return np.array(
            [
                float(bool(self._trade_now)),
                float(self.current_price),
                float(remaining) / denom,
                float(self.buyer_pool.val) / capacity,
            ],
            dtype=np.float32,
        )

    def _seller_obs(self):
        remaining = max(self.offer_chances - self._offer_events_used, 0)
        denom = max(float(self.offer_chances), 1.0)
        capacity = max(float(self.max_replenish), 1.0)
        return np.array(
            [
                float(bool(self._trade_now)),
                float(self.current_price),
                float(remaining) / denom,
                float(self.buyer_pool.val) / capacity,
            ],
            dtype=np.float32,
        )

    def _obs(self):
        return OrderedDict({
            self.SELLER: self._seller_obs(),
            self.BUYER: {
                "image": (
                    self._buyer_obs[self._buyer_inner][self._buyer_image_key]
                    if self._buyer_image_key is not None
                    else self._buyer_obs[self._buyer_inner]
                ),
                "econ": self._buyer_econ_obs(),
            },
        })

    @staticmethod
    def _extract_buyer_action(action):
        if isinstance(action, dict):
            game_action = action.get("game_action", action.get("action", 0))
            threshold = action.get("threshold", 0.0)
        else:
            values = np.asarray(action).reshape(-1)
            if len(values) < 2:
                raise ValueError("buyer action must contain game action and threshold")
            game_action = values[0]
            threshold = values[1]
        return int(np.asarray(game_action).reshape(-1)[0]), float(np.asarray(threshold).reshape(-1)[0])

    def _inner_action(self, game_action):
        if self._buyer_game_action_key is None:
            return {self._buyer_inner: int(game_action)}
        return {self._buyer_inner: {self._buyer_game_action_key: int(game_action)}}

    def reset(self):
        self._buyer_done = False
        self.step_count = 0
        self.current_price = 0.0
        self._trade_now = self.offer_chances > 0 and self.max_replenish > 0
        self._offer_events_used = 0
        self._accepted_trades = 0
        self.cumulative_revenue = 0.0
        self._buyer_obs = self.buyer_env.reset()
        return self._obs()

    def step(self, actions_dict):
        seller_price = float(
            np.clip(np.asarray(actions_dict[self.SELLER]).reshape(-1)[0], 0.0, self.price_max)
        )
        buyer_game_action, buyer_threshold = self._extract_buyer_action(actions_dict[self.BUYER])
        buyer_threshold = float(np.clip(buyer_threshold, 0.0, self.price_max))

        trade_event = bool(
            self._offer_events_used < self.offer_chances
            and self._accepted_trades < self.max_replenish
        )
        offered_price = seller_price if trade_event else 0.0
        if trade_event:
            self._offer_events_used += 1

        traded = False
        revenue = 0.0
        if trade_event and offered_price <= buyer_threshold:
            self.buyer_pool.inc(1)
            self._accepted_trades += 1
            traded = True
            revenue = offered_price
        self.cumulative_revenue += revenue

        if self._buyer_done:
            buyer_obs = None
            buyer_game_reward = 0.0
            buyer_done = {self._buyer_inner: True, "__all__": True}
            buyer_info = {self._buyer_inner: {"inactive": True}}
        else:
            buyer_obs, buyer_reward, buyer_done, buyer_info = self.buyer_env.step(
                self._inner_action(buyer_game_action)
            )
            buyer_game_reward = float(buyer_reward.get(self._buyer_inner, 0.0))

        if buyer_obs is not None:
            self._buyer_obs = buyer_obs
        self._buyer_done = self._buyer_done or bool(buyer_done.get("__all__", False))
        self.step_count += 1

        buyer_payment = self.payment_penalty_lambda * revenue
        rewards = {
            self.SELLER: revenue,
            self.BUYER: self.buyer_game_reward_scale * buyer_game_reward - buyer_payment,
        }
        common = {
            "trade_event": trade_event,
            "trade_this_step": traded,
            "accept_this_step": traded,
            "price_offered": offered_price,
            "threshold": buyer_threshold,
            "buyer_bullets": float(self.buyer_pool.val),
            "offer_events_used": float(self._offer_events_used),
            "accepted_trades": float(self._accepted_trades),
        }
        buyer_inner_info = dict(buyer_info.get(self._buyer_inner, {}))
        info = {
            "reward_generated": True,
            "surplus": rewards[self.SELLER],
            "utilities": rewards,
            "seller_reward": rewards[self.SELLER],
            "buyer_reward": rewards[self.BUYER],
            "buyer_game_reward_unscaled": buyer_game_reward,
            "buyer_game_reward_scaled": self.buyer_game_reward_scale * buyer_game_reward,
            "revenue_this_step": revenue,
            "buyer_paid_this_step": buyer_payment,
            "cumulative_revenue": self.cumulative_revenue,
            "buyer_info": buyer_inner_info,
            **common,
        }

        next_trade_now = bool(
            self._offer_events_used < self.offer_chances
            and self._accepted_trades < self.max_replenish
        )
        self._trade_now = next_trade_now
        self.current_price = seller_price if next_trade_now else 0.0

        done = self._buyer_done
        if self.max_steps is not None and self.step_count >= self.max_steps:
            done = True
        return self._obs(), rewards, bool(done), info

    def render(self, mode="rgb_array"):
        return self.buyer_env.render(mode=mode)

    def close(self):
        close = getattr(self.buyer_env, "close", None)
        if close is not None:
            close()


class BaseFrozenGameplayAtariBulletPricingEnv(BaseAtariBulletPricingEnv):
    """Frozen-gameplay version used by the threshold diagnostic.

    The economic game is the same as :class:`BaseAtariBulletPricingEnv`, but
    the buyer no longer chooses an Atari action. The env loads a frozen
    five-bullet Space Invaders policy (legacy image-only or ammo-aware) and uses
    its argmax action each step. The trainable buyer/follower action is only the
    scalar threshold.
    """

    def __init__(
            self,
            *,
            game_checkpoint=None,
            fixed_game_action=0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.game_checkpoint = game_checkpoint
        self.fixed_game_action = int(fixed_game_action)
        self._frozen_game_model = None
        self._torch = None
        self._frozen_game_ammo_aware = False
        self._frozen_ammo_denominator = 5.0
        self._frozen_fire_action_indices = ()

        self.followers_action_space = {
            self.BUYER: Box(0.0, self.price_max, shape=(1,), dtype=np.float32)
        }
        if game_checkpoint is not None:
            self._load_frozen_game_model(game_checkpoint)

    def _load_frozen_game_model(self, checkpoint_path):
        import os
        import pickle

        try:
            import torch
            from stackerlberg.train.atari_models import NatureCNNTorch
        except ImportError as exc:
            raise ImportError(
                "BaseFrozenGameplayAtariBulletPricingEnv requires torch and "
                "StackeRLberg's NatureCNNTorch to load a frozen game policy."
            ) from exc

        path = os.path.expanduser(checkpoint_path)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        metadata = payload.get("metadata", {})
        source = (
            payload.get(self.BUYER)
            or payload.get(self.SELLER)
            or payload.get("default_policy")
        )
        if source is None:
            raise ValueError(f"{path} has no usable policy weights: {list(payload)}")

        self._frozen_game_ammo_aware = bool(metadata.get("ammo_aware", False))
        if self._frozen_game_ammo_aware:
            from stackelberg_pomdp.atari_models import AmmoAwareNatureCNNTorch

            model_class = AmmoAwareNatureCNNTorch
            model_observation_space = Dict(OrderedDict([
                ("image", self.buyer_image_space),
                (
                    "ammo_fraction",
                    Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                ),
                (
                    "action_mask",
                    Box(
                        0.0,
                        1.0,
                        shape=(self.buyer_game_action_space.n,),
                        dtype=np.float32,
                    ),
                ),
            ]))
            self._frozen_ammo_denominator = float(
                metadata.get("ammo_fraction_denominator") or 5.0
            )
            self._frozen_fire_action_indices = tuple(
                idx
                for idx, meaning in enumerate(
                    self.buyer_env.unwrapped.get_action_meanings()
                )
                if "FIRE" in meaning
            )
            model_config = {
                "vf_share_layers": True,
                "custom_model_config": {
                    "ammo_hidden": int(metadata.get("ammo_hidden") or 32),
                },
            }
        else:
            model_class = NatureCNNTorch
            model_observation_space = self.buyer_image_space
            model_config = {"vf_share_layers": True}

        model = model_class(
            model_observation_space,
            self.buyer_game_action_space,
            self.buyer_game_action_space.n,
            model_config,
            "frozen_threshold_buyer_game",
        )
        current = model.state_dict()
        patched = {}
        copied = 0
        for key, value in current.items():
            if key in source and tuple(source[key].shape) == tuple(value.shape):
                patched[key] = torch.as_tensor(source[key])
                copied += 1
            else:
                patched[key] = value
        if copied == 0:
            raise ValueError(f"no compatible frozen-game tensors copied from {path}")
        model.load_state_dict(patched)
        model.eval()
        self._torch = torch
        self._frozen_game_model = model

    def _buyer_image(self):
        if self._buyer_image_key is None:
            return self._buyer_obs[self._buyer_inner]
        return self._buyer_obs[self._buyer_inner][self._buyer_image_key]

    def _frozen_game_action(self):
        if self._frozen_game_model is None:
            return self.fixed_game_action
        image = self._torch.as_tensor(self._buyer_image()[None, ...])
        if self._frozen_game_ammo_aware:
            ram = self.buyer_env.unwrapped.ale.getRAM()
            projectile_active = any(
                ram[slot]
                != AmmoAwareSinglePlayerAtariBulletEnv.INACTIVE_PROJECTILE_RAM_VALUE
                for slot in AmmoAwareSinglePlayerAtariBulletEnv.PROJECTILE_RAM_SLOTS
            )
            action_mask = np.ones(
                self.buyer_game_action_space.n, dtype=np.float32
            )
            if projectile_active or not self.buyer_pool.available_bool(1):
                action_mask[list(self._frozen_fire_action_indices)] = 0.0
            ammo_fraction = np.array([
                np.clip(
                    float(self.buyer_pool.val) / self._frozen_ammo_denominator,
                    0.0,
                    1.0,
                )
            ], dtype=np.float32)
            model_observation = {
                "image": image,
                "ammo_fraction": self._torch.as_tensor(ammo_fraction[None, ...]),
                "action_mask": self._torch.as_tensor(action_mask[None, ...]),
            }
        else:
            model_observation = image
        with self._torch.no_grad():
            logits, _ = self._frozen_game_model(
                {"obs": model_observation},
                [],
                None,
            )
        return int(self._torch.argmax(logits[0]).item())

    def _extract_buyer_action(self, action):
        threshold = float(np.asarray(action).reshape(-1)[0])
        return self._frozen_game_action(), threshold
