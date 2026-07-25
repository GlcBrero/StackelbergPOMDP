"""Composable Gym wrappers for scarce-ammo and priced Atari gameplay."""

from collections import deque, OrderedDict
from dataclasses import dataclass

import cv2
import gym
from gym import spaces
import numpy as np


cv2.ocl.setUseOpenCL(False)


class AmmoLedger:
    """Shared, explicit bullet inventory used by ammo and market wrappers."""

    def __init__(self, *, initial_ammo, capacity):
        self.initial_ammo = int(initial_ammo)
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("ammo capacity must be positive")
        if not 0 <= self.initial_ammo <= self.capacity:
            raise ValueError("initial_ammo must lie in [0, capacity]")
        self.value = self.initial_ammo

    def reset(self):
        self.value = self.initial_ammo

    def available(self, amount=1):
        return self.value >= amount

    def consume(self, amount=1):
        if not self.available(amount):
            raise RuntimeError("cannot consume unavailable ammunition")
        self.value -= amount

    def grant(self, amount=1):
        amount = int(amount)
        granted = min(amount, self.capacity - self.value)
        self.value += granted
        return granted

    @property
    def fraction(self):
        return float(self.value) / float(self.capacity)


class NoopResetWrapper(gym.Wrapper):
    def __init__(self, env, *, noop_max=30, seed=1):
        super().__init__(env)
        self.noop_max = int(noop_max)
        if self.noop_max <= 0:
            raise ValueError("noop_max must be positive")
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.env.reset()
        observation = None
        noops = int(self.rng.integers(1, self.noop_max + 1))
        for _ in range(noops):
            observation, _, done, _ = self.env.step(0)
            if done:
                observation = self.env.reset()
        return observation


class EpisodicLifeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = -1
        self.was_real_done = True

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.was_real_done = bool(done)
        lives = int(info.get("ale.lives", -1))
        life_lost = lives < self.lives and lives > -1
        self.lives = lives
        return observation, reward, bool(done or life_lost), info

    def reset(self):
        if self.was_real_done:
            observation = self.env.reset()
        else:
            observation, _, _, _ = self.env.step(0)
        lives = self.env.unwrapped.ale.allLives()
        self.lives = int(lives[0]) if len(lives) else -1
        return observation


class ClipGameRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))


class ScarceAmmoWrapper(gym.Wrapper):
    """Enforce a finite bullet pool and count actual projectiles via ALE RAM."""

    PROJECTILE_RAM_SLOTS = (0x55, 0x56)
    INACTIVE_PROJECTILE_RAM_VALUE = 0xF6
    NEW_PROJECTILE_RAM_VALUE = 0x55

    def __init__(self, env, *, ledger):
        super().__init__(env)
        self.ledger = ledger
        meanings = tuple(self.env.unwrapped.get_action_meanings())
        self.action_meanings = meanings
        self.fire_action_indices = tuple(
            index for index, meaning in enumerate(meanings) if "FIRE" in meaning
        )
        self._meaning_to_index = {
            meaning: index for index, meaning in enumerate(meanings)
        }
        self._previous_projectile_active = {
            slot: False for slot in self.PROJECTILE_RAM_SLOTS
        }
        self.shots_fired_total = 0

    def projectile_active(self):
        ram = self.env.unwrapped.ale.getRAM()
        return any(
            ram[slot] != self.INACTIVE_PROJECTILE_RAM_VALUE
            for slot in self.PROJECTILE_RAM_SLOTS
        )

    def action_mask(self):
        mask = np.ones(self.action_space.n, dtype=np.float32)
        if self.projectile_active() or not self.ledger.available(1):
            mask[list(self.fire_action_indices)] = 0.0
        return mask

    def _remove_fire(self, action):
        meaning = self.action_meanings[action]
        if "FIRE" not in meaning:
            return action
        non_fire_meaning = meaning.replace("FIRE", "") or "NOOP"
        return self._meaning_to_index.get(non_fire_meaning, 0)

    def reset(self):
        observation = self.env.reset()
        self.ledger.reset()
        ram = self.env.unwrapped.ale.getRAM()
        self._previous_projectile_active = {
            slot: bool(ram[slot] == self.NEW_PROJECTILE_RAM_VALUE)
            for slot in self.PROJECTILE_RAM_SLOTS
        }
        self.shots_fired_total = 0
        return observation

    def step(self, action):
        action = int(np.asarray(action).reshape(-1)[0])
        blocked_fire = False
        if (
                "FIRE" in self.action_meanings[action]
                and not self.ledger.available(1)
        ):
            action = self._remove_fire(action)
            blocked_fire = True

        observation, reward, done, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        shots_this_step = 0
        for slot in self.PROJECTILE_RAM_SLOTS:
            active = bool(ram[slot] == self.NEW_PROJECTILE_RAM_VALUE)
            if active and not self._previous_projectile_active[slot]:
                shots_this_step += 1
            self._previous_projectile_active[slot] = active

        if shots_this_step:
            if shots_this_step > self.ledger.value:
                raise RuntimeError("ALE fired more bullets than the available pool")
            self.ledger.consume(shots_this_step)
            self.shots_fired_total += shots_this_step

        suppressed_reward = 0.0
        if self.shots_fired_total == 0 and reward > 0:
            suppressed_reward = float(reward)
            reward = 0.0

        info = dict(info)
        info.update({
            "shot_did_fire": bool(shots_this_step),
            "shots_fired_this_step": int(shots_this_step),
            "shots_fired_total": int(self.shots_fired_total),
            "blocked_fire_this_step": bool(blocked_fire),
            "blocked_fire_count": int(blocked_fire),
            "suppressed_unowned_reward": suppressed_reward,
            "ammo_remaining": int(self.ledger.value),
        })
        return observation, reward, done, info


class MaxAndSkipWrapper(gym.Wrapper):
    def __init__(self, env, *, skip=4):
        super().__init__(env)
        self.skip = int(skip)
        if self.skip <= 0:
            raise ValueError("skip must be positive")
        self._observation_buffer = np.zeros(
            (2,) + self.observation_space.shape,
            dtype=self.observation_space.dtype,
        )

    @staticmethod
    def _merge_info(accumulated, current):
        merged = dict(accumulated)
        for key, value in current.items():
            if key in ("shot_did_fire", "blocked_fire_this_step"):
                merged[key] = bool(merged.get(key, False)) or bool(value)
            elif key in ("shots_fired_this_step", "blocked_fire_count"):
                merged[key] = int(merged.get(key, 0)) + int(value)
            elif key == "suppressed_unowned_reward":
                merged[key] = float(merged.get(key, 0.0)) + float(value)
            else:
                merged[key] = value
        return merged

    def reset(self):
        # Preserve the historical DeepMind/StackeRLberg buffer semantics.  In
        # particular, an episodic-life termination before the final two raw
        # frames leaves the corresponding prior buffer entry intact.
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        observation = None
        for index in range(self.skip):
            observation, reward, done, frame_info = self.env.step(action)
            total_reward += float(reward)
            info = self._merge_info(info, frame_info)
            if index >= self.skip - 2:
                self._observation_buffer[index - (self.skip - 2)] = observation
            if done:
                break
        max_frame = self._observation_buffer.max(axis=0)
        return max_frame, total_reward, done, info


class WarpFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env, *, width=84, height=84):
        super().__init__(env)
        self.width = int(width)
        self.height = int(height)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8,
        )

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        return frame[:, :, None]


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, *, frames=4):
        super().__init__(env)
        self.frames_count = int(frames)
        if self.frames_count <= 0:
            raise ValueError("frames must be positive")
        self.frames = deque(maxlen=self.frames_count)
        shape = list(self.observation_space.shape)
        shape[-1] *= self.frames_count
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=tuple(shape),
            dtype=self.observation_space.dtype,
        )

    def _observation(self):
        return np.concatenate(tuple(self.frames), axis=-1)

    def reset(self):
        observation = self.env.reset()
        self.frames.clear()
        for _ in range(self.frames_count):
            self.frames.append(observation)
        return self._observation()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._observation(), reward, done, info


class PriceProcess:
    def reset(self, seed=None):
        return

    def sample(self):
        raise NotImplementedError


class NoPriceProcess(PriceProcess):
    def sample(self):
        return 0.0


class FixedPriceProcess(PriceProcess):
    def __init__(self, price):
        self.price = float(price)

    def sample(self):
        return self.price


class UniformPriceProcess(PriceProcess):
    def __init__(self, low=0.0, high=1.0, *, seed=1):
        self.low = float(low)
        self.high = float(high)
        if not 0.0 <= self.low <= self.high:
            raise ValueError("uniform prices require 0 <= low <= high")
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)

    def reset(self, seed=None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)

    def sample(self):
        return float(self.rng.uniform(self.low, self.high))


class OfferTimingProcess:
    """Decide whether the seller makes its one-shot offer this decision."""

    def reset(self, seed=None):
        return

    def should_offer(self, decision_step):
        raise NotImplementedError


class ImmediateOfferTimingProcess(OfferTimingProcess):
    """Historical control: use the next opportunity immediately."""

    def should_offer(self, decision_step):
        del decision_step
        return True


class BernoulliOfferTimingProcess(OfferTimingProcess):
    """Independent stochastic seller wait/offer decisions."""

    def __init__(self, probability=0.04, *, seed=1):
        self.probability = float(probability)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("offer probability must lie in [0, 1]")
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)

    def reset(self, seed=None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)

    def should_offer(self, decision_step):
        del decision_step
        return bool(self.rng.random() < self.probability)


@dataclass
class MarketState:
    offer_active: bool = False
    current_price: float = 0.0
    opportunities_used: int = 0
    accepted_trades: int = 0
    decision_step: int = 0


class BulletMarketWrapper(gym.Wrapper):
    """Add finite one-bullet offers around otherwise ordinary gameplay."""

    def __init__(
            self,
            env,
            *,
            ledger,
            price_process,
            timing_process=None,
            trade_enabled,
            offer_chances=5,
            max_purchases=5,
            price_max=1.0,
            episode_horizon=None,
    ):
        super().__init__(env)
        self.ledger = ledger
        self.price_process = price_process
        self.timing_process = timing_process or ImmediateOfferTimingProcess()
        self.trade_enabled = bool(trade_enabled)
        self.offer_chances = int(offer_chances)
        self.max_purchases = int(max_purchases)
        self.price_max = float(price_max)
        self.episode_horizon = (
            None if episode_horizon is None else int(episode_horizon)
        )
        if self.offer_chances < 0 or self.max_purchases < 0:
            raise ValueError("offer and purchase limits must be nonnegative")
        if self.max_purchases > self.offer_chances:
            raise ValueError("max_purchases cannot exceed offer_chances")
        if self.price_max <= 0:
            raise ValueError("price_max must be positive")
        if self.episode_horizon is not None and self.episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive")
        self.game_action_count = int(self.env.action_space.n)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array(
                [float(self.game_action_count - 1), self.price_max],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.state = MarketState()

    @property
    def remaining_opportunities(self):
        return max(self.offer_chances - self.state.opportunities_used, 0)

    @property
    def normalized_timestep(self):
        if self.episode_horizon is None:
            return 0.0
        return float(np.clip(
            self.state.decision_step / float(self.episode_horizon),
            0.0,
            1.0,
        ))

    @property
    def time_remaining(self):
        return 1.0 - self.normalized_timestep

    @staticmethod
    def time_bin(normalized_timestep):
        value = float(normalized_timestep)
        if value < 1.0 / 3.0:
            return "early"
        if value < 2.0 / 3.0:
            return "middle"
        return "late"

    def _can_offer(self):
        return bool(
            self.trade_enabled
            and self.state.opportunities_used < self.offer_chances
            and self.state.accepted_trades < self.max_purchases
            and (
                self.episode_horizon is None
                or self.state.decision_step < self.episode_horizon
            )
        )

    def _prepare_offer(self):
        self.state.offer_active = bool(
            self._can_offer()
            and self.timing_process.should_offer(self.state.decision_step)
        )
        self.state.current_price = (
            float(np.clip(self.price_process.sample(), 0.0, self.price_max))
            if self.state.offer_active
            else 0.0
        )

    def reset(self):
        observation = self.env.reset()
        self.state = MarketState()
        self.price_process.reset()
        self.timing_process.reset()
        self._prepare_offer()
        return observation

    def step(self, action):
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if len(values) != 2:
            raise ValueError("Atari buyer action must be [game_action, threshold]")
        game_action = int(
            np.clip(np.rint(values[0]), 0, self.game_action_count - 1)
        )
        threshold = float(np.clip(values[1], 0.0, self.price_max))

        trade_event = bool(self.state.offer_active)
        offered_price = self.state.current_price if trade_event else 0.0
        offer_step = int(self.state.decision_step)
        normalized_timestep = float(self.normalized_timestep)
        time_remaining = float(self.time_remaining)
        time_bin = self.time_bin(normalized_timestep)
        opportunities_remaining_before = int(self.remaining_opportunities)
        ammo_before_trade = int(self.ledger.value)
        traded = False
        payment = 0.0
        if trade_event:
            self.state.opportunities_used += 1
            if offered_price <= threshold:
                granted = self.ledger.grant(1)
                if granted != 1:
                    raise RuntimeError("accepted trade could not transfer one bullet")
                self.state.accepted_trades += 1
                traded = True
                payment = offered_price
        ammo_after_trade = int(self.ledger.value)
        opportunities_remaining_after = int(self.remaining_opportunities)

        observation, game_reward, done, info = self.env.step(game_action)
        net_reward = float(game_reward) - payment
        info = dict(info)
        info.update({
            "game_reward": float(game_reward),
            "payment": float(payment),
            "net_reward": float(net_reward),
            "trade_event": trade_event,
            "trade_this_step": traded,
            "price_offered": float(offered_price),
            "threshold": threshold,
            "offer_step": offer_step,
            "normalized_timestep": normalized_timestep,
            "time_remaining": time_remaining,
            "time_bin": time_bin,
            "ammo_before_trade": ammo_before_trade,
            "ammo_after_trade": ammo_after_trade,
            "opportunities_remaining_before": opportunities_remaining_before,
            "opportunities_remaining_after": opportunities_remaining_after,
            "offer_events_used": int(self.state.opportunities_used),
            "accepted_trades": int(self.state.accepted_trades),
            "ammo_remaining": int(self.ledger.value),
        })

        self.state.decision_step += 1
        if done:
            self.state.offer_active = False
            self.state.current_price = 0.0
        else:
            self._prepare_offer()
        return observation, net_reward, done, info


class AsymmetricBuyerObservationWrapper(gym.ObservationWrapper):
    """Keep gameplay inputs stable while optionally exposing economic context."""

    def __init__(
            self,
            env,
            *,
            ledger,
            ammo_wrapper,
            market_wrapper,
            actor_economic_context=False,
    ):
        super().__init__(env)
        self.ledger = ledger
        self.ammo_wrapper = ammo_wrapper
        self.market_wrapper = market_wrapper
        self.actor_economic_context = bool(actor_economic_context)
        if (
                self.actor_economic_context
                and self.market_wrapper.episode_horizon is None
        ):
            raise ValueError(
                "actor economic context requires a finite episode horizon"
            )
        game_actions = self.ammo_wrapper.action_space.n
        observation_spaces = OrderedDict([
            ("image", self.env.observation_space),
            (
                "ammo_fraction",
                spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "projectile_active",
                spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "action_mask",
                spaces.Box(
                    0.0, 1.0, shape=(game_actions,), dtype=np.float32
                ),
            ),
            (
                "offer_active",
                spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "opportunities_remaining",
                spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
        ])
        if self.actor_economic_context:
            observation_spaces.update([
                (
                    "price",
                    spaces.Box(
                        0.0,
                        self.market_wrapper.price_max,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                ),
                (
                    "normalized_timestep",
                    spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                ),
                (
                    "time_remaining",
                    spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                ),
            ])
        observation_spaces["critic:price"] = spaces.Box(
            0.0,
            self.market_wrapper.price_max,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(observation_spaces)

    def observation(self, observation):
        market = self.market_wrapper
        remaining_fraction = (
            float(market.remaining_opportunities) / float(market.offer_chances)
            if market.offer_chances
            else 0.0
        )
        result = OrderedDict([
            ("image", observation),
            (
                "ammo_fraction",
                np.array([self.ledger.fraction], dtype=np.float32),
            ),
            (
                "projectile_active",
                np.array(
                    [float(self.ammo_wrapper.projectile_active())],
                    dtype=np.float32,
                ),
            ),
            ("action_mask", self.ammo_wrapper.action_mask()),
            (
                "offer_active",
                np.array(
                    [float(market.state.offer_active)], dtype=np.float32
                ),
            ),
            (
                "opportunities_remaining",
                np.array([remaining_fraction], dtype=np.float32),
            ),
        ])
        visible_price = (
            market.state.current_price if market.state.offer_active else 0.0
        )
        if self.actor_economic_context:
            result.update([
                (
                    "price",
                    np.array([visible_price], dtype=np.float32),
                ),
                (
                    "normalized_timestep",
                    np.array(
                        [market.normalized_timestep], dtype=np.float32
                    ),
                ),
                (
                    "time_remaining",
                    np.array([market.time_remaining], dtype=np.float32),
                ),
            ])
        result["critic:price"] = np.array(
            [visible_price], dtype=np.float32
        )
        return result


class AtariEpisodeMetricsWrapper(gym.Wrapper):
    def __init__(self, env, *, ledger, max_steps=None):
        super().__init__(env)
        self.ledger = ledger
        self.max_steps = None if max_steps is None else int(max_steps)
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._reset_metrics()

    def _reset_metrics(self):
        self.episode_reward = 0.0
        self.episode_game_reward = 0.0
        self.episode_payments = 0.0
        self.episode_steps = 0
        self.episode_shots = 0
        self.episode_purchases = 0
        self.episode_opportunities = 0
        self.trade_events = []
        self.offer_counts_by_time = {
            name: 0 for name in ("early", "middle", "late")
        }
        self.purchase_counts_by_time = {
            name: 0 for name in ("early", "middle", "late")
        }

    def _timing_metrics(self):
        offer_steps = [event["offer_step"] for event in self.trade_events]
        offer_times = [
            event["normalized_timestep"] for event in self.trade_events
        ]
        purchases = [event for event in self.trade_events if event["accepted"]]
        purchase_steps = [event["offer_step"] for event in purchases]
        purchase_times = [
            event["normalized_timestep"] for event in purchases
        ]
        result = {
            "offer_step_sum": float(sum(offer_steps)),
            "offer_normalized_timestep_sum": float(sum(offer_times)),
            "purchase_step_sum": float(sum(purchase_steps)),
            "purchase_normalized_timestep_sum": float(sum(purchase_times)),
            "mean_offer_step": (
                float(np.mean(offer_steps)) if offer_steps else 0.0
            ),
            "mean_offer_normalized_timestep": (
                float(np.mean(offer_times)) if offer_times else 0.0
            ),
            "mean_purchase_step": (
                float(np.mean(purchase_steps)) if purchase_steps else 0.0
            ),
            "mean_purchase_normalized_timestep": (
                float(np.mean(purchase_times)) if purchase_times else 0.0
            ),
            "last_offer_accepted": (
                float(self.trade_events[-1]["accepted"])
                if self.trade_events
                else 0.0
            ),
        }
        for name in ("early", "middle", "late"):
            offers = int(self.offer_counts_by_time[name])
            accepted = int(self.purchase_counts_by_time[name])
            result[f"offer_count_{name}"] = offers
            result[f"purchase_count_{name}"] = accepted
            result[f"rejection_count_{name}"] = offers - accepted
            result[f"acceptance_rate_{name}"] = (
                float(accepted) / float(offers) if offers else 0.0
            )
        late_offers = result["offer_count_late"]
        result["late_rejection_rate"] = (
            float(result["rejection_count_late"]) / float(late_offers)
            if late_offers
            else 0.0
        )
        return result

    def reset(self):
        self._reset_metrics()
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info = dict(info)
        self.episode_reward += float(reward)
        self.episode_game_reward += float(info.get("game_reward", reward))
        self.episode_payments += float(info.get("payment", 0.0))
        self.episode_steps += 1
        self.episode_shots += int(info.get("shots_fired_this_step", 0))
        self.episode_purchases += int(bool(info.get("trade_this_step", False)))
        self.episode_opportunities += int(bool(info.get("trade_event", False)))
        if info.get("trade_event", False):
            time_bin = str(info["time_bin"])
            if time_bin not in self.offer_counts_by_time:
                raise RuntimeError(f"unknown trade time bin: {time_bin}")
            accepted = bool(info.get("trade_this_step", False))
            event = {
                "opportunity_index": int(self.episode_opportunities - 1),
                "offer_step": int(info["offer_step"]),
                "normalized_timestep": float(info["normalized_timestep"]),
                "time_remaining": float(info["time_remaining"]),
                "time_bin": time_bin,
                "price": float(info["price_offered"]),
                "threshold": float(info["threshold"]),
                "accepted": accepted,
                "ammo_before_trade": int(info["ammo_before_trade"]),
                "ammo_after_trade": int(info["ammo_after_trade"]),
                "ammo_end_step": int(info.get("ammo_remaining", 0)),
                "shot_this_step": bool(info.get("shot_did_fire", False)),
                "shots_fired_this_step": int(
                    info.get("shots_fired_this_step", 0)
                ),
                "opportunities_remaining_before": int(
                    info["opportunities_remaining_before"]
                ),
                "opportunities_remaining_after": int(
                    info["opportunities_remaining_after"]
                ),
            }
            self.trade_events.append(event)
            self.offer_counts_by_time[time_bin] += 1
            self.purchase_counts_by_time[time_bin] += int(accepted)
        if self.max_steps is not None and self.episode_steps >= self.max_steps:
            done = True

        reward_per_bullet = (
            self.episode_game_reward / self.episode_shots
            if self.episode_shots
            else 0.0
        )
        timing_metrics = self._timing_metrics()
        unused_purchased_bullets = max(
            int(self.episode_purchases - self.episode_shots), 0
        )
        info.update({
            "episode_reward": float(self.episode_reward),
            "episode_game_reward": float(self.episode_game_reward),
            "episode_payments": float(self.episode_payments),
            "episode_length": int(self.episode_steps),
            "shots_fired": int(self.episode_shots),
            "purchases": int(self.episode_purchases),
            "trade_opportunities": int(self.episode_opportunities),
            "final_ammo": int(self.ledger.value),
            "reward_per_bullet": float(reward_per_bullet),
            "fired_fraction_of_purchases": (
                float(self.episode_shots) / float(self.episode_purchases)
                if self.episode_purchases
                else 0.0
            ),
            "unused_purchased_bullets": unused_purchased_bullets,
            **timing_metrics,
        })
        if done:
            acceptance_rate = (
                float(self.episode_purchases) / float(self.episode_opportunities)
                if self.episode_opportunities
                else 0.0
            )
            fired_fraction = (
                float(self.episode_shots) / float(self.episode_purchases)
                if self.episode_purchases
                else 0.0
            )
            info["trade_events"] = list(self.trade_events)
            info["episode"] = {
                "r": float(self.episode_reward),
                "l": int(self.episode_steps),
                "game_reward": float(self.episode_game_reward),
                "payments": float(self.episode_payments),
                "shots_fired": int(self.episode_shots),
                "purchases": int(self.episode_purchases),
                "trade_opportunities": int(self.episode_opportunities),
                "final_ammo": int(self.ledger.value),
                "reward_per_bullet": float(reward_per_bullet),
                "acceptance_rate": acceptance_rate,
                "fired_fraction_of_purchases": fired_fraction,
                "unused_purchased_bullets": unused_purchased_bullets,
                **timing_metrics,
            }
            for count in range(6):
                info["episode"][f"opportunity_count_is_{count}"] = float(
                    self.episode_opportunities == count
                )
        return observation, reward, done, info
