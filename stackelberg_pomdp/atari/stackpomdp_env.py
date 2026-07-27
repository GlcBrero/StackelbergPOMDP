"""Native two-player Atari economics with frozen gameplay controllers.

This module contains the environment-side pieces used by the Atari
Stackelberg-POMDP experiments.  It deliberately does not depend on the legacy
RLlib Atari environments.  Each player owns an independent native Space
Invaders instance and the two games are coupled only by a five-event bullet
market.

The economic training adapter is event driven: PPO acts only at the five trade
events.  Between events both Atari games are advanced by immutable,
deterministic E0 controllers.  This is exactly equivalent to exposing all game
steps while assigning zero log probability to the frozen game action, but is
considerably faster and makes accidental Atari fine-tuning impossible.
"""

from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import gym
from gym import spaces
import numpy as np

from stackelberg_pomdp.atari.factory import (
    AtariBuyerEnvConfig,
    make_atari_buyer_env,
)


SELLER = "seller"
BUYER = "buyer"
ROLES = {SELLER, BUYER}
NUM_TRADE_EVENTS = 5
CRITIC_STATE_DIM = 12


@dataclass(frozen=True)
class BilateralAtariConfig:
    """Configuration shared by response-head and leader experiments."""

    seed: int = 1
    gameplay_horizon: int = 200
    event_tail_steps: int = 50
    num_trade_events: int = NUM_TRADE_EVENTS
    seller_game_reward_scale: float = 0.1
    buyer_game_reward_scale: float = 1.0
    noop_max: int = 30
    frame_skip: int = 4
    frame_stack: int = 4
    episodic_life: bool = True
    clip_game_rewards: bool = True
    max_frames: int = 100_000
    rom_path: Optional[str] = None
    fixed_event_steps: Optional[Sequence[int]] = None

    def resolved(self):
        if self.num_trade_events != NUM_TRADE_EVENTS:
            raise ValueError(
                "the canonical Atari protocol requires exactly five events"
            )
        if self.gameplay_horizon <= self.num_trade_events:
            raise ValueError("gameplay_horizon is too short")
        if not 1 <= self.event_tail_steps < self.gameplay_horizon:
            raise ValueError(
                "event_tail_steps must leave positive gameplay after event five"
            )
        event_window = self.gameplay_horizon - self.event_tail_steps
        if event_window < self.num_trade_events:
            raise ValueError(
                "the pre-tail event window must contain at least five steps"
            )
        if self.seller_game_reward_scale < 0:
            raise ValueError("seller_game_reward_scale must be nonnegative")
        if self.buyer_game_reward_scale < 0:
            raise ValueError("buyer_game_reward_scale must be nonnegative")
        fixed = self.fixed_event_steps
        if fixed is not None:
            fixed = tuple(int(step) for step in fixed)
            if len(fixed) != self.num_trade_events:
                raise ValueError("fixed_event_steps must contain five entries")
            if tuple(sorted(set(fixed))) != fixed:
                raise ValueError(
                    "fixed_event_steps must be strictly increasing"
                )
            latest = self.gameplay_horizon - self.event_tail_steps - 1
            if fixed[0] < 0 or fixed[-1] > latest:
                raise ValueError(
                    "fixed_event_steps must fit before the reserved tail"
                )
            return replace(self, fixed_event_steps=fixed)
        return self


class ExactFiveEventSchedule:
    """Sample five exogenous event times before a reserved gameplay tail."""

    def __init__(self, config: BilateralAtariConfig):
        self.config = config.resolved()

    def sample(self, rng):
        fixed = self.config.fixed_event_steps
        if fixed is not None:
            return tuple(fixed)
        event_stop = (
            self.config.gameplay_horizon - self.config.event_tail_steps
        )
        candidates = np.arange(event_stop, dtype=np.int64)
        sampled = rng.choice(
            candidates,
            size=self.config.num_trade_events,
            replace=False,
        )
        return tuple(int(value) for value in np.sort(sampled))


class FrozenE0Controller:
    """Read-only deterministic adapter around the selected native SB3 E0."""

    def __init__(self, checkpoint, *, device="cpu"):
        from stable_baselines3 import PPO

        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.model = PPO.load(self.checkpoint, device=device)
        self.model.policy.set_training_mode(False)
        for parameter in self.model.policy.parameters():
            parameter.requires_grad = False

    def __call__(self, observation):
        return self.actions([observation])[0]

    def actions(self, observations):
        batch = {
            key: np.stack([observation[key] for observation in observations])
            for key in observations[0]
        }
        action, _ = self.model.predict(batch, deterministic=True)
        values = np.asarray(action, dtype=np.float32).reshape(-1, 2)
        if values.shape[0] != len(observations):
            raise RuntimeError("E0 controller returned the wrong batch size")
        if values.shape[1] != 2:
            raise RuntimeError(
                "the canonical E0 checkpoint must return [game_action, scalar]"
            )
        return [int(np.rint(value[0])) for value in values]

    def _legacy_single_action(self, observation):
        """Retained as an explicit checkpoint-shape assertion for diagnostics."""
        action, _ = self.model.predict(observation, deterministic=True)
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if len(values) != 2:
            raise RuntimeError(
                "the canonical E0 checkpoint must return [game_action, scalar]"
            )
        return int(np.rint(values[0]))


class NativeGameplaySide:
    """One native Space Invaders game with persistent outer-episode ammo."""

    def __init__(
            self,
            *,
            seed,
            config,
            controller,
            env_factory=make_atari_buyer_env,
    ):
        self.seed = int(seed)
        self.config = config
        self.controller = controller
        env_config = AtariBuyerEnvConfig(
            stage="gameplay",
            seed=self.seed,
            initial_bullets=0,
            bullet_capacity=NUM_TRADE_EVENTS,
            offer_chances=NUM_TRADE_EVENTS,
            max_purchases=NUM_TRADE_EVENTS,
            noop_max=config.noop_max,
            frame_skip=config.frame_skip,
            frame_stack=config.frame_stack,
            episodic_life=config.episodic_life,
            clip_game_rewards=config.clip_game_rewards,
            max_steps=None,
            max_frames=config.max_frames,
            rom_path=config.rom_path,
        )
        self.env = env_factory(env_config)
        self.ledger = self.env.ammo_ledger
        self.ammo_wrapper = self.env.ammo_wrapper
        self.observation_space = self.env.observation_space
        self.game_action_count = int(self.env.action_space.high[0]) + 1
        self.observation = None
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0

    def _refreshed_observation(self):
        """Refresh ledger-dependent fields without advancing the emulator."""
        if self.observation is None:
            raise RuntimeError("gameplay side has not been reset")
        result = OrderedDict(
            (key, np.array(value, copy=True))
            for key, value in self.observation.items()
        )
        result["ammo_fraction"] = np.array(
            [self.ledger.fraction], dtype=np.float32
        )
        result["projectile_active"] = np.array(
            [float(self.ammo_wrapper.projectile_active())], dtype=np.float32
        )
        result["action_mask"] = np.asarray(
            self.ammo_wrapper.action_mask(), dtype=np.float32
        )
        # These are the exact neutral values seen by the selected E0 policy.
        result["offer_active"] = np.array([0.0], dtype=np.float32)
        result["opportunities_remaining"] = np.array(
            [1.0], dtype=np.float32
        )
        result["critic:price"] = np.array([0.0], dtype=np.float32)
        self.observation = result
        return result

    def reset(self):
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0
        self.observation = self.env.reset()
        return self._refreshed_observation()

    def grant(self, amount=1):
        granted = self.ledger.grant(amount)
        self._refreshed_observation()
        return granted

    def consume(self, amount=1):
        self.ledger.consume(amount)
        self._refreshed_observation()

    @property
    def ammo(self):
        return int(self.ledger.value)

    def _reset_life_preserving_ammo(self):
        ammo = self.ammo
        self.observation = self.env.reset()
        self.ledger.value = ammo
        self.life_resets += 1
        self._refreshed_observation()

    def step(self, action=None):
        if action is None:
            action = int(self.controller(self._refreshed_observation()))
        else:
            action = int(action)
        observation, reward, done, info = self.env.step(
            np.array([action, 0.0], dtype=np.float32)
        )
        self.observation = observation
        reward = float(reward)
        shots = int(info.get("shots_fired_this_step", 0))
        self.game_reward += reward
        self.shots_fired += shots
        if done:
            self._reset_life_preserving_ammo()
        else:
            self._refreshed_observation()
        return reward, shots, dict(info)

    def close(self):
        self.env.close()


class DualAtariTradeCore:
    """Two independent frozen Atari games coupled by bullet transfers."""

    def __init__(
            self,
            config,
            *,
            game_checkpoint=None,
            controller_factory=None,
            side_factory=NativeGameplaySide,
            env_factory=make_atari_buyer_env,
    ):
        self.config = config.resolved()
        if controller_factory is None:
            if game_checkpoint is None:
                raise ValueError(
                    "game_checkpoint or controller_factory is required"
                )

            shared_controller = FrozenE0Controller(game_checkpoint)
            seller_controller = shared_controller
            buyer_controller = shared_controller
        else:
            seller_controller = controller_factory()
            buyer_controller = controller_factory()

        self.seller = side_factory(
            seed=self.config.seed,
            config=self.config,
            controller=seller_controller,
            env_factory=env_factory,
        )
        self.buyer = side_factory(
            seed=self.config.seed + 100_003,
            config=self.config,
            controller=buyer_controller,
            env_factory=env_factory,
        )
        if self.seller.observation_space != self.buyer.observation_space:
            raise ValueError("seller and buyer Atari spaces must match")
        if self.seller.game_action_count != self.buyer.game_action_count:
            raise ValueError("seller and buyer action spaces must match")
        self.observation_space = self.seller.observation_space
        self.game_action_count = self.seller.game_action_count
        self.rng = np.random.default_rng(self.config.seed + 31_337)
        self.schedule_sampler = ExactFiveEventSchedule(self.config)
        self.event_steps = ()
        self.game_step = 0
        self.next_event = 0
        self.bullets_arrived = 0
        self.transfers = 0
        self.payments = 0.0
        self.seller_payoff = 0.0
        self.buyer_payoff = 0.0
        self.events = []
        self._prepared_event_index = None

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.seller.reset()
        self.buyer.reset()
        self.event_steps = self.schedule_sampler.sample(self.rng)
        self.game_step = 0
        self.next_event = 0
        self.bullets_arrived = 0
        self.transfers = 0
        self.payments = 0.0
        self.seller_payoff = 0.0
        self.buyer_payoff = 0.0
        self.events = []
        self._prepared_event_index = None
        return self.event_steps

    @property
    def at_event(self):
        return bool(
            self.next_event < self.config.num_trade_events
            and self.game_step == self.event_steps[self.next_event]
        )

    @property
    def done(self):
        return self.game_step >= self.config.gameplay_horizon

    def advance_to_event_or_end(self):
        """Run frozen gameplay until the next event or the outer horizon."""
        seller_delta = 0.0
        buyer_delta = 0.0
        while not self.done and not self.at_event:
            shared_controller = (
                self.seller.controller
                if self.seller.controller is self.buyer.controller
                else None
            )
            if shared_controller is not None and hasattr(
                    shared_controller, "actions"
            ):
                game_actions = shared_controller.actions([
                    self.seller._refreshed_observation(),
                    self.buyer._refreshed_observation(),
                ])
                seller_reward, _, _ = self.seller.step(game_actions[0])
                buyer_reward, _, _ = self.buyer.step(game_actions[1])
            else:
                seller_reward, _, _ = self.seller.step()
                buyer_reward, _, _ = self.buyer.step()
            scaled_seller = (
                self.config.seller_game_reward_scale * seller_reward
            )
            scaled_buyer = self.config.buyer_game_reward_scale * buyer_reward
            seller_delta += scaled_seller
            buyer_delta += scaled_buyer
            self.seller_payoff += scaled_seller
            self.buyer_payoff += scaled_buyer
            self.game_step += 1
        if self.at_event:
            self.prepare_event()
        return seller_delta, buyer_delta

    def prepare_event(self):
        """Make the event bullet available before either economic decision."""
        if not self.at_event:
            raise RuntimeError("prepare_event called outside a scheduled event")
        if self._prepared_event_index == self.next_event:
            return
        granted = self.seller.grant(1)
        if granted != 1:
            raise RuntimeError("seller could not receive the exogenous bullet")
        self.bullets_arrived += 1
        self._prepared_event_index = self.next_event

    def trade(self, *, price, threshold):
        """Execute one paused, one-shot event and atomically transfer a bullet."""
        if not self.at_event:
            raise RuntimeError("trade called outside a scheduled event")
        if self._prepared_event_index != self.next_event:
            raise RuntimeError("event bullet was not prepared before trade")
        price = float(np.clip(price, 0.0, 1.0))
        threshold = float(np.clip(threshold, 0.0, 1.0))
        seller_before = self.seller.ammo
        buyer_before = self.buyer.ammo
        accepted = bool(price <= threshold)
        if accepted:
            self.seller.consume(1)
            granted = self.buyer.grant(1)
            if granted != 1:
                raise RuntimeError("buyer could not receive the accepted bullet")
            self.transfers += 1
            self.payments += price
            self.seller_payoff += price
            self.buyer_payoff -= price
        event = {
            "event_index": int(self.next_event),
            "game_step": int(self.game_step),
            "price": price,
            "threshold": threshold,
            "accepted": accepted,
            "seller_ammo_before": seller_before,
            "buyer_ammo_before": buyer_before,
            "seller_ammo_after": self.seller.ammo,
            "buyer_ammo_after": self.buyer.ammo,
        }
        self.events.append(event)
        self.next_event += 1
        self._prepared_event_index = None
        return event

    def accounting(self):
        seller_error = (
            self.bullets_arrived
            - self.seller.shots_fired
            - self.transfers
            - self.seller.ammo
        )
        buyer_error = (
            self.transfers - self.buyer.shots_fired - self.buyer.ammo
        )
        payoff_error = (
            self.seller_payoff
            - (
                self.config.seller_game_reward_scale
                * self.seller.game_reward
                + self.payments
            )
        )
        buyer_payoff_error = (
            self.buyer_payoff
            - (
                self.config.buyer_game_reward_scale
                * self.buyer.game_reward
                - self.payments
            )
        )
        return {
            "seller_bullet_error": int(seller_error),
            "buyer_bullet_error": int(buyer_error),
            "seller_payoff_error": float(payoff_error),
            "buyer_payoff_error": float(buyer_payoff_error),
        }

    def assert_accounting(self):
        accounting = self.accounting()
        if accounting["seller_bullet_error"] != 0:
            raise RuntimeError(f"seller bullet accounting failed: {accounting}")
        if accounting["buyer_bullet_error"] != 0:
            raise RuntimeError(f"buyer bullet accounting failed: {accounting}")
        if abs(accounting["seller_payoff_error"]) > 1.0e-6:
            raise RuntimeError(f"seller payoff accounting failed: {accounting}")
        if abs(accounting["buyer_payoff_error"]) > 1.0e-6:
            raise RuntimeError(f"buyer payoff accounting failed: {accounting}")

    def close(self):
        self.seller.close()
        self.buyer.close()


class MetaEconomicResponseEnv(gym.Env):
    """Train one economic response head against random five-action contexts.

    ``controlled_role='buyer'`` learns thresholds against a full seller price
    vector.  ``controlled_role='seller'`` learns prices against a full buyer
    threshold vector.  The observation is the same full Atari-shaped mapping
    used by the composite policy, but trade observations contain a fixed zero
    image and the game component of the action is ignored.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            *,
            controlled_role,
            game_checkpoint,
            config=None,
            context_sampler: Optional[Callable] = None,
            core_factory=DualAtariTradeCore,
            controller_factory=None,
    ):
        super().__init__()
        if controlled_role not in ROLES:
            raise ValueError(f"controlled_role must be one of {sorted(ROLES)}")
        self.controlled_role = controlled_role
        self.config = (config or BilateralAtariConfig()).resolved()
        self.context_sampler = context_sampler
        self.rng = np.random.default_rng(self.config.seed + 74_711)
        self.core = core_factory(
            self.config,
            game_checkpoint=game_checkpoint,
            controller_factory=controller_factory,
        )
        self.game_action_count = self.core.game_action_count
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array(
                [float(self.game_action_count - 1), 1.0], dtype=np.float32
            ),
            dtype=np.float32,
        )
        base_spaces = self.core.observation_space.spaces
        observation_spaces = OrderedDict(
            (key, value) for key, value in base_spaces.items()
        )
        observation_spaces.update([
            (
                "event_active",
                spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "event_one_hot",
                spaces.Box(
                    0.0,
                    1.0,
                    shape=(NUM_TRADE_EVENTS,),
                    dtype=np.float32,
                ),
            ),
            (
                "opponent_context",
                spaces.Box(
                    0.0,
                    1.0,
                    shape=(NUM_TRADE_EVENTS,),
                    dtype=np.float32,
                ),
            ),
            (
                "critic:state",
                spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(CRITIC_STATE_DIM,),
                    dtype=np.float32,
                ),
            ),
        ])
        self.observation_space = spaces.Dict(observation_spaces)
        image_space = base_spaces["image"]
        self.dummy_image = np.zeros(image_space.shape, dtype=image_space.dtype)
        self.opponent_context = np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        self._pending_pre_event_payoffs = (0.0, 0.0)
        self._done = False

    def seed(self, seed=None):
        seed = self.config.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(seed + 74_711)
        return [seed]

    def _sample_context(self):
        if self.context_sampler is None:
            values = self.rng.uniform(0.0, 1.0, size=NUM_TRADE_EVENTS)
        else:
            values = self.context_sampler(self.rng)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape != (NUM_TRADE_EVENTS,):
            raise ValueError("context sampler must return five scalars")
        return np.clip(values, 0.0, 1.0)

    def _controlled_side(self):
        return (
            self.core.buyer
            if self.controlled_role == BUYER
            else self.core.seller
        )

    def _critic_state(self):
        core = self.core
        event_index = min(core.next_event, NUM_TRADE_EVENTS - 1)
        values = np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
        values[:9] = (
            float(core.game_step) / float(self.config.gameplay_horizon),
            float(NUM_TRADE_EVENTS - core.next_event) / NUM_TRADE_EVENTS,
            float(core.seller.ammo) / NUM_TRADE_EVENTS,
            float(core.buyer.ammo) / NUM_TRADE_EVENTS,
            float(core.transfers) / NUM_TRADE_EVENTS,
            float(core.payments) / NUM_TRADE_EVENTS,
            float(core.seller.game_reward) / NUM_TRADE_EVENTS,
            float(core.buyer.game_reward) / NUM_TRADE_EVENTS,
            float(event_index) / float(NUM_TRADE_EVENTS - 1),
        )
        return values

    def _trade_observation(self):
        if self._done or not self.core.at_event:
            raise RuntimeError("trade observation requested outside an event")
        side = self._controlled_side()
        event_one_hot = np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        event_one_hot[self.core.next_event] = 1.0
        # Every E0 input retains its gameplay-stage neutral value.  The game
        # action is ignored on this paused trade substep.
        return OrderedDict([
            ("image", np.array(self.dummy_image, copy=True)),
            (
                "ammo_fraction",
                np.array([side.ammo / NUM_TRADE_EVENTS], dtype=np.float32),
            ),
            ("projectile_active", np.array([0.0], dtype=np.float32)),
            (
                "action_mask",
                np.ones(self.game_action_count, dtype=np.float32),
            ),
            ("offer_active", np.array([0.0], dtype=np.float32)),
            (
                "opportunities_remaining",
                np.array([1.0], dtype=np.float32),
            ),
            ("critic:price", np.array([0.0], dtype=np.float32)),
            ("event_active", np.array([1.0], dtype=np.float32)),
            ("event_one_hot", event_one_hot),
            (
                "opponent_context",
                np.array(self.opponent_context, copy=True),
            ),
            ("critic:state", self._critic_state()),
        ])

    def reset(self):
        self._done = False
        self.opponent_context = self._sample_context()
        self.core.reset(seed=int(self.rng.integers(0, 2 ** 31 - 1)))
        self._pending_pre_event_payoffs = self.core.advance_to_event_or_end()
        if not self.core.at_event:
            raise RuntimeError("the exact-five schedule did not reach event one")
        return self._trade_observation()

    def _episode_info(self):
        core = self.core
        accounting = core.accounting()
        return {
            "controlled_role": self.controlled_role,
            "event_steps": tuple(core.event_steps),
            "opponent_context": tuple(float(x) for x in self.opponent_context),
            "events": tuple(dict(event) for event in core.events),
            "trade_opportunities": int(core.next_event),
            "bullets_arrived": int(core.bullets_arrived),
            "purchases": int(core.transfers),
            "payments": float(core.payments),
            "seller_game_reward": float(core.seller.game_reward),
            "buyer_game_reward": float(core.buyer.game_reward),
            "seller_reward": float(core.seller_payoff),
            "buyer_reward": float(core.buyer_payoff),
            "seller_shots_fired": int(core.seller.shots_fired),
            "buyer_shots_fired": int(core.buyer.shots_fired),
            "seller_final_ammo": int(core.seller.ammo),
            "buyer_final_ammo": int(core.buyer.ammo),
            "seller_life_resets": int(core.seller.life_resets),
            "buyer_life_resets": int(core.buyer.life_resets),
            **accounting,
        }

    def step(self, action):
        if self._done:
            raise RuntimeError("step called after episode termination")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if len(values) != 2:
            raise ValueError("action must be [ignored_game_action, economic_action]")
        economic_action = float(np.clip(values[1], 0.0, 1.0))
        event_index = self.core.next_event
        opponent_action = float(self.opponent_context[event_index])
        if self.controlled_role == BUYER:
            price, threshold = opponent_action, economic_action
        else:
            price, threshold = economic_action, opponent_action

        previous_seller = self.core.seller_payoff
        previous_buyer = self.core.buyer_payoff
        self.core.trade(price=price, threshold=threshold)
        self.core.advance_to_event_or_end()
        seller_delta = self.core.seller_payoff - previous_seller
        buyer_delta = self.core.buyer_payoff - previous_buyer
        # Gameplay before event one is action independent.  Include it once in
        # the episode return and metrics without dropping real Atari rewards.
        if event_index == 0:
            seller_delta += self._pending_pre_event_payoffs[0]
            buyer_delta += self._pending_pre_event_payoffs[1]

        self._done = self.core.done
        if self._done:
            if self.core.next_event != NUM_TRADE_EVENTS:
                raise RuntimeError("outer horizon ended before all five events")
            self.core.assert_accounting()
            observation = self._terminal_observation()
        else:
            if not self.core.at_event:
                raise RuntimeError("gameplay stopped without reaching an event")
            observation = self._trade_observation()
        reward = buyer_delta if self.controlled_role == BUYER else seller_delta
        return observation, float(reward), self._done, self._episode_info()

    def _terminal_observation(self):
        result = self._trade_observation_template()
        result["critic:state"] = self._critic_state()
        return result

    def _trade_observation_template(self):
        side = self._controlled_side()
        return OrderedDict([
            ("image", np.array(self.dummy_image, copy=True)),
            (
                "ammo_fraction",
                np.array([side.ammo / NUM_TRADE_EVENTS], dtype=np.float32),
            ),
            ("projectile_active", np.array([0.0], dtype=np.float32)),
            (
                "action_mask",
                np.ones(self.game_action_count, dtype=np.float32),
            ),
            ("offer_active", np.array([0.0], dtype=np.float32)),
            (
                "opportunities_remaining",
                np.array([1.0], dtype=np.float32),
            ),
            ("critic:price", np.array([0.0], dtype=np.float32)),
            ("event_active", np.array([0.0], dtype=np.float32)),
            (
                "event_one_hot",
                np.zeros(NUM_TRADE_EVENTS, dtype=np.float32),
            ),
            (
                "opponent_context",
                np.array(self.opponent_context, copy=True),
            ),
            (
                "critic:state",
                np.zeros(CRITIC_STATE_DIM, dtype=np.float32),
            ),
        ])

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")
        seller = self.core.seller.env.render(mode=mode)
        buyer = self.core.buyer.env.render(mode=mode)
        return np.concatenate([seller, buyer], axis=1)

    def close(self):
        self.core.close()
