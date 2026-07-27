"""Clean bilateral Atari market and full-trajectory E1 response environment."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import gym
import numpy as np

from stackelberg_pomdp.atari.gameplay import AtariGameplaySide
from stackelberg_pomdp.atari.protocol import (
    CRITIC_STATE_DIM,
    FOLLOWER_TRADE,
    GAMEPLAY,
    NUM_TRADE_EVENTS,
    TERMINAL,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.schedule import ExactFiveEventSchedule


SELLER = "seller"
BUYER = "buyer"
ROLES = {SELLER, BUYER}


@dataclass(frozen=True)
class BilateralAtariConfig:
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
        if int(self.num_trade_events) != NUM_TRADE_EVENTS:
            raise ValueError("the Atari protocol requires exactly five events")
        if float(self.seller_game_reward_scale) < 0.0:
            raise ValueError("seller_game_reward_scale must be nonnegative")
        if float(self.buyer_game_reward_scale) < 0.0:
            raise ValueError("buyer_game_reward_scale must be nonnegative")
        schedule = ExactFiveEventSchedule(
            gameplay_horizon=self.gameplay_horizon,
            tail_steps=self.event_tail_steps,
            fixed_event_steps=self.fixed_event_steps,
        )
        return replace(self, fixed_event_steps=schedule.fixed_event_steps)


class DualAtariTradeCore:
    """Two independent Atari games coupled only by atomic bullet trades."""

    def __init__(
            self,
            config,
            *,
            side_factory=AtariGameplaySide,
            env_factory=None,
    ):
        self.config = config.resolved()
        common = {
            "initial_ammo": 0,
            "capacity": NUM_TRADE_EVENTS,
            "noop_max": self.config.noop_max,
            "frame_skip": self.config.frame_skip,
            "frame_stack": self.config.frame_stack,
            "episodic_life": self.config.episodic_life,
            "clip_game_rewards": self.config.clip_game_rewards,
            "max_frames": self.config.max_frames,
            "rom_path": self.config.rom_path,
        }
        if env_factory is not None:
            common["env_factory"] = env_factory
        self.seller = side_factory(seed=self.config.seed, **common)
        self.buyer = side_factory(seed=self.config.seed + 100_003, **common)
        if self.seller.image_space != self.buyer.image_space:
            raise ValueError("seller and buyer image spaces must match")
        if self.seller.game_action_count != self.buyer.game_action_count:
            raise ValueError("seller and buyer Atari action spaces must match")
        self.image_space = self.seller.image_space
        self.game_action_count = self.seller.game_action_count
        self.rng = np.random.default_rng(self.config.seed + 31_337)
        self.schedule = ExactFiveEventSchedule(
            gameplay_horizon=self.config.gameplay_horizon,
            tail_steps=self.config.event_tail_steps,
            fixed_event_steps=self.config.fixed_event_steps,
        )
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

    def side(self, role):
        if role == SELLER:
            return self.seller
        if role == BUYER:
            return self.buyer
        raise ValueError(f"unknown Atari role: {role!r}")

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.seller.reset()
        self.buyer.reset()
        self.event_steps = self.schedule.sample(self.rng)
        self.game_step = 0
        self.next_event = 0
        self.bullets_arrived = 0
        self.transfers = 0
        self.payments = 0.0
        self.seller_payoff = 0.0
        self.buyer_payoff = 0.0
        self.events = []
        self._prepared_event_index = None
        if self.at_event:
            self.prepare_event()
        return self.event_steps

    @property
    def at_event(self):
        return bool(
            self.next_event < NUM_TRADE_EVENTS
            and self.game_step == self.event_steps[self.next_event]
        )

    @property
    def done(self):
        return self.game_step >= self.config.gameplay_horizon

    def prepare_event(self):
        """Grant the event bullet to the seller without advancing Atari."""

        if not self.at_event:
            raise RuntimeError("prepare_event called outside a scheduled event")
        if self._prepared_event_index == self.next_event:
            return
        granted = self.seller.grant(1)
        if granted != 1:
            raise RuntimeError("seller could not receive the exogenous bullet")
        self.bullets_arrived += 1
        self._prepared_event_index = self.next_event

    def step_gameplay(self, *, seller_action, buyer_action):
        """Advance each emulator by exactly one policy decision."""

        if self.done:
            raise RuntimeError("gameplay requested after the outer horizon")
        if self.at_event:
            raise RuntimeError("trade must resolve before gameplay advances")
        seller_reward, seller_shots, seller_info = self.seller.step(seller_action)
        buyer_reward, buyer_shots, buyer_info = self.buyer.step(buyer_action)
        seller_delta = self.config.seller_game_reward_scale * seller_reward
        buyer_delta = self.config.buyer_game_reward_scale * buyer_reward
        self.seller_payoff += seller_delta
        self.buyer_payoff += buyer_delta
        self.game_step += 1
        if self.at_event:
            self.prepare_event()
        return {
            "seller_game_action": int(np.rint(seller_action)),
            "buyer_game_action": int(np.rint(buyer_action)),
            "seller_game_reward": float(seller_reward),
            "buyer_game_reward": float(buyer_reward),
            "seller_reward_delta": float(seller_delta),
            "buyer_reward_delta": float(buyer_delta),
            "seller_shots_fired": int(seller_shots),
            "buyer_shots_fired": int(buyer_shots),
            "seller_info": dict(seller_info),
            "buyer_info": dict(buyer_info),
        }

    def trade(self, *, price, threshold):
        """Execute one paused offer and its immediate payment/transfer."""

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
            if self.buyer.grant(1) != 1:
                raise RuntimeError("accepted bullet could not reach the buyer")
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
            "seller_ammo_before": int(seller_before),
            "buyer_ammo_before": int(buyer_before),
            "seller_ammo_after": int(self.seller.ammo),
            "buyer_ammo_after": int(self.buyer.ammo),
        }
        self.events.append(event)
        self.next_event += 1
        self._prepared_event_index = None
        return event

    def accounting(self):
        return {
            "seller_bullet_error": int(
                self.bullets_arrived
                - self.seller.shots_fired
                - self.transfers
                - self.seller.ammo
            ),
            "buyer_bullet_error": int(
                self.transfers - self.buyer.shots_fired - self.buyer.ammo
            ),
            "seller_payoff_error": float(
                self.seller_payoff
                - (
                    self.config.seller_game_reward_scale * self.seller.game_reward
                    + self.payments
                )
            ),
            "buyer_payoff_error": float(
                self.buyer_payoff
                - (
                    self.config.buyer_game_reward_scale * self.buyer.game_reward
                    - self.payments
                )
            ),
        }

    def assert_accounting(self):
        values = self.accounting()
        if values["seller_bullet_error"] != 0:
            raise RuntimeError(f"seller bullet accounting failed: {values}")
        if values["buyer_bullet_error"] != 0:
            raise RuntimeError(f"buyer bullet accounting failed: {values}")
        if abs(values["seller_payoff_error"]) > 1.0e-6:
            raise RuntimeError(f"seller payoff accounting failed: {values}")
        if abs(values["buyer_payoff_error"]) > 1.0e-6:
            raise RuntimeError(f"buyer payoff accounting failed: {values}")

    def close(self):
        self.seller.close()
        self.buyer.close()


class FrozenAtariPolicyController:
    """Deterministic clean-policy controller for the non-learning Atari side."""

    def __init__(self, checkpoint, *, device="cpu"):
        from stable_baselines3 import PPO

        path = Path(checkpoint).expanduser()
        if not path.is_file() and Path(f"{path}.zip").is_file():
            path = Path(f"{path}.zip")
        if not path.is_file():
            raise FileNotFoundError(f"Atari checkpoint does not exist: {path}")
        self.model = PPO.load(str(path), device=device)
        self.model.policy.set_training_mode(False)
        for parameter in self.model.policy.parameters():
            parameter.requires_grad = False

    def __call__(self, values):
        action, _ = self.model.predict(values, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (2,):
            raise RuntimeError("clean Atari controller must return two actions")
        return int(np.rint(action[0]))


class MetaAtariResponseEnv(gym.Env):
    """Full E1 trajectory for a buyer or seller meta-response.

    A random opponent commitment ``omega in [0,1]^5`` is sampled at reset.
    The controlled policy acts at all 200 gameplay steps and five paused trade
    steps, enabling joint gameplay/economic fine-tuning.  The other Atari side
    uses a deterministic E0b controller and never supplies an economic choice;
    its five economic actions are exactly the sampled commitment.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            *,
            controlled_role,
            e0b_checkpoint=None,
            config=None,
            context_sampler: Optional[Callable] = None,
            core_factory=DualAtariTradeCore,
            controller_factory=None,
            side_factory=AtariGameplaySide,
            env_factory=None,
            device="cpu",
    ):
        super().__init__()
        if controlled_role not in ROLES:
            raise ValueError(f"controlled_role must be one of {sorted(ROLES)}")
        self.controlled_role = controlled_role
        self.other_role = BUYER if controlled_role == SELLER else SELLER
        self.config = (config or BilateralAtariConfig()).resolved()
        self.context_sampler = context_sampler
        self.rng = np.random.default_rng(self.config.seed + 74_711)
        self.core = core_factory(
            self.config,
            side_factory=side_factory,
            env_factory=env_factory,
        )
        if controller_factory is None:
            if e0b_checkpoint is None:
                raise ValueError("e0b_checkpoint or controller_factory is required")
            self.other_controller = FrozenAtariPolicyController(
                e0b_checkpoint, device=device
            )
        else:
            self.other_controller = controller_factory()
        self.action_space = action_space(self.core.game_action_count)
        self.observation_space = observation_space(
            self.core.image_space, self.core.game_action_count
        )
        self.dummy_image = np.zeros(
            self.core.image_space.shape, dtype=self.core.image_space.dtype
        )
        self.opponent_commitment = np.zeros(
            NUM_TRADE_EVENTS, dtype=np.float32
        )
        self._done = False
        self.gameplay_transitions = 0
        self.trade_transitions = 0

    def seed(self, seed=None):
        seed = self.config.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(seed + 74_711)
        return [seed]

    def _sample_context(self):
        values = (
            self.rng.uniform(0.0, 1.0, size=NUM_TRADE_EVENTS)
            if self.context_sampler is None
            else self.context_sampler(self.rng)
        )
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape != (NUM_TRADE_EVENTS,):
            raise ValueError("context sampler must return exactly five scalars")
        return np.clip(values, 0.0, 1.0)

    def _critic_state(self):
        core = self.core
        values = np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
        values[:15] = (
            float(core.game_step) / self.config.gameplay_horizon,
            float(core.next_event) / NUM_TRADE_EVENTS,
            float(core.at_event),
            float(core.seller.ammo) / NUM_TRADE_EVENTS,
            float(core.buyer.ammo) / NUM_TRADE_EVENTS,
            float(core.transfers) / NUM_TRADE_EVENTS,
            float(core.payments),
            float(core.seller.game_reward),
            float(core.buyer.game_reward),
            float(core.seller_payoff),
            float(core.buyer_payoff),
            float(core.bullets_arrived) / NUM_TRADE_EVENTS,
            float(self.controlled_role == SELLER),
            float(self.gameplay_transitions) / self.config.gameplay_horizon,
            float(self.trade_transitions) / NUM_TRADE_EVENTS,
        )
        values[15:20] = (
            np.asarray(core.event_steps, dtype=np.float32)
            / self.config.gameplay_horizon
        )
        values[20:25] = self.opponent_commitment
        return values

    def _side_observation(self, role, *, controlled, decision_kind):
        side = self.core.side(role)
        trade_mode = self.core.at_event and not self._done
        event_index = (
            self.core.next_event
            if self.core.next_event < NUM_TRADE_EVENTS
            else None
        )
        context = (
            self.opponent_commitment
            if controlled
            else np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        )
        return observation(
            image=self.dummy_image if trade_mode else side.image,
            state=actor_state(
                ammo_fraction=float(side.ammo) / NUM_TRADE_EVENTS,
                projectile_active=float(side.projectile_active),
                normalized_time=float(self.core.game_step)
                / self.config.gameplay_horizon,
                trade_mode=float(trade_mode),
                event_index=event_index,
                opponent_commitment=context,
            ),
            action_mask=side.action_mask,
            decision_kind=decision_kind,
            critic_state=self._critic_state(),
        )

    def _controlled_observation(self):
        if self._done:
            kind = TERMINAL
        else:
            kind = FOLLOWER_TRADE if self.core.at_event else GAMEPLAY
        return self._side_observation(
            self.controlled_role, controlled=True, decision_kind=kind
        )

    def _other_game_action(self):
        values = self._side_observation(
            self.other_role, controlled=False, decision_kind=GAMEPLAY
        )
        result = self.other_controller(values)
        return int(np.clip(np.rint(result), 0, self.core.game_action_count - 1))

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.seed(seed)
        self.opponent_commitment = self._sample_context()
        self.core.reset(seed=int(self.rng.integers(0, 2 ** 31 - 1)))
        self._done = False
        self.gameplay_transitions = 0
        self.trade_transitions = 0
        return self._controlled_observation()

    @staticmethod
    def _validated_action(action, game_action_count):
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (2,):
            raise ValueError("action must be [game_action, economic_action]")
        return np.array([
            np.clip(values[0], 0.0, game_action_count - 1),
            np.clip(values[1], 0.0, 1.0),
        ], dtype=np.float32)

    def _controlled_payoff(self):
        return float(
            self.core.seller_payoff
            if self.controlled_role == SELLER
            else self.core.buyer_payoff
        )

    def _episode_info(self):
        core = self.core
        return {
            "controlled_role": self.controlled_role,
            "opponent_commitment": tuple(
                float(value) for value in self.opponent_commitment
            ),
            "event_steps": tuple(int(value) for value in core.event_steps),
            "events": tuple(dict(event) for event in core.events),
            "gameplay_transitions": int(self.gameplay_transitions),
            "trade_transitions": int(self.trade_transitions),
            "outer_transition_count": int(
                self.gameplay_transitions + self.trade_transitions
            ),
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
            **core.accounting(),
        }

    def step(self, action):
        if self._done:
            raise RuntimeError("step called after E1 outer episode termination")
        values = self._validated_action(action, self.core.game_action_count)
        previous = self._controlled_payoff()
        if self.core.at_event:
            event_index = self.core.next_event
            opponent = float(self.opponent_commitment[event_index])
            controlled_economic = float(values[1])
            if self.controlled_role == BUYER:
                price, threshold = opponent, controlled_economic
            else:
                price, threshold = controlled_economic, opponent
            seller_calls = self.core.seller.step_calls
            buyer_calls = self.core.buyer.step_calls
            event = self.core.trade(price=price, threshold=threshold)
            if (
                    seller_calls != self.core.seller.step_calls
                    or buyer_calls != self.core.buyer.step_calls
            ):
                raise RuntimeError("an Atari emulator advanced during trade")
            self.trade_transitions += 1
            substep = FOLLOWER_TRADE
            detail = {"trade_event": dict(event), "emulator_advanced": False}
        else:
            other_action = self._other_game_action()
            if self.controlled_role == SELLER:
                seller_action, buyer_action = values[0], other_action
            else:
                seller_action, buyer_action = other_action, values[0]
            transition = self.core.step_gameplay(
                seller_action=seller_action,
                buyer_action=buyer_action,
            )
            self.gameplay_transitions += 1
            substep = GAMEPLAY
            detail = {"gameplay": transition, "emulator_advanced": True}

        reward = self._controlled_payoff() - previous
        self._done = bool(self.core.done)
        if self._done:
            if self.core.next_event != NUM_TRADE_EVENTS:
                raise RuntimeError("E1 horizon ended before all five events")
            self.core.assert_accounting()
        info = {
            "substep_type": substep,
            "controlled_reward_delta": float(reward),
            "game_step": int(self.core.game_step),
            "next_event": int(self.core.next_event),
            **detail,
        }
        if self._done:
            episode = self._episode_info()
            info.update(episode)
            info["episode"] = {
                "r": self._controlled_payoff(),
                "l": episode["outer_transition_count"],
                **episode,
            }
        return self._controlled_observation(), float(reward), self._done, info

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")
        seller = self.core.seller.env.render(mode=mode)
        buyer = self.core.buyer.env.render(mode=mode)
        return np.concatenate([seller, buyer], axis=1)

    def close(self):
        self.core.close()


__all__ = [
    "BUYER",
    "BilateralAtariConfig",
    "DualAtariTradeCore",
    "FrozenAtariPolicyController",
    "MetaAtariResponseEnv",
    "ROLES",
    "SELLER",
]
