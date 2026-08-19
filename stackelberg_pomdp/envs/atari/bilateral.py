"""Atari-specific bilateral bullet-trade base environment."""

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from stackelberg_pomdp.envs.atari.gameplay import AtariGameplaySide
from stackelberg_pomdp.atari.protocol import (
    FOLLOWER_TRADE,
    GAMEPLAY,
    NUM_TRADE_EVENTS,
    TERMINAL,
    action_space,
    actor_state,
    observation,
    observation_space,
    validate_action,
)
from stackelberg_pomdp.atari.sampling import ExactFiveEventSchedule
from stackelberg_pomdp.envs.base import BaseEnv


SELLER = "seller"
BUYER = "buyer"
ROLES = {SELLER, BUYER}


@dataclass(frozen=True)
class BilateralAtariConfig:
    seed: int = 1
    gameplay_horizon: int = 200
    event_tail_steps: int = 0
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

    def reset(self, *, seed=None, event_steps=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.seller.reset()
        self.buyer.reset()
        if event_steps is None:
            self.event_steps = self.schedule.sample(self.rng)
        else:
            self.event_steps = ExactFiveEventSchedule(
                gameplay_horizon=self.config.gameplay_horizon,
                tail_steps=self.config.event_tail_steps,
                fixed_event_steps=event_steps,
            ).sample(self.rng)
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
            "seller_emulator_advanced": bool(
                seller_info.get("emulator_advanced", True)
            ),
            "buyer_emulator_advanced": bool(
                buyer_info.get("emulator_advanced", True)
            ),
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
            "seller_true_game_over_resets": int(
                self.seller.true_game_over_resets
            ),
            "buyer_true_game_over_resets": int(
                self.buyer.true_game_over_resets
            ),
            "seller_time_limit_resets": int(self.seller.time_limit_resets),
            "buyer_time_limit_resets": int(self.buyer.time_limit_resets),
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


class BilateralAtariRewardEnv(BaseEnv):
    """Two-player Atari base game shared by response and leader training.

    The environment owns all emulator, trade, payoff, and accounting state.
    Like the other StackPOMDP base environments, ``step`` accepts the complete
    leader/follower action mapping, returns the leader's scalar reward, and
    exposes every player's reward through ``info["utilities"]``.  Response and
    StackPOMDP phase logic belong to wrappers above this class.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            *,
            leader_role,
            config=None,
            core_factory=DualAtariTradeCore,
            side_factory=None,
            env_factory=None,
    ):
        if leader_role not in ROLES:
            raise ValueError(f"leader_role must be one of {sorted(ROLES)}")
        follower_role = BUYER if leader_role == SELLER else SELLER
        self.config = (config or BilateralAtariConfig()).resolved()
        super().__init__(
            leader=leader_role,
            followers_list=[follower_role],
            seed=self.config.seed,
        )
        self.leader_role = leader_role
        self.follower_role = follower_role

        core_kwargs = {}
        if side_factory is not None:
            core_kwargs["side_factory"] = side_factory
        if env_factory is not None:
            core_kwargs["env_factory"] = env_factory
        self.core = core_factory(self.config, **core_kwargs)

        role_action_space = action_space(self.core.game_action_count)
        role_observation_space = observation_space(
            self.core.image_space, self.core.game_action_count
        )
        self.action_space = role_action_space
        self.observation_space = role_observation_space
        self.followers_action_space = {follower_role: role_action_space}
        self.followers_observation_space = {
            follower_role: role_observation_space
        }
        self.dummy_image = np.zeros(
            self.core.image_space.shape, dtype=self.core.image_space.dtype
        )
        self._done = False
        self.gameplay_transitions = 0
        self.trade_transitions = 0

    def _validated_joint_action(self, actions):
        if not isinstance(actions, Mapping) or set(actions) != ROLES:
            raise ValueError("joint Atari action must contain seller and buyer")
        return {
            role: validate_action(actions[role], self.core.game_action_count)
            for role in (SELLER, BUYER)
        }

    def role_observation(
            self,
            role,
            *,
            opponent_commitment=None,
            decision_kind=None,
            critic_state=None,
    ):
        """Return the current canonical observation for one Atari role."""

        side = self.core.side(role)
        trade_mode = self.core.at_event and not self._done
        event_index = (
            self.core.next_event
            if self.core.next_event < NUM_TRADE_EVENTS
            else None
        )
        if decision_kind is None:
            decision_kind = (
                TERMINAL
                if self._done
                else FOLLOWER_TRADE
                if trade_mode
                else GAMEPLAY
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
                opponent_commitment=opponent_commitment,
            ),
            action_mask=side.action_mask,
            decision_kind=decision_kind,
            critic_state=critic_state,
        )

    def _observations(self):
        return OrderedDict(
            (role, self.role_observation(role)) for role in (SELLER, BUYER)
        )

    def leader_observation(self, follower_actions=None):
        del follower_actions
        return self.role_observation(self.leader_role)

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else dict(options)
        unknown = set(options) - {"event_steps"}
        if unknown:
            raise ValueError(
                f"unknown bilateral Atari reset options: {sorted(unknown)}"
            )
        if "event_steps" in options:
            self.core.reset(seed=seed, event_steps=options["event_steps"])
        else:
            self.core.reset(seed=seed)
        self._done = False
        self.gameplay_transitions = 0
        self.trade_transitions = 0
        return self._observations()

    def role_payoff(self, role):
        if role == SELLER:
            return float(self.core.seller_payoff)
        if role == BUYER:
            return float(self.core.buyer_payoff)
        raise ValueError(f"unknown Atari role: {role!r}")

    def leader_reward(self):
        return self.role_payoff(self.leader_role)

    def episode_info(self):
        core = self.core
        final_event_step = int(core.event_steps[-1])
        seller_before_fifth = bool(any(
            step <= final_event_step
            for step in core.seller.true_game_over_reset_steps
        ))
        buyer_before_fifth = bool(any(
            step <= final_event_step
            for step in core.buyer.true_game_over_reset_steps
        ))
        return {
            "leader_role": self.leader_role,
            "follower_role": self.follower_role,
            "event_steps": tuple(int(value) for value in core.event_steps),
            "events": tuple(dict(event) for event in core.events),
            "gameplay_transitions": int(self.gameplay_transitions),
            "trade_transitions": int(self.trade_transitions),
            "reward_transition_count": int(
                self.gameplay_transitions + self.trade_transitions
            ),
            "bullets_arrived": int(core.bullets_arrived),
            "purchases": int(core.transfers),
            "payments": float(core.payments),
            "seller_game_reward": float(core.seller.game_reward),
            "buyer_game_reward": float(core.buyer.game_reward),
            "seller_reward": float(core.seller_payoff),
            "buyer_reward": float(core.buyer_payoff),
            "leader_reward": self.leader_reward(),
            "seller_shots_fired": int(core.seller.shots_fired),
            "buyer_shots_fired": int(core.buyer.shots_fired),
            "seller_final_ammo": int(core.seller.ammo),
            "buyer_final_ammo": int(core.buyer.ammo),
            "seller_emulator_step_calls": int(core.seller.step_calls),
            "buyer_emulator_step_calls": int(core.buyer.step_calls),
            "seller_life_resets": int(core.seller.life_resets),
            "buyer_life_resets": int(core.buyer.life_resets),
            "seller_real_terminal_resets": int(
                core.seller.real_terminal_resets
            ),
            "buyer_real_terminal_resets": int(core.buyer.real_terminal_resets),
            "seller_real_terminal_reset_steps": tuple(
                int(step) for step in core.seller.real_terminal_reset_steps
            ),
            "buyer_real_terminal_reset_steps": tuple(
                int(step) for step in core.buyer.real_terminal_reset_steps
            ),
            "seller_true_game_over_resets": int(
                core.seller.true_game_over_resets
            ),
            "buyer_true_game_over_resets": int(
                core.buyer.true_game_over_resets
            ),
            "seller_true_game_over_reset_rate": float(
                core.seller.true_game_over_resets
                / max(core.seller.step_calls, 1)
            ),
            "buyer_true_game_over_reset_rate": float(
                core.buyer.true_game_over_resets
                / max(core.buyer.step_calls, 1)
            ),
            "seller_true_game_over_reset_steps": tuple(
                int(step) for step in core.seller.true_game_over_reset_steps
            ),
            "buyer_true_game_over_reset_steps": tuple(
                int(step) for step in core.buyer.true_game_over_reset_steps
            ),
            "seller_time_limit_resets": int(core.seller.time_limit_resets),
            "buyer_time_limit_resets": int(core.buyer.time_limit_resets),
            "seller_time_limit_reset_steps": tuple(
                int(step) for step in core.seller.time_limit_reset_steps
            ),
            "buyer_time_limit_reset_steps": tuple(
                int(step) for step in core.buyer.time_limit_reset_steps
            ),
            "seller_true_game_over_before_fifth_event": seller_before_fifth,
            "buyer_true_game_over_before_fifth_event": buyer_before_fifth,
            "any_true_game_over_before_fifth_event": bool(
                seller_before_fifth or buyer_before_fifth
            ),
            **core.accounting(),
        }

    def step(self, actions):
        if self._done:
            raise RuntimeError("reward-game step called after Atari termination")
        values = self._validated_joint_action(actions)
        previous = {
            role: self.role_payoff(role) for role in (SELLER, BUYER)
        }
        if self.core.at_event:
            seller_calls = self.core.seller.step_calls
            buyer_calls = self.core.buyer.step_calls
            event = self.core.trade(
                price=float(values[SELLER][1]),
                threshold=float(values[BUYER][1]),
            )
            if (
                    seller_calls != self.core.seller.step_calls
                    or buyer_calls != self.core.buyer.step_calls
            ):
                raise RuntimeError("an Atari emulator advanced during trade")
            self.trade_transitions += 1
            detail = {
                "substep_type": "trade",
                "trade_event": dict(event),
                "emulator_advanced": False,
            }
        else:
            transition = self.core.step_gameplay(
                seller_action=values[SELLER][0],
                buyer_action=values[BUYER][0],
            )
            self.gameplay_transitions += 1
            both_emulators_advanced = bool(
                transition["seller_emulator_advanced"]
                and transition["buyer_emulator_advanced"]
            )
            detail = {
                "substep_type": GAMEPLAY,
                "gameplay": transition,
                "seller_emulator_advanced": bool(
                    transition["seller_emulator_advanced"]
                ),
                "buyer_emulator_advanced": bool(
                    transition["buyer_emulator_advanced"]
                ),
                "emulator_advanced": both_emulators_advanced,
            }

        rewards = {
            role: float(self.role_payoff(role) - previous[role])
            for role in (SELLER, BUYER)
        }
        self._done = bool(self.core.done)
        if self._done:
            if self.core.next_event != NUM_TRADE_EVENTS:
                raise RuntimeError("Atari horizon ended before all five trades")
            self.core.assert_accounting()
        leader_reward = float(rewards[self.leader_role])
        info = {
            **detail,
            "reward_generated": True,
            "utilities": dict(rewards),
            "surplus": leader_reward,
            "game_step": int(self.core.game_step),
            "next_event": int(self.core.next_event),
        }
        if self._done:
            info.update(self.episode_info())
        return self._observations(), leader_reward, self._done, info

    def reward_phase_length(self, default_length):
        del default_length
        return int(self.config.gameplay_horizon + NUM_TRADE_EVENTS)

    def max_reward_phase_length(self, default_length):
        return self.reward_phase_length(default_length)

    def max_subepisode_transitions(self):
        return 1

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
    "BilateralAtariRewardEnv",
    "DualAtariTradeCore",
    "ROLES",
    "SELLER",
]
