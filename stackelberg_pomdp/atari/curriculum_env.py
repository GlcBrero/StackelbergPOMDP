"""Clean E0a/E0b Atari pretraining environments."""

from dataclasses import dataclass

import gym
import numpy as np

from stackelberg_pomdp.atari.gameplay import AtariGameplaySide
from stackelberg_pomdp.atari.protocol import (
    AUTOMATIC_TRANSFER,
    CRITIC_STATE_DIM,
    GAMEPLAY,
    NUM_TRADE_EVENTS,
    TERMINAL,
    action_space,
    actor_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.schedule import ExactFiveEventSchedule


E0_STAGES = {"e0a", "e0b"}


@dataclass(frozen=True)
class AtariCurriculumConfig:
    stage: str = "e0a"
    seed: int = 1
    gameplay_horizon: int = 200
    event_tail_steps: int = 50
    noop_max: int = 30
    frame_skip: int = 4
    frame_stack: int = 4
    episodic_life: bool = True
    clip_game_rewards: bool = True
    max_frames: int = 100_000
    rom_path: str = None
    fixed_event_steps: tuple = None

    def validate(self):
        if self.stage not in E0_STAGES:
            raise ValueError(f"stage must be one of {sorted(E0_STAGES)}")
        if int(self.gameplay_horizon) <= 0:
            raise ValueError("gameplay_horizon must be positive")
        return self


class AtariCurriculumEnv(gym.Env):
    """Single-player E0 with a stable final actor interface.

    E0a starts with five bullets and contains only gameplay transitions.  E0b
    starts empty and inserts five paused, randomly timed, automatic free-bullet
    transfers.  E0b transfer actions are ignored and receive no actor credit.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            config=None,
            *,
            side_factory=AtariGameplaySide,
            env_factory=None,
    ):
        super().__init__()
        self.config = (config or AtariCurriculumConfig()).validate()
        self.rng = np.random.default_rng(self.config.seed + 41_009)
        side_kwargs = {}
        if env_factory is not None:
            side_kwargs["env_factory"] = env_factory
        self.side = side_factory(
            seed=self.config.seed,
            initial_ammo=NUM_TRADE_EVENTS if self.config.stage == "e0a" else 0,
            capacity=NUM_TRADE_EVENTS,
            noop_max=self.config.noop_max,
            frame_skip=self.config.frame_skip,
            frame_stack=self.config.frame_stack,
            episodic_life=self.config.episodic_life,
            clip_game_rewards=self.config.clip_game_rewards,
            max_frames=self.config.max_frames,
            rom_path=self.config.rom_path,
            **side_kwargs,
        )
        self.action_space = action_space(self.side.game_action_count)
        self.observation_space = observation_space(
            self.side.image_space, self.side.game_action_count
        )
        self.dummy_image = np.zeros(
            self.side.image_space.shape, dtype=self.side.image_space.dtype
        )
        self.schedule = (
            ExactFiveEventSchedule(
                gameplay_horizon=self.config.gameplay_horizon,
                tail_steps=self.config.event_tail_steps,
                fixed_event_steps=self.config.fixed_event_steps,
            )
            if self.config.stage == "e0b"
            else None
        )
        self.game_step = 0
        self.next_event = 0
        self.event_steps = ()
        self.free_transfers = 0
        self._done = False

    @property
    def at_event(self):
        return bool(
            self.config.stage == "e0b"
            and self.next_event < NUM_TRADE_EVENTS
            and self.game_step == self.event_steps[self.next_event]
        )

    def _event_index(self):
        if self.config.stage == "e0a" or self.next_event >= NUM_TRADE_EVENTS:
            return None
        return self.next_event

    def _critic_state(self):
        values = np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
        values[:8] = (
            float(self.config.stage == "e0b"),
            float(self.game_step) / float(self.config.gameplay_horizon),
            float(self.next_event) / NUM_TRADE_EVENTS,
            float(self.at_event),
            float(self.side.ammo) / NUM_TRADE_EVENTS,
            float(self.side.shots_fired) / NUM_TRADE_EVENTS,
            float(self.side.game_reward) / NUM_TRADE_EVENTS,
            float(self.free_transfers) / NUM_TRADE_EVENTS,
        )
        if self.event_steps:
            values[8:13] = (
                np.asarray(self.event_steps, dtype=np.float32)
                / float(self.config.gameplay_horizon)
            )
        return values

    def _observation(self):
        if self._done:
            kind = TERMINAL
            trade_mode = False
        else:
            trade_mode = self.at_event
            kind = AUTOMATIC_TRANSFER if trade_mode else GAMEPLAY
        state = actor_state(
            ammo_fraction=float(self.side.ammo) / NUM_TRADE_EVENTS,
            projectile_active=float(self.side.projectile_active),
            normalized_time=float(self.game_step) / self.config.gameplay_horizon,
            trade_mode=float(trade_mode),
            event_index=self._event_index(),
            opponent_commitment=np.zeros(NUM_TRADE_EVENTS, dtype=np.float32),
        )
        return observation(
            image=self.dummy_image if trade_mode else self.side.image,
            state=state,
            action_mask=self.side.action_mask,
            decision_kind=kind,
            critic_state=self._critic_state(),
        )

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(int(seed) + 41_009)
        self.side.reset()
        self.game_step = 0
        self.next_event = 0
        self.free_transfers = 0
        self._done = False
        self.event_steps = (
            self.schedule.sample(self.rng) if self.schedule is not None else ()
        )
        return self._observation()

    @staticmethod
    def _validated_action(action, game_action_count):
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (2,):
            raise ValueError("Atari action must be [game_action, economic]")
        return np.array([
            np.clip(values[0], 0.0, game_action_count - 1),
            np.clip(values[1], 0.0, 1.0),
        ], dtype=np.float32)

    def _episode_info(self):
        expected = (
            NUM_TRADE_EVENTS
            if self.config.stage == "e0a"
            else self.free_transfers
        )
        bullet_error = expected - self.side.shots_fired - self.side.ammo
        if bullet_error != 0:
            raise RuntimeError(
                f"E0 bullet accounting failed with error {bullet_error}"
            )
        return {
            "stage": self.config.stage,
            "gameplay_steps": int(self.game_step),
            "trade_transitions": int(self.free_transfers),
            "outer_transition_count": int(
                self.game_step + self.free_transfers
            ),
            "event_steps": tuple(int(value) for value in self.event_steps),
            "game_reward": float(self.side.game_reward),
            "payments": 0.0,
            "episode_reward": float(self.side.game_reward),
            "shots_fired": int(self.side.shots_fired),
            "free_transfers": int(self.free_transfers),
            "final_ammo": int(self.side.ammo),
            "bullet_accounting_error": int(bullet_error),
            "life_resets": int(self.side.life_resets),
        }

    def step(self, action):
        if self._done:
            raise RuntimeError("step called after E0 outer episode termination")
        values = self._validated_action(action, self.side.game_action_count)
        if self.at_event:
            calls_before = self.side.step_calls
            granted = self.side.grant(1)
            if granted != 1:
                raise RuntimeError("automatic E0b transfer could not grant a bullet")
            completed_event = self.next_event
            self.next_event += 1
            self.free_transfers += 1
            if self.side.step_calls != calls_before:
                raise RuntimeError("Atari advanced during a paused E0b transfer")
            info = {
                "substep_type": AUTOMATIC_TRANSFER,
                "event_index": int(completed_event),
                "emulator_advanced": False,
                "automatic_transfer": True,
                "payment": 0.0,
                "game_reward": 0.0,
            }
            return self._observation(), 0.0, False, info

        reward, shots, gameplay_info = self.side.step(values[0])
        self.game_step += 1
        self._done = self.game_step >= self.config.gameplay_horizon
        info = dict(gameplay_info)
        info.update({
            "substep_type": GAMEPLAY,
            "emulator_advanced": True,
            "game_reward": float(reward),
            "payment": 0.0,
            "shots_fired_this_step": int(shots),
        })
        if self._done:
            if self.config.stage == "e0b" and self.next_event != NUM_TRADE_EVENTS:
                raise RuntimeError("E0b horizon ended before all five transfers")
            episode = self._episode_info()
            info.update(episode)
            info["episode"] = {
                "r": episode["episode_reward"],
                "l": episode["outer_transition_count"],
                **episode,
            }
        return self._observation(), float(reward), self._done, info

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")
        return self.side.env.render(mode=mode)

    def close(self):
        self.side.close()


__all__ = [
    "AtariCurriculumConfig",
    "AtariCurriculumEnv",
    "E0_STAGES",
]
