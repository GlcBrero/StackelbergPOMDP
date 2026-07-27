"""Reusable single-player Atari side for the clean curriculum environments."""

import numpy as np

from stackelberg_pomdp.atari.core import SinglePlayerSpaceInvadersEnv
from stackelberg_pomdp.atari.wrappers import (
    AmmoLedger,
    ClipGameRewardWrapper,
    EpisodicLifeWrapper,
    FrameStackWrapper,
    MaxAndSkipWrapper,
    NoopResetWrapper,
    ScarceAmmoWrapper,
    WarpFrameWrapper,
)


class AtariGameplaySide:
    """One Space Invaders instance with outer-episode ammunition accounting.

    Episodic-life resets restart the frame stack but preserve the market ledger.
    Trade methods update only the ledger and never advance ALE.
    """

    def __init__(
            self,
            *,
            seed,
            initial_ammo,
            capacity=5,
            noop_max=30,
            frame_skip=4,
            frame_stack=4,
            episodic_life=True,
            clip_game_rewards=True,
            max_frames=100_000,
            rom_path=None,
            env_factory=SinglePlayerSpaceInvadersEnv,
    ):
        self.seed = int(seed)
        self.initial_ammo = int(initial_ammo)
        self.capacity = int(capacity)
        self.ledger = AmmoLedger(
            initial_ammo=self.initial_ammo,
            capacity=self.capacity,
        )
        env = env_factory(
            seed=self.seed,
            max_frames=int(max_frames),
            rom_path=rom_path,
        )
        env = NoopResetWrapper(env, noop_max=int(noop_max), seed=self.seed)
        if episodic_life:
            env = EpisodicLifeWrapper(env)
        if clip_game_rewards:
            env = ClipGameRewardWrapper(env)
        self.ammo_wrapper = ScarceAmmoWrapper(env, ledger=self.ledger)
        env = MaxAndSkipWrapper(self.ammo_wrapper, skip=int(frame_skip))
        env = WarpFrameWrapper(env)
        self.env = FrameStackWrapper(env, frames=int(frame_stack))
        self.game_action_count = int(self.env.action_space.n)
        self.image_space = self.env.observation_space
        self._observation = None
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0
        self.step_calls = 0

    def _require_observation(self):
        if self._observation is None:
            raise RuntimeError("gameplay side has not been reset")

    @property
    def observation(self):
        self._require_observation()
        return np.array(self._observation, copy=True)

    @property
    def image(self):
        return self.observation

    @property
    def action_mask(self):
        self._require_observation()
        return np.asarray(self.ammo_wrapper.action_mask(), dtype=np.float32)

    @property
    def projectile_active(self):
        return bool(self.ammo_wrapper.projectile_active())

    @property
    def ammo(self):
        return int(self.ledger.value)

    def reset(self):
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0
        self.step_calls = 0
        self._observation = np.array(self.env.reset(), copy=True)
        return self.observation

    def _reset_life_preserving_ammo(self):
        ammo = self.ammo
        self._observation = np.array(self.env.reset(), copy=True)
        self.ledger.value = ammo
        self.life_resets += 1

    def step(self, game_action):
        action = int(np.clip(
            np.rint(game_action), 0, self.game_action_count - 1
        ))
        observation, reward, done, info = self.env.step(action)
        self._observation = np.array(observation, copy=True)
        reward = float(reward)
        shots = int(info.get("shots_fired_this_step", 0))
        self.game_reward += reward
        self.shots_fired += shots
        self.step_calls += 1
        if done:
            self._reset_life_preserving_ammo()
        return reward, shots, dict(info)

    def grant(self, amount=1):
        return self.ledger.grant(int(amount))

    def consume(self, amount=1):
        self.ledger.consume(int(amount))

    def close(self):
        self.env.close()


__all__ = ["AtariGameplaySide"]
