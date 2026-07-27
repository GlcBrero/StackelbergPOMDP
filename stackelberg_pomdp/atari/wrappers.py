"""Composable preprocessing and scarce-ammunition wrappers for Atari."""

from collections import deque

import cv2
import gym
from gym import spaces
import numpy as np


cv2.ocl.setUseOpenCL(False)


class AmmoLedger:
    """Shared, explicit bullet inventory for gameplay and trade ledgers."""

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

    def without_fire(self, action):
        """Return the movement-only form of an Atari action."""

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
                and (
                    self.projectile_active()
                    or not self.ledger.available(1)
                )
        ):
            action = self.without_fire(action)
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
        # Match the historical DeepMind/StackeRLberg frame-buffer semantics.
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        observation = None
        for index in range(self.skip):
            # FIRE is one policy command, not four fire presses. Repeating the
            # movement component is safe; later raw frames remove FIRE.
            frame_action = action
            if index > 0 and hasattr(self.env, "without_fire"):
                frame_action = self.env.without_fire(
                    int(np.asarray(action).reshape(-1)[0])
                )
            observation, reward, done, frame_info = self.env.step(frame_action)
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


__all__ = [
    "AmmoLedger",
    "ClipGameRewardWrapper",
    "EpisodicLifeWrapper",
    "FrameStackWrapper",
    "MaxAndSkipWrapper",
    "NoopResetWrapper",
    "ScarceAmmoWrapper",
    "WarpFrameWrapper",
]
