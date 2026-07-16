"""Minimal single-player Space Invaders ALE environment.

This module intentionally contains no pricing, ammunition, preprocessing,
RLlib, or policy code.  Those concerns are composed around the environment in
``stackelberg_pomdp.atari.wrappers``.
"""

from pathlib import Path
import os

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


ACTION_MEANINGS = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}


def default_rom_path():
    configured = os.environ.get("STACKPOMDP_SPACE_INVADERS_ROM")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(
            f"STACKPOMDP_SPACE_INVADERS_ROM does not exist: {path}"
        )

    vendored = Path(__file__).resolve().parent / "roms" / "space_invaders.bin"
    if vendored.is_file():
        return vendored

    raise FileNotFoundError(
        "Space Invaders ROM is missing. Expected "
        f"{vendored} or STACKPOMDP_SPACE_INVADERS_ROM."
    )


class SinglePlayerSpaceInvadersEnv(gym.Env):
    """Raw single-player modified Space Invaders ROM through ALE."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
            self,
            *,
            seed=1,
            rom_path=None,
            repeat_action_probability=0.0,
            max_frames=100_000,
    ):
        super().__init__()
        import multi_agent_ale_py

        multi_agent_ale_py.ALEInterface.setLoggerMode("error")
        self.ale = multi_agent_ale_py.ALEInterface()
        self.ale.setFloat(
            b"repeat_action_probability", float(repeat_action_probability)
        )
        self.rom_path = str(Path(rom_path or default_rom_path()).resolve())
        self.max_frames = int(max_frames)
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")

        self.ale.loadROM(self.rom_path)
        modes = self.ale.getAvailableModes(1)
        if len(modes) == 0:
            raise IOError(f"ROM {self.rom_path} has no single-player mode")
        self.mode = int(modes[0])
        self.ale.setMode(self.mode)
        if self.ale.numPlayersActive() != 1:
            raise RuntimeError("ALE failed to enter single-player mode")

        self.action_mapping = np.asarray(
            self.ale.getMinimalActionSet(), dtype=np.int32
        )
        self.action_space = spaces.Discrete(len(self.action_mapping))
        width, height = self.ale.getScreenDims()
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )
        self._screen = None
        self.frame = 0
        self.seed(seed)

    def seed(self, seed=None):
        if seed is None:
            seed = seeding.create_seed(seed, max_bytes=4)
        seed = int(seed)
        self.ale.setInt(b"random_seed", seed)
        self.ale.loadROM(self.rom_path)
        self.ale.setMode(self.mode)
        return [seed]

    def get_action_meanings(self):
        return [ACTION_MEANINGS[int(action)] for action in self.action_mapping]

    def reset(self):
        self.ale.reset_game()
        self.frame = 0
        return self.ale.getScreenRGB()

    def step(self, action):
        action = int(np.asarray(action).reshape(-1)[0])
        if not self.action_space.contains(action):
            raise ValueError(f"invalid Atari action index: {action}")
        reward = float(self.ale.act(np.asarray([self.action_mapping[action]])))
        self.frame += 1
        done = bool(self.ale.game_over() or self.frame >= self.max_frames)
        observation = self.ale.getScreenRGB()
        lives = self.ale.allLives()
        info = {"ale.lives": int(lives[0]) if len(lives) else -1}
        return observation, reward, done, info

    def render(self, mode="human", zoom_factor=4):
        if mode not in self.metadata["render.modes"]:
            raise ValueError(f"unsupported render mode: {mode}")
        image = self.ale.getScreenRGB()
        if mode == "rgb_array":
            return image

        import pygame

        width, height = self.ale.getScreenDims()
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode(
                (width * zoom_factor, height * zoom_factor)
            )
        surface = pygame.image.fromstring(
            image.tobytes(), image.shape[:2][::-1], "RGB"
        )
        surface = pygame.transform.scale(
            surface, (width * zoom_factor, height * zoom_factor)
        )
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()

    def close(self):
        if self._screen is not None:
            import pygame

            pygame.quit()
            self._screen = None
