"""Small deterministic Atari stand-ins; no emulator or ROM required."""
from types import SimpleNamespace
import gym
import numpy as np
from experiments.atari.game import BilateralAtariConfig

FIRE = 1


class _FakeAmmoWrapper:
    fire_action_indices = (FIRE,)


class _FakeSide:
    def __init__(self, *, seed, initial_ammo, capacity, **kwargs):
        del seed, kwargs
        self.initial_ammo = int(initial_ammo)
        self.capacity = int(capacity)
        self.game_action_count = 6
        self.image_space = gym.spaces.Box(
            0, 255, shape=(84, 84, 4), dtype=np.uint8
        )
        self.ammo_wrapper = _FakeAmmoWrapper()
        self._ammo = self.initial_ammo
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0
        self.step_calls = 0
        self.real_terminal_resets = 0
        self.real_terminal_reset_steps = []
        self.true_game_over_resets = 0
        self.true_game_over_reset_steps = []
        self.time_limit_resets = 0
        self.time_limit_reset_steps = []
        self.env = SimpleNamespace(
            render=lambda mode="rgb_array": np.zeros(
                (84, 84, 3), dtype=np.uint8
            )
        )

    @property
    def ammo(self):
        return self._ammo

    @property
    def projectile_active(self):
        return False

    @property
    def image(self):
        return np.full(
            self.image_space.shape,
            min(255, self.step_calls + 10 * self._ammo),
            dtype=np.uint8,
        )

    @property
    def action_mask(self):
        result = np.ones(self.game_action_count, dtype=np.float32)
        if self._ammo == 0:
            result[FIRE] = 0.0
        return result

    def reset(self):
        self._ammo = self.initial_ammo
        self.game_reward = 0.0
        self.shots_fired = 0
        self.life_resets = 0
        self.step_calls = 0
        self.real_terminal_resets = 0
        self.real_terminal_reset_steps = []
        self.true_game_over_resets = 0
        self.true_game_over_reset_steps = []
        self.time_limit_resets = 0
        self.time_limit_reset_steps = []

    def grant(self, amount=1):
        amount = min(int(amount), self.capacity - self._ammo)
        self._ammo += amount
        return amount

    def consume(self, amount=1):
        amount = int(amount)
        if amount > self._ammo:
            raise RuntimeError("fake ammo underflow")
        self._ammo -= amount

    def step(self, action):
        self.step_calls += 1
        fired = int(int(np.rint(action)) == FIRE and self._ammo > 0)
        if fired:
            self.consume(1)
        reward = float(fired)
        self.shots_fired += fired
        self.game_reward += reward
        return reward, fired, {
            "shots_fired_this_step": fired,
            "emulator_advanced": True,
            "real_terminal_reset": False,
            "real_game_over_reset": False,
            "time_limit_reset": False,
        }

    def close(self):
        return


class _FrozenFollower:
    def __init__(self, role, economic_action=1.0):
        self.policy = SimpleNamespace(
            economic_role=role,
            economic_input_mode="full",
            set_training_mode=lambda mode: None,
            parameters=lambda: (),
        )
        self.economic_action = float(economic_action)
        self.observations = []

    def predict(self, observation, deterministic=True):
        assert deterministic
        self.observations.append({
            key: np.array(value, copy=True)
            for key, value in observation.items()
        })
        return np.array([FIRE, self.economic_action], dtype=np.float32), None


def _bilateral_config():
    return BilateralAtariConfig(
        seed=11,
        gameplay_horizon=7,
        event_tail_steps=0,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

