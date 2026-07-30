import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from stackelberg_pomdp.baselines_utils import CustomA2C


class HiddenPrefixEnv(gym.Env):
    """Two hidden prefix steps followed by two stored reward steps."""

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=4.0, shape=(1,), dtype=np.float32
        )
        self.episode_step = 0
        self.executed_steps = 0

    def reset(self):
        self.episode_step = 0
        return np.asarray([0.0], dtype=np.float32)

    def step(self, action):
        del action
        self.episode_step += 1
        self.executed_steps += 1
        done = self.episode_step == 4
        return (
            np.asarray([float(self.episode_step)], dtype=np.float32),
            1.0 if self.episode_step >= 3 else 0.0,
            done,
            {"exclude_from_buffer": self.episode_step <= 2},
        )


def test_hidden_prefix_preserves_episode_start_for_first_stored_transition():
    raw_env = HiddenPrefixEnv()
    vec_env = DummyVecEnv([lambda: raw_env])
    model = CustomA2C(
        "MlpPolicy",
        vec_env,
        n_steps=2,
        gamma=1.0,
        learning_rate=1e-3,
        seed=0,
        verbose=0,
    )
    model.learn(total_timesteps=4)
    np.testing.assert_array_equal(
        model.rollout_buffer.episode_starts[:, 0],
        [1.0, 0.0],
    )
    assert raw_env.executed_steps == 4
    assert model.rollout_buffer.pos == 2
