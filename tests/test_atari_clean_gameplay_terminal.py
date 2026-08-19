import gym
import numpy as np

from stackelberg_pomdp.envs.atari.gameplay import AtariGameplaySide
from stackelberg_pomdp.wrappers.atari.preprocessing import MaxAndSkipWrapper


class _LifecycleALE:
    def __init__(self):
        self.ram = np.full(256, 0xF6, dtype=np.uint8)
        self.lives = 3

    def getRAM(self):
        return self.ram

    def allLives(self):
        return np.asarray([self.lives], dtype=np.int32)


class _LifecycleRawEnv(gym.Env):
    """Life loss on raw step 2, real game-over on raw step 4."""

    def __init__(self):
        super().__init__()
        self.ale = _LifecycleALE()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(8, 8, 3), dtype=np.uint8
        )
        self.raw_steps = 0
        self.reset_calls = 0

    def get_action_meanings(self):
        return ["NOOP", "FIRE"]

    def _frame(self):
        return np.full(
            self.observation_space.shape, self.raw_steps, dtype=np.uint8
        )

    def reset(self):
        self.raw_steps = 0
        self.reset_calls += 1
        self.ale.lives = 3
        self.ale.ram.fill(0xF6)
        return self._frame()

    def step(self, action):
        self.ale.ram[0x55] = 0x55 if int(action) == 1 else 0xF6
        self.raw_steps += 1
        real_game_over = self.raw_steps == 4
        if self.raw_steps == 2:
            self.ale.lives = 2
        elif real_game_over:
            self.ale.lives = 0
        # Step two also emits a positive reward. Before the first reset it is
        # owned by a shot; after a real reset the same no-shot reward must be
        # suppressed by the fresh-emulator provenance guard.
        reward = 1.0 if self.raw_steps == 2 else 0.0
        return self._frame(), reward, real_game_over, {
            "ale.lives": int(self.ale.lives),
            "ale.game_over": bool(real_game_over),
            "time_limit_reached": False,
            "terminal_reason": "ale_game_over" if real_game_over else None,
        }


def test_life_loss_and_true_game_over_reset_only_the_local_emulator():
    raw = _LifecycleRawEnv()
    side = AtariGameplaySide(
        seed=1,
        initial_ammo=2,
        capacity=5,
        noop_max=1,
        frame_skip=1,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        env_factory=lambda **kwargs: raw,
    )
    try:
        side.reset()
        assert raw.reset_calls == 1
        assert raw.raw_steps == 1  # deterministic noop-reset step

        life_reward, life_shots, life_info = side.step(1)
        assert life_reward == 1.0
        assert life_shots == 1
        assert life_info["life_lost"]
        assert not life_info["real_done"]
        assert life_info["life_reset"]
        assert life_info["shots_fired_total"] == 1
        assert life_info["shots_fired_since_emulator_reset"] == 1
        assert side.life_resets == 1
        assert side.ammo == 1
        assert side.shots_fired == 1
        assert side.ammo_wrapper.shots_fired_total == 1
        assert raw.reset_calls == 1
        assert raw.raw_steps == 3  # life reset advances one NOOP, not reset_game

        _, _, terminal_info = side.step(0)
        assert terminal_info["real_done"]
        assert terminal_info["real_game_over"]
        assert terminal_info["real_terminal_reset"]
        assert terminal_info["real_game_over_reset"]
        assert not terminal_info["time_limit_reset"]
        assert terminal_info["shots_fired_total"] == 1
        assert terminal_info["shots_fired_since_emulator_reset"] == 0
        assert side.life_resets == 1
        assert side.real_terminal_resets == 1
        assert side.real_terminal_reset_steps == [2]
        assert side.true_game_over_resets == 1
        assert side.true_game_over_reset_steps == [2]
        assert side.time_limit_resets == 0
        assert raw.reset_calls == 2
        assert raw.raw_steps == 1  # full reset plus deterministic reset NOOP
        assert side.ammo == 1
        assert side.shots_fired == 1
        assert side.ammo_wrapper.shots_fired_total == 0
        assert np.unique(side.observation).tolist() == [1]

        reward, shots, continued_info = side.step(0)
        assert reward == 0.0
        assert shots == 0
        assert continued_info["suppressed_unowned_reward"] == 1.0
        assert continued_info["shots_fired_total"] == 1
        assert continued_info["shots_fired_since_emulator_reset"] == 0
        assert raw.raw_steps == 3  # raw step 2 plus the life-reset NOOP
        assert raw.reset_calls == 2
        assert side.step_calls == 3
        assert side.ammo == 1
        assert not side.projectile_active
        np.testing.assert_array_equal(side.action_mask, [1.0, 1.0])
        assert continued_info["emulator_advanced"]
        assert continued_info["life_reset"]

        # Market inventory remains valid across local emulator resets.
        assert side.grant(1) == 1
        assert side.ammo == 2

        side.reset()
        assert side.true_game_over_resets == 0
        assert side.true_game_over_reset_steps == []
        assert side.real_terminal_resets == 0
        assert side.ammo == 2
        assert raw.reset_calls == 3
    finally:
        side.close()


class _TimeLimitRawEnv(_LifecycleRawEnv):
    """Time-limit terminal on raw step two without an ALE game-over."""

    def step(self, action):
        del action
        self.raw_steps += 1
        reached_limit = self.raw_steps == 2
        return self._frame(), 0.0, reached_limit, {
            "ale.lives": int(self.ale.lives),
            "ale.game_over": False,
            "time_limit_reached": reached_limit,
            "terminal_reason": "time_limit" if reached_limit else None,
        }


def test_time_limit_reset_is_logged_separately_from_true_game_over():
    raw = _TimeLimitRawEnv()
    side = AtariGameplaySide(
        seed=2,
        initial_ammo=2,
        capacity=5,
        noop_max=1,
        frame_skip=1,
        frame_stack=4,
        episodic_life=True,
        clip_game_rewards=True,
        env_factory=lambda **kwargs: raw,
    )
    try:
        side.reset()
        _, _, info = side.step(0)
        assert info["real_terminal_reset"]
        assert info["time_limit_reset"]
        assert not info["real_game_over_reset"]
        assert side.real_terminal_resets == 1
        assert side.time_limit_resets == 1
        assert side.time_limit_reset_steps == [1]
        assert side.true_game_over_resets == 0
        assert side.ammo == 2
        assert side.step_calls == 1
        assert raw.reset_calls == 2
        assert raw.raw_steps == 1
    finally:
        side.close()


class _ImmediateTerminalFrameEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(2, 2, 1), dtype=np.uint8
        )

    def reset(self):
        return np.full(self.observation_space.shape, 200, dtype=np.uint8)

    def step(self, action):
        del action
        return (
            np.full(self.observation_space.shape, 7, dtype=np.uint8),
            0.0,
            True,
            {"real_done": True, "real_game_over": True},
        )


def test_max_skip_uses_terminal_frame_instead_of_stale_pool_entries():
    env = MaxAndSkipWrapper(_ImmediateTerminalFrameEnv(), skip=4)
    env.reset()
    observation, _, done, info = env.step(0)
    assert done
    assert info["real_done"]
    assert info["real_game_over"]
    np.testing.assert_array_equal(
        observation,
        np.full(env.observation_space.shape, 7, dtype=np.uint8),
    )
