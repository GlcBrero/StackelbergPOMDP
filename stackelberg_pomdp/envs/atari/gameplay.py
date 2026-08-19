"""Reusable single-player Atari side for the clean curriculum environments."""

import numpy as np

from stackelberg_pomdp.envs.atari.space_invaders import (
    SinglePlayerSpaceInvadersEnv,
)
from stackelberg_pomdp.wrappers.atari.preprocessing import (
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

    Wrapper terminals are local to this emulator. Ordinary life losses and
    real terminals rebuild its frame stack immediately, while the outer market
    episode, ammunition ledger, cumulative reward, and shot count continue.
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
        self.episodic_life_wrapper = None
        if episodic_life:
            env = EpisodicLifeWrapper(env)
            self.episodic_life_wrapper = env
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
        self.real_terminal_resets = 0
        self.real_terminal_reset_steps = []
        self.true_game_over_resets = 0
        self.true_game_over_reset_steps = []
        self.time_limit_resets = 0
        self.time_limit_reset_steps = []

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
        self.real_terminal_resets = 0
        self.real_terminal_reset_steps = []
        self.true_game_over_resets = 0
        self.true_game_over_reset_steps = []
        self.time_limit_resets = 0
        self.time_limit_reset_steps = []
        # An outer-episode reset is always a fresh ALE game. EpisodicLife's
        # normal reset path intentionally emits a single NOOP after a lost life,
        # so force its full-reset branch here.
        if self.episodic_life_wrapper is not None:
            self.episodic_life_wrapper.was_real_done = True
        self._observation = np.array(self.env.reset(), copy=True)
        return self.observation

    def _reset_preserving_outer_state(
            self,
            reset_kind,
            *,
            real_game_over=False,
            time_limit_reached=False,
    ):
        """Reset this wrapper chain without resetting outer-episode ledgers."""

        ammo = self.ammo
        wrapper_shots = int(self.ammo_wrapper.shots_fired_total)
        self._observation = np.array(self.env.reset(), copy=True)
        self.ledger.value = ammo
        # A life-loss reset remains within the same ALE game, so retain the
        # wrapper's reward-provenance flag. A real terminal starts a fresh game:
        # its no-bullet reward guard must remain reset even though the side-level
        # diagnostic shot count continues across the outer market episode.
        if reset_kind == "life":
            self.ammo_wrapper.shots_fired_total = wrapper_shots

        step = int(self.step_calls)
        if reset_kind == "life":
            self.life_resets += 1
        else:
            self.real_terminal_resets += 1
            self.real_terminal_reset_steps.append(step)
            if real_game_over or reset_kind == "ale_game_over":
                self.true_game_over_resets += 1
                self.true_game_over_reset_steps.append(step)
            if time_limit_reached or reset_kind == "time_limit":
                self.time_limit_resets += 1
                self.time_limit_reset_steps.append(step)

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
        info = dict(info)
        reset_kind = None
        reset_real_game_over = False
        reset_time_limit = False
        if done:
            if bool(info.get("real_done", True)):
                reset_real_game_over = bool(
                    info.get("real_game_over", info.get("ale.game_over", True))
                )
                reset_time_limit = bool(info.get("time_limit_reached", False))
                reset_kind = info.get("terminal_reason") or (
                    "ale_game_over"
                    if reset_real_game_over
                    else "time_limit" if reset_time_limit else "terminal"
                )
                self._reset_preserving_outer_state(
                    reset_kind,
                    real_game_over=reset_real_game_over,
                    time_limit_reached=reset_time_limit,
                )
            else:
                reset_kind = "life"
                self._reset_preserving_outer_state(reset_kind)
        info.update({
            "emulator_advanced": True,
            "shots_fired_total": int(self.shots_fired),
            "shots_fired_since_emulator_reset": int(
                self.ammo_wrapper.shots_fired_total
            ),
            "life_reset": reset_kind == "life",
            "real_terminal_reset": bool(done and reset_kind != "life"),
            "real_game_over_reset": reset_real_game_over,
            "time_limit_reset": reset_time_limit,
            "life_resets": int(self.life_resets),
            "real_terminal_resets": int(self.real_terminal_resets),
            "true_game_over_resets": int(self.true_game_over_resets),
            "time_limit_resets": int(self.time_limit_resets),
        })
        return reward, shots, info

    def grant(self, amount=1):
        return self.ledger.grant(int(amount))

    def consume(self, amount=1):
        self.ledger.consume(int(amount))

    def close(self):
        self.env.close()


__all__ = ["AtariGameplaySide"]
