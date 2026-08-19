"""Single-agent adapter for Atari meta-response training.

The bilateral environment remains the sole owner of game and market dynamics.
This wrapper samples one five-action opponent commitment, supplies the frozen
opponent's gameplay actions, and exposes only the controlled role to SB3.
"""

from typing import Callable, Optional

import gym
import numpy as np

from stackelberg_pomdp.atari.envs.bilateral import (
    SELLER,
    BilateralAtariRewardEnv,
    DualAtariTradeCore,
)
from stackelberg_pomdp.atari.policies.loading import (
    FrozenAtariPolicyController,
)
from stackelberg_pomdp.atari.protocol import (
    CRITIC_STATE_DIM,
    FOLLOWER_TRADE,
    GAMEPLAY,
    NUM_TRADE_EVENTS,
    TERMINAL,
    action_space,
    observation_space,
    validate_action,
)
from stackelberg_pomdp.atari.sampling import (
    ALL_EQUAL_E1_SAMPLER,
    CONTEXT_STRATA,
    E1_SAMPLER_MODES,
    SCHEDULE_STRATA,
    TEMPORAL_MIX_E1_SAMPLER,
    UNIFORM_E1_SAMPLER,
    TemporalMarginalE1Sampler,
    sample_all_equal_e1_context,
)


class AtariFixedCommitmentResponseWrapper(gym.Wrapper):
    """Train one response policy against a sampled five-action commitment."""

    def __init__(
            self,
            env,
            *,
            e0b_checkpoint=None,
            context_sampler: Optional[Callable] = None,
            e1_sampler_mode=UNIFORM_E1_SAMPLER,
            controller_factory=None,
            device="cpu",
    ):
        if not isinstance(env, BilateralAtariRewardEnv):
            raise TypeError(
                "AtariFixedCommitmentResponseWrapper requires "
                "BilateralAtariRewardEnv"
            )
        super().__init__(env)
        self.controlled_role = env.leader_role
        self.other_role = env.follower_role
        self.config = env.config
        self.context_sampler = context_sampler
        self.e1_sampler_mode = str(e1_sampler_mode)
        if self.e1_sampler_mode not in E1_SAMPLER_MODES:
            raise ValueError(
                f"unknown E1 sampler mode: {self.e1_sampler_mode!r}"
            )
        if (
                self.e1_sampler_mode in (
                    ALL_EQUAL_E1_SAMPLER,
                    TEMPORAL_MIX_E1_SAMPLER,
                )
                and context_sampler is not None
        ):
            raise ValueError(
                f"{self.e1_sampler_mode} supplies its own context and is "
                "incompatible with context_sampler"
            )
        self.rng = np.random.default_rng(self.config.seed + 74_711)
        self.temporal_sampler = (
            TemporalMarginalE1Sampler(
                seed=self.config.seed,
                gameplay_horizon=self.config.gameplay_horizon,
                event_tail_steps=self.config.event_tail_steps,
                fixed_event_steps=self.config.fixed_event_steps,
            )
            if self.e1_sampler_mode == TEMPORAL_MIX_E1_SAMPLER
            else None
        )
        self.schedule_stratum = None
        self.context_stratum = None
        self.schedule_stratum_counts = {
            name: 0 for name in SCHEDULE_STRATA
        }
        self.context_stratum_counts = {
            name: 0 for name in CONTEXT_STRATA
        }
        self.sampler_episode_count = 0
        self.action_space = action_space(self.core.game_action_count)
        self.observation_space = observation_space(
            self.core.image_space, self.core.game_action_count
        )
        if controller_factory is None:
            if e0b_checkpoint is None:
                raise ValueError("e0b_checkpoint or controller_factory is required")
            self.other_controller = FrozenAtariPolicyController(
                e0b_checkpoint, device=device
            )
        else:
            self.other_controller = controller_factory()
        self.opponent_commitment = np.zeros(
            NUM_TRADE_EVENTS, dtype=np.float32
        )

    @property
    def core(self):
        return self.env.core

    def seed(self, seed=None):
        seed = self.config.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(seed + 74_711)
        if self.temporal_sampler is not None:
            self.temporal_sampler.seed(seed)
        return [seed]

    def _sample_context(self):
        if self.e1_sampler_mode == ALL_EQUAL_E1_SAMPLER:
            values = sample_all_equal_e1_context(self.rng)
        else:
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
            float(self.env.gameplay_transitions) / self.config.gameplay_horizon,
            float(self.env.trade_transitions) / NUM_TRADE_EVENTS,
        )
        values[15:20] = (
            np.asarray(core.event_steps, dtype=np.float32)
            / self.config.gameplay_horizon
        )
        values[20:25] = self.opponent_commitment
        return values

    def _role_observation(self, role, *, controlled, decision_kind):
        context = (
            self.opponent_commitment
            if controlled
            else np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        )
        return self.env.role_observation(
            role,
            opponent_commitment=context,
            decision_kind=decision_kind,
            critic_state=self._critic_state(),
        )

    def _controlled_observation(self):
        if self.env._done:
            kind = TERMINAL
        else:
            kind = FOLLOWER_TRADE if self.core.at_event else GAMEPLAY
        return self._role_observation(
            self.controlled_role, controlled=True, decision_kind=kind
        )

    def _other_game_action(self):
        values = self._role_observation(
            self.other_role,
            controlled=False,
            decision_kind=GAMEPLAY,
        )
        result = self.other_controller(values)
        return int(np.clip(np.rint(result), 0, self.core.game_action_count - 1))

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.seed(seed)
        reset_options = None
        if self.temporal_sampler is None:
            self.opponent_commitment = self._sample_context()
            self.schedule_stratum = (
                "fixed"
                if self.config.fixed_event_steps is not None
                else "unconditional"
            )
            self.context_stratum = (
                "external"
                if self.context_sampler is not None
                else (
                    "all_equal"
                    if self.e1_sampler_mode == ALL_EQUAL_E1_SAMPLER
                    else "uniform"
                )
            )
        else:
            draw = self.temporal_sampler.sample()
            self.opponent_commitment = np.array(
                draw.opponent_commitment, copy=True
            )
            self.schedule_stratum = draw.schedule_stratum
            self.context_stratum = draw.context_stratum
            reset_options = {"event_steps": draw.event_steps}
        self.sampler_episode_count += 1
        self.schedule_stratum_counts[self.schedule_stratum] += 1
        self.context_stratum_counts[self.context_stratum] += 1
        inner_seed = int(self.rng.integers(0, 2 ** 31 - 1))
        if reset_options is None:
            self.env.reset(seed=inner_seed)
        else:
            self.env.reset(seed=inner_seed, options=reset_options)
        return self._controlled_observation()

    def _joint_action(self, values, *, trade):
        if trade:
            event_index = self.core.next_event
            opponent_action = np.array(
                [0.0, self.opponent_commitment[event_index]],
                dtype=np.float32,
            )
        else:
            opponent_action = np.array(
                [self._other_game_action(), 0.0], dtype=np.float32
            )
        return {
            self.controlled_role: values,
            self.other_role: opponent_action,
        }

    def _episode_info(self):
        base = self.env.episode_info()
        sampler = {
            "e1_sampler_mode": self.e1_sampler_mode,
            "e1_sampler_episode_count_per_env": int(
                self.sampler_episode_count
            ),
            "e1_schedule_stratum": self.schedule_stratum,
            "e1_context_stratum": self.context_stratum,
        }
        sampler.update({
            f"e1_schedule_stratum_one_hot_{name}": int(
                self.schedule_stratum == name
            )
            for name in SCHEDULE_STRATA
        })
        sampler.update({
            f"e1_context_stratum_one_hot_{name}": int(
                self.context_stratum == name
            )
            for name in CONTEXT_STRATA
        })
        sampler.update({
            f"e1_schedule_stratum_per_env_count_{name}": int(count)
            for name, count in self.schedule_stratum_counts.items()
        })
        sampler.update({
            f"e1_context_stratum_per_env_count_{name}": int(count)
            for name, count in self.context_stratum_counts.items()
        })
        if self.e1_sampler_mode == ALL_EQUAL_E1_SAMPLER:
            sampler.update({
                "e1_context_entries_all_equal": int(np.all(
                    self.opponent_commitment == self.opponent_commitment[0]
                )),
                "e1_context_shared_value": float(
                    self.opponent_commitment[0]
                ),
            })
        return {
            **base,
            **sampler,
            "controlled_role": self.controlled_role,
            "opponent_commitment": tuple(
                float(value) for value in self.opponent_commitment
            ),
            "outer_transition_count": int(base["reward_transition_count"]),
        }

    def step(self, action):
        if self.env._done:
            raise RuntimeError(
                "step called after meta-response episode termination"
            )
        values = validate_action(action, self.core.game_action_count)
        trade = bool(self.core.at_event)
        _, reward, done, info = self.env.step(
            self._joint_action(values, trade=trade)
        )
        info.update({
            "substep_type": FOLLOWER_TRADE if trade else GAMEPLAY,
            "controlled_reward_delta": float(reward),
        })
        if done:
            episode = self._episode_info()
            info.update(episode)
            info["episode"] = {
                "r": self.env.role_payoff(self.controlled_role),
                "l": episode["outer_transition_count"],
                **episode,
            }
        return self._controlled_observation(), float(reward), done, info


def make_atari_meta_response_env(
        *,
        controlled_role,
        e0b_checkpoint=None,
        config=None,
        context_sampler=None,
        e1_sampler_mode=UNIFORM_E1_SAMPLER,
        core_factory=DualAtariTradeCore,
        controller_factory=None,
        side_factory=None,
        env_factory=None,
        device="cpu",
):
    """Compose the bilateral game with the fixed-commitment adapter."""

    reward_env = BilateralAtariRewardEnv(
        leader_role=controlled_role,
        config=config,
        core_factory=core_factory,
        side_factory=side_factory,
        env_factory=env_factory,
    )
    return AtariFixedCommitmentResponseWrapper(
        reward_env,
        e0b_checkpoint=e0b_checkpoint,
        context_sampler=context_sampler,
        e1_sampler_mode=e1_sampler_mode,
        controller_factory=controller_factory,
        device=device,
    )


__all__ = [
    "AtariFixedCommitmentResponseWrapper",
    "make_atari_meta_response_env",
]
