"""Five-query Stackelberg-POMDP leader environment for native Atari.

One outer episode consists of exactly five zero-reward policy queries followed
by a fresh bilateral Atari reward game.  The five queried economic actions are
cached and reused at the corresponding reward-game trade events.  Thus the
leader is restricted to a clean event-indexed commitment while both gameplay
controllers remain the immutable E0 policy.
"""

from collections import OrderedDict
import hashlib
from pathlib import Path

import gym
from gym import spaces
import numpy as np

from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    CRITIC_STATE_DIM,
    NUM_TRADE_EVENTS,
    ROLES,
    SELLER,
    BilateralAtariConfig,
    DualAtariTradeCore,
)


class StackPOMDPAtariLeaderEnv(gym.Env):
    """Train an event-indexed seller or buyer commitment against a meta-response."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            *,
            leader_role,
            response_checkpoint,
            game_checkpoint,
            config=None,
            core_factory=DualAtariTradeCore,
            controller_factory=None,
            response_model_factory=None,
            device="cpu",
    ):
        super().__init__()
        if leader_role not in ROLES:
            raise ValueError(f"leader_role must be one of {sorted(ROLES)}")
        self.leader_role = leader_role
        self.follower_role = BUYER if leader_role == SELLER else SELLER
        self.config = (config or BilateralAtariConfig()).resolved()
        self.rng = np.random.default_rng(self.config.seed + 91_019)
        self.core = core_factory(
            self.config,
            game_checkpoint=game_checkpoint,
            controller_factory=controller_factory,
        )
        self.game_action_count = self.core.game_action_count
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array(
                [float(self.game_action_count - 1), 1.0], dtype=np.float32
            ),
            dtype=np.float32,
        )
        observation_spaces = OrderedDict(
            (key, value)
            for key, value in self.core.observation_space.spaces.items()
        )
        observation_spaces.update([
            (
                "event_active",
                spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "event_one_hot",
                spaces.Box(
                    0.0,
                    1.0,
                    shape=(NUM_TRADE_EVENTS,),
                    dtype=np.float32,
                ),
            ),
            (
                "opponent_context",
                spaces.Box(
                    0.0,
                    1.0,
                    shape=(NUM_TRADE_EVENTS,),
                    dtype=np.float32,
                ),
            ),
            (
                "critic:state",
                spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(CRITIC_STATE_DIM,),
                    dtype=np.float32,
                ),
            ),
        ])
        self.observation_space = spaces.Dict(observation_spaces)
        image_space = self.core.observation_space.spaces["image"]
        self.dummy_image = np.zeros(image_space.shape, dtype=image_space.dtype)

        loaded_real_response = response_model_factory is None
        if loaded_real_response:
            from stable_baselines3 import PPO

            response_path = Path(response_checkpoint).expanduser().resolve()
            self.response_model = PPO.load(str(response_path), device=device)
        else:
            self.response_model = response_model_factory(
                response_checkpoint, device=device
            )
        response_role = getattr(
            getattr(self.response_model, "policy", None),
            "economic_role",
            None,
        )
        if response_role is None:
            raise TypeError(
                "response checkpoint lacks required economic_role metadata"
            )
        if response_role != self.follower_role:
            raise ValueError(
                f"response checkpoint role {response_role!r} does not match "
                f"required follower {self.follower_role!r}"
            )
        if loaded_real_response:
            response_fingerprint = getattr(
                self.response_model.policy,
                "gameplay_fingerprint",
                None,
            )
            game_path = Path(game_checkpoint).expanduser()
            if not game_path.is_file():
                zip_path = Path(f"{game_path}.zip")
                if zip_path.is_file():
                    game_path = zip_path
            if not game_path.is_file():
                raise FileNotFoundError(
                    f"game checkpoint does not exist: {game_checkpoint}"
                )
            expected_fingerprint = hashlib.sha256(
                game_path.read_bytes()
            ).hexdigest()
            if response_fingerprint != expected_fingerprint:
                raise ValueError(
                    "response/gameplay checkpoint fingerprint mismatch: "
                    f"{response_fingerprint!r} != {expected_fingerprint!r}"
                )

        self.query_index = 0
        self.leader_context = np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        self.leader_full_actions = np.zeros(
            (NUM_TRADE_EVENTS, 2), dtype=np.float32
        )
        self.follower_actions = np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        self._done = False

    def seed(self, seed=None):
        seed = self.config.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(seed + 91_019)
        return [seed]

    def _base_trade_observation(
            self,
            *,
            role,
            event_index,
            opponent_context,
            ammo_fraction,
            critic_state,
            event_active=1.0,
    ):
        del role  # Role affects values supplied by callers, not the schema.
        one_hot = np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        if event_active > 0.5:
            one_hot[int(event_index)] = 1.0
        return OrderedDict([
            ("image", np.array(self.dummy_image, copy=True)),
            (
                "ammo_fraction",
                np.array([ammo_fraction], dtype=np.float32),
            ),
            ("projectile_active", np.array([0.0], dtype=np.float32)),
            (
                "action_mask",
                np.ones(self.game_action_count, dtype=np.float32),
            ),
            # Preserve the exact neutral market inputs used by E0 gameplay.
            ("offer_active", np.array([0.0], dtype=np.float32)),
            (
                "opportunities_remaining",
                np.array([1.0], dtype=np.float32),
            ),
            ("critic:price", np.array([0.0], dtype=np.float32)),
            (
                "event_active",
                np.array([event_active], dtype=np.float32),
            ),
            ("event_one_hot", one_hot),
            (
                "opponent_context",
                np.asarray(opponent_context, dtype=np.float32).copy(),
            ),
            (
                "critic:state",
                np.asarray(critic_state, dtype=np.float32).copy(),
            ),
        ])

    def _query_critic_state(self, event_index):
        state = np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
        state[0] = 0.0  # response/query phase; critic only
        state[1] = float(event_index) / float(NUM_TRADE_EVENTS - 1)
        state[2] = float(event_index) / float(NUM_TRADE_EVENTS)
        state[3:3 + NUM_TRADE_EVENTS] = self.leader_context
        return state

    def _query_observation(self, event_index):
        # The commitment is intentionally event-indexed only.  Ammo is a fixed
        # canonical dummy value, identical whenever this event is queried or
        # its cached action is deployed in the reward game.
        return self._base_trade_observation(
            role=self.leader_role,
            event_index=event_index,
            opponent_context=np.zeros(NUM_TRADE_EVENTS, dtype=np.float32),
            ammo_fraction=0.0,
            critic_state=self._query_critic_state(event_index),
        )

    def canonical_reward_observation(self, event_index):
        """Diagnostic copy of the actor-visible deployment observation."""
        return self._query_observation(event_index)

    def _response_critic_state(self, event_index):
        core = self.core
        state = np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
        state[:9] = (
            1.0,  # reward phase; critic only
            float(core.game_step) / float(self.config.gameplay_horizon),
            float(NUM_TRADE_EVENTS - event_index) / NUM_TRADE_EVENTS,
            float(core.seller.ammo) / NUM_TRADE_EVENTS,
            float(core.buyer.ammo) / NUM_TRADE_EVENTS,
            float(core.transfers) / NUM_TRADE_EVENTS,
            float(core.payments) / NUM_TRADE_EVENTS,
            float(core.seller.game_reward) / NUM_TRADE_EVENTS,
            float(core.buyer.game_reward) / NUM_TRADE_EVENTS,
        )
        return state

    def _response_observation(self, event_index):
        side = (
            self.core.buyer
            if self.follower_role == BUYER
            else self.core.seller
        )
        return self._base_trade_observation(
            role=self.follower_role,
            event_index=event_index,
            opponent_context=self.leader_context,
            ammo_fraction=float(side.ammo) / NUM_TRADE_EVENTS,
            critic_state=self._response_critic_state(event_index),
        )

    def reset(self):
        # No Atari reset or progression occurs in the query phase.  The reward
        # game is created from a fresh reset only after query five.
        self.query_index = 0
        self.leader_context.fill(0.0)
        self.leader_full_actions.fill(0.0)
        self.follower_actions.fill(0.0)
        self._done = False
        return self._query_observation(0)

    def _run_reward_game(self):
        self.core.reset(seed=int(self.rng.integers(0, 2 ** 31 - 1)))
        while not self.core.done:
            self.core.advance_to_event_or_end()
            if self.core.done:
                break
            if not self.core.at_event:
                raise RuntimeError("reward game stopped outside event or horizon")
            event_index = self.core.next_event
            response_observation = self._response_observation(event_index)
            response_action, _ = self.response_model.predict(
                response_observation,
                deterministic=True,
            )
            response_values = np.asarray(
                response_action, dtype=np.float32
            ).reshape(-1)
            if len(response_values) != 2:
                raise RuntimeError(
                    "meta-response must return [game_action, economic_action]"
                )
            follower_economic_action = float(
                np.clip(response_values[1], 0.0, 1.0)
            )
            self.follower_actions[event_index] = follower_economic_action
            leader_economic_action = float(self.leader_context[event_index])
            # The complete queried action is cached.  Its game component is
            # deliberately ignored on the paused trade state, exactly as it
            # would be if the policy were called again here.
            cached_full_action = self.leader_full_actions[event_index]
            if not np.isclose(
                    cached_full_action[1], leader_economic_action
            ):
                raise RuntimeError("cached leader action/context mismatch")
            if self.leader_role == SELLER:
                price = leader_economic_action
                threshold = follower_economic_action
            else:
                price = follower_economic_action
                threshold = leader_economic_action
            self.core.trade(price=price, threshold=threshold)

        if self.core.next_event != NUM_TRADE_EVENTS:
            raise RuntimeError("reward game ended before all five events")
        self.core.assert_accounting()

    def _terminal_observation(self):
        return self._base_trade_observation(
            role=self.leader_role,
            event_index=0,
            opponent_context=np.zeros(NUM_TRADE_EVENTS, dtype=np.float32),
            ammo_fraction=0.0,
            critic_state=np.zeros(CRITIC_STATE_DIM, dtype=np.float32),
            event_active=0.0,
        )

    def _episode_info(self):
        core = self.core
        return {
            "leader_role": self.leader_role,
            "follower_role": self.follower_role,
            "leader_context": tuple(float(x) for x in self.leader_context),
            "leader_full_actions": tuple(
                tuple(float(x) for x in action)
                for action in self.leader_full_actions
            ),
            "follower_actions": tuple(float(x) for x in self.follower_actions),
            "event_steps": tuple(core.event_steps),
            "events": tuple(dict(event) for event in core.events),
            "trade_opportunities": int(core.next_event),
            "bullets_arrived": int(core.bullets_arrived),
            "purchases": int(core.transfers),
            "payments": float(core.payments),
            "seller_game_reward": float(core.seller.game_reward),
            "buyer_game_reward": float(core.buyer.game_reward),
            "seller_reward": float(core.seller_payoff),
            "buyer_reward": float(core.buyer_payoff),
            "leader_reward": float(
                core.seller_payoff
                if self.leader_role == SELLER
                else core.buyer_payoff
            ),
            "seller_shots_fired": int(core.seller.shots_fired),
            "buyer_shots_fired": int(core.buyer.shots_fired),
            "seller_final_ammo": int(core.seller.ammo),
            "buyer_final_ammo": int(core.buyer.ammo),
            **core.accounting(),
        }

    def step(self, action):
        if self._done:
            raise RuntimeError("step called after outer episode termination")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if len(values) != 2:
            raise ValueError("leader action must be [game_action, economic_action]")
        economic_action = float(np.clip(values[1], 0.0, 1.0))
        self.leader_full_actions[self.query_index] = np.array(
            [
                np.clip(values[0], 0.0, self.game_action_count - 1),
                economic_action,
            ],
            dtype=np.float32,
        )
        self.leader_context[self.query_index] = economic_action
        self.query_index += 1

        if self.query_index < NUM_TRADE_EVENTS:
            info = {
                "is_query": True,
                "query_index": self.query_index - 1,
                "query_count": self.query_index,
                "reward_game_started": False,
            }
            return (
                self._query_observation(self.query_index),
                0.0,
                False,
                info,
            )

        self._run_reward_game()
        self._done = True
        info = self._episode_info()
        info.update({
            "is_query": True,
            "query_index": NUM_TRADE_EVENTS - 1,
            "query_count": NUM_TRADE_EVENTS,
            "reward_game_started": True,
        })
        reward = info["leader_reward"]
        return self._terminal_observation(), float(reward), True, info

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")
        seller = self.core.seller.env.render(mode=mode)
        buyer = self.core.buyer.env.render(mode=mode)
        return np.concatenate([seller, buyer], axis=1)

    def close(self):
        self.core.close()
