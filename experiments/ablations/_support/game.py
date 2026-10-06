"""Self-contained matrix environments for the paper diagnostics.

The repeated-game path mirrors the clean Atari meta-RL protocol without
importing Atari code:

* E1 trains a follower against a sampled deterministic leader commitment.
* E2 queries the leader once at every canonical state, freezes that ordered
  action table as context, and evaluates a frozen E1 follower in a fresh game.

The tabular-Q path intentionally lives here as well.  Its one-shot update,
initialization, exploration, and reset semantics are explicit, rather than
being inherited from the pricing-specific Q-learning implementation.
"""

from support.checkpoint_compat import load_checkpoint
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import gym
from gym.spaces import Dict, Discrete
import numpy as np

from experiments.ablations._support.profiles import (
    get_profile,
    profile_for_spec,
)


LEADER = "leader"
FOLLOWER = "follower_0"
RESPONSE_CONTRACT_SCHEMA_VERSION = 2
RESPONSE_CONTRACT_FILENAME = "response_contract.json"
MEMORY_MODES = ("none", "opponent", "joint")


@dataclass(frozen=True)
class MatrixGameSpec:
    """Complete specification of a two-player simultaneous matrix game.

    ``memory_mode='opponent'`` reproduces the legacy ``small_memory=True``
    encoding (Start plus the other player's previous action).  ``'joint'``
    uses Start plus the previous joint action and matches the five-state
    description in the paper for two-by-two repeated games.
    """

    name: str
    payoffs: np.ndarray
    episode_length: int
    reward_offset: float = 0.0
    memory_mode: str = "none"

    def __post_init__(self):
        payoffs = np.asarray(self.payoffs, dtype=np.float64)
        if payoffs.ndim != 3 or payoffs.shape[-1] != 2:
            raise ValueError(
                "payoffs must have shape (leader actions, follower actions, 2)"
            )
        if self.episode_length < 1:
            raise ValueError("episode_length must be positive")
        if self.memory_mode not in MEMORY_MODES:
            raise ValueError(
                "memory_mode must be one of {}".format(", ".join(MEMORY_MODES))
            )
        if self.episode_length > 1 and self.memory_mode == "none":
            raise ValueError("repeated paper games require an explicit memory mode")
        object.__setattr__(self, "payoffs", payoffs)

    @property
    def num_leader_actions(self):
        return int(self.payoffs.shape[0])

    @property
    def num_follower_actions(self):
        return int(self.payoffs.shape[1])

    @property
    def num_joint_actions(self):
        return self.num_leader_actions * self.num_follower_actions

    @property
    def num_leader_states(self):
        if self.memory_mode == "none":
            return 1
        if self.memory_mode == "opponent":
            return self.num_follower_actions + 1
        return self.num_joint_actions + 1

    @property
    def num_follower_states(self):
        if self.memory_mode == "none":
            return 1
        if self.memory_mode == "opponent":
            return self.num_leader_actions + 1
        return self.num_joint_actions + 1

    @property
    def centered_payoffs(self):
        return self.payoffs + float(self.reward_offset)

    @property
    def centered_follower_payoffs(self):
        return self.payoffs[:, :, 1] + float(self.reward_offset)

    def with_memory_mode(self, memory_mode):
        return replace(self, memory_mode=memory_mode)


MATRIX_GAMES = {
    "prisoners_dilemma": MatrixGameSpec(
        name="prisoners_dilemma",
        payoffs=np.array([
            [[3.0, 3.0], [1.0, 4.0]],
            [[4.0, 1.0], [2.0, 2.0]],
        ]),
        episode_length=5,
        reward_offset=-4.0,
        memory_mode="joint",
    ),
    "battle_of_the_sexes": MatrixGameSpec(
        name="battle_of_the_sexes",
        payoffs=np.array([
            [[2.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 2.0]],
        ]),
        episode_length=1,
    ),
    "coordination_zero_miscoordination": MatrixGameSpec(
        name="coordination_zero_miscoordination",
        payoffs=np.array([
            [[2.0, 0.001], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 2.0]],
        ]),
        episode_length=1,
    ),
    "coordination_penalized_miscoordination": MatrixGameSpec(
        name="coordination_penalized_miscoordination",
        payoffs=np.array([
            [[2.0, 0.001], [-5.0, 0.0]],
            [[-5.0, 0.0], [1.0, 2.0]],
        ]),
        episode_length=1,
    ),
}


def get_matrix_game(name, memory_mode=None, profile_id=None):
    """Return an independent specification under an explicit named profile."""

    try:
        source = MATRIX_GAMES[name]
    except KeyError as exc:
        raise ValueError(
            "unknown matrix game {!r}; available: {}".format(
                name, ", ".join(sorted(MATRIX_GAMES))
            )
        ) from exc
    reward_offset = source.reward_offset
    if profile_id is not None:
        profile = get_profile(profile_id)
        if source.episode_length <= 1:
            raise ValueError("named repeated-game profiles require a repeated game")
        if memory_mode is not None and memory_mode != profile.memory_mode:
            raise ValueError(
                "memory_mode {!r} conflicts with profile {!r}".format(
                    memory_mode, profile.profile_id
                )
            )
        memory_mode = profile.memory_mode
        reward_offset = profile.training_reward_offset
    spec = MatrixGameSpec(
        name=source.name,
        payoffs=np.array(source.payoffs, copy=True),
        episode_length=source.episode_length,
        reward_offset=reward_offset,
        memory_mode=source.memory_mode,
    )
    return spec if memory_mode is None else spec.with_memory_mode(memory_mode)


class RepeatedMatrixGame:
    """Small deterministic game state used by both E1 and E2."""

    def __init__(self, spec):
        self.spec = spec
        self.current_step = 0
        self.leader_state = 0
        self.follower_state = 0

    def reset(self):
        self.current_step = 0
        self.leader_state = 0
        self.follower_state = 0
        return self.observations()

    def observations(self):
        return {
            LEADER: int(self.leader_state),
            FOLLOWER: int(self.follower_state),
        }

    def _next_states(self, leader_action, follower_action):
        if self.spec.memory_mode == "none":
            return 0, 0
        if self.spec.memory_mode == "opponent":
            return 1 + follower_action, 1 + leader_action
        joint_state = (
            1 + leader_action * self.spec.num_follower_actions + follower_action
        )
        return joint_state, joint_state

    def step(self, leader_action, follower_action):
        if self.current_step >= self.spec.episode_length:
            raise RuntimeError("step called after matrix episode terminated")
        leader_action = int(leader_action)
        follower_action = int(follower_action)
        if not 0 <= leader_action < self.spec.num_leader_actions:
            raise ValueError("invalid leader action: {}".format(leader_action))
        if not 0 <= follower_action < self.spec.num_follower_actions:
            raise ValueError("invalid follower action: {}".format(follower_action))

        centered = self.spec.centered_payoffs[leader_action, follower_action]
        rewards = {LEADER: float(centered[0]), FOLLOWER: float(centered[1])}
        previous_states = self.observations()
        self.current_step += 1
        self.leader_state, self.follower_state = self._next_states(
            leader_action, follower_action
        )
        done = self.current_step >= self.spec.episode_length
        info = {
            "matrix_game": self.spec.name,
            "matrix_step": int(self.current_step),
            "previous_states": previous_states,
            "leader_action": leader_action,
            "follower_action": follower_action,
            "utilities": rewards,
            "reward_generated": True,
        }
        return self.observations(), rewards, done, info


def meta_follower_observation_space_n(spec):
    """Number of encoded follower-state/commitment contexts."""

    return int(
        spec.num_follower_states
        * spec.num_leader_actions ** spec.num_leader_states
    )


def encode_meta_follower_observation(spec, follower_state, commitment):
    """Encode state and ordered commitment using the legacy mixed radix.

    This is the layout from StackeRLberg's
    ``wrappers/dict_to_discrete_obs_wrapper.py``: the original follower state
    is most significant, followed by the ordered query actions.  For the
    paper-spec joint-memory game this yields ``Discrete(5 * 2**5)``.
    """

    follower_state = int(follower_state)
    if not 0 <= follower_state < spec.num_follower_states:
        raise ValueError("invalid follower state: {}".format(follower_state))
    commitment = np.asarray(commitment, dtype=np.int64).reshape(-1)
    if commitment.shape != (spec.num_leader_states,):
        raise ValueError(
            "commitment must contain {} actions".format(spec.num_leader_states)
        )
    if np.any(commitment < 0) or np.any(
            commitment >= spec.num_leader_actions
    ):
        raise ValueError("commitment contains an invalid leader action")

    encoded = follower_state
    for leader_action in commitment:
        encoded *= spec.num_leader_actions
        encoded += int(leader_action)
    return int(encoded)


def decode_meta_follower_observation(spec, encoded):
    """Invert :func:`encode_meta_follower_observation` exactly."""

    encoded = int(encoded)
    space_n = meta_follower_observation_space_n(spec)
    if not 0 <= encoded < space_n:
        raise ValueError("encoded observation is outside Discrete({})".format(space_n))
    actions = []
    remainder = encoded
    for _ in range(spec.num_leader_states):
        actions.append(remainder % spec.num_leader_actions)
        remainder //= spec.num_leader_actions
    actions.reverse()
    follower_state = remainder
    return int(follower_state), tuple(int(action) for action in actions)


class MatrixFixedCommitmentResponseEnv(gym.Env):
    """E1 follower environment conditioned on a full leader policy table."""

    metadata = {"render.modes": []}

    def __init__(
            self,
            spec,
            seed=None,
            commitment_sampler: Optional[Callable] = None,
            fixed_commitment: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if spec.episode_length <= 1 or spec.memory_mode == "none":
            raise ValueError("meta-response training requires a repeated memory game")
        # Gym reserves ``env.spec`` for an EnvSpec; Monitor reads ``spec.id``.
        self.game_spec = spec
        self.game = RepeatedMatrixGame(spec)
        self.action_space = Discrete(spec.num_follower_actions)
        self.observation_space = Discrete(
            meta_follower_observation_space_n(spec)
        )
        self.commitment_sampler = commitment_sampler
        self.fixed_commitment = (
            None if fixed_commitment is None
            else self._validated_commitment(fixed_commitment)
        )
        self.rng = np.random.default_rng(seed)
        self.commitment = np.zeros(spec.num_leader_states, dtype=np.int64)
        self.episode_return = 0.0
        self.episode_steps = 0

    @classmethod
    def commitments(cls, spec):
        return tuple(product(
            range(spec.num_leader_actions), repeat=spec.num_leader_states
        ))

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def _validated_commitment(self, values):
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        if values.shape != (self.game_spec.num_leader_states,):
            raise ValueError(
                "commitment must contain {} actions".format(
                    self.game_spec.num_leader_states
                )
            )
        if np.any(
            values < 0
        ) or np.any(values >= self.game_spec.num_leader_actions):
            raise ValueError("commitment contains an invalid leader action")
        return values

    def _sample_commitment(self):
        if self.fixed_commitment is not None:
            return np.array(self.fixed_commitment, copy=True)
        if self.commitment_sampler is not None:
            return self._validated_commitment(self.commitment_sampler(self.rng))
        # The legacy trainer randomized every bias-free linear leader weight
        # iid Uniform[-1, 1] and then acted deterministically.  With two
        # actions, symmetry makes every state action an independent fair bit,
        # exactly the uniform deterministic-table distribution sampled here.
        return self.rng.integers(
            0,
            self.game_spec.num_leader_actions,
            size=self.game_spec.num_leader_states,
            dtype=np.int64,
        )

    def _observation(self):
        return encode_meta_follower_observation(
            self.game_spec,
            self.game.follower_state,
            self.commitment,
        )

    def reset(self):
        self.commitment = self._sample_commitment()
        self.game.reset()
        self.episode_return = 0.0
        self.episode_steps = 0
        return self._observation()

    def step(self, action):
        follower_action = int(np.asarray(action).reshape(-1)[0])
        leader_action = int(self.commitment[self.game.leader_state])
        _, rewards, done, info = self.game.step(leader_action, follower_action)
        reward = float(rewards[FOLLOWER])
        self.episode_return += reward
        self.episode_steps += 1
        info.update({
            "controlled_role": FOLLOWER,
            "leader_commitment": tuple(int(value) for value in self.commitment),
        })
        if done:
            info["episode"] = {
                "r": float(self.episode_return),
                "l": int(self.episode_steps),
            }
        return self._observation(), reward, done, info


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def response_game_contract(spec):
    """Return the behaviorally relevant part of an E1 checkpoint contract."""

    profile = profile_for_spec(spec)
    profile_payload = profile.to_dict()
    profile_sha256 = hashlib.sha256(
        json.dumps(
            profile_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "profile": profile_payload,
        "profile_id": profile.profile_id,
        "profile_sha256": profile_sha256,
        "episode_length": int(spec.episode_length),
        "memory_mode": spec.memory_mode,
        "query_states": list(range(spec.num_leader_states)),
        "num_leader_actions": int(spec.num_leader_actions),
        "num_follower_actions": int(spec.num_follower_actions),
        "num_follower_states": int(spec.num_follower_states),
        "raw_follower_payoffs": spec.payoffs[:, :, 1].tolist(),
        "training_reward_offset": float(spec.reward_offset),
        "centered_follower_payoffs": spec.centered_follower_payoffs.tolist(),
        "follower_observation_layout": [
            "follower_state",
            "ordered_leader_commitment",
        ],
        "follower_observation_encoding": (
            profile.context_encoding_version
        ),
        "follower_observation_space_n": meta_follower_observation_space_n(spec),
    }


def build_response_checkpoint_contract(spec, checkpoint, algorithm, metadata=None):
    checkpoint = Path(checkpoint)
    payload = {
        "schema_version": RESPONSE_CONTRACT_SCHEMA_VERSION,
        "kind": "matrix_meta_follower",
        "algorithm": str(algorithm),
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "response_game": response_game_contract(spec),
    }
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    return payload


def write_response_checkpoint_contract(
        spec, checkpoint, algorithm, path=None, metadata=None
):
    checkpoint = Path(checkpoint)
    path = Path(path) if path is not None else checkpoint.with_name(
        RESPONSE_CONTRACT_FILENAME
    )
    payload = build_response_checkpoint_contract(
        spec, checkpoint, algorithm, metadata=metadata
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
    return path


def validate_response_checkpoint_contract(
        spec, checkpoint, algorithm=None, contract_path=None
):
    checkpoint = Path(checkpoint)
    contract_path = (
        Path(contract_path) if contract_path is not None
        else checkpoint.with_name(RESPONSE_CONTRACT_FILENAME)
    )
    if not contract_path.exists():
        raise FileNotFoundError(
            "matrix response contract is missing: {}".format(contract_path)
        )
    with contract_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != RESPONSE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported matrix response contract schema")
    if payload.get("kind") != "matrix_meta_follower":
        raise ValueError("checkpoint is not a matrix meta-follower")
    if algorithm is not None and payload.get("algorithm") != algorithm:
        raise ValueError("response algorithm does not match checkpoint contract")
    if payload.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("response checkpoint SHA256 does not match its contract")
    expected = response_game_contract(spec)
    if payload.get("response_game") != expected:
        raise ValueError(
            "response checkpoint is incompatible with the requested matrix game"
        )
    return payload


def load_response_model(checkpoint, algorithm="REINFORCE", device="cpu"):
    if algorithm != "REINFORCE":
        raise ValueError("The paper contextual follower uses REINFORCE")
    from .reinforce import Reinforce
    return load_checkpoint(Reinforce, checkpoint, device=device)







# These domain states remain local to the experiment. The shared environment
# owns scheduling, repeated-query reuse, and the per-episode commitment.
from stackelberg_pomdp.stackelberg import (StackelbergPOMDPEnv, ResponsePhase,
    ResponseReady, RewardPhase, RewardDone, LeaderQuery, Reward)










class MatrixExperiment(StackelbergPOMDPEnv):
    def max_episode_transitions(self):
        return self.response.max_episode_transitions(self.game)

    def max_executed_transitions(self):
        return self.response.max_executed_transitions(self.game)






def make_meta_leader_env(spec, *, deterministic_leader=True, **kwargs):
    from experiments.ablations.commitment_consistency.experiment import build
    return build(spec, deterministic_leader=deterministic_leader, **kwargs)


def make_tabular_q_leader_env(spec, *, deterministic_leader=True, **kwargs):
    from experiments.ablations.reward_timing.experiment import build
    return build(spec, deterministic_leader=deterministic_leader, **kwargs)


def __getattr__(name):
    if name == "MatrixMetaLeaderEnv":
        from experiments.ablations.commitment_consistency.game import Game
        return Game
    if name == "LegacyMatrixQLeaderEnv":
        from experiments.ablations.reward_timing.game import Game
        return Game
    raise AttributeError(name)
