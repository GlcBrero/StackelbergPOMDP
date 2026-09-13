"""Stable evaluation schemas, thresholds, and canonical value conversions."""

import hashlib
import json
from pathlib import Path

import numpy as np

from stackelberg_pomdp.atari.protocol import NUM_TRADE_EVENTS, actor_observation
from stackelberg_pomdp.checkpoints.files import checkpoint_sha256
from stackelberg_pomdp.envs.atari.bilateral import BUYER, SELLER
from stackelberg_pomdp.envs.atari.space_invaders import default_rom_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "replication/atari/results/e2_selections"


PROTOCOL_ATOL = 1.0e-6


EVALUATOR_NAME = "clean_atari_e2_selector_v2"


CANONICAL_GAMEPLAY_HORIZON = 200


SCREEN_EPISODES = 20


CONFIRMATION_EPISODES = 100


ECONOMIC_INTERVENTION_SCHEMA = "clean_atari_e2_commitment_intervention_v1"


ECONOMIC_CONTROL_COMMITMENTS = {
    "all_zero": (0.0,) * NUM_TRADE_EVENTS,
    "all_one": (1.0,) * NUM_TRADE_EVENTS,
}


ECONOMIC_GATE_MIN_FACTUAL_PAYOFF = 0.25


ECONOMIC_GATE_MIN_MEAN_ADVANTAGE = 0.25


ECONOMIC_GATE_MIN_MEDIAN_ADVANTAGE = 0.0


ECONOMIC_GATE_MIN_WIN_RATE = 0.60


ECONOMIC_GATE_MIN_TOTAL_SHOTS = 4.0


CANONICAL_EVENT_ACTION_MASK = (1.0, 0.0, 1.0, 1.0, 0.0, 0.0)


ECONOMIC_GATE_HYPOTHESIS = (
    "predeclared five-bullet interior-surplus hypothesis: the learned "
    "commitment outperforms both endpoint commitments; this is diagnostic "
    "for this experiment and is not a universal equilibrium condition"
)


E1_BUYER_INIT_MEAN = 0.95


E1_BUYER_INIT_CONCENTRATION = 10.0


E1_SELLER_INIT_MEAN = 0.5


E1_SELLER_INIT_CONCENTRATION = 2.0


E2_INIT_MEAN = 0.5


E2_INIT_CONCENTRATION = 2.0


SELECTION_RULE = (
    "exclude any checkpoint with a screen protocol violation",
    "exclude any checkpoint that fails this experiment's predeclared five-bullet interior-surplus endpoint-dominance hypothesis",
    "maximize mean leader payoff",
    "maximize median leader payoff",
    "maximize minimum leader payoff",
    "minimize leader-payoff standard deviation",
    "prefer fewer training timesteps",
    "prefer lexicographically smaller checkpoint SHA-256",
    "confirm only the screen-preselected top candidate on disjoint seeds",
    "create the selected alias only after confirmation passes",
)


def opposite_role(role):
    if role == SELLER:
        return BUYER
    if role == BUYER:
        return SELLER
    raise ValueError(f"unknown Atari role: {role!r}")


def _canonical_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")


def environment_config(args):
    """Return the exact actor-visible and economic environment contract."""

    fixed = getattr(args, "fixed_event_steps", None)
    rom_path = getattr(args, "rom_path", None)
    rom = (
        Path(rom_path).expanduser().resolve()
        if rom_path is not None
        else Path(default_rom_path()).resolve()
    )
    if not rom.is_file():
        raise FileNotFoundError(f"Space Invaders ROM does not exist: {rom}")
    rom_hash = checkpoint_sha256(rom)
    return {
        "leader_role": str(args.leader_role),
        "follower_role": opposite_role(args.leader_role),
        "gameplay_horizon": int(args.gameplay_horizon),
        "event_tail_steps": int(args.event_tail_steps),
        "fixed_event_steps": None if fixed is None else list(fixed),
        "seller_game_reward_scale": 0.1,
        "buyer_game_reward_scale": 1.0,
        "noop_max": int(args.noop_max),
        "frame_skip": 4,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
        "max_frames": int(args.max_frames),
        "rom_path": str(rom),
        "rom_sha256": rom_hash,
        "deterministic_actor": True,
        "leader_action_cache": True,
        "leader_economic_input": "event_only",
        "response_economic_input": "full",
        "response_algorithm": "frozen_meta_policy",
    }


def environment_config_sha256(config):
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def actor_observation_sha256(observation):
    """Hash exactly the fields entering the actor and action-cache key."""

    digest = hashlib.sha256()
    for name, value in actor_observation(observation).items():
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _action_list(action):
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"E2 policy produced an invalid full action: {action!r}")
    return [float(value) for value in values]


def _number(value):
    if isinstance(value, (bool, int, float, np.number)):
        return float(value)
    return None


def _close(left, right, *, atol=PROTOCOL_ATOL):
    return bool(abs(float(left) - float(right)) <= float(atol))


def _ranges_overlap(start_a, count_a, start_b, count_b):
    end_a = int(start_a) + int(count_a) - 1
    end_b = int(start_b) + int(count_b) - 1
    return max(int(start_a), int(start_b)) <= min(end_a, end_b)
