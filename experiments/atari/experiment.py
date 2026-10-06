"""Assemble the bilateral game, frozen response, reward play and custom policy."""
from stackelberg_pomdp import StackelbergPOMDPEnv
from .game import BilateralAtariConfig, Game, DualAtariTradeCore
from .response import AtariResponse
from ._support.reward import AtariReward
from ._support.leader_policy import POLICY


def build(
        *,
        leader_role,
        response_checkpoint=None,
        config=None,
        core_factory=DualAtariTradeCore,
        response_model_factory=None,
        side_factory=None,
        env_factory=None,
        device="cpu",
        deterministic_leader=True,
        lazy=True,
):
    """Compose Atari leader training with the shared StackPOMDP wrapper."""

    if not lazy or not deterministic_leader:
        raise ValueError("The paper Atari response requires lazy=True for economic commitments; a fresh-sampling response is not implemented")
    resolved = (config or BilateralAtariConfig()).resolved()
    game = Game(
        leader_role=leader_role,
        config=resolved,
        core_factory=core_factory,
        side_factory=side_factory,
        env_factory=env_factory,
    )
    response = AtariResponse(
        game,
        response_checkpoint=response_checkpoint,
        response_model_factory=response_model_factory,
        device=device,
        seed=resolved.seed,
        deterministic_leader=deterministic_leader,
    )
    result = StackelbergPOMDPEnv(game, response, AtariReward(response),
                                lazy=lazy, add_phase_observation=False)
    return result
