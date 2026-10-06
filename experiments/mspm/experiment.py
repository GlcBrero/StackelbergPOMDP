"""Assemble the MSPM game, certified MW response and exact welfare reward."""
from stackelberg_pomdp import StackelbergPOMDPEnv
from .game import Game
from .response import MSPMResponse
from ._support.reward import MSPMReward


def build(config=None, **kwargs):
    """Construct one paper experiment; the shared runner chooses its policy."""
    config = {**(config or {}), **kwargs}
    if not config.get("lazy", True):
        raise ValueError("MSPM's certified fixed-payoff response requires lazy=True; a fresh-sampling response is not implemented")
    game = Game(config.get("num_types", 2), config.get("num_messages", 2),
                   seed=config.get("seed", 0))
    response = MSPMResponse(
        epsilon=config.get("mw_epsilon", .1),
        tolerance=config.get("response_bcce_threshold", .05),
        fixed_seed=config.get("mw_fixed_seed", 0),
        max_extra_updates=config.get("response_bcce_max_extra_updates", 10000),
    )
    return StackelbergPOMDPEnv(game, response, MSPMReward(), lazy=config.get("lazy", True))
