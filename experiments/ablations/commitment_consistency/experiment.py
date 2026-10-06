"""Connect the matrix POMG, frozen meta-response and diagnostic evaluation."""
from experiments.ablations._support.game import MatrixExperiment
from .game import Game
from .response import MetaResponse
from ._support.reward import MetaReward


def build(spec=None, *, lazy=None, deterministic_leader=True, phase_observable=None, **kwargs):
    lazy = deterministic_leader if lazy is None else lazy
    game = Game(spec, phase_observable=phase_observable)
    response = MetaResponse(game.game_spec, **kwargs)
    return MatrixExperiment(game, response, MetaReward(), lazy=lazy, add_phase_observation=False)
