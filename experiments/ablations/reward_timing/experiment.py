"""Connect the coordination POMG, Q-learning response and reward evaluation."""
from experiments.ablations._support.game import MatrixExperiment
from .game import Game
from .response import QResponse
from ._support.reward import QReward


def build(spec=None, *, lazy=None, deterministic_leader=True, **kwargs):
    lazy = deterministic_leader if lazy is None else lazy
    game = Game(spec)
    response = QResponse(game.game_spec, **kwargs)
    return MatrixExperiment(game, response, QReward(), lazy=lazy, add_phase_observation=False)
