"""Connect the matching game and follower response to the shared executor.

Copy this folder to experiments/my_game/ and run:
    python run.py my_game --steps 1024 --output results/my-game
"""
from stackelberg_pomdp import StackelbergPOMDPEnv
from .game import Game
from .response import CopyLeaderResponse


def build(*, seed=0, lazy=True):
    """The shared runner supplies the default PPO policy and reward executor."""
    env = StackelbergPOMDPEnv(Game(), CopyLeaderResponse(), lazy=lazy)
    env.seed(seed)
    return env
