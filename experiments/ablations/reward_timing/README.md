# Rewards during the follower response

`game.py` defines the one-shot `Game(POMG)` with joint actions and per-agent
rewards. `response.py` runs adaptation games, applies the terminal Q update,
and returns a fixed follower `PolicyProfile`. Exploration, initialization and
Q-table storage live in `_support/q_learning.py`. `_support/reward.py` resets
the game and plays against the returned strategy. Its additional evaluation
context carries Q diagnostics and replay records, not a gameplay starting state.
`experiment.py` selects whether adaptation rewards contribute to leader return.
This is the deliberate exception to the shared framework's zero-response-reward rule.

The matrix definitions, SimpleQ leader and paper runners are shared in
`experiments/ablations/_support/`. Change the game's payoffs through its specification,
follower adaptation and phase reward handling in `response.py`.

Run `python run.py reward-timing --help` from the code root.
The root README lists both payoff conditions and paper seeds/budgets.
