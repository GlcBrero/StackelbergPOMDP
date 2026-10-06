# Message-space sequential posted-price mechanism

The visible files are `game.py`, `response.py`, `experiment.py`, this README
and `__init__.py`. Supporting implementation, settings and assets are in
`_support/`; `experiment.py` shows how the pieces connect.

Read `experiment.py`, then `game.py`, `response.py` and `_support/reward.py`.
The `Game(POMG)` has two privately informed buyers. A message round is followed
by at most two offers; `reset()` returns private observations, and `step()`
accepts all three actions and returns per-agent rewards. Buyer acceptance is
strict and automatic; the follower strategies choose messages. The response runs multiplicative
weights and tests the last mixed, last deterministic and empirical-average
candidates in that order, stopping at the first exact BCCE certificate.
The reward is exact expected welfare relative to efficient allocation.

The response returns a `MessageProfile` containing the full joint message
distribution conditional on types. Exact reward integration preserves its correlation. It is a strategy
representation, separate from the returned certificate and diagnostics.

Certification starts after one complete MW update and permits at most 10,000
additional updates by default (`response_bcce_max_extra_updates`). This cap is
also present in all 125 retained paper run configurations. At the cap, failure
to certify any candidate raises an error and aborts the run; no uncertified
profile is passed to reward evaluation. The finite query bound and update cap
do not prove that certification succeeds for every possible leader mechanism.

- `response.py` performs the initial MW sweep and queries every payoff needed
  for exact certification through the shared episode commitment.
- `_support/mw.cpp` performs subsequent MW updates, keeps strategy history, and selects
  the first certified candidate. Python supplies one type draw per update,
  preserving its random stream. Once the payoff table is complete, these
  updates need no new leader actions.
- `_support/bcce.cpp` is the original native certificate kernel. It uses full strategy
  snapshots to preserve empirical correlation. `_support/_native.py` compiles both
  native sources and handles the Python/C++ boundary.
- `_support/mechanism.py` drives `game.reset()` and `game.step()` at fixed
  type/message profiles for exact payoff calculations.
- `_support/_critic.py` encodes private training inputs from `response.state`;
  multiplicative-weight arrays are not game state.
- `_support/paper.json` contains the paper run plan; `_support/_runner.py` handles its CLI and outputs.

The paper runner selects `EpisodeWeightedPPO`: it multiplies the actor and
entropy losses by the average number of new decisions per episode. This
episode normalization is optional training behavior, implemented alongside
platform's variant in `support/episode_weighting.py`; the shared default PPO
has no such weighting. Both use the library's PPO training method.

From the code root, use `python run.py mspm --help` for the paper command.
A C++17 compiler (`clang++` or `g++`) is required for the native response.
Compiled files go into a temporary cache, outside this source folder.
To change the setting, edit buyer values/type probabilities and mechanism
transitions in `game.py`. `response.py` controls payoff queries, learning
parameters and the update budget; `_support/mw.cpp` contains the numerical update and
candidate-selection rules. A different follower algorithm can replace
`response.py` using the same phase interface, without C++ or changes to the
common executor.
