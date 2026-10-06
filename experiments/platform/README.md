# Platform intervention with learning sellers

The visible files are `game.py`, `response.py`, `experiment.py`, this README
and `__init__.py`. Supporting implementation, settings and assets are in
`_support/`; `experiment.py` shows how the pieces connect.

`game.py` defines `Game(POMG)`: sellers choose prices in one step, then the
platform chooses display in the next. Both steps take a joint-action mapping
and return per-agent observations/rewards; actions for inactive roles are
ignored. Sellers observe their cost and price history; the leader sees prices.
`_support/market.py` supplies demand, profit and consumer-surplus arithmetic.
`response.py` starts a resumable seller Q-learning response for each of the two hidden-cost
conditions with the same leader commitment and seller seed. `_support/q_learning.cpp`
contains the tabular Q update loop; `_support/q_learning.py` provides its Python adapter; `_support/_native.py` handles compilation and ctypes.
`_support/reward.py` evaluates the recurrent cycles of the frozen seller policies.
`experiment.py` connects these pieces; `_support/_critic.py` handles private observations.

The response returns a `SellerPolicies` strategy profile, with one pair of
frozen pricing tables per hidden cost. The terminal price states are separate
in `ResponseReady.evaluation_context`; full training records remain diagnostics.
Reward evaluation selects the recurrent states reached from those terminal
prices, plays each state through the same POMG, and averages exact cycle
surplus. The ordinary POMG also supports fresh finite-horizon market play. This is a response-dependent,
cycle-average evaluation variant. The basic paper theorem concerns a fresh,
independently initialized game and finite-horizon cumulative reward, so it
does not directly cover this variant. The reward objective is unchanged.

The response asks for a leader action only when a seller price pair is first
encountered. The C++ learner pauses before using that unknown payoff and resumes
with the same Q tables, stopping counters and random stream after the answer.
The shared executor caches the answer for both cost conditions and reward
evaluation; gameplay can extend the same commitment at a new price pair.
Unvisited entries remain absent. This changes the training query order and is
recorded as a new scientific cohort, not an exact replay of the older upfront protocol.

The PPO budget and learning-rate schedule count complete episodes, because query
counts can vary. Held-out archive candidates alone fill any unvisited entries
with the current greedy policy; this never adds queries or training rows.

`_support/paper.json` and `_support/search.json` contain the run settings. `_support/` holds
training, evaluation and uniform-search runners. From the code root:
`python run.py platform --help` or `python run.py platform-search --help`.
A C++17 compiler is required (`CXX` can select it); generated libraries belong in run output folders.
Change market transitions in `game.py`, payoff arithmetic in `_support/market.py`,
follower behavior in `response.py` and
`_support/q_learning.cpp`, and the evaluation objective in `_support/reward.py`.

The paper runner selects `EpisodeWeightedPPO` from
`support/episode_weighting.py`. Its actor and entropy factor is the average
number of new decisions per episode divided by `n_queries + 1` (121 on the
paper grid), preserving the earlier normalization denominator. This is an explicit
experiment choice; the default `EpisodePPO` uses the ordinary library loss.
