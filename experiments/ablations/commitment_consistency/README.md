# Commitment consistency and phase information

This folder owns the joint-action `Game(POMG)` (`game.py`), frozen follower
response (`response.py`), reward play (`_support/reward.py`) and assembly (`experiment.py`).
`build()` selects whether the leader observes the phase. The paper uses
opponent memory (Start, C, D) and a separately trained contextual follower.
The response returns a `PolicyProfile` binding that follower to the queried
commitment. The reward phase resets the game and executes that profile on
private observation histories; it also records consistency diagnostics.
Checkpoint loading and follower encoding live in `_support/follower.py`.

The reusable matrix engine, follower pretraining, REINFORCE and evaluation
are shared with the other matrix ablation in `experiments/ablations/_support/`.
Change this setting's observations in `game.py` and query/replay rules in
`response.py` and `_support/reward.py`. Shared commitment bookkeeping stays in the core.

Run `python run.py commitment-consistency --help` from the code root.
Use the root README for training budgets, follower certification and seeds.

Train a paper follower with `python run.py normal-form meta-follower --certify
--memory-mode opponent --seed 0 --output-root results/normal-form` (one line).
The learner is checked every 100 updates, stops at the first max regret at most
0.25 after at least 500 updates, and fails after 10,000 if still uncertified.
Repeat for seeds 0–24. Use each seed's returned `model.zip` for all six leader
curves for that seed; the root README lists their learning rates and budgets.
