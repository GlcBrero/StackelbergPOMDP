# Focused replication checks

Run `make test` from the code root (or `python -m pytest tests -q`).
A C++17 compiler is needed. The Atari checks use deterministic stand-ins,
so this focused suite does not require ROMs or downloaded policies.

The retained tests cover the paper applications' joint-action POMG contracts,
private observations and per-agent rewards, lazy reuse and standard fresh queries across both
phases, fresh joint-action gameplay with private observation histories,
execution of returned follower profiles, new queries during reward,
default training, terminal critic observations and episode diagnostics,
MSPM's archived mechanisms, native MW updates
and BCCE arithmetic, platform admission payoffs and Atari phase execution. The small
JSON fixture contains MSPM reference results used to detect numerical changes.

The broader development suite (checkpoint selection pipelines, cluster-run
summaries, plotting and historical variants) is preserved in the author's
external archive. It passed before the public test suite was reduced.
These checks validate implementation contracts; full scientific reruns are
documented in the root README.md and reproduction/README.md.

`test_follower_certification.py` checks minimum-update enforcement, continuation
to the first eligible certificate, and failure without a usable checkpoint at
the cap. `test_statistical_comparisons.py` checks the paper's sign tests,
multiple-comparison corrections and resampling across leader and seller seeds.
It runs with the rest of this suite, or separately with `make test-statistics`.
