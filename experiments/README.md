# Write a game and its follower response

Start with [example/](example/). In this two-action matching game, the response
queries the leader once and returns a follower policy that always plays the
observed action. Gameplay executes that policy. Copy the folder and run from
the code root:

```bash
cp -R experiments/example experiments/my_game
python run.py my_game --check
python run.py my_game --steps 1024 --output results/my-game
```

The launcher discovers the folder automatically. `--check` validates one
episode with sampled actions before you start training. Choose a fresh output
directory for each run. These defaults help you develop a new experiment;
the paper's training and evaluation commands are in the root README.

`run.py` only lists and dispatches commands. Its `TARGETS` table names the paper
runners, and `ALIASES` lists shortcuts to those commands. Experiment-specific
defaults belong in the runners' argument parsers. Custom experiments use the
shared trainer in `support/custom.py` and need no entry in the launcher.

## The files you write

```text
my_game/
    __init__.py
    game.py          # Rules, observations, actions and transitions
    response.py      # Follower response algorithm
    experiment.py    # build(): connect game, response and reward
    _support/        # Optional reward/policy helpers, native code, settings, assets
```

1. **`game.py`** extends `POMG`: define per-agent `action_spaces` and
   `observation_spaces`, the `leader` id, and finite `horizon`. `reset()` returns
   fresh per-agent observations; `step(actions)` takes a joint action and returns
   `(observations, rewards, done, info)`, with observations/rewards keyed by agent.
   The game computes transitions and payoffs; it does not execute policies.
2. **`response.py`** extends `ResponsePhase`. Its `run(game)` requests leader
   actions, computes the follower response and returns
   `ResponseReady(PolicyProfile({agent: policy, ...}))`. Each policy implements
   `act(observation_history)` and uses only that follower's observations.
3. **`experiment.py`** defines `build(seed=0, ...)`, returning
   `StackelbergPOMDPEnv(game, response)`. The default reward phase plays the
   game against those followers. It resets the game and all game histories,
   assembles joint actions, and sums the leader's rewards up to the horizon.
   Pass game settings through `--config settings.json`.

Define `leader_observation(history)` and `leader_observation_space` in the game
to encode the leader's own history for the neural policy. Response queries use
the same encoding. In a one-state or memoryless setting the last observation
may suffice; general POMGs need the history or a sufficient statistic. This
encoding also determines cache keys. Neither phase identity nor a follower's
private history belongs in the leader's actor input.

`PolicyProfile` selects each agent's action from its own history. For correlated
followers, override the profile's `act(histories)` to retain the joint sampling
rule and information restrictions; do not replace a joint distribution with
independent marginals. Policies receive complete gameplay histories, so any
recurrent implementation must compute its memory from these histories rather
than retain simulation memory from the response phase.

Use `_support/reward.py` when the paper needs an exact expectation, a recurrent-cycle
calculation or another custom evaluation. Extend `RewardPhase`; its
`run(game, response_result)` can request actions, yield `Reward` events, and
return `RewardDone`. Pass this phase as the third argument to the executor.
Such evaluators may consume an exact strategy representation: MSPM integrates
its returned joint message distribution without sampling away correlation.
Platform returns `SellerPolicies` plus a separate `evaluation_context` containing
the terminal price states. Its custom evaluator retains the recorded cycle
objective. The default fresh-game evaluator rejects such response-dependent
evaluation state. All paper game-based settings implement `POMG`, including the native-backed
settings. Their exact/cycle evaluators execute that same game interface.
MSPM retains an exact conditional joint-distribution representation instead
of sampling the returned response. The scalar response-credit diagnostic is
the intentional standalone exception.

Most experiments use the default neural policy. For a custom policy, define
`POLICY` in a helper under `_support/` and import it in `experiment.py`. The default trainer
also accepts `POLICY_KWARGS` from `experiment.py`. Atari demonstrates this;
its neural network components are shared with pretraining through its local
`_support/` directory.

The default learner is `EpisodePPO`, a subclass of Stable-Baselines3 PPO.
It inherits the library's `train()` method and extends rollout collection to
finish complete episodes. You do not need to write training code for a new
setting. MSPM and platform explicitly select `EpisodeWeightedPPO` from the
shared `support/episode_weighting.py` helper to preserve their paper loss
normalization. `stackelberg_pomdp/training/rollout_buffer.py` stores episodes
and calculates returns across internally executed steps. Its optional
action-credit field is metadata; Atari's masking/normalization helpers
and losses live in `_support/_ppo.py`, and REINFORCE's masked loss stays
with the matrix ablations.

Load older checkpoints through `support.checkpoint_compat.load_checkpoint`: 

```python
from support.checkpoint_compat import load_checkpoint
model = load_checkpoint(EpisodePPO, checkpoint_path, env=env)
```

This enables the historical class-path redirects only when loading a
checkpoint. Importing `stackelberg_pomdp` alone does not enable them.

## How an action request works

In `stackelberg_pomdp/stackelberg.py`, `_episode()` defines the two phases:
run the response algorithm, then evaluate its returned follower strategies.
`_run_until_action_needed()` resumes that computation, answers cached queries,
and pauses for a new leader action. `Commitment` owns the shared episode map.

The supporting details are separate: `stackelberg_pomdp/episode.py` contains
`EpisodeLogger` (diagnostics only) and `RewardAccumulator` (rewards and elapsed
time). `stackelberg_pomdp/observations.py` adds centralized-critic inputs through
`CriticObservation`. This adapter runs when an observation is emitted, including
at termination, so it records the correct phase and state. Experiment authors
still use the same constructor and response/game interfaces.

```python
action = yield LeaderQuery(observation)
```

`yield` pauses the response or reward procedure at that line. Select the mode
in your setting's `experiment.py`:

```python
env = StackelbergPOMDPEnv(game, response, reward, lazy=True)
```

With `lazy=True`, the executor checks the episode's private action map. It
reuses an existing answer, or requests and stores a new one. Reward gameplay
can extend the same map; reset clears it. With `lazy=False`, every request
goes to the policy for a fresh action, in both phases. The same policy maps
observations to action distributions, although its sampled actions can differ.
Response code uses `LeaderQuery(...)` in either mode. A response for standard
execution must support stochastic answers; it cannot silently reuse a payoff
table that assumes fixed leader actions.

The paper's basic theorem uses fresh sampling (`lazy=False`) and independently
initialized finite-horizon gameplay. Cached execution (`lazy=True`) selects a
sampled deterministic commitment for the episode; it generally defines a
different response distribution from repeatedly sampling a stochastic policy.

The actor receives no phase flag, follower state or cache. Optional `critic:`
fields go only to the centralized critic. A response can implement
`critic_observation(game, commitment, phase)` to expose private training state;
that state stays with the response. `_critic.py` keeps that encoding
separate from the scientific procedure. Custom policies must preserve the
same separation, as Atari's policy does.

The response separately chooses when to query: upfront enumeration and queries
made as needed both use this interface. For fresh actions even in lazy mode, use
`LeaderQuery(observation, key=None)`, as Atari does for live visual decisions.
Its economic actions still use the episode commitment. Ordinary response
queries earn zero leader reward; the reward-timing ablation explicitly changes
that objective to measure its effect.

The example and matrix ablations support both query modes. The paper MSPM,
platform and Atari builders currently require lazy execution and reject
`lazy=False`: their certified payoff table, native fixed rule, or frozen
economic context assumes cached answers. They need different response
implementations before supporting standard execution. The independent scalar
response-credit ablation already samples freshly without the shared executor.

## Reading the paper experiments

| Folder | Scientific procedure |
| --- | --- |
| `mspm/` | Mechanism rules in `game.py`; payoff queries in `response.py`; repeated multiplicative-weight updates and candidate selection in `_support/mw.cpp`, using `_support/bcce.cpp`; exact welfare in `_support/reward.py`. |
| `platform/` | Joint-action transitions in `game.py`, demand/payoff arithmetic in `_support/market.py`; `QLearningResponse` in `response.py` drives resumable seller learning; the native update loop is `_support/q_learning.cpp`; `_support/reward.py` evaluates frozen seller cycles. |
| `atari/` | Bilateral trading game, five economic queries to a frozen meta-follower, live reward gameplay and a custom policy. |
| `ablations/commitment_consistency/` | Matrix game, pretrained follower response and reward play for the phase-information ablation. |
| `ablations/reward_timing/` | Matrix game, follower Q-learning adaptation and the response-reward ablation. |
| `ablations/response_credit/` | Independent scalar REINFORCE diagnostic in `experiment.py`; it needs no game or phase executor. |

Platform seller Q-learning requests a leader action when it first encounters
an uncommitted price pair. The native learner pauses for the answer, then resumes
with its Q-tables and random stream intact. Response and reward evaluation share
the same episode commitment; the paper's completed cohort uses this protocol.

Keep every setting-specific helper, input, native source and configuration
inside its experiment folder. Each main application exposes the same five
files shown above; everything else lives in its `_support/` directory. The
README explains what to edit and how to run it. The matrix ablations share
their engine and learners
in `ablations/_support/`. Root `support/` holds shared output/training utilities.
Atari checkpoint generation is kept
separate in `pretrained/` and imports the same experiment implementation.
MSPM and platform each use `_support/_native.py` for compilation and the
Python/C++ boundary. New settings can be entirely Python. Start with
`experiment.py`, then read the game and response; follow its imports for reward details.
