# One query, one follower strategy, one round of gameplay

Both players choose 0 or 1. The follower earns 1 for matching the leader.
The leader earns 1 when both choose 1, and 0 otherwise.

Read `response.py` first. It asks the leader for one action and returns a
`PolicyProfile` containing a `FollowerPolicy` that always plays that action.
A constant action is a complete
strategy here: the follower has only one decision and observes no changing state.

`game.py` is the underlying multi-agent game. `reset()` returns an observation
for each player. `step({"leader": ..., "follower": ...})` takes both actions
and returns both rewards. It has no reference to a follower policy.

The shared reward executor starts a fresh game, calls the follower's
`act(observation_history)`, queries the leader, and passes their joint action
to the game. It reports the leader's reward. The follower uses the strategy
selected during the response phase and never sees the leader's final action
before choosing its own action.

`experiment.py` connects the game and response to the shared executor:

```python
StackelbergPOMDPEnv(Game(), CopyLeaderResponse(), lazy=True)
```

With `lazy=True`, the leader's sampled action is reused in gameplay. Copying it
is an exact best response. For example, a response query returning 1 produces
the strategy "always play 1"; gameplay then plays (1, 1), giving leader reward 1.

With `lazy=False`, gameplay requests a fresh leader action from the same policy.
The follower still plays its chosen action, so the players can differ. This
deliberately simple response uses one sample; it is not generally an exact best
response to a stochastic policy. The executor supports either query mode.

From the code root:

```bash
python run.py example --check
python run.py example --steps 1024 --output results/example
```

To select standard execution, put `{"lazy": false}` in `settings.json` and pass
`--config settings.json`. Copy this folder to start a new setting; the launcher
discovers it automatically. This is a teaching example, not a paper experiment.
