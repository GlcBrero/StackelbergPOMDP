# MSPM MW Response and BCCE Verification

This note summarizes the MSPM setting, the follower strategy obtained from multiplicative weights (MW), and the BCCE-style verification diagnostic used for the current StackelbergPOMDP code.

## MSPM Setting

The tested experiment is the sequential posted-price mechanism (MSPM) with:

- 2 followers: `A0`, `A1`
- 3 private types per follower
- 2 possible follower messages
- Uniform type distribution
- A learned leader mechanism/policy

In each StackelbergPOMDP episode, the response phase runs the follower response algorithm against the current leader policy. For this setting, one full MW update requires:

```text
2 followers * 2 messages = 4 reward-generating response subepisodes
```

Therefore:

```text
300 response subepisodes = 75 full MW updates
400 response subepisodes = 100 full MW updates
```

## Final MW Strategy After 100 Updates

For the checkpoint:

```text
rl_model_19678_steps.zip
```

the final MW mixed strategy after 100 full MW updates was:

```text
A0:
  type 0 -> message 0: 0.5000, message 1: 0.5000
  type 1 -> message 0: 0.4976, message 1: 0.5024
  type 2 -> message 0: 0.4922, message 1: 0.5078

A1:
  type 0 -> message 0: 0.5000, message 1: 0.5000
  type 1 -> message 0: 0.5070, message 1: 0.4930
  type 2 -> message 0: 0.5066, message 1: 0.4934
```

The strategy is close to uniform randomization. The largest tilt is about 1.6 percentage points away from 50/50.

If converted to a pure argmax strategy, this becomes:

```text
A0: type 0 -> 0, type 1 -> 1, type 2 -> 1
A1: type 0 -> 0, type 1 -> 0, type 2 -> 0
```

This argmax conversion is stronger than what MW theory directly supports. MW naturally produces a randomized or empirical-play object, not necessarily a pure strategy.

## BCCE Verification

The BCCE diagnostic checks whether a follower type can gain by committing ex ante to a fixed alternative message.

For each follower `i`, type `theta_i`, and deviation message `a_i'`, we compare:

```text
expected payoff from MW randomized play
vs.
expected payoff from always deviating to message a_i'
```

The expectation averages over:

- the other follower's type,
- the other follower's MW mixed action,
- the follower's own MW mixed action,
- and, for empirical verification, the MW update snapshots.

The type-level deviation gap is:

```text
gap(i, theta_i)
=
max_a_i' E[u_i(a_i', a_-i)]
-
E[u_i(a_i, a_-i)]
```

where `a_i` and `a_-i` are drawn from the MW randomized strategy.

The reported BCCE gap is:

```text
max_type_BCCE_gap = max_i,theta_i max(0, gap(i, theta_i))
```

If this value is zero, no follower type can profitably deviate to a fixed message before seeing the randomized recommendation. Small positive values measure approximate BCCE error.

## Diagnostic Results

For the screenshot-style checkpoint after 100 MW updates:

```text
pure_BNE_max_deviation = 0.001653
max_type_BCCE_gap      = 0.000823
max_ex_ante_BCCE_gap   = 0.000819
```

For the empirical distribution over MW updates, the learned policy was already very close to BCCE:

```text
updates   max BCCE gap
1         0.000811
2         0.001710
5         0.000811
10        0.000812
20        0.000813
50        0.000815
75        0
100       0
```

Using a tolerance of `1e-3`, the empirical MW play is essentially converged almost immediately, with small oscillations around the tolerance. Using `2e-3`, it is within tolerance throughout the tested horizon.

## Interpretation

The MW response in this MSPM run is not strongly selecting a deterministic bidding strategy. It keeps followers nearly indifferent across messages and remains close to 50/50 randomization. This is consistent with the theory if MW is interpreted as inducing an approximate BCCE through randomized or empirical play.

The main code implication is that reward-phase play and equilibrium verification should ideally use the MW randomized or empirical strategy, not only the pure `argmax(weights)` strategy.
