# Paper ablations

- `commitment_consistency/`: makes the response/reward phase visible or hidden
  to test the leader's commitment consistency.
- `reward_timing/`: includes or excludes leader rewards during the follower
  response procedure.
- `response_credit/`: independent scalar REINFORCE diagnostic for credit from
  response queries.
- `_support/`: matrix games, learner implementations, training and evaluation
  shared by the first two ablations.

The existing commands still work from the code root:

```bash
python run.py commitment-consistency --help
python run.py reward-timing --help
python run.py response-credit --help
```

Each experiment has its own README. The matrix ablations use the PG and
SimpleQ procedures described there; the scalar diagnostic runs independently
of the shared phase executor. Full paper commands are in the root README.

The matrix experiment builders accept `lazy=True` for cached actions or
`lazy=False` for fresh queries in both phases. Existing paper commands retain
their `deterministic_leader` setting as an alias when `lazy` is unspecified.
The same policy distribution is used in both phases in standard mode; equal
sampled actions are not required. Changing mode creates a different run cohort.
