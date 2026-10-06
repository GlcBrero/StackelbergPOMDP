# Legacy provenance for the matrix diagnostics

This package is the maintained implementation.  Matthias Gerstgrasser's
`StackeRLberg` repository is a read-only behavioral reference; the maintained
code does not import it or depend on its Ray/RLlib environment.  References
below are repository-relative paths inside that read-only checkout.

## Paper-panel map

| Paper output | Historical run definition | Historical plotted cohort | Maintained interpretation |
|---|---|---|---|
| `fig_memory_pg` | historical `smipd_tellleader_pg_pg`; phase-aware commit `8421178`, phase-unaware commit `b437644` | intended 10 seeds per LR/condition, with two missing phase-unaware LR `.03` runs | Phase bit visible versus unavailable; “memory” never meant action-history memory |
| `fig_bots_leaderreward` | `bots_dqn_tabularq_out_of_eq`, group `test_11_rllib_bugfix`; zero-penalty prefix `a1699` near `8db223b`, penalty prefix `ead38` near `a538714` | 10 seeds per matrix/reward-inclusion cell | Reward-game performance with response-phase leader reward excluded versus included |

The authoritative legacy locations are:

- experiment settings: `stackerlberg/train/experiments/configurations.py`;
- query and Q-response construction: `stackerlberg/train/make_env.py`;
- query execution: `stackerlberg/wrappers/observed_queries_wrapper.py`;
- tabular response updates: `stackerlberg/wrappers/tabularq_wrapper.py`;
- alternating PG training: `stackerlberg/trainers/stackerlberg_trainable.py`;
- leader-query deletion: `stackerlberg/trainers/callbacks.py`;
- random commitment sampling: `stackerlberg/trainers/utils.py`;
- linear no-bias model: `stackerlberg/models/linear_torch_model.py`;
- final run filters and transforms: `plotting/plots_iclr_response.ipynb`.

## Ported policy-gradient response

The maintained categorical response copies the behaviorally relevant RLlib
PG mechanics without adding Ray as a dependency:

- a no-bias linear two-action softmax over one jointly encoded categorical
  observation;
- uniformly sampled deterministic leader tables (equivalent to independently
  randomizing each legacy linear leader logit and taking deterministic argmax);
- stochastic follower actions during training and deterministic argmax during
  certification;
- undiscounted reward-to-go (`gamma=1`), no critic, no baseline, and loss
  `-mean(log pi(a|s) * return_to_go)`;
- Adam at the declared learning rate;
- complete five-step episodes with a 100 executed-transition collection target
  (100 actual transitions for the paper five-query profile and 104 for the
  historical three-query profile);
- immutable checkpoints and exhaustive deterministic response certification.

RLlib 2.0.1 implements these details in `PGTorchPolicy.loss`,
`post_process_advantages`, `compute_advantages`, and
`TorchPolicyV2.optimizer`.  The legacy follower used LR `.02`, 100-transition
batches, and 500 pretraining updates.

## Maintained leader recipes

Commitment consistency uses linear PG leader training through
`matrix_ablations/reinforce.py`: Adam, undiscounted
reward-to-go, no baseline, no entropy bonus, and complete episodes collected
to at least 100 transitions per update. The leader accepts categorical Dict
observations and caches actions by the complete actor-visible observation.
The phase key therefore distinguishes response from reward play only in the
visible treatment. Follower REINFORCE remains uncached during its training.

Reward timing uses `matrix_ablations/simple_q.py`, an SB3-based implementation
of the historical SimpleQ mechanics: bias-free linear Q values, uniform
50,000-transition replay, batch size 1024, Adam at LR 0.1, Huber loss,
undiscounted one-step terminal-masked targets, learning after 100 transitions,
and target copies every 500 transitions. Gaussian parameter noise starts at
standard deviation 1 and adapts by a factor of 1.01 against target KL zero;
one noisy commitment is used for a complete outer episode. Clean network
weights are used for learning and deterministic evaluation.

The `bots_dqn_tabularq_out_of_eq` configuration enables follower parameter
noise; `make_matrix_tabularq_env` fixes its scale at 0.1. The maintained
reward-timing preset specifies this scale, a Q learning rate of 0.2,
a zero initial Q-table, and follower payoff 0.001 at the leader-preferred
coordination outcome.

These are corrected training recipes, **not exact historical training
replays**. In addition to the state/reward differences below, the maintained
PG leader fixes sampled actions within an episode, while historical PG
training sampled at repeated visits. SimpleQ's SB3 collector and replay RNG
differ from RLlib's scheduling and RNG. Each held-out episode uses an
independently initialized response. Learning curves for both algorithms are
evaluated after optimizer updates. Each run records these
choices in `leader_protocol` and `evaluation_response_state`.

## Deliberate paper-spec differences

The historical three-state profile uses `small_memory=True`: three states
(`Start` plus the opponent's previous action), three leader queries, and eight
deterministic leader tables.  The current paper instead specifies five states
(`Start`, `CC`, `CD`, `DC`, `DD`), five queries, and 32 tables.  The maintained
paper profile follows the paper and jointly encodes 160 follower contexts
(five follower states times 32 tables).  A legacy three-state profile is only
a parity diagnostic; it cannot be presented as evidence for the five-state
paper experiment.

The historical repeated-game environment used its default reward offset
`-2.5`, while the plotting notebook transformed leader return by `/5 - 1.5`,
yielding the paper's effective `-4` display scale.  The maintained paper
profile uses the paper-centered rewards directly.  The constant shift leaves
exact best responses unchanged but can change finite-sample PG variance, so it
is recorded in every run config.
