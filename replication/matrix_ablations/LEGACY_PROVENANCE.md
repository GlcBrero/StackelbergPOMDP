# Legacy provenance for the matrix diagnostics

This package is the maintained implementation.  Matthias Gerstgrasser's
`StackeRLberg` repository is a read-only behavioral reference; the maintained
code does not import it or depend on its Ray/RLlib environment.  References
below are repository-relative paths inside that read-only checkout.

## Paper-panel map

| Paper output | Historical run definition | Historical plotted cohort | Maintained interpretation |
|---|---|---|---|
| `fig_hidden` (PG) | `smipd_hiddenqueries_pg_pg_new`, closest source commit `c603134` | 10 seeds per observed/hidden condition | Policy-gradient leader with rolled-out versus deleted query transitions |
| `fig_hidden` (ES) | `smipd_es_pg_new`, source commit `bd971a8` | LR `.004`; only seeds 1 and 2 completed per condition | ES leader with the same frozen response interface |
| `fig_memory_pg` | historical `smipd_tellleader_pg_pg`; phase-aware commit `8421178`, phase-unaware commit `b437644` | intended 10 seeds per LR/condition, with two missing phase-unaware LR `.03` runs | Phase bit visible versus unavailable; “memory” never meant action-history memory |
| `fig_reset` | `bots_pg_tabularq`; reset group `test_2_longer` at `32dc86f`, ongoing group `test_4_noreset` at `6551100` | notebook filters to seeds 1--3 and LRs `.008`, `.015`, `.03` | Fresh versus carried response state, using a correct terminal Q target |
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

## Ported leader policy gradient

The maintained `PG` leader is a separate treatment from both the response
REINFORCE model and the A2C/PPO redesign.  It ports the behaviorally relevant
settings from `smipd_hiddenqueries_pg_pg_new`:

- RLlib PG with LR `.008`, complete episodes, and a minimum of 100 executed
  environment transitions per update;
- a bias-free linear categorical policy with row-wise normc initialization at
  scale `.01` and Adam epsilon `1e-8`;
- `gamma=1`, undiscounted reward-to-go, and exact loss
  `-mean(log pi(a|s) * return_to_go)`;
- no trained value function, baseline, entropy term, advantage normalization,
  gradient clipping, or minibatch epochs; and
- 2,000 historical outer-loop leader updates.

The hidden-query callback deleted query transitions from the leader batch only
after they executed.  Complete-episode collection consequently gives the
paper five-query game 10 episodes and 100 executed transitions per update,
with 100 observed-query samples versus 50 hidden-query samples.  The final
legacy three-query configuration gives 13 episodes and 104 executed
transitions, with 104 versus 65 stored samples.

Crucially, historical matrix leader training did not cache a sampled action
table between query and reward calls.  Each visit sampled independently from
the same stationary policy distribution.  The `deterministic_leader` flag in
the alternating trainer applied while the follower was being trained, not
during the leader's own PG update.  The maintained PG port preserves this:
training visits are independent, while deterministic evaluation uses the same
argmax policy in both phases.  Atari's deliberate exact-action replay is a
different protocol and must not be imported into this matrix treatment.

## Deliberate paper-spec differences

The final historical hidden-query runs set `small_memory=True`: three states
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

## Historical ES forensic correction

The ES curves in the archived `fig_hidden` cohort cannot support the current
Modified-Prisoner's-Dilemma caption.  All four plotted ES run configurations
record `matrix_name=prisoners_dilemma`; they were launched on 2022-11-08 from
the ordinary-PD configuration.  The ES configuration was changed to modified
PD only on 2022-11-12, after those runs, and the plotting notebook did not
filter by matrix.  The historical PG curves did use modified PD.

Two additional configuration details make the old labels misleading.  The
legacy sweep set `leader_config["lr"]` to `.004` or `.015`, but Ray RLlib 2.0.1
ES reads `stepsize`, not `lr`; the effective Adam stepsize was therefore its
unchanged default `.01`.  The trainer also removed the complete `multiagent`
configuration before constructing ES.  Consequently the declared no-bias
linear model was not installed and Ray's default two-layer, 256-unit tanh
network was effective.

The maintained ES treatment is explicitly a new qualitative validation.  It
uses modified PD, joint five-state memory, and the certified finite follower
response, while restoring native RLlib 2.0.1 ES: the default two-layer
256-unit tanh stochastic policy, `MeanStdFilter`, sigma `.02`, centered ranks,
Adam stepsize `.01`, and L2 coefficient `.005`.  Every run records Ray's native
`episode_reward_mean`, explicit stochastic evaluation, measured batch
overshoot, and a hashed terminal RLlib checkpoint containing both weights and
the synchronized filter.  Historical ES artifacts remain forensic evidence
only.

## Known legacy issues retained only as forensic evidence

- All four legacy notebook panels set Seaborn `ci="sd"`; their bands are
  standard deviations even though the revised captions say standard error.
  Maintained plots compute sample SEM across independent seeds explicitly.
- The September 2022 reset/ongoing runs predate commit `f6189ac`, which fixed
  terminal Q updates.  Those runs bootstrapped from a terminal next-state Q
  value with `gamma=1`, creating sticky carried values.  The maintained code
  uses the correct terminal target and must not recreate that bug as evidence.
- The response-reward runs postdate the terminal fix and used parameter noise,
  ten Q-response episodes, `alpha=.2`, zero Q initialization, and a fresh
  response table per leader episode.  Evaluation excluded response-phase
  leader reward even when training included it.

These differences are scientific metadata, not compatibility quirks.  Every
maintained artifact records the profile, state order, reward offset, response
protocol, seed, source hash, and uncertainty definition.
