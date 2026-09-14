# Reading the implementation

For a first run, follow the [one-episode walkthrough](README.md#follow-one-episode).
It traces a matrix game through the follower response, phase wrapper,
policy/cache, and PPO rollout collection, with a runnable example.

Start with the experiment you want to reproduce in
[`replication/targets.json`](replication/targets.json). Its command points to
the experiment entry point; the learning and environment code lives in
`stackelberg_pomdp/`.

## Environments and the StackPOMDP wrapper

[`envs/base.py`](stackelberg_pomdp/envs/base.py) defines the common
`BaseEnv` interface: leader and follower observations, actions, and episode
evaluation. Concrete domains are separate:

| Module under `stackelberg_pomdp/envs/` | Responsibility |
| --- | --- |
| `normal_form.py` | `BaseEnvSimpleMatrixGame`: normal-form matrix games |
| `matrix_design.py` | `BaseEnvMatrixDesignGame`: leader chooses payoff entries |
| `simple_allocation.py` | `BaseSimpleAllocation`: indirect allocation mechanism |
| `spm.py` | `BaseSPM` and `BaseMessageSPM`: sequential posted-price mechanisms |
| `bertrand.py` | `BertrandCompetitionEnv`: pricing and platform intervention |
| `matrix.py` | Repeated-matrix diagnostics and ablation environments |
| `atari/` | Space Invaders emulator, gameplay, and bilateral trade |

[`env_setups.py`](stackelberg_pomdp/env_setups.py) assembles domains and
their wrappers. Read
[`wrappers/core.py`](stackelberg_pomdp/wrappers/core.py) next for the shared
response/reward phase logic and
[`policies/cache.py`](stackelberg_pomdp/policies/cache.py) for cached leader
actions. Domain classes implement game rules; wrappers implement how the
leader interacts with the follower response algorithm.

## Normal-form appendix diagnostics

The [diagnostic CLI](stackelberg_pomdp/experiments/matrix_ablations.py) accepts
the paper's two figure names. Its implementation lives under
`stackelberg_pomdp/matrix_ablations/`: `presets.py` defines treatment curves,
`meta_training.py` trains the contextual follower, `leader_training.py` runs
the leader treatment, `evaluation.py` measures reward play, and `artifacts.py`
saves manifests and hashes. `reinforce.py` is shared by PG leaders and the
REINFORCE follower; `simple_q.py` supplies the reward-timing leader.
The [replication guide](replication/matrix_ablations/README.md) follows one run
and explains full-cohort planning and plotting.

## Atari leader evaluation

The command remains
[`replication/atari/evaluate_atari_stackpomdp_leader_sb3.py`](replication/atari/evaluate_atari_stackpomdp_leader_sb3.py).
Its `main` parses arguments, checks output paths, runs selection, and writes
the report. The implementation is organized by responsibility:

| Module under `stackelberg_pomdp/` | Responsibility |
| --- | --- |
| `evaluation/atari/workflow.py` | Screen candidates, confirm the winner on fresh seeds, create the selected checkpoint alias |
| `evaluation/atari/rollouts.py` | Build evaluation environments and retain episode, transition, and decision rows |
| `evaluation/atari/protocol_audit.py` | Audit phase order, cached actions, trade, inventory, and payoff identities |
| `evaluation/atari/interventions.py` | Apply factual or fixed economic commitments while preserving gameplay actions |
| `evaluation/atari/selection.py` | Paired economic gate, candidate ranking, and screen/confirmation consistency |
| `evaluation/atari/reporting.py` | Assemble CSV/JSON tables and publish complete artifact sets |
| `evaluation/atari/contracts.py` | Shared evaluation constants, configuration, and canonical data representations |
| `checkpoints/atari_evaluation.py` | Load and validate E1/E2 evaluation checkpoints |
| `checkpoints/atari_provenance.py` | Source/runtime fingerprints, manifest validation, and frozen gameplay actor checks |
| `checkpoints/atari.py` | Frozen policy loading, actor transfer, and parameter hashes |
| `checkpoints/files.py` | File hashes, checkpoint aliases, and safe rollback |

Follow `workflow.run_selection` for the complete evaluation sequence. The
scientific rules and command examples are in the
[Atari README](replication/atari/README.md).

## Compatibility code

Old checkpoints can contain Python module paths, so a few small adapters are
intentional. `envs.base` lazily resolves its six former concrete classes;
the Atari evaluator resolves historical helper imports through
`evaluation/atari/legacy.py`; and trainer modules re-export moved checkpoint
helpers. New code imports the owning modules directly. The existing
`baselines_utils.py` and `atari/stackpomdp_policy.py` also preserve serialized
policy paths.

Provenance records include source hashes, so moving code changes the current
fingerprint. The validator also accepts the exact public-release and
pre-extraction fingerprints recorded in `checkpoints/atari_provenance.py`.
Other mismatches still fail validation. Import/pickle regression tests and
the checkpoint round-trip tests exercise these compatibility boundaries.
