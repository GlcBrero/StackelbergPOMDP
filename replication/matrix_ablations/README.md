# Matrix-game qualitative replications

This directory reruns the four normal-form diagnostics in the paper.  Most
treatments use the current Stable-Baselines3 codebase; hidden-query ES alone
uses native RLlib 2.0.1 in a dedicated optional environment.

The implementation is isolated in `stackelberg_pomdp/matrix_ablations/`.  It
does not import or modify Atari environments, trainers, checkpoints, or logs.
The historical Matthias `StackeRLberg` checkout is used only as a read-only
behavioral source.  The panel-to-source map, version differences, and known
historical issues are recorded in [LEGACY_PROVENANCE.md](LEGACY_PROVENANCE.md).

## Experimental contract

- Repeated Prisoner's Dilemma uses five rounds and, by default, five canonical
  states: `Start`, `CC`, `CD`, `DC`, and `DD`.  This matches the current paper.
  The legacy three-state opponent-memory encoding remains available through
  `--memory-mode opponent` for sensitivity checks.
- The maintained run profile is named `paper_joint_v1`.  Checkpoint contracts
  include the complete immutable profile, state order, query order, context
  encoding, reward convention, training protocol, and measured step counters;
  artifacts from different profiles cannot be combined silently.
- Repeated-game rewards use the paper's centered matrices directly (the
  positive legacy source matrices plus an offset of -4), so the optimal
  per-stage leader reward is zero without a plotting-time translation.
- E1 trains a contextual follower against uniformly sampled deterministic
  leader tables.  E2 queries the ordered table once, freezes it as context,
  resets the game, and calls that frozen follower deterministically.
- A checkpoint contract binds E1 to the centered follower payoffs, horizon,
  memory mode, state order, spaces, SHA256, and measured response regret.  The
  modified and canonical Prisoner's Dilemma share the same follower game, so a
  seed-matched E1 checkpoint is intentionally reused across both diagnostics.
- The one-shot Q follower has an explicit terminal target.  The reset study
  uses small-normal initialization, alpha 0.1, and fixed epsilon 0.1.  The
  response-reward study uses zero initialization, alpha 0.2, and per-response
  parameter noise with standard deviation 0.1.  Both run ten response updates
  followed by one noiseless argmax reward game.
- ES is native RLlib 2.0.1 with one rollout worker, its default two-layer
  256-unit tanh policy and `MeanStdFilter`, sigma `.02`, Adam stepsize `.01`,
  and L2 coefficient `.005`.  Native batches can overshoot their configured
  minima, so progress records measured executed transitions.
- Evaluation always reports reward-game performance only.  Carried-Q
  evaluation is marked as a dependent response stream and uses an explicit
  warmup; uncertainty in paper plots is computed across independent training
  seeds, not across that stream.

E1 uses a clean Ray-free port of the historical vanilla REINFORCE response: a
bias-free linear softmax over the complete 160-way categorical context,
undiscounted return-to-go, no baseline, and Adam.  Its 100 executed-step
collection geometry yields 50 follower samples per optimizer update for the
paper profile.  Checkpoints are saved only after completed updates and record
both follower samples and executed-equivalent query-plus-game transitions.

The leader CLI also provides a separate, Ray-free `PG` treatment matching the
historical matrix leader: a bias-free linear categorical policy, row-normc
initialization at `.01`, Adam (`eps=1e-8`), undiscounted reward-to-go, and no
critic, baseline, entropy term, normalization, or clipping.  Complete episodes
are collected until at least 100 environment transitions execute.  For the
five-state paper game, observed queries therefore store 100 samples and hidden
queries store 50 samples per update, while both execute 100 transitions.  For
the historical three-state game, the corresponding counts are 104 and 65
stored samples, with 104 executed transitions.

Matrix PG intentionally does **not** use Atari-style exact action replay.
During training, every visit independently samples from one stationary policy
distribution shared by the query and reward phases.  Hidden queries execute
but are removed from the gradient rollout; deterministic evaluation uses the
same policy's argmax in both phases.  This distinction is necessary for the
hidden-query contrast and is recorded in every PG config and manifest.

The faithful PG CLI is currently restricted to `hidden_queries`.  The other
diagnostics had different historical response and observation protocols; they
must be audited separately instead of silently reusing this batch geometry.

## Native RLlib hidden-query ES

The maintained `ES` treatment uses the actual
`ray.rllib.algorithms.es.ES` implementation from Ray 2.0.1.  Ray is an
optional dependency used only by
`stackelberg_pomdp/matrix_ablations/rllib_es.py`; the main environment remains
Ray-free.  Create the dedicated environment with
`environment-ray-es.yml`.

The effective settings match the historical ES runtime: Torch, one rollout
worker, RLlib's two-layer 256-unit tanh policy, `MeanStdFilter`, parameter-noise
standard deviation `.02`, Adam stepsize `.01`, L2 coefficient `.005`,
`episodes_per_batch=1000`, `train_batch_size=1000`, `eval_prob=.03`, and a
trailing report window of ten.  The historical `lr` sweep did not control ES;
RLlib reads `stepsize`.

The paper-profile rerun deliberately keeps the current modified-PD payoff
matrix, joint five-state memory, and certified finite follower response.  The
driver evaluates that response on every commitment/state pair, validates the
lookup exhaustively against the checkpoint, and sends only the resulting plain
action table to Ray workers.  The leader remains stochastic on every visit,
including repeated visits to the same state.

Each progress row records both native
`result["episode_reward_mean"] / 5` and a separately seeded explicit
stochastic evaluation.  RLlib's time-based worker batches may exceed the two
configured batch minima.  The paper curve therefore aligns independent seeds
by native ES iteration; each row and manifest also retains measured cumulative
timesteps for resource accounting.  `--timesteps` is not an ES control and is
recorded as null; `--es-iterations` controls training.  The terminal native
RLlib checkpoint contains both policy weights and the synchronized
`MeanStdFilter`.

There is one ES implementation and no backend selector.  Earlier experimental
prototypes have been removed from active code.  Their raw outputs and the
historical StackeRLberg reports remain provenance, not release treatments.

One smoke run:

```bash
conda activate stackelberg-pomdp-ray-es
python -m stackelberg_pomdp.experiments.matrix_ablations leader \
  --experiment hidden_queries --condition observed --algorithm ES \
  --profile-id paper_joint_v1 \
  --response-checkpoint PATH/TO/PAPER/model.zip \
  --response-algorithm REINFORCE --seed 1
```

The final cluster design is
`cluster/hidden_queries_rllib_es_s1to25.sbatch`: 25 array tasks, one per
training seed, with the observed and hidden conditions executed sequentially
on the same node.  This yields 50 independently recorded run artifacts while
preserving the seed pairing.  The batch script enables W&B only for live trend
monitoring; the immutable local manifests and native checkpoints are the
authoritative artifacts.
## One run

Train a seed-matched E1 response:

```bash
python -m stackelberg_pomdp.experiments.matrix_ablations meta-follower \
  --matrix modified_pd --memory-mode joint --algorithm REINFORCE --seed 1
```

Then run one E2 condition:

```bash
python -m stackelberg_pomdp.experiments.matrix_ablations leader \
  --experiment hidden_queries --condition observed --algorithm PG \
  --response-checkpoint PATH/TO/model.zip --seed 1
```

The historical leader schedule is 2,000 PG updates at LR `.008`.  With the
paper geometry this is exactly 200,000 executed transitions.  `progress.jsonl`,
`evaluation.json`, and `run_manifest.json` separately record completed updates,
stored gradient samples, and executed environment transitions.

W&B is off by default.  Each invocation writes an immutable config, progress
JSONL, final evaluation, model/parameters, and an atomic completion manifest.
Existing run artifacts are never overwritten.

Superseded custom-ES pilots and calibration outputs remain available only as
historical raw evidence.  Their implementation and launchers are deliberately
absent from the maintained package, and they must not be combined with native
RLlib ES results.  See [LEGACY_PROVENANCE.md](LEGACY_PROVENANCE.md) for the
historical interpretation.  The only maintained ES paper cohort is the
separate native-Ray array described above.

## Ray-free ten-seed base sweep

Planning writes commands but executes nothing:

```bash
python replication/matrix_ablations/sweep.py plan \
  --sweep-id qualitative-v1 --seeds 1-10
```

Execution is explicit and stage ordered:

```bash
python replication/matrix_ablations/sweep.py run \
  --sweep-id qualitative-v1 --stage meta-follower

python replication/matrix_ablations/sweep.py run \
  --sweep-id qualitative-v1 --stage leader
```

The runner continues past independent failures and reports them together at the
end.  After correcting a failure, `--retry-failed` creates a new numbered,
immutable attempt while retaining the failed attempt for diagnosis.  A missing
E1 checkpoint blocks only its seed-matched meta-response cells; unrelated Q
diagnostics and other seeds can still run.

The plan contains 10 E1 runs and 120 leader runs:

- 40 hidden-query runs: A2C/PPO x observed/hidden x 10 seeds;
- 20 phase-observability runs;
- 20 reset-versus-carried-Q runs; and
- 40 response-reward runs across the two coordination matrices.

Native ES is intentionally excluded from this base sweep because it runs in
the dedicated Ray environment and paired 25-task launcher documented above.

No sweep command submits SLURM jobs.  `status` is read-only:

```bash
python replication/matrix_ablations/sweep.py status --sweep-id qualitative-v1
```

### Unity launch

The maintained cluster workflow uses an isolated source snapshot and five
submissions: ten E1 tasks, forty response-independent leader tasks, one E1
certification gate, eighty meta-response leader tasks, and one final plotting
task.  The 130 logical experiments therefore use 132 SLURM task executions.
Array workers select exactly one plan record with `--record-index`; they never
race by trying to execute an entire stage.

The scripts are under `replication/matrix_ablations/cluster/`.  Submit the E1
and independent arrays together, gate the meta-dependent array on certified E1
responses, and gate plotting on both leader arrays:

```bash
E1_JOB=$(sbatch --parsable replication/matrix_ablations/cluster/e1.sbatch)
Q_JOB=$(sbatch --parsable \
  replication/matrix_ablations/cluster/independent_leaders.sbatch)
GATE_JOB=$(sbatch --parsable --dependency=afterok:${E1_JOB} \
  replication/matrix_ablations/cluster/gate.sbatch)
META_JOB=$(sbatch --parsable --dependency=afterok:${GATE_JOB} \
  replication/matrix_ablations/cluster/meta_leaders.sbatch)
sbatch --dependency=afterok:${Q_JOB}:${META_JOB} \
  replication/matrix_ablations/cluster/plot.sbatch
```

Training arrays log evaluation metrics to the W&B group `qualitative-v1` and
also retain complete local manifests.  `gate_e1.py` requires every response
checkpoint to pass its contract and maximum-regret threshold before any of the
80 dependent jobs can start.  `trend.py` gives a read-only provisional summary
from completed manifests; final interpretation still requires the complete
planned seed cohort.

## Uniform figures and paper logs

```bash
python replication/matrix_ablations/plot.py \
  --input replication/matrix_ablations/results/hidden_queries_pg_es_calibration_s1to3 \
  --additional-input replication/matrix_ablations/results/hidden_queries_pg_es_extension_s4to10 \
  --additional-input replication/matrix_ablations/results/hidden_queries_pg_es_extension_s11to25 \
  --additional-input replication/matrix_ablations/results/hidden_queries_rllib_es_s1to25 \
  --figure fig_hidden \
  --seeds 1-25 \
  --require-seeds-per-cell 25 \
  --paper-logs-root /path/to/journal-artifact/paper/data/paper_logs
```

For policy gradient, the plotter records `evaluation_mean` against environment
transitions.  For native RLlib ES, it records
`native_episode_reward_mean / 5` against native optimizer iteration and retains
both measured cumulative environment transitions and the explicit stochastic
evaluations as diagnostics.  Iteration is the common ES coordinate across
seeds because RLlib's time-based batches overshoot by different amounts.
It averages those seed-level values and draws sample standard error,
`std(ddof=1) / sqrt(n_independent_seeds)`.  It does not pool the within-run
evaluation SEM into that band.  With only one completed seed, sample SEM is
undefined and is exported as `NaN` rather than incorrectly shown as zero;
final paper rendering requires the complete seed cohort.

The plotter retains numeric `learning_rate` for PG and the other RL algorithms,
while corrected ES retains null `learning_rate` and is grouped by its effective
`es_stepsize`.  A figure root that mixes effective step sizes for one algorithm
is rejected rather than silently pooling incompatible cohorts.  A parent root
containing sibling batch subtrees (for example, `seeds1to3/` and
`seeds4to10/`) is discovered recursively; when that parent has no `plan.json`,
pass the complete cohort explicitly (the current hidden-query paper cohort uses
`--seeds 1-25`) for final validation.

All maintained panels use the same publication style: sentence-case,
normal-weight labels; concise semantic panel titles rather than serialized
matrix definitions; one frameless legend outside the plotting area; and compact
training-step ticks such as `100k` and `1M`.  Each figure receives a vector PDF,
PNG, and summary CSV output, plus figure-grouped `history.csv`, `runs.csv`, and
`configs.csv`.  Paper logs go directly under the paper basenames `fig_hidden/`,
`fig_memory_pg/`, `fig_reset/`, and `fig_bots_leaderreward/`, outside
`legacy_appendix/`.  Generated PDFs use those same basenames and are drop-in
replacements for the current LaTeX includes.
