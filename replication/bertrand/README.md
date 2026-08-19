# Bertrand / Price-Collusion Replication

This folder contains paper-specific scripts for the collusion and platform
intervention experiment family.

Core reusable code remains in `stackelberg_pomdp`:

- `BertrandCompetitionEnv`
- `QLearningFollowersWrapper`
- StackPOMDP training and policy machinery

Paper-specific code lives here:

- `price_collusion.py`: constants and gain-index helpers.
- `calibrate_price_learners.py`: borrowed Q-learning price-collusion calibration.
- `punishment_diagnostic.py`: deviation/punishment diagnostic for `deviation_m4`.
- `evaluate_intervention.py`: post-training intervention evaluation.

Run through the top-level manifest:

```bash
python replication/run.py calibration_price_collusion_m4_grid --dry-run
python replication/run.py fig_price_collusion_m4_calibration --dry-run
python replication/run.py fig_price_collusion_punishment_m4 --dry-run
python replication/run.py fig_collusion_learning_state --seed 1 --dry-run
python replication/run.py fig_collusion_fixed_policy_training --variant pdp --dry-run
```

The calibration target expands to the historical grid
$\alpha\in\{0.05,0.15,0.25\}$ and
$\beta\in\{4\times10^{-5},10^{-4},4\times10^{-4},10^{-3},4\times10^{-3}\}$,
with five seeds per cell. Results are written to disjoint cell directories so
parallel runs cannot overwrite one another.

The deviation target retains 15 post-deviation periods (16 rows including the
pre-deviation state), matching the archived figure input.  The learned PDA
target records the 50,000-step seller response, 30-step reward phase, 50M A2C
budget, and 25-seed cohort.  Fixed-policy target variants train the
no-intervention, PDP, and DPDP seller cohorts; cycle aggregation is performed
by the journal reproducibility pipeline.
