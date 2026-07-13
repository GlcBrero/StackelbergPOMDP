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
python replication/run.py fig_price_collusion_m4_calibration --dry-run
python replication/run.py fig_price_collusion_punishment_m4 --dry-run
python replication/run.py fig_collusion_learning_state --seed 1 --dry-run
```
