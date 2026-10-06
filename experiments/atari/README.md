# Bilateral ammunition trade in Atari

The visible files are `game.py`, `response.py`, `experiment.py`, this README
and `__init__.py`. Supporting implementation, settings and assets are in
`_support/`; `experiment.py` shows how the pieces connect.

`game.py` defines the POMG and its bilateral trading rules in one place.
Start with `Game.reset()` and `Game.step()`: each player supplies a joystick
action and a price or acceptance threshold. Scheduled trades pause both games;
otherwise the joystick actions advance them. `DualAtariTradeCore.trade()` below
shows the purchase condition, bullet transfer and payment; `step_gameplay()`
applies the scaled Atari scores. Per-player observations, rewards and accounting
are also in this file. `BilateralAtariConfig` holds the settings.

`response.py` asks the five canonical economic questions that condition the
frozen meta-follower. `_support/reward.py`
runs the joint-action game using that same economic commitment and the
returned follower `PolicyProfile`. `_support/leader_policy.py` selects the
custom neural policy; `experiment.py` connects the setting to the shared core.

Implementation helpers remain in `_support/`: single-emulator adapters,
observation encoding, phase wrappers, neural components, training and evaluation.
The six released policies, checksum manifest and downloader are in
[`pretrained/atari/checkpoints/`](../../pretrained/atari/checkpoints/README.md).
`_support/follower_policy.py` binds the selected context to a frozen follower
policy acting on its own observations. Response bookkeeping belongs to
`AtariResponse`, not to the game. Checkpoint generation remains a separate entry point in `pretrained/atari/`
and imports this same game and helper implementation.

Atari's training extensions remain local. `_support/_ppo.py` masks actor
terms on pretraining rows where that actor did not act; `_support/training.py`
adds learning-rate groups and, for the balanced variant, separate averages
for gameplay and economic decisions. These extend the common episode
collector; their losses are not imposed on new experiment folders.

From the code root:

```bash
python pretrained/atari/checkpoints/download.py --verify-only
python run.py atari-leader --help
python run.py atari-evaluate-leader --help
```

The Space Invaders ROM is an external dependency; set
`STACKPOMDP_SPACE_INVADERS_ROM` to its path. The root README gives the exact
paper commands. Change trading rules in `game.py`, response conditioning in
`response.py`, and network architecture in `_support/policy/` as needed.
