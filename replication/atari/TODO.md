# Atari Replication TODO

The Atari Space Invaders bilateral-trade experiment is part of the paper, but
the current implementation lives in:

- `Code/StackeRLberg`
- `atari experiment/`

The paper target is represented in `replication/targets.json` as
`fig_atari_bilateral_trade` with status `companion_stackerlberg`.

The five-bullet buyer gameplay pretraining stage has now been ported. Run:

```bash
PYTHONPATH=.:../StackeRLberg \
  python -m replication.atari.train_five_bullet_gameplay \
  --seed 1 --timesteps 10000000
```

This trains PPO with the Nature CNN in the same preprocessing and bullet
resource chain used by the new trade environment. It writes the frozen buyer
checkpoint and deterministic 20-episode evaluation under
`replication/atari/checkpoints/` and logs to W&B project `StackPOMDP`.

Evaluate any exported PPO/A3C checkpoint without starting Ray:

```bash
PYTHONPATH=.:../StackeRLberg \
  python -m replication.atari.evaluate_five_bullet_checkpoint \
  --checkpoint replication/atari/checkpoints/<checkpoint>.pkl \
  --episodes 20
```

The evaluator uses the fixed fresh-seed suite, deterministic argmax actions,
the identical five-bullet preprocessing chain, and atomically writes a JSON
record beside the checkpoint. The `*_best.json` manifests identify protected
PPO and A3C deployment checkpoints independently of volatile latest weights.

TODO:

- Add the exact StackeRLberg commands for stage-1 meta-follower training and
  stage-2 leader training.
- Add evaluation/plot commands for `fig_atari_ppo_ppo_byvar`.
- Port the pricing-policy stages and final figure generation into this repo.
