# Pretrained Atari policies

This folder groups the retained models, their SHA-256 checksums, and the optional
scripts that produce the gameplay and meta-follower models.

- `atari/gameplay.py`: train the E0 gameplay policies (`python run.py atari-gameplay`).
- `atari/meta_response.py`: train the E1 meta-followers (`python run.py atari-response`).
- [`atari/checkpoints/`](atari/checkpoints/README.md): the six retained policy ZIPs.
- [`atari/checkpoints/manifest.csv`](atari/checkpoints/manifest.csv): each model's
  SHA-256, size, role and training stage. The two leader ZIPs are historical examples.

The paper's leader runs reuse the four E0/E1 models. Their pretraining does not
need to be repeated. Leader training commands are in the [main README](../README.md#atari-bilateral-trade).
New training outputs belong in `results/`, preserving these retained files.

From the code root, verify all six model checksums with:

```bash
make verify-checkpoints
```

## External Space Invaders ROM

The modified ROM is supplied separately and is not included in this package.
The ROM used by the paper runs has this SHA-256:

```text
7224b17462b992d67f4e06a3c85f269c9822b06df6015bf038b55f384ced0301
```

Set its path and check the digest against the value above:

```bash
export STACKPOMDP_SPACE_INVADERS_ROM=/absolute/path/to/space_invaders.bin
shasum -a 256 "$STACKPOMDP_SPACE_INVADERS_ROM"
```

## Optional pretraining

The included E0/E1 models are sufficient to rerun the paper's leader experiments.
To regenerate the upstream policies, run these commands from the package root
after installing the full dependencies and providing the external ROM:

```bash
python run.py atari-gameplay --stage e0a --seed 1 --timesteps 50000000 --num-envs 4 \
  --checkpoint results/atari/gameplay-e0a.zip
python run.py atari-gameplay --stage e0b --seed 1 --timesteps 2000000 --num-envs 4 \
  --init-checkpoint pretrained/atari/checkpoints/gameplay_fixed_ammunition.zip --checkpoint \
  results/atari/gameplay-e0b.zip
python run.py atari-response --role buyer --seed 1 --timesteps 2000800 --num-envs 4 \
  --actor-loss-mode balanced --e1-sampler-mode uniform --e0b-checkpoint \
  pretrained/atari/checkpoints/gameplay_delayed_ammunition.zip --checkpoint \
  results/atari/meta-buyer-uniform.zip
python run.py atari-response --role buyer --seed 1 --timesteps 400160 --num-envs 4 \
  --actor-loss-mode balanced --e1-sampler-mode temporal-marginal-v1 --resume \
  results/atari/meta-buyer-uniform.zip --e0b-checkpoint \
  pretrained/atari/checkpoints/gameplay_delayed_ammunition.zip --checkpoint \
  results/atari/meta-buyer.zip
python run.py atari-response --role seller --seed 1 --timesteps 2000800 --num-envs 4 \
  --learning-rate 0.0005 --actor-loss-mode balanced --economic-architecture \
  seller_shared_context_beta_v5 --e0b-checkpoint \
  pretrained/atari/checkpoints/gameplay_delayed_ammunition.zip --checkpoint \
  results/atari/meta-seller.zip
```

E0a learns gameplay at fixed ammunition; E0b adapts it to delayed ammunition.
The next stages train contextual buyer and seller responses. The buyer's second
command continues its first checkpoint under the temporal-marginal sampler.
These commands use the released gameplay models as documented inputs; to replace
them with newly trained models, supply those paths explicitly. New upstream models
must pass their corresponding evaluator before use in downstream experiments.
See `python run.py atari-evaluate-response --help` and
`experiments/atari/_support/runners/evaluate_atari_e0b_sb3.py`.
