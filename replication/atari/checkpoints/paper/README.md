# Atari paper checkpoints

These six Stable-Baselines3 archives are exact copies of the
[JAIR reproducibility release](https://github.com/GlcBrero/Stackelberg-Journal-Version/releases/tag/jair-2026-v1.0.0)
assets. They total **128,307,831 bytes (122.4 MiB)**. Each model was trained
with seed 1 and saved with Stable-Baselines3 1.8.0.

The checkpoints are licensed under **CC BY 4.0**, with attribution to the
Stackelberg POMDP authors; see [LICENSE-DATA](LICENSE-DATA). The repository's
MIT license covers software.

## Which file to use

| File | Stage | Saved training transitions | Use |
| --- | --- | ---: | --- |
| `gameplay_fixed_ammunition.zip` | E0a | 2,000,000 | Five bullets at reset; initializes delayed-ammunition gameplay |
| `gameplay_delayed_ammunition.zip` | E0b | 2,000,800 | Randomly timed free bullets; initializes both meta-followers |
| `meta_buyer.zip` | E1 buyer | 2,400,960 | Frozen response to a seller leader; gameplay initialization for a buyer leader |
| `meta_seller.zip` | E1 seller | 2,000,800 | Frozen response to a buyer leader; gameplay initialization for a seller leader |
| `buyer_leader_seed1.zip` | E2 buyer | 2,000,040 | Selected buyer leader; evaluate against `meta_seller.zip` |
| `seller_leader_seed1.zip` | E2 seller | 1,600,200 | Selected seller leader; evaluate against `meta_buyer.zip` |

The seller leader is the selected 1,600,200-transition checkpoint, not the
terminal policy. The counts above are read from the saved archives. The
meta-buyer selection and its timing-response limitation are recorded in the
[retained selection protocol](../../results/e1_selections/e1_buyer_temporal_mix_v1_primary_economic_protocol_v1.json).

## Verify the files

From the repository root, run:

```bash
cd replication/atari/checkpoints/paper
LC_ALL=C shasum -a 256 -c SHA256SUMS
```

All six lines should end in `OK`. On systems with GNU coreutils, the equivalent
command is `sha256sum --check SHA256SUMS`. Neither command needs the RL
environment or a ROM.

[`manifest.csv`](manifest.csv) is copied unchanged from the
[versioned release manifest](https://github.com/GlcBrero/Stackelberg-Journal-Version/blob/jair-2026-v1.0.0/reproducibility/artifacts/atari/checkpoints/manifest.csv).
It records filenames, byte sizes, SHA-256 digests, stages, roles, release scope,
training transitions, and the SB3 version. `SHA256SUMS` contains the same
digests in the standard checksum-tool format. The release assets are also a
download source if a local archive needs to be restored.

## Coverage and running the models

This folder supplies both curriculum policies, both learned follower
responses, and one selected leader per role. It does **not** contain all
100 leader snapshots behind the ten-seed training curves. The released
[training-curve data](https://github.com/GlcBrero/Stackelberg-Journal-Version/tree/jair-2026-v1.0.0/reproducibility)
cover those snapshots. The current manuscript additionally includes initialization
data and updated plotters, which must be combined with that data release. See
the [current-paper coverage and release status](../../../PAPER_COVERAGE.md).

Install the repository's pinned environment and supply the local ROM as
described in the [Atari replication guide](../../README.md#runtime-inputs).
That guide's training commands use these files as inputs. To evaluate an E2
archive, pass it as `--checkpoint` to
`replication.atari.evaluate_atari_stackpomdp_leader_sb3`, together with its
opposite-role `--response-checkpoint` from the table. Write the evaluator's
`--selected-checkpoint` alias and reports outside this frozen folder.

The archives retain their original serialized metadata and hashes. Keep the
repository's compatibility imports when loading them. Newly trained models
belong under `replication/atari/checkpoints/clean/` or another ignored output
folder; only the six named paper archives are exempted from Git's ZIP ignore
rule. The ROM is supplied separately.
