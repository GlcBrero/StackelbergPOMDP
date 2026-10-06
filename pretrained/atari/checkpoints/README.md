# Retained Atari checkpoints

All six ZIPs listed in `manifest.csv` are included in the repository and release
archive and verified by byte count and SHA-256.

- `gameplay_fixed_ammunition.zip`, `gameplay_delayed_ammunition.zip`: retained E0 gameplay.
- `meta_buyer.zip`, `meta_seller.zip`: the frozen E1 responses reused for the new 25-seed leader cohorts.
- `buyer_leader_seed1.zip`, `seller_leader_seed1.zip`: historical illustrative leaders, not the newly trained cohort or necessarily terminal checkpoints.

The current paper figures use the included `reproduction/data/` records.
To rerun leader training, start from the E0/E1 policies above; no pretraining
rerun is required. New models belong in a fresh output directory.

From the package root, run `make verify-checkpoints`. The models are regular
repository files, so a clone includes them without Git LFS or another download.
The optional downloader accepts an explicit `--base-url` for a checkpoint mirror;
verification of the included files requires no network access.

The [checkpoint license](../../../LICENSE-DATA) is CC BY 4.0. The Space
Invaders ROM is an external input: set `STACKPOMDP_SPACE_INVADERS_ROM` to its
path. Training/evaluation instructions are in the [root README](../../../README.md).

The `sha256` column in [manifest.csv](manifest.csv) records the checksum of each ZIP.
ROM setup and its required checksum are in [pretrained/README.md](../../README.md).
