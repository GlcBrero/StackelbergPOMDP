# Space Invaders ROM

The modified single-player-compatible `space_invaders.bin` ROM is not tracked
in this repository. Supply it in one of two ways:

1. Copy the reference ROM to this directory as `space_invaders.bin`.
2. Set `STACKPOMDP_SPACE_INVADERS_ROM` to its absolute path.

For the adjacent reference checkout used by this replication:

```bash
cp ../StackeRLberg/stackerlberg/envs/roms/space_invaders.bin \
  stackelberg_pomdp/atari/roms/space_invaders.bin
```
