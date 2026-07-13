# Stackelberg POMDP: A Reinforcement Learning Approach for Economic Design

This repository hosts the source code associated with the paper "Stackelberg POMDP: A Reinforcement Learning Approach for Economic Design." 
The arXiv version of the paper can be accessed at: https://arxiv.org/abs/2210.03852

## Companion codebase

This repository implements the Stackelberg POMDP framework with **tabular follower oracles and a centralized critic**, using Stable-Baselines3. It covers the platform-market (Bertrand pricing) and indirect mechanism-design experiments, and absorbs the earlier `ai_collusion` codebase in full.

For the matrix-game and Atari bilateral-trade experiments — which use **neural-network followers trained via external alternating optimization** in Ray RLlib — see the companion repository [StackeRLberg](https://github.com/mgerstgrasser/StackeRLberg).

The split reflects an architectural trade-off: centralized critics are natural for tabular follower state but not for high-dimensional pixel observations.

### Installation

Create the conda environment:

```
conda env create -f environment.yml
conda activate stackelberg-pomdp
```

Run commands from the repository root:

```
python replication/run.py --list
```

The environment sets `PYTHONNOUSERSITE=1` so Python does not accidentally import
packages from `~/.local`.

### Normal Form Games

You can run normal form games in two modes: deterministic and randomized. 
In the deterministic mode, the leader must choose a single, specific matrix row. Conversely, in the randomized mode, they may employ a probabilistic strategy, allowing them to play any row with certain probabilities.
- To run the Escape game in deterministic mode, use the following command:
```
python -m stackelberg_pomdp.experiments.normal_form --game_name game_1 --randomized false
```
- For randomized mode, use the following command:
```
python -m stackelberg_pomdp.experiments.normal_form --game_name game_1 --randomized true
```
The Maintain game can be run in the same way by replacing `game_1` with `game_2`.

### Matrix Design Games

To run matrix design games, use the following command:
```
python -m stackelberg_pomdp.experiments.matrix_design
```
You can specify the observation type for the critic by replacing `critic_obs` with `full` 
for MAPPO or `none` for PPO. You can also specify the POMDP construction by
setting `pomdp_mode` to `stackelberg`, `hidden_queries`, or
`reward_during_response`.
Use `--tot_num_response_episodes` to set the follower-response horizon.

### Simple Allocation Mechanisms

To run simple allocation mechanisms with a message space size of `i`, use the following command:
```
python -m stackelberg_pomdp.experiments.simple_allocation --num_messages i
```

### Sequential Price Mechanisms

To run a sequential price mechanism with `t` types and `i` messages, use the following command:
```
python -m stackelberg_pomdp.experiments.mspm --setting MSGSpace --num_types t --num_messages i
```
Use `--seed SEED` to control both environment randomness and learner initialization for replication runs.
