# Stackelberg POMDP: A Reinforcement Learning Approach for Economic Design

This repository hosts the source code associated with the paper "Stackelberg POMDP: A Reinforcement Learning Approach for Economic Design." 
The arXiv version of the paper can be accessed at: https://arxiv.org/abs/2210.03852

### Installation

First, install the necessary packages by running the following command:

```
pip install -r requirements.txt
```

### Normal Form Games

You can run normal form games in two modes: deterministic and randomized. 
In the deterministic mode, the leader must choose a single, specific matrix row. Conversely, in the randomized mode, they may employ a probabilistic strategy, allowing them to play any row with certain probabilities.
- To run the Escape game in deterministic mode, use the following command:
```
python stackelberg_pomdp/main_args.py --experiment_type normal_form:game_1:False
```
- For randomized mode, use the following command:
```
python stackelberg_pomdp/main_args.py --experiment_type normal_form:game_1:True
```
The Maintain game can be run in the same way by replacing `game_1` with `game_2`.

### Matrix Design Games

To run matrix design games, use the following command:
```
python stackelberg_pomdp/main_args.py --experiment_type matrix_design
```
You can specify the observation type for the critic by replacing `critic_obs` with `full` 
for MAPPO or `none` for PPO. You can also specify the response phase probability by 
replacing `response_phase_prob` with `0` for Basic POMDP or `1` for Stackelberg POMDP.

### Simple Allocation Mechanisms

To run simple allocation mechanisms with a message space size of `i`, use the following command:
```
python stackelberg_pomdp/main_args.py --experiment_type simple_allocation:i --tot_num_reward_episodes 30
```

### Sequential Price Mechanisms

To run a sequential price mechanism with `t` types and `i` messages, use the following command:
```
python stackelberg_pomdp/main_args.py --learning_method RL:StopOnThreshold --experiment_type mspm:MSGSpace:t:i --tot_num_eq_episodes 1000 --tot_num_reward_episodes 100
```
For a fixed training environment set by using `--seed SEED`, you can test different initialization weights by varying `--training_seed`.
Each unique value provided to `--training_seed` will result in a different set of initial weights. Remember to replace SEED with your actual seed value. For example, if your seed value is 42, the command would be `--seed 42`.