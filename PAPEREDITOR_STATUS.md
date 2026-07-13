# Papereditor Status

- Timestamp: 2026-07-13 (America/New_York)
- Phase: MSPM PI debugging — default-PPO response-100 validation active.
- Completed: verified exact profile weights and rewards, fixed the two-transition subepisode horizon and full critic state, confirmed 300 response games are 75 MW updates, and confirmed the hardcoded optimal MSPM returns exactly 0. The same hardcoded mechanism also returned 0 in 200/200 seeds with 100 response games (25 MW updates). Deterministic checkpoint inspection showed that all three failed diagnostics learned an uninformative-message mechanism worth exactly -0.08: two always sell to A0, while the large-rollout policy price-screens A0 and loses only on the low-low profile.
- Active run: tmux `mspm_pi_response100_defaults_10m_0713`; 10M steps, response 100, exact reward 4, gamma 1, learning rate `3e-4`, one 208-transition StackPOMDP episode per rollout, batch 64, 10 epochs, entropy 0. Training W&B: `678tcqce`; deterministic leader-mode/follower-argmax W&B: `piarg100`.
- Current result: the first 10-episode deterministic evaluation at 50,108 steps returned reward 0 in all episodes. Continue monitoring whether argmax reward remains at 0 rather than returning to the -0.08 no-information basin.
- Current blocker: none.
- Evidence: `simulation_logs/pi_training_plot/mspm_pi_2messages_seed1_10m_exact_reward_response100_ppo_defaults_gamma1.log`.
