# Papereditor Status

- Timestamp: 2026-07-13 (America/New_York)
- Phase: Ten-seed Unity validation running for clean PI SPM versus MSPM.
- Completed: restored the economic SPM termination rule (end when inventory is exhausted or all buyers are visited); retained exact four-profile expected rewards and the full Markov critic state; verified response 100 produces 25 MW updates; and confirmed the hardcoded optimal MSPM returns exactly 0 in 200/200 seeds under early termination. Those outer episodes used 128--161 transitions, confirming the horizon is variable.
- Rollout safety: PPO uses the conservative maximum outer-episode length, `(100 + 4) * 2 = 208`, without padding the environment. Every 208-step rollout must contain at least one completed reward phase; the custom collector counts completions and raises if this invariant fails. The first live rollout stored 208 steps and completed one outer episode.
- Active run: tmux `mspm_pi_clean_response100_10m_0713`; 10M steps, response 100, exact reward 4, gamma 1, learning rate `3e-4`, batch 64, 10 epochs, entropy 0. Training W&B: `5xhgdpgt`; deterministic leader-mode/follower-argmax W&B: `piargcln`.
- Figure completed: compared clean seed-1 SPM run `6ijiy88e` with clean seed-1 MSPM run `5xhgdpgt`, using identically defined training reward, 10,000-step bins, a centered 100,000-step smooth, and a fixed shared horizon of 1.4M steps. Generated and visually verified vector PDF and 300-dpi PNG artifacts at `output/pdf/spm_vs_mspm_pi_reward_seed1.{pdf,png}`; copied matching files to the paper `Figures/` directory and the Desktop.
- Running jobs: Unity SLURM array `61798498`; tasks 0--9 run MSPM seeds 1--10 and tasks 10--19 run SPM seeds 1--10. All 20 tasks reached training and initialized W&B without a startup error. The W&B group is `pi_spm_vs_mspm_10seeds_20260713`.
- Source: working MSPM marker commit `d8a709a`; ten-seed array manifest commit `d6dcbc1`.
- Current blocker: none. The array is still running, so the existing chart remains interim single-seed evidence until all 20 runs complete.
- Evidence: `simulation_logs/pi_training_plot/spm_pi_seed1_10m_reward_only.log` and `simulation_logs/pi_training_plot/mspm_pi_2messages_seed1_10m_response100_defaultppo_gamma1_earlytermination.log`.
- Verification: rendered the PDF with Poppler at 220 dpi, inspected the render, confirmed a 432 x 252 pt single-page vector PDF with embedded DejaVu Sans, checked matching SHA-256 hashes after copying, and ran `git diff --check`.
- Next action: monitor array `61798498`, sync completed logs/results, aggregate ten seeds per method, and regenerate the paper comparison with uncertainty bands.
