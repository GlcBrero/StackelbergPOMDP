# Autonomous Run Report

## 2026-07-13 — PI SPM versus MSPM ten-seed Unity launch

- Working MSPM implementation committed as `d8a709a`; reproducible Unity array
  manifest committed as `d6dcbc1`.
- Submitted SLURM array `61798498` with 20 tasks: tasks 0--9 are MSPM seeds
  1--10 and tasks 10--19 are SPM seeds 1--10, each for 10M training steps.
- W&B group: `pi_spm_vs_mspm_10seeds_20260713`. MSPM run IDs are
  `3vqh4jad`, `9ic017js`, `wtg9nzex`, `o8kt1bgy`, `d5hnhwmo`, `ayzjxfxx`,
  `a6rtlw1p`, `esz18gc4`, `nxee73a9`, and `61rzh15c`. SPM run IDs are
  `2kzm425e`, `nctxnwlv`, `zj9xq68m`, `ntvmo7u9`, `tdr60qek`, `0famcbr5`,
  `7aqpjy6x`, `a4grx2bi`, `flczyzr5`, and `l05qz1cd`.
- Startup verification: all 20 tasks entered training, all 20 initialized W&B,
  and no task log contained a traceback or import/startup error.
- Unity logs:
  `simulation_logs/cluster_pi_spm_vs_mspm_10seeds/slurm_61798498_<task>.{out,err}`.
- Next: monitor completion, retrieve logs/results, aggregate ten seeds per
  method, and regenerate the paper figure with cross-seed uncertainty.

## 2026-07-13 — PI SPM versus MSPM paper figure

- Task: create a paper-style training-step versus leader-reward comparison and place it on the Desktop.
- Sources: clean seed-1 SPM run `6ijiy88e` from `simulation_logs/pi_training_plot/spm_pi_seed1_10m_reward_only.log`; clean seed-1 MSPM run `5xhgdpgt` from `simulation_logs/pi_training_plot/mspm_pi_2messages_seed1_10m_response100_defaultppo_gamma1_earlytermination.log`.
- Processing: stream reward rows into 10,000-step bins; plot a faint binned trace and a centered 100,000-step moving average; crop both series to a fixed shared 1.4M-step horizon. No uncertainty band is shown because there is one seed per method.
- Script: `tools/plot_pi_spm_vs_mspm.py`.
- Results: `output/pdf/spm_vs_mspm_pi_reward_seed1.pdf`, `.png`, `.csv`, and `.json`; matching PDF/PNG copied to `Stackelberg-Journal-Version/JAIR/Figures/` and `/Users/gbrero/Desktop/`.
- Verification: script help smoke test under the project environment; `git diff --check`; PDF metadata and embedded-font inspection with `pdfinfo` and `pdffonts`; Poppler render at 220 dpi followed by visual inspection; SHA-256 equality checks for repository, paper, and Desktop copies.
- Limitations: this is an interim single-seed figure. MSPM was still running when the 1.4M-step snapshot was fixed. Regenerate at the completed 10M horizon and add independent seeds before using the chart as final empirical evidence.
- TODO/GB comments: none modified or removed.
