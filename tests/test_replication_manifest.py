import contextlib
import io
import sys
import unittest
from pathlib import Path

from replication import run
from stackelberg_pomdp.run_setups import _effective_tot_num_response_episodes


MANIFEST = Path(__file__).resolve().parents[1] / "replication" / "targets.json"


class ReplicationManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets = run.load_targets(MANIFEST)

    def test_every_runnable_manifest_expansion_parser_validates(self):
        labels = run.validate_targets(self.targets, seed=7)
        self.assertEqual(len(labels), 37)
        self.assertIn("fig_matrix_design_ablation:basic_ppo", labels)
        self.assertIn("fig_collusion_fixed_policy_training:dpdp", labels)
        self.assertIn(
            "calibration_price_collusion_m4_grid:alpha0p25_beta1em4",
            labels,
        )

    def test_named_variants_and_positional_subcommands_expand_structurally(self):
        target = {
            "module": "example.module",
            "command": ["leader"],
            "args": {"seed": "{seed}"},
            "variants": [
                {"name": "observed", "args": {"condition": "observed"}},
                {"name": "hidden", "args": {"condition": "hidden"}},
            ],
        }
        commands = run.build_commands(target, seed=13)
        self.assertEqual([name for name, _ in commands], ["observed", "hidden"])
        self.assertEqual(commands[0][1][:4], [sys.executable, "-m", "example.module", "leader"])
        self.assertIn("13", commands[0][1])

    def test_unsupported_brace_expansion_fails_before_launch(self):
        target = {
            "module": "example.module",
            "args": {"condition": "{observed|hidden}"},
        }
        with self.assertRaisesRegex(run.ManifestError, "named manifest variant"):
            run.build_commands(target, seed=1)

    def test_manifest_matches_retained_simple_allocation_sampling_protocol(self):
        target = self.targets["fig_simple_allocation_stackpomdp_mappo"]
        self.assertEqual(target["args"]["max_steps"], 5_000_000)
        self.assertEqual(target["args"]["tot_num_reward_episodes"], 30)
        self.assertEqual(target["args"]["eval_reward_episodes"], 30)
        self.assertEqual(target["args"]["eval_freq"], 5_000)
        self.assertFalse(target["args"]["align_mw_response_phase"])
        self.assertEqual(
            target["args"]["ppo_rollout_geometry"],
            "historical_ratio_scaled",
        )
        self.assertEqual(target["cohort"]["plot_max_steps"], 1_000_000)

    def test_historical_unaligned_prefix_and_clean_alignment_are_distinguished(self):
        def effective(experiment_type, requested):
            return _effective_tot_num_response_episodes({
                "experiment_type": experiment_type,
                "tot_num_response_episodes": requested,
                "followers_algorithm": "MW",
                "align_mw_response_phase": True,
            })

        self.assertEqual(effective("simple_allocation:1", 100), 100)
        self.assertEqual(effective("simple_allocation:2", 100), 100)
        self.assertEqual(effective("simple_allocation:3", 100), 99)
        self.assertEqual(effective("matrix_design", 30), 28)

        simple = self.targets["fig_simple_allocation_stackpomdp_mappo"]
        matrix = self.targets["fig_matrix_design_ablation"]
        self.assertFalse(simple["args"]["align_mw_response_phase"])
        self.assertEqual(simple["cohort"]["executed_response_games"], 100)
        self.assertEqual(simple["cohort"]["complete_mw_updates"], 33)
        self.assertEqual(simple["cohort"]["incomplete_final_cycle_queries"], 1)
        self.assertFalse(matrix["args"]["align_mw_response_phase"])
        self.assertEqual(
            matrix["args"]["ppo_rollout_geometry"],
            "historical_ratio_scaled",
        )
        self.assertEqual(matrix["cohort"]["executed_response_games"], 30)
        self.assertEqual(matrix["cohort"]["complete_mw_updates"], 7)
        self.assertEqual(matrix["cohort"]["incomplete_final_cycle_queries"], 2)

    def test_obsolete_nonpaper_targets_are_not_exposed(self):
        obsolete = {
            "fig_maintain_randomized",
            "table_mspm_5types_3messages",
            "baseline_mspm_evolutionary_3types_2messages",
            "baseline_mspm_evolutionary_4types_2messages",
            "fig_simple_allocation_reward_during_response_mappo",
            "fig_simple_allocation_reward_during_response_ppo",
            "fig_collusion_learning_no_state",
            "fig_atari_bilateral_trade",
        }
        self.assertTrue(obsolete.isdisjoint(self.targets))

    def test_atari_figures_use_the_maintained_sb3_workflow(self):
        expected = {
            "fig_atari_response_diagnostics": {
                "fig:atari_response_diagnostics",
                "atari_e1_responses",
            },
            "fig_atari_meta_stackpomdp": {
                "fig:atari_meta_stackpomdp",
                "atari_meta_stackpomdp",
            },
        }
        for name, figure_names in expected.items():
            target = self.targets[name]
            self.assertEqual(target["status"], "specialized_workflow")
            self.assertEqual(target["owner"], "replication/atari/README.md")
            self.assertTrue(
                figure_names.issubset(set(target["figure_or_table"].split("; ")))
            )
            for relative_path in target["entrypoints"]:
                self.assertTrue(
                    (MANIFEST.parents[1] / relative_path).is_file(),
                    relative_path,
                )

        response = self.targets["fig_atari_response_diagnostics"]["protocol"]
        self.assertEqual(response["framework"], "Stable-Baselines3 PPO")
        self.assertEqual(response["roles"], ["buyer", "seller"])
        self.assertEqual(response["trade_events"], 5)
        self.assertEqual(response["discount_factor"], 1.0)

        leaders = self.targets["fig_atari_meta_stackpomdp"]["protocol"]
        self.assertEqual(leaders["independent_seeds_per_role"], 10)
        self.assertEqual(leaders["training_transitions"], 2_000_040)
        self.assertEqual(
            leaders["scheduled_checkpoint_transitions"],
            [400_680, 800_520, 1_200_360, 1_600_200],
        )
        self.assertEqual(leaders["held_out_screen_episodes_per_checkpoint"], 20)
        self.assertEqual(leaders["fresh_confirmation_episodes"], 100)

    def test_specialized_atari_target_prints_exact_handoff(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run.main([
                "fig_atari_meta_stackpomdp",
                "--manifest",
                str(MANIFEST),
            ])
        rendered = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("status: specialized_workflow", rendered)
        self.assertIn("owner: replication/atari/README.md", rendered)
        self.assertIn(
            "replication/atari/automation/unity_atari_e2_multiseed.sbatch",
            rendered,
        )

    def test_collusion_calibration_grid_matches_historical_launcher(self):
        target = self.targets["calibration_price_collusion_m4_grid"]
        cells = {
            (variant["args"]["alpha"], float(variant["args"]["beta"]))
            for variant in target["variants"]
        }
        self.assertEqual(
            cells,
            {
                (alpha, beta)
                for alpha in (0.05, 0.15, 0.25)
                for beta in (4e-5, 1e-4, 4e-4, 1e-3, 4e-3)
            },
        )
        self.assertEqual(target["args"]["n_sessions"], 5)
        self.assertEqual(target["args"]["max_steps"], 10_000_000)


if __name__ == "__main__":
    unittest.main()
