import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from replication.cluster import cohort


class ClusterCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = cohort.make_records(
            cohort.load_targets(cohort.ROOT / "replication/targets.json")
        )

    def test_paper_cohorts_have_no_duplicate_runs_or_output_paths(self):
        self.assertEqual(len(self.records), 210)
        self.assertEqual(len({r["key"] for r in self.records}), 210)
        self.assertEqual(len({r["log_directory"] for r in self.records}), 210)
        self.assertEqual({r["variant"] for r in self.records if r["group"] == "matrix_design"},
                         {"basic_mappo", "basic_ppo"})
        for group, count, steps in (("simple_allocation", 150, 5_000_000),
                                    ("matrix_design", 50, 5_000_000), ("spm", 10, 1_000_000)):
            rows = [r for r in self.records if r["group"] == group]
            self.assertEqual(len(rows), count)
            self.assertEqual({r["resolved_config"]["max_steps"] for r in rows}, {steps})

    def test_planning_does_not_train_or_overwrite_a_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(cohort.subprocess, "run", side_effect=AssertionError("training during planning")):
                cohort.prepare(directory)
                with self.assertRaises(FileExistsError):
                    cohort.prepare(directory)
            plan = json.loads((Path(directory) / "mechanism_plan.json").read_text())
            self.assertEqual(plan["counts"], {"simple_allocation": 150, "matrix_design": 50, "spm": 10})

    def test_worker_rejects_changed_manifest_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort.prepare(directory)
            path = Path(directory) / "mechanism_plan.json"
            plan = json.loads(path.read_text())
            plan["targets_sha256"] = "changed"
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "changed"):
                cohort.run_record(directory, "spm", 0)

    def test_worker_rejects_existing_artifacts_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort.prepare(directory)
            with patch.object(Path, "exists", return_value=True):
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    cohort.run_record(directory, "spm", 0)


if __name__ == "__main__":
    unittest.main()
