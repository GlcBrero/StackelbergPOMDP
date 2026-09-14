import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_plotter():
    path = REPO_ROOT / "replication/matrix_ablations/plot.py"
    spec = importlib.util.spec_from_file_location("matrix_plot_publication", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary_rows(experiment, matrices, algorithms, conditions):
    rows = []
    for matrix in matrices:
        for algorithm in algorithms:
            for condition_index, condition in enumerate(conditions):
                for step in (0, 100_000, 1_000_000):
                    mean = -2.0 + condition_index * 0.25 + step / 1_000_000
                    rows.append({
                        "experiment": experiment,
                        "matrix": matrix,
                        "algorithm": algorithm,
                        "condition": condition,
                        "learning_rate": 0.008,
                        "evaluation_target_step": step,
                        "mean": mean,
                        "std": np.sqrt(2.0),
                        "sem": 1.0,
                        "n": 2,
                        "n_independent_seeds": 2,
                        "lower": mean - 1.0,
                        "upper": mean + 1.0,
                    })
    return pd.DataFrame(rows)


def test_sample_sem_is_across_unique_seed_values_and_not_zero_for_one_seed():
    plot = load_plotter()
    base = {
        "experiment": "phase_observability",
        "matrix": "prisoners_dilemma",
        "algorithm": "A2C",
        "condition": "visible",
        "learning_rate": 0.008,
        "evaluation_target_step": 100,
    }
    single = plot.summarize(pd.DataFrame([{**base, "seed": 1,
                                           "leader_reward": 2.0}])).iloc[0]
    assert single["n_independent_seeds"] == 1
    assert np.isnan(single["std"])
    assert np.isnan(single["sem"])
    assert np.isnan(single["lower"])
    assert np.isnan(single["upper"])

    paired = plot.summarize(pd.DataFrame([
        {**base, "seed": 1, "leader_reward": 1.0},
        {**base, "seed": 2, "leader_reward": 3.0},
    ])).iloc[0]
    assert paired["n_independent_seeds"] == 2
    assert np.isclose(paired["std"], np.sqrt(2.0))
    assert np.isclose(paired["sem"], 1.0)

    duplicate_seed = pd.DataFrame([
        {**base, "seed": 1, "leader_reward": 1.0},
        {**base, "seed": 1, "leader_reward": 3.0},
    ])
    try:
        plot.summarize(duplicate_seed)
    except ValueError as error:
        assert "one independent value per training seed" in str(error)
    else:
        raise AssertionError("duplicate seed values must not enter sample SEM")


def test_training_step_ticks_are_compact_and_unambiguous():
    plot = load_plotter()
    assert plot.format_training_step(0) == "0"
    assert plot.format_training_step(500) == "500"
    assert plot.format_training_step(2_500) == "2.5k"
    assert plot.format_training_step(100_000) == "100k"
    assert plot.format_training_step(1_000_000) == "1M"
    assert plot.format_training_step(1_500_000) == "1.5M"


def test_all_plot_types_use_normal_weight_semantic_titles_and_figure_legends():
    plot = load_plotter()
    plot.configure_style()
    figures = [
        plot.plot_phase(summary_rows(
            "phase_observability", ["prisoners_dilemma"], ["A2C"],
            ["visible", "hidden"],
        )),
        plot.plot_response_reward(summary_rows(
            "response_reward",
            [
                "coordination_zero_miscoordination",
                "coordination_penalized_miscoordination",
            ],
            ["A2C"], ["excluded", "included"],
        )),
    ]
    try:
        for figure in figures:
            assert len(figure.legends) == 1
            assert figure.legends[0].get_frame_on() is False
            for ax in figure.axes:
                assert ax.get_legend() is None
                assert ax.xaxis.label.get_text() == "Training steps"
                assert ax.xaxis.label.get_fontweight() == "normal"
                assert ax.yaxis.label.get_fontweight() == "normal"
                assert ax.title.get_fontweight() == "normal"
                assert "[[" not in ax.title.get_text()
                assert "coordination_" not in ax.title.get_text()
        response_titles = [ax.title.get_text() for ax in figures[-1].axes]
        assert response_titles == [
            "No coordination penalty", "Coordination penalty (−5)",
        ]
    finally:
        for figure in figures:
            plt.close(figure)


def test_render_keeps_vector_pdf_and_figure_grouped_paper_logs(tmp_path):
    plot = load_plotter()
    plot.configure_style()
    runs = []
    history = []
    configs = []
    for condition_index, condition in enumerate(("visible", "hidden")):
        seeds = (1,) if condition == "visible" else (1, 2)
        for seed in seeds:
            run_id = "{}-seed{}".format(condition, seed)
            base = {
                "run_id": run_id,
                "run_dir": "/fixture/{}".format(run_id),
                "status": "completed",
                "profile_id": "paper_joint_v1",
                "experiment": "phase_observability",
                "condition": condition,
                "matrix": "prisoners_dilemma",
                "algorithm": "A2C",
                "seed": seed,
                "learning_rate": 0.008,
            }
            runs.append(base)
            configs.append({**base, "config_json": "{}"})
            for step in (0, 100_000, 1_000_000):
                history.append({
                    **base,
                    "evaluation_target_step": step,
                    "global_step": step,
                    "leader_reward": (
                        -2.0 + condition_index * 0.25 + seed * 0.1
                        + step / 1_000_000
                    ),
                    "within_run_eval_sem": 0.05,
                    "evaluation_episodes": 20,
                })

    output = tmp_path / "output"
    paper_logs = tmp_path / "paper_logs"
    plot.render_figure(
        "fig_phase_observability", output, paper_logs,
        pd.DataFrame(runs), pd.DataFrame(history), pd.DataFrame(configs),
    )

    pdf = output / "figures/fig_memory_pg.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert b"/Subtype /Image" not in pdf.read_bytes()
    assert (output / "figures/fig_memory_pg.png").is_file()
    assert (output / "figures/fig_memory_pg_summary.csv").is_file()

    logs_dir = paper_logs / "fig_memory_pg"
    assert {path.name for path in logs_dir.iterdir()} == {
        "configs.csv", "history.csv", "manifest.json", "runs.csv",
    }
    manifest = json.loads((logs_dir / "manifest.json").read_text())
    spec = manifest["uncertainty_specification"]
    assert spec["sampling_unit"] == "independent leader-training seed"
    assert spec["formula"] == (
        "sample_std(ddof=1) / sqrt(n_independent_seeds)"
    )
    assert spec["within_run_evaluation_sem_pooled"] is False
    summary_path = output / "figures/fig_memory_pg_summary.csv"
    summary = pd.read_csv(summary_path)
    assert set(summary["n_independent_seeds"]) == {1, 2}
    assert "NaN" in summary_path.read_text()
