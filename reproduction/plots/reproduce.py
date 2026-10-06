#!/usr/bin/env python3
"""Regenerate the current paper's figures and numerical tables.

Outputs stay in build/ unless --install and --paper-dir are supplied.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from validation.check import split_generated_block
FIGURE_STEMS = (
    "response_credit_ablation",
    "pi_spm_vs_mspm_exact_eval_allseeds", "mspm_multitype_training", "hidden_cost_policy",
    "hidden_cost_learning", "atari_e1_responses", "atari_meta_stackpomdp",
    "fig_memory_pg", "fig_bots_leaderreward",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "build")
    parser.add_argument("--check", action="store_true",
                        help="Check regenerated figures and numbers against the paper reference.")
    parser.add_argument("--compare-rendered", action="store_true",
                        help="Also require pixel-identical figures (needs the reference Poppler version).")
    parser.add_argument("--install", action="store_true",
                        help="Copy regenerated PDFs and numbers into --paper-dir after all commands succeed.")
    parser.add_argument("--paper-dir", type=Path,
                        help="Manuscript directory containing StackelbergPOMDP.tex and figures/.")
    args = parser.parse_args()
    if args.install and (args.paper_dir is None or not (args.paper_dir / "StackelbergPOMDP.tex").is_file()):
        parser.error("--install requires --paper-dir pointing to the manuscript directory")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    commands = [
        ("build_figures.py", output),
        ("paper_results.py", "--output-dir", output),
        ("statistical_comparisons.py", "--output-dir", output),
    ]
    for script, *arguments in commands:
        command = [sys.executable, str(HERE / script), *map(str, arguments)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, env=env, check=True)
    for stem in FIGURE_STEMS:
        path = output / "figures" / (stem + ".pdf")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.check:
        command = [sys.executable, str(HERE.parent / "validation/check.py"), "--build-dir",
                   str(output)]
        if args.compare_rendered:
            command.append("--compare-rendered")
        if args.paper_dir is not None:
            command += ["--paper-dir", str(args.paper_dir)]
        subprocess.run(command, env=env, check=True)
    if args.install:
        manuscript = args.paper_dir / "StackelbergPOMDP.tex"
        source = manuscript.read_text(encoding="utf-8")
        updated = source
        for block, filename in (("PLATFORM NUMBERS", "hidden_cost_numbers.tex"),
                                ("STATISTICAL COMPARISONS", "statistical_comparisons.tex")):
            before, _, after = split_generated_block(updated, block)
            updated = before + (output / filename).read_text(encoding="utf-8") + after
        for stem in FIGURE_STEMS:
            shutil.copy2(output / "figures" / (stem + ".pdf"), args.paper_dir / "figures" / (stem + ".pdf"))
        if updated != source:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             dir=manuscript.parent, delete=False) as handle:
                handle.write(updated)
                temporary = Path(handle.name)
            try:
                shutil.copymode(manuscript, temporary)
                temporary.replace(manuscript)
            finally:
                temporary.unlink(missing_ok=True)
    print(f"Regenerated {len(FIGURE_STEMS)} figures and numerical tables in {output}")


if __name__ == "__main__":
    main()
