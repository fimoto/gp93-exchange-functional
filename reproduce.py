#!/usr/bin/env python3
"""Optional end-to-end integration check for the documented manual workflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GENERATED_DIRECTORIES = (
    ROOT / "results" / "analytic",
    ROOT / "results" / "calibration",
    ROOT / "results" / "atomic",
    ROOT / "results" / "h2",
    ROOT / "results" / "potentials",
    ROOT / "figures",
)


def run(command: list[str], environment: dict[str, str]) -> None:
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def verify_selected_parameters() -> None:
    selected_path = ROOT / "results" / "calibration" / "selected_parameters.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    parameters = json.loads(
        (ROOT / "functional" / "parameters.json").read_text(encoding="utf-8")
    )
    if (
        float(selected["s0"]) != float(parameters["selected_s0"])
        or int(selected["p"]) != int(parameters["selected_p"])
    ):
        raise RuntimeError(
            "The calibrated selection does not match "
            "functional/parameters.json"
        )


def clean_generated_outputs() -> None:
    """Remove only pipeline-owned artifacts before a complete recalculation."""

    for path in GENERATED_DIRECTORIES:
        if path.exists():
            shutil.rmtree(path)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "postprocess"),
        default="all",
        help=(
            "'all' recomputes every data set; 'postprocess' rebuilds figures "
            "and checks existing results"
        ),
    )
    args = parser.parse_args()

    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = "1"
    python = sys.executable

    if args.stage == "all":
        clean_generated_outputs()
        run([python, "calculations/calibration/run_calibration.py"], environment)
        verify_selected_parameters()
        commands = (
            [python, "calculations/analytic/run_analytic.py"],
            [python, "calculations/analytic/run_hydrogenic.py"],
            [python, "calculations/atomic/run_hydrogen_exact.py"],
            [python, "calculations/atomic/run_atomic.py"],
            [python, "calculations/potentials/run_potentials.py"],
            [python, "calculations/h2/run_h2.py"],
            [python, "calculations/h2/validate_grid.py"],
        )
        for command in commands:
            run(command, environment)

    # Calculation drivers write data only. Figures are always rebuilt from
    # their saved CSV/JSON records by the plot-only program.
    run([python, "calculations/plot_results.py", "all"], environment)
    run([python, "calculations/validate_results.py"], environment)
    print("\nReproduction completed successfully.", flush=True)


if __name__ == "__main__":
    main()
