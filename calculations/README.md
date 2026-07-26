# Calculation-by-calculation workflow

Run commands from the repository root. The public reproduction route is:

1. inspect the stated JSON input;
2. run one data-producing calculation;
3. inspect the named CSV/JSON output;
4. create one figure from that saved output, where applicable; and
5. compare regenerated numerical data and figures with the committed
   canonical files.

Calculation drivers never create figures. `plot_results.py` never performs an
electronic-structure calculation. Paper tables are read directly from five
canonical CSV files; there is no table-export or document-build stage.

## Ordered stages

| Stage | Inspect first | Calculate | Principal result | Next check |
|---|---|---|---|---|
| switch calibration | `inputs/calibration.json`, `inputs/atomic.json` | `python calculations/calibration/run_calibration.py` | `results/calibration/calibration_summary.csv` | `plot_results.py fig-calibration` |
| GP93/H analytic tests | `inputs/analytic.json`, `functional/parameters.json` | `python calculations/analytic/run_analytic.py` | `results/analytic/derivative_checks.csv` | two analytic figure targets |
| hydrogenic ions | `inputs/hydrogenic.json` | `python calculations/analytic/run_hydrogenic.py` | `results/analytic/hydrogenic_ions.csv` | paper table 1, directly from CSV |
| unregularized H SCF | `inputs/hydrogen_exact_scf.json` | `python calculations/atomic/run_hydrogen_exact.py` | `results/atomic/hydrogen_exact_scf.json` | inspect final zero-damping stage |
| matched atoms and He basis | `inputs/atomic.json` | `python calculations/atomic/run_atomic.py` | four atomic CSV files | three figures and paper tables 2–4 |
| radial potentials | `inputs/potentials.json` | `python calculations/potentials/run_potentials.py` | `results/potentials/radial_potentials.csv` | radial-potential figure |
| H2 curve and minima | `inputs/h2.json` | `python calculations/h2/run_h2.py` | `results/h2/h2_curve.csv`, `h2_minima.csv` | H2 figure and paper table 5 |
| H2 grid check | `inputs/h2.json` | `python calculations/h2/validate_grid.py` | `results/h2/h2_grid_validation.csv` | inspect level-to-level differences |

The calibration and atomic drivers intentionally keep their complete matched
sets together: 45 calibration cases and 15 atom/functional cases. Every case
has its own row, convergence flag, settings, and diagnostics. This prevents a
failed case from being silently omitted from a comparison.

## One figure at a time

```bash
python calculations/plot_results.py fig-calibration
python calculations/plot_results.py fig-gp93-building-block
python calculations/plot_results.py fig-hydrogen-regularization
python calculations/plot_results.py fig-atomic-systematics
python calculations/plot_results.py fig-he-levels
python calculations/plot_results.py fig-he-comparison
python calculations/plot_results.py fig-radial-potentials
python calculations/plot_results.py fig-h2
```

Each command reads only saved result files. The complete source mapping is in
`results/README.md`.
The PNGs are written at 300 dpi and at their intended physical placement
widths: approximately 3.4 inches for one-column figures and 7.05 inches for
full-width figures. This avoids shrinking the plotted text during typesetting.

## One paper table at a time

There is no table-generation command. Regenerate the corresponding
calculation, open the CSV, verify convergence where applicable, and select
the documented columns:

| Table | Recompute | Inspect |
|---|---|---|
| hydrogenic ions | `python calculations/analytic/run_hydrogenic.py` | `results/analytic/hydrogenic_ions.csv` |
| atomic systematics | `python calculations/atomic/run_atomic.py` | `results/atomic/atomic_systematics.csv` |
| He negative levels | same atomic command | `results/atomic/he_negative_virtual_levels.csv` |
| He basis sensitivity | same atomic command | `results/atomic/he_basis_sensitivity.csv` |
| H2 minima | `python calculations/h2/run_h2.py` | `results/h2/h2_minima.csv` |

The exact row order, column selection, units, and display rounding are
specified under “Paper-table reconstruction” in
`REPRODUCTION_GUIDE.md`. The full-precision CSV is authoritative; rounding
is applied only when values are displayed in the paper.

## Validation

After all calculations:

```bash
python calculations/validate_results.py
```

For an optional clean integration check:

```bash
python reproduce.py
```

For the non-SCF postprocessing check:

```bash
python reproduce.py --stage postprocess
```

`reproduce.py` is intended for CI and final release auditing. It does not
replace the individual, inspectable steps above.
