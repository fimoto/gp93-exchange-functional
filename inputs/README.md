# Calculation input files

These JSON files are the human-readable calculation settings consumed by the
stage drivers.  They contain numerical grids, basis/ECP choices, convergence
stages, geometry lists, counting tolerances, and the systems included in each
calculation.

- `calibration.json`: scanned \(s_0\) values, fixed \(p\), systems, and
  selection rule.
- `analytic.json`: enhancement-factor and exact-H radial grids, reported
  error windows, and the regularization-transition check window.
- `hydrogenic.json`: ions and scaled radial quadrature.
- `hydrogen_exact_scf.json`: basis, DFT grid, and every SCF stabilization
  stage for the unregularized H calculation.
- `atomic.json`: atomic systems, basis/ECP choices, functionals, DFT/HF
  tolerances, one-electron optimizer, virtual-state definition, and He basis
  study.
- `h2.json`: bond-length grid, electronic-structure tolerances, spline
  resolution, and grid-convergence subset.
- `potentials.json`: common reference densities, basis/ECP choices, radial
  grids, and trusted-density masks.

The analytic definition and fixed regularization parameters of the functional
are separate from calculation settings and remain in
`functional/parameters.json`.  Its `numerical_implementation` object also
records the GP93 quadrature, Frobenius order, continuation spline, Fmix
log-\(s\) spline, switch cap, and numerical floors actually consumed by the
functional modules.  Experimental reference values remain in `data/*.csv`.

All lengths and energies use the units written in the corresponding key.
Dimensionless quantities and PySCF grid levels are labeled explicitly.
Each result metadata JSON contains a snapshot of the input block used for
that run.  `calculations/validate_results.py` compares those snapshots with
the current files, so changing an input without recomputing its results is a
validation error.

Fields named `purpose`, `definition`, `*_rule`, or similar prose are
human-readable method documentation.  Numerical fields, enumerated systems,
basis/ECP names, grids, tolerances, and stage lists are executable settings.
