# GP93 hydrogen-exact exchange

This repository is the computational and data companion to
“Hydrogen-exact exchange as a semilocal building block for
intermediate-range potentials and diffuse bound states.”

The project separates two physically distinct objects:

1. the bare GP93 GGA building block, which satisfies
   \(v_x[n_{\mathrm H}]=-v_H[n_{\mathrm H}]\) and has the formal
   hydrogen-exact large-gradient branch; and
2. the practical regularized meta-GGA, which retains that potential shape in
   the intermediate radial region controlling low-lying diffuse states and
   smoothly closes the numerical far tail.

The repository is intentionally limited to calculations, numerical data, and
figures. The article source is maintained separately. Each paper table is
reproduced directly from a named canonical CSV file rather than through a
formatting or document-build program.

## Start here

[`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md) gives the complete manual
route. For every scientific result it identifies:

1. the JSON input deck to inspect;
2. the calculation command to run;
3. the CSV/JSON output to inspect;
4. the independent figure command, where applicable; and
5. the exact row selection, columns, and display rounding used for each
   paper table.

The calculation drivers write numerical data only. The plot-only program
reads saved CSV/JSON files and does not import PySCF, a functional module, or
a calculation driver. This separation lets a third party inspect the actual
numbers before deciding how to display them.

Two compact references complement the main guide:

- [`calculations/README.md`](calculations/README.md): ordered command
  checklist;
- [`results/README.md`](results/README.md): data dictionary, expected row
  counts, units, definitions, and figure/table provenance.

## Canonical design

The practical exchange enhancement is

\[
F_{\rm mix}
=F_{\rm PBE}
+X_n G(\alpha)\,[F_{\rm cross}-F_{\rm PBE}],
\]

where \(F_{\rm cross}\) crosses from PBE to the GP93 building block,
\(G(\alpha)\) localizes that contribution to iso-orbital regions, and \(X_n\)
is a broad cubic smoothstep in log spin density. The crossover power is fixed
at \(p=8\), while the switching scale \(s_0\) is selected by the repository
calibration rather than copied into the paper by hand. Fixed regularization
constants are stored in `functional/parameters.json`.

## Repository map

- `functional/`: GP93 continuation and canonical regularized PySCF callback.
- `inputs/`: human-readable settings consumed directly by the drivers.
- `data/`: versioned NIST reference data.
- `calculations/analytic/`: GP93, exact-H, derivative, and hydrogenic tests.
- `calculations/calibration/`: five-system parameter selection.
- `calculations/atomic/`: matched exchange-only atoms, diffuse virtual-state
  analysis, basis sensitivity, and the unregularized exact-H SCF test.
- `calculations/h2/`: near-equilibrium H2 curve and grid validation.
- `calculations/potentials/`: common-density radial-potential diagnostics.
- `calculations/plot_results.py`: figures from saved numerical results only.
- `calculations/validate_results.py`: cross-file scientific and provenance
  checks.
- `results/`: canonical machine-readable outputs and data dictionary.
- `figures/`: figures regenerated from the canonical outputs.
- `requirements-lock.txt`: exact direct and transitive Python package
  versions used for the canonical run.

## Quick installation

Run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
```

Then follow the stages in
[`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md). `make help` prints the same
commands in compact form.

## Five paper tables: direct numerical sources

No separate table generator is needed. Open these files after running their
producer:

| Paper table | Canonical data |
|---|---|
| hydrogenic fixed-density test | `results/analytic/hydrogenic_ions.csv` |
| matched atomic comparison | `results/atomic/atomic_systematics.csv` |
| He negative virtual levels | `results/atomic/he_negative_virtual_levels.csv` |
| He basis sensitivity | `results/atomic/he_basis_sensitivity.csv` |
| H2 spline minima | `results/h2/h2_minima.csv` |

The row order, column selection, and rounding used in the paper are specified
in the guide and in `results/README.md`. Full-precision CSV values remain the
authoritative output.

## Optional integration check

After the individual stages have been understood and inspected, run:

```bash
python reproduce.py
```

This optional CI/release check deletes pipeline-owned outputs, recomputes all
data, rebuilds every figure from the saved results, and runs the validator.
To rebuild figures and validate already-existing results without electronic-
structure calculations, run:

```bash
python reproduce.py --stage postprocess
```

## Validation policy

`python calculations/validate_results.py` fails unless, among other checks:

- every calibration and atomic calculation converges;
- the selected parameter is the minimum of the complete calibration scan;
- the unregularized hydrogen-exact crossover has a hydrogenic residual below
  \(10^{-5}\) mHa, while the independently tested raw GP93 hydrogen-potential
  identity is accurate to \(5\times10^{-5}\);
- the regularized final functional's hydrogenic error is below 0.01 mHa;
- analytic functional derivatives agree with finite differences;
- the regularized exact-H potential is accurate through 6 bohr and its
  density-window transition remains smooth;
- every H2 point is converged, spin symmetric, and stable to grid refinement;
- each result metadata snapshot agrees with the current input deck; and
- every downstream stage uses the calibrated \(s_0,p\).

## Interpretation notes

- RKS He spatial orbitals are counted once.
- Open-shell H and ECP alkalis report alpha and beta spin-orbital diagnostics
  separately.
- Every one-electron atom and every compared functional uses the same
  symmetry-preserving variational orbital optimizer.
- Li, Na, and K are one-valence-electron ECP models; they are not presented as
  all-electron core benchmarks.
- “Rydberg-like” denotes a finite low-lying diffuse bound manifold. Strict
  \(-1/r\) asymptotics belongs to the bare GP93 building block; the practical
  regularization targets the physically important intermediate range.

## External reference data

Neutral-atom ionization energies are from NIST ASD 5.12 and are stored with
uncertainties and access date in `data/experimental_ip.csv`. The experimental
H2 equilibrium-distance marker is stored with its NIST CCCBDB provenance in
`data/h2_reference.csv`. Calculations do not access the network.

## License

No software license has yet been selected. Add the author-approved license
before public release so reuse terms are explicit.
