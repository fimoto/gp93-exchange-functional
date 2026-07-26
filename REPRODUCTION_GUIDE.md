# Step-by-step reproduction guide

This is the primary route for reproducing the numerical results and figures
reported in the paper. It deliberately keeps the scientific stages separate:

1. read the input deck;
2. run one calculation;
3. inspect its full-precision CSV/JSON result;
4. regenerate the corresponding figure from that saved result; and
5. reconstruct each reported table directly from its canonical CSV.

`reproduce.py` is retained as an optional end-to-end integration check. It is
not a substitute for the manual route below.

The repository contains calculations, data, and figures only. The article
source is maintained separately, so there is no document-build or table-
export stage in this workflow.

## 1. Software setup

Run every command from the repository root. Python 3.12 and the exact package
versions in `requirements-lock.txt` define the canonical software stack.

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

The one-thread settings remove avoidable BLAS scheduling variation. Small
last-digit differences can still occur on a different CPU or linear algebra
library. The locked canonical calculation uses PySCF 2.14.0.

For the cleanest audit, begin with a fresh clone and work on a temporary
branch:

```bash
git switch -c reproduce-check
git status --short
```

The repository includes canonical outputs. Each calculation driver removes
only the files it owns before starting, so an interrupted run leaves a
missing or explicitly incomplete result rather than silently retaining an
older completed file.

After each stage, use `git diff` on the named result directory. This gives a
transparent comparison between the committed canonical data and the
independent rerun.

## 2. Input-deck inventory

These files are executable inputs, not retrospective descriptions. Each
driver opens the named JSON file at runtime.

| Input | Controls |
|---|---|
| `functional/parameters.json` | selected \(s_0,p\), GP93 constant, regularization, descriptor floors/clips, quadrature, and spline settings |
| `inputs/calibration.json` | fixed \(p\), nine \(s_0\) points, systems, objective, and selection rule |
| `inputs/analytic.json` | enhancement and radial grids, continuation boundary, derivative checks, and convergence spacings |
| `inputs/hydrogenic.json` | hydrogenic ions and uniform \(x=Zr\) quadrature |
| `inputs/hydrogen_exact_scf.json` | H geometry, basis, grid, UHF seed, and every unregularized-H SCF stage |
| `inputs/atomic.json` | atoms, bases/ECPs, functionals, tolerances, stabilization stages, orbital optimizer, state-count definitions, and He basis study |
| `inputs/h2.json` | geometries, basis, grid, SCF/FCI tolerances, spline sampling, spin check, and grid-refinement subset |
| `inputs/potentials.json` | reference densities, bases/ECPs, radial grids, masks, derivatives, and Hartree integration |
| `data/experimental_ip.csv` | versioned NIST atomic ionization energies |
| `data/h2_reference.csv` | versioned experimental H2 equilibrium-distance marker |

Inspect an input before executing its driver:

```bash
python -m json.tool inputs/calibration.json
python -m json.tool inputs/atomic.json
```

PySCF basis and ECP names are stored in these inputs. Their definitions come
from the locked PySCF installation.

## 3. Stage 1 — calibrate the outer switch

Scientific question: with \(p=8\) fixed as a design choice, which scanned
\(s_0\) minimizes the unweighted mean absolute error in
\(-\epsilon_{\rm HOMO}\) for H, He, Li, Na, and K, provided all five
calculations converge?

### 3.1 Inspect the settings

```bash
python -m json.tool inputs/calibration.json
python -m json.tool inputs/atomic.json
```

Confirm the nine-point \(s_0\) grid, five-system list, reference-data path,
and convergence policy.

### 3.2 Calculate

```bash
python calculations/calibration/run_calibration.py
```

### 3.3 Inspect every trial and the selected minimum

```bash
python - <<'PY'
import csv

with open("results/calibration/calibration_summary.csv", newline="") as f:
    for row in csv.DictReader(f):
        print(
            f"s0={float(row['s0']):.2f}  "
            f"converged={row['n_converged']}/5  "
            f"MAE={row['mae_5system_eV']} eV"
        )
PY

python -m json.tool results/calibration/selected_parameters.json
```

Also inspect `results/calibration/calibration_results.csv`: it contains all
45 system/scan-point calculations. The canonical selection is \(s_0=0.16\),
\(p=8\), with a five-system MAE of approximately 0.790 eV. A failed
calculation is never removed from the objective.

### 3.4 Verify the explicit parameter handoff

```bash
python - <<'PY'
import json
from pathlib import Path

selected = json.loads(
    Path("results/calibration/selected_parameters.json").read_text()
)
fixed = json.loads(Path("functional/parameters.json").read_text())
assert float(selected["s0"]) == float(fixed["selected_s0"])
assert int(selected["p"]) == int(fixed["selected_p"])
print(f"verified: s0={selected['s0']}, p={selected['p']}")
PY
```

Downstream drivers read `functional/parameters.json` directly. Their metadata
records the effective pair, and the final validator compares it with the
calibration.

### 3.5 Recreate the calibration figure

```bash
python calculations/plot_results.py fig-calibration
git diff -- results/calibration/ figures/fig_calibration.png
```

## 4. Stage 2 — GP93 continuation and exact-H regularization

Scientific questions:

- Does the continued GP93 building block exhibit the derived large-\(s\)
  behavior?
- Does the unregularized crossover reproduce the exact H local exchange
  potential?
- Does the practical density window retain that shape at intermediate
  distance while smoothly closing the numerical far tail?
- Are the coded derivatives consistent with finite differences?

### 4.1 Inspect and calculate

```bash
python -m json.tool inputs/analytic.json
python -m json.tool functional/parameters.json
python calculations/analytic/run_analytic.py
```

### 4.2 Inspect the saved diagnostics

```bash
head -n 6 results/analytic/gp93_building_block.csv
head -n 6 results/analytic/hydrogen_regularization.csv
head -n 10 results/analytic/derivative_checks.csv
less results/analytic/DERIVATION_NOTES.md
python -m json.tool results/analytic/analytic_metadata.json
```

The primary records are:

- `gp93_building_block.csv`: \(s,z,F_{\rm PBE},F_{\rm GP93}\), crossover,
  clipping, and sampled implementation;
- `hydrogen_regularization.csv`: radius, exact-H descriptors, density window,
  and bare/unregularized/regularized potentials;
- `derivative_checks.csv`: analytic-versus-finite-difference and radial-grid
  checks.

### 4.3 Recreate the two figures independently

```bash
python calculations/plot_results.py fig-gp93-building-block
python calculations/plot_results.py fig-hydrogen-regularization
git diff -- results/analytic/ figures/fig_gp93_building_block.png \
  figures/fig_hydrogen_regularization.png
```

## 5. Stage 3 — fixed-density hydrogenic ions

Scientific question: on exact one-electron \(1s\) densities for
\(Z=1,\ldots,5\), how closely do the unregularized hydrogen-exact building
block, practical regularized functional, and PBE-X reproduce
\(-\epsilon_{1s}=Z^2/2\)?

### 5.1 Inspect and calculate

```bash
python -m json.tool inputs/hydrogenic.json
python calculations/analytic/run_hydrogenic.py
```

### 5.2 Inspect all 15 rows

```bash
python - <<'PY'
import csv

with open("results/analytic/hydrogenic_ions.csv", newline="") as f:
    for row in csv.DictReader(f):
        print(
            row["system"],
            row["method"],
            f"absolute error={float(row['absolute_error_mHa']):.9g} mHa",
        )
PY
```

This is a fixed-density radial identity test, not a Gaussian-basis SCF
calculation. The CSV is also the direct source of paper table 1; its exact
column selection is specified in Section 11.

## 6. Stage 4 — self-consistent unregularized H

Scientific question: does the unregularized hydrogen-exact crossover remain
usable in a one-electron all-electron self-consistent calculation without the
density-window regularization used for multi-electron applications?

### 6.1 Inspect, calculate, and check convergence

```bash
python -m json.tool inputs/hydrogen_exact_scf.json
python calculations/atomic/run_hydrogen_exact.py
python -m json.tool results/atomic/hydrogen_exact_scf.json
cat results/atomic/hydrogen_exact_scf.csv
```

Inspect every stabilization stage in the JSON. The final stage must converge
with zero damping and zero level shift. The canonical aug-cc-pV5Z result
gives \(-\epsilon_{\rm HOMO}\approx13.6053\) eV. This calculation is
intentionally separate from the regularized atomic comparison.

## 7. Stage 5 — matched atoms and He diffuse levels

Scientific questions:

- Under one matched protocol, how do PBE-X, SCAN-X, and
  \(F_{\rm mix}\)-X compare in \(-\epsilon_{\rm HOMO}\)?
- How many negative-energy virtual spatial/spin orbitals occur under the
  stated counting definition?
- What are the degeneracy, angular character, effective quantum number, and
  radial extent of the He levels?
- Is the He conclusion stable to increasingly diffuse basis sets?

### 7.1 Inspect and calculate

```bash
python -m json.tool inputs/atomic.json
python calculations/atomic/run_atomic.py
```

Run the complete command; the He basis-sensitivity portion is required for
paper table 4.

### 7.2 Inspect all atom/functional pairs

```bash
python - <<'PY'
import csv

with open("results/atomic/atomic_systematics.csv", newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    print(
        row["system"],
        row["functional"],
        f"converged={row['converged']}",
        f"-HOMO={row['minus_homo_ev']} eV",
        row["count_display"],
    )

assert len(rows) == 15
assert all(row["converged"] == "True" for row in rows)
assert all(
    row["converged"] == "True"
    for row in rows
    if row["system"] == "Li"
)
PY
```

Li is a one-valence-electron CRENBL ECP model and is required to converge.
The released run contains all 15 converged atom/functional pairs.

Inspect the two He level groups and all four basis rows:

```bash
cat results/atomic/he_negative_virtual_levels.csv
cat results/atomic/he_basis_sensitivity.csv
```

### 7.3 Recreate the atomic figures independently

```bash
python calculations/plot_results.py fig-atomic-systematics
python calculations/plot_results.py fig-he-levels
python calculations/plot_results.py fig-he-comparison
git diff -- results/atomic/ figures/fig_atomic_systematics.png \
  figures/fig_he_levels.png figures/fig_he_comparison.png
```

`fig_he_comparison.png` is an additional audit figure.

The atomic CSV files are the direct sources of paper tables 2–4; Section 11
gives the exact selections.

## 8. Stage 6 — common-density radial potentials

Scientific question: when each approximation is evaluated on the same
reference density, is the hydrogen-derived potential shape transferred into
the intermediate radial region of He and the Na 3s ECP model?

### 8.1 Inspect, calculate, and plot

```bash
python -m json.tool inputs/potentials.json
python calculations/potentials/run_potentials.py
python -m json.tool results/potentials/radial_potentials_metadata.json
head -n 6 results/potentials/radial_potentials.csv
python calculations/plot_results.py fig-radial-potentials
git diff -- results/potentials/ figures/fig_radial_potentials.png
```

The CSV records every plotted radial point and density-threshold mask. The He
and Na metrics compare regularized \(F_{\rm mix}\)-X and PBE-X on one common
HF density. They diagnose transfer of the intermediate-range potential shape
rather than strict far-tail exactness.

## 9. Stage 7 — H2 binding region

Scientific question: does \(F_{\rm mix}\)+LYP give a stable, shallow
near-equilibrium H2 well while preserving the intended atomic-potential
behavior?

### 9.1 Inspect and calculate the curve

```bash
python -m json.tool inputs/h2.json
python calculations/h2/run_h2.py
```

This command writes three complementary records:

- `results/h2/h2_curve.csv`: all 19 calculated geometries;
- `results/h2/h2_minima.csv`: the two spline minima reported in paper
  table 5;
- `results/h2/h2_metadata.json`: thresholds, settings, reference, and
  convergence summary.

### 9.2 Inspect convergence, spin symmetry, and minima

```bash
python - <<'PY'
import csv

with open("results/h2/h2_curve.csv", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 19
for row in rows:
    assert row["fmix_converged"] == "True"
    assert row["rhf_converged"] == "True"
    assert row["fci_converged"] == "True"
    print(
        f"R={float(row['R_A']):.2f} A  "
        f"Fmix={float(row['Erel_fmix_eV']): .7f} eV  "
        f"FCI={float(row['Erel_fci_eV']): .7f} eV  "
        f"spin norm={float(row['spin_density_matrix_norm']):.3e}"
    )
PY

cat results/h2/h2_minima.csv
python -m json.tool results/h2/h2_metadata.json
```

Each curve is referenced to its own same-basis \(2E_{\rm H}\) threshold. The
canonical spline minima are about 0.712 Å / 3.58 eV for
\(F_{\rm mix}\)+LYP and 0.743 Å / 4.70 eV for same-basis FCI.

### 9.3 Recreate the H2 figure

```bash
python calculations/plot_results.py fig-h2
git diff -- results/h2/h2_curve.csv results/h2/h2_minima.csv \
  results/h2/h2_metadata.json figures/fig_h2.png
```

## 10. Stage 8 — H2 grid refinement

Scientific question: are the near-minimum relative energies stable when the
numerical quadrature grid is increased?

```bash
python -m json.tool inputs/h2.json
python calculations/h2/validate_grid.py
cat results/h2/h2_grid_validation.csv
python -m json.tool results/h2/h2_grid_validation.json
```

The CSV has six rows: two grid levels at three near-minimum distances. Check
the Fmix, RHF, and FCI convergence flags and spin-density norm on every row.
The JSON reports each level-6-minus-level-5 difference and its maximum
absolute value.

## 11. Paper-table reconstruction

The five tables require no generated intermediate. Their numerical sources
are ordinary CSV files written directly by the calculation drivers. The
full-precision CSV values are authoritative; the rounding below applies only
to display.

### Table 1 — hydrogenic fixed-density errors

Producer and source:

```bash
python calculations/analytic/run_hydrogenic.py
cat results/analytic/hydrogenic_ions.csv
```

Use systems in this order:

1. `H`
2. `He+`
3. `Li2+`
4. `Be3+`
5. `B4+`

For each system, pivot the three method rows:

| Display column | CSV row/column | Display rule |
|---|---|---|
| ion | `system` | exact string |
| \(Z\) | any row, `Z` | integer |
| unregularized H-exact error | `method=unregularized_H_exact`, `absolute_error_mHa` | 6 decimal places |
| regularized \(F_{\rm mix}\) error | `method=regularized_Fmix`, `absolute_error_mHa` | 6 decimal places |
| PBE-X error | `method=PBE_X`, `absolute_error_mHa` | 3 decimal places |

The unit of the three error columns is mHa. Confirm that there are exactly
15 rows and one row for every system/method pair.

### Table 2 — matched atomic comparison

Producer and source:

```bash
python calculations/atomic/run_atomic.py
cat results/atomic/atomic_systematics.csv
```

Use systems `H`, `He`, `Li`, `Na`, `K` in that order. Require
`converged=True` for all 15 rows. Pivot `functional` over `PBE_X`, `SCAN_X`,
and `Fmix_X`.

| Display column | CSV row/column | Display rule |
|---|---|---|
| system | `system` | exact string |
| NIST IP | any functional row, `experimental_ip_ev` | 6 decimal places |
| PBE-X \(-\epsilon_{\rm HOMO}\) | `functional=PBE_X`, `minus_homo_ev` | 3 decimal places |
| SCAN-X \(-\epsilon_{\rm HOMO}\) | `functional=SCAN_X`, `minus_homo_ev` | 3 decimal places |
| \(F_{\rm mix}\)-X \(-\epsilon_{\rm HOMO}\) | `functional=Fmix_X`, `minus_homo_ev` | 3 decimal places |
| PBE-X negative virtual diagnostic | `functional=PBE_X`, `count_display` | exact string; no rounding |
| \(F_{\rm mix}\)-X negative virtual diagnostic | `functional=Fmix_X`, `count_display` | exact string; no rounding |

Energy columns are in eV. For He, `count_display` is the RKS spatial-orbital
count. For H/Li/Na/K, it reports separate alpha and beta UKS spin-orbital
counts. Preserve that distinction in the table caption or notes.

### Table 3 — He negative virtual levels

Producer and source:

```bash
python calculations/atomic/run_atomic.py
cat results/atomic/he_negative_virtual_levels.csv
```

Use all rows in ascending `group_index`; the released result has two. Confirm
that every row has `system=He`, `functional=Fmix_X`, and `method=RKS`.

| Display column | CSV column | Display rule |
|---|---|---|
| character | `l_symbol` | append “-like” |
| orbital energy | `energy_ha` | 6 decimal places |
| degeneracy | `spatial_degeneracy` | integer |
| effective quantum number | `n_eff` | 3 decimal places |
| radial extent | `r_rms_bohr` | 3 decimal places |

Orbital energy is in Ha and radial extent
\(\sqrt{\langle r^2\rangle}\) is in bohr.

### Table 4 — He basis sensitivity

Producer and source:

```bash
python calculations/atomic/run_atomic.py
cat results/atomic/he_basis_sensitivity.csv
```

Use all four rows in file order and require `converged=True`.

| Display column | CSV column | Display rule |
|---|---|---|
| basis | `basis` | use exact label, or abbreviate `aug-cc-pV` as `aV` consistently |
| \(-\epsilon_{\rm HOMO}\) | `minus_homo_ev` | 6 decimal places |
| negative spatial orbitals | `negative_virtual_spatial_count` | integer |

The energy unit is eV. `generated_even_tempered=True` identifies the
`aug-cc-pV5Z+ET1` row. The semicolon-separated `group_*` fields provide the
full level-group audit but are not additional display columns in this table.

### Table 5 — H2 spline minima

Producer and source:

```bash
python calculations/h2/run_h2.py
cat results/h2/h2_minima.csv
```

Use both rows in file order:

| Display method | CSV `method` | \(R_e\) | \(D_e\) |
|---|---|---|---|
| \(F_{\rm mix}\)+LYP | `implemented_Fmix_LYP` | `R_e_angstrom`, 5 decimal places | `D_e_eV`, 5 decimal places |
| FCI | `FCI` | `R_e_angstrom`, 5 decimal places | `D_e_eV`, 5 decimal places |

\(R_e\) is in angstrom and \(D_e\) in eV. The values are produced by the
spline search specified in `inputs/h2.json`, with each curve referenced to
its own same-basis atomic threshold.

## 12. Figure provenance

Every figure is created by `calculations/plot_results.py` from the listed
saved numerical records. Plotting never imports the functional code or a
calculation driver.

| Figure | Individual target | Saved numerical source |
|---|---|---|
| `fig_calibration.png` | `fig-calibration` | `results/calibration/calibration_results.json` |
| `fig_gp93_building_block.png` | `fig-gp93-building-block` | `results/analytic/gp93_building_block.csv`, `analytic_metadata.json` |
| `fig_hydrogen_regularization.png` | `fig-hydrogen-regularization` | `results/analytic/hydrogen_regularization.csv`, `analytic_metadata.json` |
| `fig_atomic_systematics.png` | `fig-atomic-systematics` | `results/atomic/atomic_systematics.csv` |
| `fig_he_levels.png` | `fig-he-levels` | `results/atomic/he_negative_virtual_levels.csv` |
| `fig_he_comparison.png` | `fig-he-comparison` | He rows of `results/atomic/atomic_systematics.csv` |
| `fig_radial_potentials.png` | `fig-radial-potentials` | `results/potentials/radial_potentials.csv` |
| `fig_h2.png` | `fig-h2` | `results/h2/h2_curve.csv`, `h2_metadata.json` |

Run all eight commands individually:

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

## 13. Cross-file validation

After all numerical stages:

```bash
python calculations/validate_results.py
```

The validator checks convergence, calibration selection, input snapshots,
calibrated-parameter propagation, exact-H identities, derivatives, radial
potential ranges, atomic state counts, table-source inventories, H2 spin
symmetry, spline minima, and grid refinement.

Then inspect all differences:

```bash
git status --short
git diff -- results/calibration/
git diff -- results/analytic/
git diff -- results/atomic/
git diff -- results/potentials/
git diff -- results/h2/
git diff -- figures/
```

Small last-digit numerical differences can occur across CPUs or BLAS
libraries. The validator applies explicit scientific tolerances rather than
requiring byte-identical floating-point text.

## 14. Optional integration/CI route

After the individual workflow is understood, the complete producer graph can
be checked with:

```bash
python reproduce.py
```

This cleans pipeline-owned results, executes every calculation in the same
order documented above, rebuilds all figures, and runs validation.

To test only the data-to-figure and validation stages on existing results:

```bash
python reproduce.py --stage postprocess
```
