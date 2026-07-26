# Numerical results and provenance

This directory contains the machine-readable numerical results used for the
paper's figures and tables. Run commands from the repository root.
Calculation programs write CSV/JSON data here; `calculations/plot_results.py`
reads those saved data and writes figures in a separate step.

Paths beginning with `analytic/`, `atomic/`, `calibration/`, `h2/`, or
`potentials/` are relative to this directory. All other paths are relative to
the repository root.

The row counts below correspond to the released input decks and provide
simple checks after a rerun. `Ha` means hartree, `mHa` millihartree, `eV`
electronvolt, `bohr` the atomic unit of length, and `angstrom` angstrom.
Quantities without a stated unit are dimensionless or are counts.

## Producer sequence

| Step | Calculation command | Principal inputs | Numerical outputs |
|---|---|---|---|
| calibration | `python calculations/calibration/run_calibration.py` | `inputs/calibration.json`, `inputs/atomic.json`, `functional/parameters.json`, `data/experimental_ip.csv` | `calibration/calibration_results.csv`, `calibration/calibration_summary.csv`, `calibration/calibration_results.json`, `calibration/selected_parameters.json` |
| GP93 and exact-H diagnostics | `python calculations/analytic/run_analytic.py` | `inputs/analytic.json`, `functional/parameters.json`, functional modules | `analytic/gp93_building_block.csv`, `analytic/hydrogen_regularization.csv`, `analytic/derivative_checks.csv`, `analytic/analytic_metadata.json`, `analytic/DERIVATION_NOTES.md` |
| hydrogenic ions | `python calculations/analytic/run_hydrogenic.py` | `inputs/hydrogenic.json`, selected switch parameters, functional modules | `analytic/hydrogenic_ions.csv`, `analytic/hydrogenic_ions.json` |
| unregularized H SCF | `python calculations/atomic/run_hydrogen_exact.py` | `inputs/hydrogen_exact_scf.json`, selected switch parameters, `data/experimental_ip.csv` | `atomic/hydrogen_exact_scf.csv`, `atomic/hydrogen_exact_scf.json` |
| matched atoms and He basis study | `python calculations/atomic/run_atomic.py` | `inputs/atomic.json`, `functional/parameters.json`, `data/experimental_ip.csv` | `atomic/atomic_systematics.csv`, `atomic/state_groups.csv`, `atomic/he_negative_virtual_levels.csv`, `atomic/he_basis_sensitivity.csv`, `atomic/results.json` |
| radial potentials | `python calculations/potentials/run_potentials.py` | `inputs/potentials.json`, `functional/parameters.json` | `potentials/radial_potentials.csv`, `potentials/radial_potentials_metadata.json` |
| H2 curve and minima | `python calculations/h2/run_h2.py` | `inputs/h2.json`, `functional/parameters.json`, `data/h2_reference.csv` | `h2/h2_curve.csv`, `h2/h2_minima.csv`, `h2/h2_metadata.json` |
| H2 grid check | `python calculations/h2/validate_grid.py` | grid-validation block in `inputs/h2.json` and the same H2 settings | `h2/h2_grid_validation.csv`, `h2/h2_grid_validation.json` |

The selected switch parameters are calibrated values, not additional fitted
inputs. Before downstream stages, verify that
`calibration/selected_parameters.json` agrees with
`functional/parameters.json`, as shown in `REPRODUCTION_GUIDE.md`.

## Calibration

### `calibration/calibration_results.csv`

- **Expected rows:** 45 = 9 values of \(s_0\) times 5 systems.
- **One row means:** one atom at one trial \(s_0\), with \(p\) fixed.
- **Key columns:** `calculated_ip_eV` is
  \(-\epsilon_{\rm HOMO}\); `absolute_error_eV` is
  \(\lvert-\epsilon_{\rm HOMO}-\mathrm{IP}_{\rm expt}\rvert\);
  `spin` is \(2S\); and `attempt_count`, `converged`, and `exception`
  preserve numerical status instead of dropping failed cases.
- **Use:** detailed audit of the complete parameter scan.

### `calibration/calibration_summary.csv`

- **Expected rows:** 9, one for each trial \(s_0\).
- **Key columns:** `mae_5system_eV` is the unweighted mean of the five
  `absolute_error_eV` values and is present only when all five calculations
  converge. `failed_systems` names any failed systems; the selection rule
  never averages over a smaller set.
- **Use:** direct inspection of the objective and its minimum.

### `calibration/calibration_results.json`

This is the complete structured calibration record: protocol, reference HF
calculations, every SCF/optimizer attempt, all 45 results, software
provenance, and the final minimum. Its `scan[].s0` and
`scan[].mae_5system_eV` arrays produce `figures/fig_calibration.png`.

### `calibration/selected_parameters.json`

This compact handoff records the minimizing `s0`, fixed `p`, five-system MAE,
and selection rule. It must agree with the production defaults in
`functional/parameters.json`. Each downstream metadata JSON records the
effective pair independently, allowing the validator to detect a mismatched
run.

```bash
python calculations/plot_results.py fig-calibration
```

## Analytic GP93 and hydrogen tests

### `analytic/gp93_building_block.csv`

- **Expected rows:** 1,401 on the logarithmic \(s\) grid.
- **Units:** \(s\), \(z\), enhancement factors, and crossover \(\chi\) are
  dimensionless.
- **Definitions:** `z_raw` is \(3\ln s+\ln(6\pi)\);
  `F_GP93_bare_analytic` is the unclipped analytic continuation;
  `F_GP93_fmix_py_zclipped` is the finite implementation spline;
  `Fmix_direct_selected_parameters` is the selected crossover;
  `Fmix_direct_reconstructed` independently reconstructs it;
  `s_after_mgga_clip` and `Fmix_sampled_by_mgga_ch` record values seen by the
  production meta-GGA channel.
- **Figure:** all three panels of
  `figures/fig_gp93_building_block.png`. Annotations use the effective
  \(s_0,p\) snapshot in `analytic/analytic_metadata.json`.

```bash
python calculations/plot_results.py fig-gp93-building-block
```

### `analytic/hydrogen_regularization.csv`

- **Expected rows:** 7,401 radial points from 0.2 to 15 bohr.
- **Units:** `r_bohr` and density-derived quantities are in atomic units;
  `minus_r_vx_*` columns are the numerical products \(-r v_x(r)\).
- **Definitions:** `n_H`, `dn_dr`, and `tau_H` are exact-H descriptors;
  `alpha`, `G_alpha`, and `X_n` expose the gates;
  `minus_r_vx_exact`, `minus_r_vx_bare_GP93`,
  `minus_r_vx_unregularized_H_exact`, and
  `minus_r_vx_implemented` separate the exact reference, bare building
  block, unregularized crossover, and practical regularized functional.
  `vrho_implemented`, `vsigma_implemented`, and `vtau_implemented` are the
  derivatives returned by the production channel.
- **Figure:** `figures/fig_hydrogen_regularization.png`.
- **Validation:** exact-H intermediate-range accuracy and smooth
  density-window transition.

```bash
python calculations/plot_results.py fig-hydrogen-regularization
```

### `analytic/derivative_checks.csv`

- **Expected rows:** 65.
- **One row means:** one analytic-versus-finite-difference or
  grid-convergence check.
- **Key columns:** `check_type` identifies the test;
  `code_or_analytic` and `finite_difference_or_reference` are the compared
  values; `abs_error` and `rel_error` are their differences.
- **Use:** numerical validation.

### `analytic/analytic_metadata.json`

This provenance companion records the complete `inputs/analytic.json`
settings, functional-parameter snapshot, and effective calibrated \(s_0,p\).
Both analytic figures read annotations from this saved metadata, so plotting
does not read a mutable input deck.

### `analytic/hydrogenic_ions.csv`

- **Expected rows:** 15 = 5 ions times 3 methods.
- **Units:** eigenvalues are in Ha and eV; errors are in mHa.
- **Definitions:** every row uses the exact normalized hydrogenic \(1s\)
  density. `exact_Z2_over_2_Ha` is \(Z^2/2\);
  `hartree_plus_exchange_expectation_Ha` is
  \(\langle v_H+v_x\rangle\); `signed_error_mHa` is
  \(1000[-\epsilon_{1s}-Z^2/2]\).
- **Paper table:** direct source of the hydrogenic fixed-density table.

Table reconstruction:

1. order systems as `H`, `He+`, `Li2+`, `Be3+`, `B4+`;
2. pivot the `method` rows `unregularized_H_exact`,
   `regularized_Fmix`, and `PBE_X`;
3. display `absolute_error_mHa`;
4. use 6 decimal places for the first two methods and 3 for PBE-X.

The full-precision values remain in the CSV.

### `analytic/hydrogenic_ions.json`

This companion contains the same 15 rows plus method definitions, selected
\(s_0,p\), input path, and scaled radial-grid specification. It is the
preferred audit record for the CSV labels.

## Atomic calculations

### `atomic/hydrogen_exact_scf.csv`

- **Expected rows:** 1.
- **Units:** suffix `_ha` denotes Ha and `_ev` denotes eV.
- **Definition:** final zero-damping, zero-level-shift self-consistent
  H/aug-cc-pV5Z calculation with the unregularized hydrogen-exact GGA.
  Alpha and beta negative-virtual counts are separate.
- **Use:** compact numerical summary of the exact-H SCF test.

### `atomic/hydrogen_exact_scf.json`

This companion preserves the functional definition, complete input snapshot,
selected parameters, every stabilization stage, and final CSV row. It is the
primary convergence audit for the unregularized-H calculation.

### `atomic/atomic_systematics.csv`

- **Expected rows:** 15 = 5 systems times 3 exchange-only functionals.
- **Units:** total and orbital energies with `_ha` are in Ha; HOMO and
  experimental ionization energies with `_ev` are in eV.
- **Definitions:** `minus_homo_ev` is
  \(-\epsilon_{\rm HOMO}\). For He/RKS,
  `negative_virtual_spatial_count` counts spatial orbitals once. For
  H/Li/Na/K UKS calculations, alpha and beta spin-orbital counts are
  separate. `count_definition` records the convention.
- **Figures:** `figures/fig_atomic_systematics.png` and
  `figures/fig_he_comparison.png`.
- **Paper table:** direct source of the matched atomic comparison.

Table reconstruction:

1. require `converged=True` for all 15 rows;
2. order systems as `H`, `He`, `Li`, `Na`, `K`;
3. pivot `functional` over `PBE_X`, `SCAN_X`, and `Fmix_X`;
4. display `experimental_ip_ev` to 6 decimal places;
5. display each `minus_homo_ev` to 3 decimal places;
6. copy `count_display` exactly for PBE-X and Fmix-X.

System and functional display order is not inferred from alphabetical
sorting.

```bash
python calculations/plot_results.py fig-atomic-systematics
python calculations/plot_results.py fig-he-comparison
```

### `atomic/he_negative_virtual_levels.csv`

- **Expected rows:** 2 degenerate level groups in the released calculation.
- **Units:** `energy_ha`/`energy_ev` are orbital energies;
  `r_rms_bohr` is \(\sqrt{\langle r^2\rangle}\) in bohr.
- **Definitions:** `spatial_degeneracy` counts RKS spatial orbitals;
  `l_est` and `l_symbol` come from AO angular-momentum projections;
  `l_confidence` is the dominant projected fraction; and
  `n_eff=1/sqrt(-2*energy_ha)`.
- **Figure:** `figures/fig_he_levels.png`.
- **Paper table:** direct source of the He negative-level table.

Use all rows in ascending `group_index`. Display `l_symbol` with “-like”,
`energy_ha` to 6 decimals, `spatial_degeneracy` as an integer, and `n_eff`
and `r_rms_bohr` to 3 decimals.

```bash
python calculations/plot_results.py fig-he-levels
```

### `atomic/he_basis_sensitivity.csv`

- **Expected rows:** 4 basis choices.
- **Units:** HOMO values are in eV and group energies in Ha.
- **Definitions:** semicolon-separated `group_*` fields list distinct
  negative-energy spatial groups; `generated_even_tempered` identifies the
  ET1 basis.
- **Paper table:** direct source of the He basis-sensitivity table.

Use all four rows in file order and require `converged=True`. Display `basis`
exactly, or consistently abbreviate `aug-cc-pV` as `aV`; display
`minus_homo_ev` to 6 decimals and
`negative_virtual_spatial_count` as an integer.

### `atomic/state_groups.csv`

- **Expected rows:** 17 groups across all released systems, functionals, and
  spin channels.
- **Purpose:** expanded audit record for every negative virtual group,
  including energy, within-channel degeneracy, angular assignment, and
  effective quantum number.
- **Use:** audit data; the He table uses the more detailed dedicated CSV.

### `atomic/results.json`

This is the full structured atomic record. It contains settings, convergence
diagnostics, every orbital energy and occupation, grouped negative virtual
states, the He basis study, generated ET1 exponents, and software provenance.
The smaller CSV files are explicit projections of this record.

## H2

### `h2/h2_curve.csv`

- **Expected rows:** 19 bond lengths.
- **Units:** `R_A` is in angstrom; total energies are in Ha; relative
  energies are in eV; `spin_density_matrix_norm` is a dimensionless
  Frobenius norm.
- **Definitions:**
  `Erel_fmix_eV=(E_fmix_lyp_Ha-2*E_H_fmix_lyp)*Ha_to_eV` and
  `Erel_fci_eV=(E_fci_Ha-2*E_H_same_basis)*Ha_to_eV`.
  `fmix_converged`, `rhf_converged`, and `fci_converged` retain numerical
  status at every geometry.
- **Figure:** calculated points and spline curves in
  `figures/fig_h2.png`.

### `h2/h2_minima.csv`

- **Expected rows:** 2.
- **Columns:** `method`, `R_e_angstrom`, `D_e_eV`.
- **Definitions:** spline minima evaluated with the search density in
  `inputs/h2.json`; each curve uses its own same-basis atomic threshold.
- **Paper table:** direct source of the H2 minima table.

Use both rows in file order. Display `implemented_Fmix_LYP` as
\(F_{\rm mix}\)+LYP and `FCI` as FCI. Display both `R_e_angstrom` and
`D_e_eV` to 5 decimal places.

### `h2/h2_metadata.json`

This file records the functional, basis, grid, spin treatment, experimental
reference, two separately calculated atomic thresholds, spline minima,
all-point convergence flag, and maximum spin-density norm.

- **Figure:** minima and experimental marker in `figures/fig_h2.png`.
- **Validation:** cross-check against `h2_minima.csv`, convergence, and spin
  symmetry.

```bash
python calculations/plot_results.py fig-h2
```

### `h2/h2_grid_validation.csv`

- **Expected rows:** 6 = 2 grid levels times 3 near-minimum distances.
- **Units:** distance is in angstrom and relative energy in eV.
- **Purpose:** every value in the production-grid refinement comparison,
  including Fmix/RHF/FCI convergence flags and spin-density norm.

### `h2/h2_grid_validation.json`

This companion records the per-distance level-6-minus-level-5 differences
and their maximum absolute value.

- **Use:** numerical validation of quadrature-grid stability.

## Common-density radial potentials

### `potentials/radial_potentials.csv`

- **Expected rows:** 4,136 total for exact H, He/RHF/aug-cc-pV5Z, and the
  Na/UHF/SBKJC 3s ECP model.
- **Units:** `r_bohr` is in bohr and `density` in atomic density units; all
  `minus_r_vx_*` columns are numerical products \(-r v_x\).
- **Definitions:** `minus_r_vx_hydrogen_exact_gp93` is populated only for
  exact H; `minus_r_vx_implemented` and `minus_r_vx_pbe_x` are evaluated on
  the same density for each system;
  `minus_r_vx_self_interaction_free_reference` is \(r v_H\) for
  one-electron H/Na and \(r v_H/2\) for He;
  `density_threshold_mask` identifies the trusted plotted/averaged range.
- **Figure:** all three panels of
  `figures/fig_radial_potentials.png`.

```bash
python calculations/plot_results.py fig-radial-potentials
```

### `potentials/radial_potentials_metadata.json`

This file records the input snapshot, selected \(s_0,p\), included
regularization, reference convergence and energy, radial ranges, and
Fmix/PBE mean absolute distances from the stated reference.

- **Use:** interpretation and numerical validation of the H/He/Na
  potential diagnostics.

## Figure source map

| Figure | Numerical source |
|---|---|
| `fig_calibration.png` | `calibration/calibration_results.json` |
| `fig_gp93_building_block.png` | `analytic/gp93_building_block.csv`, `analytic/analytic_metadata.json` |
| `fig_hydrogen_regularization.png` | `analytic/hydrogen_regularization.csv`, `analytic/analytic_metadata.json` |
| `fig_atomic_systematics.png` | `atomic/atomic_systematics.csv` |
| `fig_he_levels.png` | `atomic/he_negative_virtual_levels.csv` |
| `fig_he_comparison.png` | He rows of `atomic/atomic_systematics.csv` |
| `fig_radial_potentials.png` | `potentials/radial_potentials.csv` |
| `fig_h2.png` | `h2/h2_curve.csv`, `h2/h2_metadata.json` |

## Paper-table source map

| Paper table | Direct CSV source | Producer |
|---|---|---|
| hydrogenic fixed-density errors | `analytic/hydrogenic_ions.csv` | `run_hydrogenic.py` |
| matched atomic comparison | `atomic/atomic_systematics.csv` | `run_atomic.py` |
| He negative virtual levels | `atomic/he_negative_virtual_levels.csv` | `run_atomic.py` |
| He basis sensitivity | `atomic/he_basis_sensitivity.csv` | `run_atomic.py` |
| H2 spline minima | `h2/h2_minima.csv` | `run_h2.py` |

No secondary table file is generated. This keeps the connection between
calculation settings, full-precision data, and displayed values visible to
the reader.

## Inspecting a rerun

After each calculation:

```bash
git status --short
git diff -- results/calibration/
git diff -- results/analytic/
git diff -- results/atomic/
git diff -- results/potentials/
git diff -- results/h2/
```

Small last-digit floating-point differences may occur across CPUs or BLAS
libraries. Scientific acceptance criteria are checked with explicit
tolerances:

```bash
python calculations/validate_results.py
```
