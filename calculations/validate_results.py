#!/usr/bin/env python3
"""Fail fast if canonical numerical results and figures are inconsistent."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def rows(relative: str) -> list[dict[str, str]]:
    with (RESULTS / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(relative: str):
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_files(relative_paths: tuple[str, ...]) -> None:
    """Check the documented input/result inventory without opaque hashes."""

    missing = [
        relative for relative in relative_paths if not (ROOT / relative).is_file()
    ]
    require(not missing, "missing required files: " + ", ".join(missing))


def main() -> None:
    require_files(
        (
            "functional/parameters.json",
            "inputs/calibration.json",
            "inputs/analytic.json",
            "inputs/hydrogenic.json",
            "inputs/hydrogen_exact_scf.json",
            "inputs/atomic.json",
            "inputs/h2.json",
            "inputs/potentials.json",
            "data/experimental_ip.csv",
            "data/h2_reference.csv",
            "results/calibration/calibration_results.csv",
            "results/calibration/calibration_summary.csv",
            "results/calibration/calibration_results.json",
            "results/calibration/selected_parameters.json",
            "results/analytic/gp93_building_block.csv",
            "results/analytic/hydrogen_regularization.csv",
            "results/analytic/derivative_checks.csv",
            "results/analytic/hydrogenic_ions.csv",
            "results/analytic/hydrogenic_ions.json",
            "results/analytic/analytic_metadata.json",
            "results/analytic/DERIVATION_NOTES.md",
            "results/atomic/atomic_systematics.csv",
            "results/atomic/state_groups.csv",
            "results/atomic/he_negative_virtual_levels.csv",
            "results/atomic/he_basis_sensitivity.csv",
            "results/atomic/results.json",
            "results/atomic/hydrogen_exact_scf.csv",
            "results/atomic/hydrogen_exact_scf.json",
            "results/h2/h2_curve.csv",
            "results/h2/h2_minima.csv",
            "results/h2/h2_metadata.json",
            "results/h2/h2_grid_validation.csv",
            "results/h2/h2_grid_validation.json",
            "results/potentials/radial_potentials.csv",
            "results/potentials/radial_potentials_metadata.json",
        )
    )
    selected = load_json("calibration/selected_parameters.json")
    parameters = json.loads(
        (ROOT / "functional" / "parameters.json").read_text(encoding="utf-8")
    )
    analytic_input = json.loads(
        (ROOT / "inputs" / "analytic.json").read_text(encoding="utf-8")
    )
    calibration_input = json.loads(
        (ROOT / "inputs" / "calibration.json").read_text(encoding="utf-8")
    )
    atomic_input = json.loads(
        (ROOT / "inputs" / "atomic.json").read_text(encoding="utf-8")
    )
    hydrogenic_input = json.loads(
        (ROOT / "inputs" / "hydrogenic.json").read_text(encoding="utf-8")
    )
    hydrogen_exact_input = json.loads(
        (ROOT / "inputs" / "hydrogen_exact_scf.json").read_text(
            encoding="utf-8"
        )
    )
    h2_input = json.loads(
        (ROOT / "inputs" / "h2.json").read_text(encoding="utf-8")
    )
    potentials_input = json.loads(
        (ROOT / "inputs" / "potentials.json").read_text(encoding="utf-8")
    )
    require(float(selected["s0"]) == float(parameters["selected_s0"]), "s0 mismatch")
    require(int(selected["p"]) == int(parameters["selected_p"]), "p mismatch")

    calibration_document = load_json("calibration/calibration_results.json")
    require(
        calibration_document["input_settings"] == calibration_input,
        "calibration results do not match the current input deck",
    )
    require(
        calibration_document["atomic_input_settings"] == atomic_input,
        "calibration results do not match the current atomic input deck",
    )
    calibration = rows("calibration/calibration_summary.csv")
    expected_s0 = [float(value) for value in calibration_document["method"]["s0_values"]]
    require(calibration_document["summary"].get("status") == "complete",
            "calibration is only a partial checkpoint")
    require(len(expected_s0) == 9, "calibration must define nine s0 points")
    require(len(calibration) == len(expected_s0),
            "calibration summary does not contain every expected s0 point")
    require(
        sorted(float(row["s0"]) for row in calibration) == sorted(expected_s0),
        "calibration s0 inventory is incomplete or duplicated",
    )
    require(len(calibration_document["scan"]) == len(expected_s0),
            "calibration JSON scan is incomplete")
    calibration_detail = rows("calibration/calibration_results.csv")
    require(len(calibration_detail) == 5 * len(expected_s0),
            "calibration detail must contain five systems at every s0")
    require(all(row["converged"] == "True" for row in calibration_detail),
            "calibration detail contains an unconverged calculation")
    require(all(row["all_five_converged"] == "True" for row in calibration),
            "not every calibration point converged")
    minimum = min(calibration, key=lambda row: float(row["mae_5system_eV"]))
    require(float(minimum["s0"]) == float(selected["s0"]),
            "selected s0 is not the calibrated minimum")
    require(int(minimum["p"]) == int(selected["p"]),
            "selected p disagrees with calibration")
    final_minimum = calibration_document["summary"]["minimum_valid_point"]
    require(float(final_minimum["s0"]) == float(selected["s0"]),
            "final calibration JSON minimum disagrees with selection")

    atomic = rows("atomic/atomic_systematics.csv")
    atomic_metadata = load_json("atomic/results.json")
    require(
        atomic_metadata["input_settings"] == atomic_input,
        "atomic results do not match the current input deck",
    )
    require(
        int(calibration_document["method"]["grid_level"])
        == int(atomic_metadata["settings"]["grid_level"]),
        "calibration and final atomic grid levels differ",
    )
    require(
        math.isclose(
            float(calibration_document["method"]["conv_tol"]),
            float(atomic_metadata["settings"]["conv_tol"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "calibration and final atomic convergence tolerances differ",
    )
    expected_atomic_pairs = {
        (str(system["name"]), str(functional["name"]))
        for system in atomic_input["systems"]
        for functional in atomic_input["functionals"]
    }
    require(
        len(atomic) == len(expected_atomic_pairs),
        "atomic table does not contain every configured system/functional pair",
    )
    require(
        {(row["system"], row["functional"]) for row in atomic}
        == expected_atomic_pairs,
        "atomic table system/functional inventory differs from inputs/atomic.json",
    )
    require(all(row["converged"] == "True" for row in atomic),
            "at least one atomic calculation is unconverged")
    selected_point = next(
        point
        for point in calibration_document["scan"]
        if float(point["s0"]) == float(selected["s0"])
    )
    for system in ("H", "He", "Li", "Na", "K"):
        subset = [row for row in atomic if row["system"] == system]
        fmix = next(row for row in subset if row["functional"] == "Fmix_X")
        for baseline_name in ("PBE_X", "SCAN_X"):
            baseline = next(
                row for row in subset if row["functional"] == baseline_name
            )
            fmix_error = abs(
                float(fmix["minus_homo_ev"]) - float(fmix["experimental_ip_ev"])
            )
            baseline_error = abs(
                float(baseline["minus_homo_ev"])
                - float(baseline["experimental_ip_ev"])
            )
            require(fmix_error < baseline_error,
                    f"Fmix does not improve {system} relative to {baseline_name}")
        calibrated = next(
            result
            for result in selected_point["systems"]
            if result["symbol"] == system
        )
        require(
            abs(float(calibrated["calculated_ip_eV"]) - float(fmix["minus_homo_ev"]))
            < 5.0e-6,
            f"selected calibration and final atomic value differ for {system}",
        )
    for system in ("H", "He"):
        for functional in ("PBE_X", "SCAN_X"):
            row = next(
                item
                for item in atomic
                if item["system"] == system
                and item["functional"] == functional
            )
            count_key = (
                "negative_virtual_spatial_count"
                if system == "He"
                else "negative_virtual_alpha_count"
            )
            require(
                int(row[count_key]) == 0,
                f"zero-virtual result failed for {system}/{functional}",
            )
    for system in ("Li", "Na", "K"):
        by_functional = {
            row["functional"]: int(row["negative_virtual_alpha_count"])
            for row in atomic
            if row["system"] == system
        }
        require(
            by_functional["Fmix_X"]
            > max(by_functional["PBE_X"], by_functional["SCAN_X"]),
            f"Fmix does not enlarge the negative virtual manifold for {system}",
        )

    he_basis = rows("atomic/he_basis_sensitivity.csv")
    require(all(row["converged"] == "True" for row in he_basis),
            "He basis-sensitivity calculation is unconverged")
    he_basis_by_label = {row["basis"]: row for row in he_basis}
    require(
        int(he_basis_by_label["aug-cc-pVTZ"][
            "negative_virtual_spatial_count"
        ]) >= 1,
        "He triple-zeta basis has no negative virtual state",
    )
    for label in ("aug-cc-pVQZ", "aug-cc-pV5Z", "aug-cc-pV5Z+ET1"):
        symbols = set(he_basis_by_label[label]["group_l_symbols"].split(";"))
        require(
            {"s", "p"}.issubset(symbols),
            f"He {label} basis does not contain both s- and p-like groups",
        )
    hydrogen_exact_scf = load_json("atomic/hydrogen_exact_scf.json")
    require(
        hydrogen_exact_scf["input_settings"] == hydrogen_exact_input,
        "unregularized H SCF does not match the current input deck",
    )
    require(bool(hydrogen_exact_scf["result"]["converged"]),
            "unregularized hydrogen-exact H SCF is unconverged")
    require(float(hydrogen_exact_scf["result"]["final_level_shift_ha"]) == 0.0,
            "unregularized hydrogen-exact H result retains a level shift")
    require(float(hydrogen_exact_scf["result"]["absolute_error_ev"]) < 0.01,
            "unregularized hydrogen-exact H HOMO error exceeds 0.01 eV")
    require("no density window" in hydrogen_exact_scf["definition"],
            "H-exact SCF is not identified as unregularized")
    require(
        float(hydrogen_exact_scf["selected_parameters"]["s0"])
        == float(selected["s0"]),
        "H-exact SCF used a different calibrated s0",
    )
    require(
        int(hydrogen_exact_scf["selected_parameters"]["p"])
        == int(selected["p"]),
        "H-exact SCF used a different fixed p",
    )

    hydrogenic = rows("analytic/hydrogenic_ions.csv")
    hydrogenic_metadata = load_json("analytic/hydrogenic_ions.json")
    analytic_metadata = load_json("analytic/analytic_metadata.json")
    require(
        len(hydrogenic) == 3 * len(hydrogenic_input["systems"]),
        "hydrogenic table does not contain every configured ion/method pair",
    )
    require(
        hydrogenic_metadata["input_settings"] == hydrogenic_input,
        "hydrogenic results do not match the current input deck",
    )
    require(
        analytic_metadata["input_settings"] == analytic_input,
        "analytic results do not match the current input deck",
    )
    require(
        analytic_metadata["functional_parameters"] == parameters,
        "analytic results do not match functional/parameters.json",
    )
    regularized_error = max(
        float(row["absolute_error_mHa"])
        for row in hydrogenic
        if row["method"] == "regularized_Fmix"
    )
    require(regularized_error < 0.01,
            "regularized hydrogenic error exceeds 0.01 mHa")
    require(
        all(
            float(row["absolute_error_mHa"]) < 1.0e-5
            for row in hydrogenic
            if row["method"] == "unregularized_H_exact"
        ),
        "implemented unregularized H-exact hydrogenic identity failed",
    )

    derivative = rows("analytic/derivative_checks.csv")
    for row in derivative:
        check = row["check_type"]
        relative = float(row["rel_error"])
        absolute = float(row["abs_error"])
        if check.startswith("dFmix"):
            require(relative < 2.0e-5, f"{check} failed at {row['point']}")
        elif check == "descriptor_clip_derivative":
            require(absolute < 1.0e-12, f"{check} failed at {row['point']}")
        elif check == "d_density_window/dlogn":
            require(absolute < 1.0e-6, f"{check} failed at {row['point']}")
        elif check.startswith("implemented_v"):
            require(relative < 5.0e-9, f"{check} failed at {row['point']}")
        elif check.startswith("implemented_nonzero_alpha"):
            require(relative < 5.0e-7, f"{check} failed at {row['point']}")

    h_curve = np.genfromtxt(
        RESULTS / "analytic" / "hydrogen_regularization.csv",
        delimiter=",",
        names=True,
    )
    radius = h_curve["r_bohr"]
    regularized = h_curve["minus_r_vx_implemented"]
    exact = h_curve["minus_r_vx_exact"]
    bare = h_curve["minus_r_vx_bare_GP93"]
    unregularized = h_curve["minus_r_vx_unregularized_H_exact"]
    intermediate_start, intermediate_stop = (
        float(value)
        for value in analytic_input["regularized_intermediate_range_bohr"]
    )
    bare_start, bare_stop = (
        float(value)
        for value in analytic_input["bare_potential_error_range_bohr"]
    )
    transition_start, transition_stop = (
        float(value)
        for value in analytic_input[
            "regularization_transition_check_range_bohr"
        ]
    )
    through_six = (
        (radius >= intermediate_start) & (radius <= intermediate_stop)
    )
    transition = (radius >= transition_start) & (radius <= transition_stop)
    require(
        float(np.max(np.abs(bare[transition | through_six] - exact[transition | through_six])))
        < 5.0e-5,
        "implemented bare GP93 potential does not reproduce exact H",
    )
    direct_region = (radius >= bare_start) & (radius <= bare_stop)
    require(
        float(np.max(np.abs(unregularized[direct_region] - exact[direct_region])))
        < 5.0e-5,
        "released unregularized H-exact local potential does not reproduce exact H",
    )
    require(
        float(np.max(np.abs(regularized[through_six] - exact[through_six])))
        < 1.0e-3,
        "regularized H potential is inaccurate over the configured "
        "intermediate-range window",
    )
    require(float(np.max(regularized[transition])) < 1.25,
            "density-window transition has an excessive potential peak")
    require(float(np.min(regularized[transition])) > -0.02,
            "density-window transition has an excessive attractive artifact")

    h2 = load_json("h2/h2_metadata.json")
    h2_grid = load_json("h2/h2_grid_validation.json")
    potential = load_json("potentials/radial_potentials_metadata.json")
    require(
        h2["input_settings"] == h2_input,
        "H2 curve does not match the current input deck",
    )
    require(
        h2_grid["input_settings"] == h2_input["grid_validation"],
        "H2 grid check does not match the current input deck",
    )
    require(
        potential["input_settings"] == potentials_input,
        "radial-potential results do not match the current input deck",
    )
    selected_pair = {
        "s0": float(selected["s0"]),
        "p": int(selected["p"]),
    }
    effective_parameter_records = {
        "analytic": analytic_metadata["selected_parameters"],
        "hydrogenic": hydrogenic_metadata["selected_parameters"],
        "unregularized H SCF": hydrogen_exact_scf["selected_parameters"],
        "H2 curve": h2["selected_parameters"],
        "H2 grid check": h2_grid["selected_parameters"],
        "radial potentials": potential["selected_parameters"],
        "atomic": {
            "s0": atomic_metadata["settings"]["fmix_s0"],
            "p": atomic_metadata["settings"]["fmix_p"],
        },
    }
    for stage, record in effective_parameter_records.items():
        actual = {"s0": float(record["s0"]), "p": int(record["p"])}
        require(
            actual == selected_pair,
            f"{stage} used parameters that differ from the calibration",
        )
    require(bool(h2["all_points_converged"]), "H2 curve has an unconverged point")
    h2_curve = rows("h2/h2_curve.csv")
    require(
        len(h2_curve) == len(h2_input["bond_lengths_angstrom"]),
        "H2 curve does not contain every configured bond length",
    )
    h2_minima = rows("h2/h2_minima.csv")
    require(
        {row["method"] for row in h2_minima}
        == {"implemented_Fmix_LYP", "FCI"},
        "H2 minima CSV does not contain exactly the two reported methods",
    )
    h2_radius = np.asarray([float(row["R_A"]) for row in h2_curve])
    spline_grid = np.linspace(
        float(h2_radius.min()),
        float(h2_radius.max()),
        int(h2_input["spline_search_points"]),
    )
    energy_columns = {
        "implemented_Fmix_LYP": "Erel_fmix_eV",
        "FCI": "Erel_fci_eV",
    }
    for row in h2_minima:
        metadata_minimum = h2["spline_minima"][row["method"]]
        spline = CubicSpline(
            h2_radius,
            np.asarray(
                [float(point[energy_columns[row["method"]]]) for point in h2_curve]
            ),
        )
        spline_values = spline(spline_grid)
        minimum_index = int(np.argmin(spline_values))
        reconstructed_r = float(spline_grid[minimum_index])
        reconstructed_depth = float(-spline_values[minimum_index])
        require(
            math.isclose(
                float(row["R_e_angstrom"]),
                reconstructed_r,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"H2 equilibrium distance is not reproduced from the curve for "
            f"{row['method']}",
        )
        require(
            math.isclose(
                float(row["D_e_eV"]),
                reconstructed_depth,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"H2 well depth is not reproduced from the curve for "
            f"{row['method']}",
        )
        require(
            math.isclose(
                float(row["R_e_angstrom"]),
                float(metadata_minimum["R_A"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                float(row["D_e_eV"]),
                float(metadata_minimum["depth_eV"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"H2 minima CSV and metadata disagree for {row['method']}",
        )
    h2_grid_rows = rows("h2/h2_grid_validation.csv")
    require(
        len(h2_grid_rows)
        == len(h2_input["grid_validation"]["grid_levels"])
        * len(h2_input["grid_validation"]["bond_lengths_angstrom"]),
        "H2 grid check does not contain every configured level/distance pair",
    )
    require(float(h2["maximum_spin_density_matrix_norm"]) < 1.0e-8,
            "H2 solution is not spin symmetric")
    require(float(h2_grid["maximum_absolute_difference_eV"]) < 0.005,
            "H2 grid sensitivity exceeds 5 meV")
    require(bool(h2_grid["all_points_converged"]),
            "H2 grid-validation set has an unconverged point")
    require(
        float(h2_grid["maximum_spin_density_matrix_norm"])
        < float(h2_input["spin_density_norm_tolerance"]),
        "H2 grid-validation set is not spin symmetric",
    )

    require(
        potential["systems"]["He"]["mean_abs_error_minus_r_vx"]["regularized_Fmix"]
        < potential["systems"]["He"]["mean_abs_error_minus_r_vx"]["PBE_X"],
        "He fixed-density potential is not improved",
    )
    require(
        potential["systems"]["Na"]["mean_abs_error_minus_r_vx"]["regularized_Fmix"]
        < potential["systems"]["Na"]["mean_abs_error_minus_r_vx"]["PBE_X"],
        "Na fixed-density potential is not improved",
    )

    expected_figures = (
        "fig_gp93_building_block.png",
        "fig_hydrogen_regularization.png",
        "fig_calibration.png",
        "fig_atomic_systematics.png",
        "fig_he_levels.png",
        "fig_he_comparison.png",
        "fig_h2.png",
        "fig_radial_potentials.png",
    )
    for name in expected_figures:
        path = ROOT / "figures" / name
        require(path.is_file() and path.stat().st_size > 10_000,
                f"missing or empty figure: {name}")

    print("All canonical numerical-result and figure checks passed.")


if __name__ == "__main__":
    main()
