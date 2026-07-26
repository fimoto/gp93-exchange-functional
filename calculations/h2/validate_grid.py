#!/usr/bin/env python3
"""Small quadrature-grid check for the H2 curve."""

from __future__ import annotations

import csv
import json

import run_h2 as H


def main() -> None:
    H.G._Fmix_override(H.S0, H.P_SWITCH)
    H.G.CORR = H.INPUT["correlation_libxc_suffix"]
    validation = H.INPUT["grid_validation"]
    levels = tuple(int(value) for value in validation["grid_levels"])
    if len(levels) != 2:
        raise ValueError(
            "grid_validation.grid_levels must contain exactly two levels"
        )
    distances = tuple(
        float(value) for value in validation["bond_lengths_angstrom"]
    )
    output = []
    for level in levels:
        H.GRID_LEVEL = level
        e_h_fmix, _, atom_ok = H.atomic_thresholds()
        if not atom_ok:
            raise RuntimeError(f"H threshold failed at grid level {level}")
        for r in distances:
            row = H.molecular_point(r)
            convergence = {
                "fmix_converged": bool(row["fmix_converged"]),
                "rhf_converged": bool(row["rhf_converged"]),
                "fci_converged": bool(row["fci_converged"]),
            }
            if not all(convergence.values()):
                failed = ", ".join(
                    name
                    for name, converged in convergence.items()
                    if not converged
                )
                raise RuntimeError(
                    f"H2 failed at grid={level}, R={r}: {failed}"
                )
            spin_norm = float(row["spin_density_matrix_norm"])
            if spin_norm > float(H.INPUT["spin_density_norm_tolerance"]):
                raise RuntimeError(
                    f"H2 spin-density norm failed at grid={level}, R={r}: "
                    f"{spin_norm}"
                )
            output.append(
                {
                    "grid_level": level,
                    "R_A": r,
                    "Erel_fmix_eV": (
                        float(row["E_fmix_lyp_Ha"]) - 2.0 * e_h_fmix
                    )
                    * H.HARTREE_TO_EV,
                    "spin_density_matrix_norm": spin_norm,
                    **convergence,
                }
            )

    outdir = H.OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("h2_grid_validation.csv", "h2_grid_validation.json"):
        (outdir / name).unlink(missing_ok=True)
    with (outdir / "h2_grid_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output)

    differences = {}
    low_level, high_level = levels
    for r in distances:
        e_low = next(
            float(x["Erel_fmix_eV"])
            for x in output
            if x["grid_level"] == low_level and x["R_A"] == r
        )
        e_high = next(
            float(x["Erel_fmix_eV"])
            for x in output
            if x["grid_level"] == high_level and x["R_A"] == r
        )
        differences[str(r)] = e_high - e_low
    summary = {
        "input_file": str(H.INPUT_PATH.relative_to(H.ROOT)),
        "input_settings": validation,
        "selected_parameters": {"s0": H.S0, "p": H.P_SWITCH},
        "grid_levels": list(levels),
        "R_A": list(distances),
        "all_points_converged": all(
            row["fmix_converged"]
            and row["rhf_converged"]
            and row["fci_converged"]
            for row in output
        ),
        "maximum_spin_density_matrix_norm": max(
            float(row["spin_density_matrix_norm"]) for row in output
        ),
        f"level{high_level}_minus_level{low_level}_eV": differences,
        "maximum_absolute_difference_eV": max(
            abs(value) for value in differences.values()
        ),
    }
    (outdir / "h2_grid_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
