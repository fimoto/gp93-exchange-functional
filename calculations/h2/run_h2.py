#!/usr/bin/env python3
"""Recompute the H2 binding-region figure.

The proposed functional is evaluated exactly as implemented in
``mgga_fmix.py``, including the alpha switch, broad log-density window,
and descriptor clipping.
The molecular calculation uses UKS with explicitly identical alpha/beta
initial densities and verifies the final spin-density norm.  No
broken-symmetry claim is made.  Both curves are referenced to their own
same-basis two-H-atom thresholds.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "functional"
OUTPUT_DIR = ROOT / "results" / "h2"
INPUT_PATH = ROOT / "inputs" / "h2.json"
INPUT = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
if int(INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/h2.json schema_version")
REFERENCE_PATH = ROOT / INPUT["experimental_reference_file"]
if not (SOURCE / "mgga_fmix.py").exists():
    raise FileNotFoundError(f"Functional source not found under {SOURCE}")
sys.path.insert(0, str(SOURCE))

from pyscf import dft, fci, gto, lib, scf  # noqa: E402

import mgga_fmix as G  # noqa: E402


def _sandbox_memory_workaround() -> None:
    """Permit execution in containers with a restricted /proc mount."""

    try:
        Path("/proc/self/statm").read_text(encoding="ascii")
    except OSError:
        lib.current_memory = lambda *args, **kwargs: (0, 0)


_sandbox_memory_workaround()
lib.num_threads(1)

HARTREE_TO_EV = 27.211386245988
BASIS = INPUT["basis"]
S0 = float(G.PARAMETERS["selected_s0"])
P_SWITCH = int(G.PARAMETERS["selected_p"])
GRID_LEVEL = int(INPUT["production_grid_level"])
FMIX_SCF = INPUT["fmix_scf"]
REFERENCE_SCF = INPUT["reference_scf"]
FCI_INPUT = INPUT["fci"]


def load_experimental_reference() -> dict[str, str]:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row["system"] == "H2"
            and row["property"] == "equilibrium_bond_length"
        ]
    if len(matches) != 1:
        raise RuntimeError("Expected one H2 equilibrium-bond-length reference")
    return matches[0]


H2_REFERENCE = load_experimental_reference()
H2_EXPERIMENTAL_RE_A = float(H2_REFERENCE["value"])

R_VALUES = np.array(INPUT["bond_lengths_angstrom"], dtype=float)


def make_fmix_mf(mol: gto.Mole, unrestricted: bool):
    cls = dft.UKS if unrestricted else dft.RKS
    mf = cls(mol)
    mf.xc = ""
    mf.define_xc_(G.eval_xc, xctype="MGGA")
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = float(FMIX_SCF["conv_tol"])
    mf.max_cycle = int(FMIX_SCF["initial_max_cycle"])
    return mf


def atomic_thresholds() -> tuple[float, float, bool]:
    mol = gto.M(
        atom="H 0 0 0",
        basis=BASIS,
        spin=1,
        unit=INPUT["coordinate_unit"],
        verbose=0,
    )
    uhf = scf.UHF(mol)
    uhf.conv_tol = float(REFERENCE_SCF["conv_tol"])
    uhf.max_cycle = int(REFERENCE_SCF["max_cycle"])
    uhf.kernel()

    mf = make_fmix_mf(mol, unrestricted=True)
    mf.damp = float(FMIX_SCF["stabilized_stage"]["damping"])
    mf.level_shift = float(
        FMIX_SCF["stabilized_stage"]["level_shift_ha"]
    )
    mf.kernel(dm0=uhf.make_rdm1())
    mf.damp = float(FMIX_SCF["final_stage"]["damping"])
    mf.level_shift = float(FMIX_SCF["final_stage"]["level_shift_ha"])
    mf.max_cycle = int(FMIX_SCF["final_stage"]["max_cycle"])
    mf.kernel(dm0=mf.make_rdm1())
    return float(mf.e_tot), float(uhf.e_tot), bool(mf.converged and uhf.converged)


def molecular_point(r_angstrom: float) -> dict[str, float | bool]:
    mol = gto.M(
        atom=f"H 0 0 {-r_angstrom / 2}; H 0 0 {r_angstrom / 2}",
        basis=BASIS,
        spin=0,
        unit=INPUT["coordinate_unit"],
        verbose=0,
    )

    rhf = scf.RHF(mol)
    rhf.conv_tol = float(REFERENCE_SCF["conv_tol"])
    rhf.max_cycle = int(REFERENCE_SCF["max_cycle"])
    rhf.kernel()

    # The custom MGGA's PySCF-2.13 RKS adapter returns spin-resolved arrays
    # that are incompatible with nr_rks.  Use UKS with an explicitly
    # spin-symmetric initial density and verify that it remains symmetric.
    dm_rhf = rhf.make_rdm1()
    dm_symmetric = np.stack((0.5 * dm_rhf, 0.5 * dm_rhf))
    mf = make_fmix_mf(mol, unrestricted=True)
    mf.damp = float(FMIX_SCF["stabilized_stage"]["damping"])
    mf.level_shift = float(
        FMIX_SCF["stabilized_stage"]["level_shift_ha"]
    )
    mf.kernel(dm0=dm_symmetric)
    mf.damp = float(FMIX_SCF["final_stage"]["damping"])
    mf.level_shift = float(FMIX_SCF["final_stage"]["level_shift_ha"])
    mf.max_cycle = int(FMIX_SCF["final_stage"]["max_cycle"])
    mf.kernel(dm0=mf.make_rdm1())

    solver = fci.FCI(rhf)
    solver.conv_tol = float(FCI_INPUT["conv_tol"])
    e_fci, _ = solver.kernel()
    dm_final = np.asarray(mf.make_rdm1())
    spin_density_norm = float(np.linalg.norm(dm_final[0] - dm_final[1]))
    return {
        "R_A": float(r_angstrom),
        "E_fmix_lyp_Ha": float(mf.e_tot),
        "fmix_converged": bool(mf.converged),
        "spin_density_matrix_norm": spin_density_norm,
        "E_fci_Ha": float(e_fci),
        "rhf_converged": bool(rhf.converged),
        "fci_converged": bool(getattr(solver, "converged", True)),
    }


def spline_minimum(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    sp = CubicSpline(x, y)
    grid = np.linspace(
        float(x.min()),
        float(x.max()),
        int(INPUT["spline_search_points"]),
    )
    values = sp(grid)
    i = int(np.argmin(values))
    return float(grid[i]), float(values[i])


def main() -> None:
    outdir = OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("h2_curve.csv", "h2_minima.csv", "h2_metadata.json"):
        (outdir / name).unlink(missing_ok=True)
    G._Fmix_override(S0, P_SWITCH)
    G.CORR = INPUT["correlation_libxc_suffix"]

    e_h_fmix, e_h_basis, atom_converged = atomic_thresholds()
    rows = []
    for r in R_VALUES:
        row = molecular_point(float(r))
        row["Erel_fmix_eV"] = (
            float(row["E_fmix_lyp_Ha"]) - 2.0 * e_h_fmix
        ) * HARTREE_TO_EV
        row["Erel_fci_eV"] = (
            float(row["E_fci_Ha"]) - 2.0 * e_h_basis
        ) * HARTREE_TO_EV
        rows.append(row)
        print(
            f"R={r:5.2f} A  Fmix={row['Erel_fmix_eV']: .7f} eV "
            f"(conv={row['fmix_converged']})  "
            f"FCI={row['Erel_fci_eV']: .7f} eV",
            flush=True,
        )

    if not atom_converged:
        raise RuntimeError("Atomic threshold calculation did not converge")
    if not all(
        bool(row["fmix_converged"])
        and bool(row["rhf_converged"])
        and bool(row["fci_converged"])
        for row in rows
    ):
        raise RuntimeError("At least one molecular point did not converge")
    if (
        max(float(row["spin_density_matrix_norm"]) for row in rows)
        > float(INPUT["spin_density_norm_tolerance"])
    ):
        raise RuntimeError("The nominally spin-symmetric H2 solution broke symmetry")

    csv_path = outdir / "h2_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    r = np.array([float(row["R_A"]) for row in rows])
    e_fmix = np.array([float(row["Erel_fmix_eV"]) for row in rows])
    e_fci = np.array([float(row["Erel_fci_eV"]) for row in rows])
    re_fmix, emin_fmix = spline_minimum(r, e_fmix)
    re_fci, emin_fci = spline_minimum(r, e_fci)
    minima_rows = [
        {
            "method": "implemented_Fmix_LYP",
            "R_e_angstrom": re_fmix,
            "D_e_eV": -emin_fmix,
        },
        {
            "method": "FCI",
            "R_e_angstrom": re_fci,
            "D_e_eV": -emin_fci,
        },
    ]
    with (outdir / "h2_minima.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "R_e_angstrom", "D_e_eV"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(minima_rows)

    metadata = {
        "input_file": str(INPUT_PATH.relative_to(ROOT)),
        "input_settings": INPUT,
        "selected_parameters": {"s0": S0, "p": P_SWITCH},
        "functional": f"implemented Fmix(s0={S0},p={P_SWITCH})+LYP",
        "implementation_includes": [
            "alpha switch",
            "broad log-density window Xn",
            "s and z clipping",
        ],
        "molecular_spin_treatment": (
            "UKS with identical alpha/beta initialization; final spin norm saved"
        ),
        "basis": BASIS,
        "grid_level": GRID_LEVEL,
        "plot_interpolation_points": int(INPUT["plot_interpolation_points"]),
        "experimental_reference": H2_REFERENCE,
        "dissociation_thresholds_Ha": {
            "implemented_Fmix_LYP_H_atom": e_h_fmix,
            "same_basis_one_electron_H_atom_for_FCI": e_h_basis,
        },
        "spline_minima": {
            "implemented_Fmix_LYP": {"R_A": re_fmix, "depth_eV": -emin_fmix},
            "FCI": {"R_A": re_fci, "depth_eV": -emin_fci},
        },
        "all_points_converged": True,
        "maximum_spin_density_matrix_norm": max(
            float(row["spin_density_matrix_norm"]) for row in rows
        ),
    }
    (outdir / "h2_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2), flush=True)
    print(
        "Plot separately with: python calculations/plot_results.py fig-h2",
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
