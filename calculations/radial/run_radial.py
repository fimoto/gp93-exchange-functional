#!/usr/bin/env python3
"""Basis-free radial exchange-only Kohn-Sham spectra of H and He.

Every ground-state spin channel of H (spin-polarized) and He (closed shell) is
a single orbital, so alpha = 0, G(alpha) = 1 and v_tau = 0: on these densities
the GP93 meta-GGA reduces to a GGA and the Kohn-Sham equations are radial.
The exchange potential is assembled from the canonical channel routine,

    v_x = v_rho - r^-2 d/dr [ 2 r^2 v_sigma dn_sigma/dr ],

and bound levels are obtained with a sparse shift-and-invert generalized
eigensolver on a logarithmic grid, which stays accurate for shallow
Rydberg-like levels where a Gaussian basis does not.

Outputs (results/radial/):
  coulomb_test.csv     solver test on the bare -1/r potential
  radial_summary.csv   one row per system/functional: -eps_HOMO, SCF status,
                       mixing and grid sensitivity of all bound levels
  radial_levels.csv    one row per bound virtual level
  radial_metadata.json settings and definitions
"""

from __future__ import annotations

import csv
import json
import sys
from math import pi
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "functional"))

import mgga_fmix as M  # noqa: E402
import gp93r  # noqa: E402

INPUT = json.loads((ROOT / "inputs" / "radial.json").read_text(encoding="utf-8"))
if int(INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/radial.json schema_version")
OUTPUT_DIR = ROOT / "results" / "radial"
HARTREE_TO_EV = 27.211386245988
OUTPUTS = ("coulomb_test.csv", "radial_summary.csv", "radial_levels.csv",
           "radial_metadata.json")
L_SYMBOL = "spdfg"


def grid(r_min: float, r_max: float, points: int):
    x = np.linspace(np.log(r_min), np.log(r_max), points)
    return x, np.exp(x), x[1] - x[0]


def solve(potential, l, x, r, dx, targets):
    """Levels of -1/2 u'' + [l(l+1)/2r^2 + V] u = e u with u = r^(1/2) phi.

    With r = e^x the equation becomes the symmetric generalized problem
    -1/2 (phi'' - phi/4) + [l(l+1)/2 + r^2 V] phi = e r^2 phi.
    """

    n = len(x)
    d = 1.0 / dx**2
    a = sparse.diags(
        [0.5 * (2.0 * d + 0.25) + l * (l + 1) / 2.0 + r**2 * potential,
         -0.5 * d * np.ones(n - 1), -0.5 * d * np.ones(n - 1)],
        [0, 1, -1], format="csc")
    b = sparse.diags(r**2, 0, format="csc")
    found = []
    for sigma, k in targets:
        w, v = eigsh(a, k=k, M=b, sigma=sigma, which="LM")
        found += list(zip(w, v.T))
    found.sort(key=lambda t: t[0])
    energies, vectors = [], []
    for e, vec in found:
        if energies and abs(e - energies[-1]) < 1e-9:
            continue
        energies.append(e)
        vectors.append(vec)
    u = np.array(vectors).T * np.sqrt(r)[:, None]
    u /= np.sqrt(np.trapezoid(u**2 * r[:, None], x, axis=0))
    return np.array(energies), u


def exchange_potential(n_sigma, kind, x, r):
    """Exchange potential of one spin channel with spin density n_sigma."""

    spline = CubicSpline(x, np.log(np.maximum(n_sigma, 1e-300)))
    dn = n_sigma * spline(x, 1) / r
    sigma = dn * dn
    tau = sigma / (8.0 * np.maximum(n_sigma, 1e-300))
    if kind == "pbe":
        # alpha >> 1 closes the gate exactly: F = F_PBE.
        tau = tau + 1e3 * M.CF * np.maximum(2.0 * n_sigma, 1e-300) ** (5 / 3) / 2
    _e, vrho, vsigma, _vtau = M._channel(n_sigma, sigma, tau)
    flux = CubicSpline(x, r**2 * 2.0 * vsigma * dn)
    return vrho - flux(x, 1) / r / r**2


def hartree(n, x, r):
    inner = cumulative_trapezoid(4 * pi * r**3 * n, x, initial=0.0)
    outer = cumulative_trapezoid(4 * pi * r**2 * n, x, initial=0.0)
    return inner / r + (outer[-1] - outer)


def scf(system, kind, mixing, grid_settings):
    x, r, dx = grid(**grid_settings)
    z = system["nuclear_charge"]
    closed = system["spin"] == "closed"
    u0 = 2.0 * z**1.5 * r * np.exp(-z * r)
    n_sigma = u0**2 / (4 * pi * r**2)
    potential = None
    settings = INPUT["scf"]
    residual = np.inf
    for cycle in range(1, int(settings["max_cycle"]) + 1):
        total = 2.0 * n_sigma if closed else n_sigma
        v_out = -z / r + hartree(total, x, r) + exchange_potential(n_sigma, kind, x, r)
        if potential is None:
            potential = v_out
        residual = float(np.max(np.abs((v_out - potential) * r)))
        potential = (1.0 - mixing) * potential + mixing * v_out
        w, u = solve(potential, 0, x, r, dx,
                     ((float(INPUT["ground_state_shift_ha"]), 1),))
        n_sigma = u[:, 0] ** 2 / (4 * pi * r**2)
        if residual < float(settings["potential_residual_tolerance"]) and cycle > 3:
            break
    # Location of the largest remaining residual (diagnostic only).
    total = 2.0 * n_sigma if closed else n_sigma
    v_out = -z / r + hartree(total, x, r) + exchange_potential(n_sigma, kind, x, r)
    worst = int(np.argmax(np.abs((v_out - potential) * r)))
    targets = [tuple(t) for t in INPUT["shift_invert_targets_ha"]]
    levels = {}
    for l in INPUT["angular_momenta"]:
        e, _u = solve(potential, l, x, r, dx, targets)
        levels[l] = [float(v) for v in e[: int(INPUT["levels_per_l"])]]
    homo = levels[0][0]
    virtual = {l: [v for v in (levels[l][1:] if l == 0 else levels[l])
                   if v < float(INPUT["bound_threshold_ha"])]
               for l in levels}
    return {
        "homo": homo, "virtual": virtual, "cycles": cycle,
        "converged": residual < float(settings["potential_residual_tolerance"]),
        "final_residual": residual, "residual_location_bohr": float(r[worst]),
    }


def max_level_change(a, b):
    change = abs(a["homo"] - b["homo"])
    for l in a["virtual"]:
        for x, y in zip(a["virtual"][l], b["virtual"][l]):
            if min(x, y) < float(INPUT["reported_threshold_ha"]):
                change = max(change, abs(x - y))
    return change


def count(run, orbitals):
    """Bound virtual levels (or orbitals of one spin) below the reported threshold."""

    threshold = float(INPUT["reported_threshold_ha"])
    return sum((2 * l + 1 if orbitals else 1) * sum(1 for v in values if v < threshold)
               for l, values in run["virtual"].items())


def coulomb_test():
    settings = INPUT["radial_grid"]
    x, r, dx = grid(settings["r_min_bohr"], settings["r_max_bohr"], settings["points"])
    targets = [tuple(t) for t in INPUT["shift_invert_targets_ha"]]
    rows = []
    for l in INPUT["coulomb_test"]["angular_momenta"]:
        e, _u = solve(-1.0 / r, l, x, r, dx, targets)
        for k, value in enumerate(e[: int(INPUT["coulomb_test"]["max_n"]) - l]):
            n = l + 1 + k
            rows.append({"n": n, "l": l, "energy_ha": value,
                         "exact_ha": -0.5 / n**2,
                         "abs_error_ha": abs(value + 0.5 / n**2)})
    return rows


def write_csv(name, rows):
    with (OUTPUT_DIR / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    M.CORR = ""
    M.set_fmix_parameters()

    coulomb = coulomb_test()
    write_csv("coulomb_test.csv", coulomb)
    print(f"Coulomb test: max error {max(r['abs_error_ha'] for r in coulomb):.2e} Ha")

    base = {"r_min": INPUT["radial_grid"]["r_min_bohr"],
            "r_max": INPUT["radial_grid"]["r_max_bohr"],
            "points": INPUT["radial_grid"]["points"]}
    check = dict(base, r_max=INPUT["grid_check"]["r_max_bohr"],
                 points=INPUT["grid_check"]["points"])
    summary, levels = [], []
    for functional in INPUT["functionals"]:
        kind = functional["kind"]
        for system in INPUT["systems"]:
            mix_main, mix_check = functional["mixing"][system["name"]]

            def run(mixing, settings):
                if kind == "gp93r":
                    with gp93r.use_gp93r():
                        return scf(system, "gp93", mixing, settings)
                return scf(system, kind, mixing, settings)

            main_run = run(mix_main, base)
            mixing_run = run(mix_check, base)
            grid_run = run(mix_main, check)
            summary.append({
                "system": system["name"], "functional": functional["name"],
                "minus_homo_ev": -main_run["homo"] * HARTREE_TO_EV,
                "homo_ha": main_run["homo"],
                "scf_cycles": main_run["cycles"],
                "potential_residual_converged": main_run["converged"],
                "final_potential_residual": main_run["final_residual"],
                "residual_location_bohr": main_run["residual_location_bohr"],
                "mixing": mix_main, "mixing_check": mix_check,
                "max_level_change_mixing_ha": max_level_change(main_run, mixing_run),
                "max_level_change_grid_ha": max_level_change(main_run, grid_run),
                "n_bound_levels": count(main_run, False),
                "n_bound_orbitals_one_spin": count(main_run, True),
                "n_marginal_levels": sum(len(v) for v in main_run["virtual"].values())
                - count(main_run, False),
            })
            for l, values in main_run["virtual"].items():
                for k, value in enumerate(values):
                    levels.append({
                        "system": system["name"], "functional": functional["name"],
                        "l": l, "label": f"{l + (2 if l == 0 else 1) + k}{L_SYMBOL[l]}",
                        "energy_ha": value, "n_eff": 1.0 / np.sqrt(-2.0 * value),
                        "above_reported_threshold": value > float(INPUT["reported_threshold_ha"]),
                    })
            print(f"{system['name']:2s} {functional['name']:8s} "
                  f"-eps_HOMO {summary[-1]['minus_homo_ev']:.4f} eV  "
                  f"bound {summary[-1]['n_bound_levels']}  "
                  f"dmix {summary[-1]['max_level_change_mixing_ha']:.1e}  "
                  f"dgrid {summary[-1]['max_level_change_grid_ha']:.1e}", flush=True)
    write_csv("radial_summary.csv", summary)
    write_csv("radial_levels.csv", levels)
    (OUTPUT_DIR / "radial_metadata.json").write_text(json.dumps({
        "input_file": "inputs/radial.json", "input_settings": INPUT,
        "selected_parameters": {"s0": M._CURRENT_S0, "p": M._CURRENT_P},
        "gp93r": {"s1": gp93r.S1_DEFAULT, "D": gp93r.D_DEFAULT,
                  "window": [gp93r.WINDOW_LOW, gp93r.WINDOW_HIGH]},
        "level_labels": "hydrogenic-like ordering label: the first virtual s level "
                        "is 2s, the first p level 2p, the first d level 3d",
        "counts": "n_bound_levels and n_bound_orbitals_one_spin count levels below "
                  "reported_threshold_ha; n_marginal_levels counts levels between "
                  "reported_threshold_ha and zero",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
