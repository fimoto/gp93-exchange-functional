#!/usr/bin/env python3
"""Radial spectra of H and He for GP93R ablation variants and CAP exchange.

GP93R changes two things that matter for one orbital per spin: the curvature
of the factor below s1 = 0.5 (F_R) and the density window (1e-6 .. 1e-4 instead
of 1e-12 .. 1e-6).  This stage switches them on one at a time, so that the
change of the bound levels between GP93-x and GP93R-x can be assigned to
either change.  It also runs CAP exchange, a GGA whose potential decays as
-1/r, as a matched comparison on the same radial grid and solver.

The SCF, solver and grid are those of run_radial.py and inputs/radial.json.

Outputs (results/radial/):
  radial_variants_summary.csv   -eps_HOMO, SCF status and sensitivities
  radial_variants_levels.csv    bound virtual levels
  radial_variants_metadata.json settings and the GGA self-check
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "functional"))
sys.path.insert(0, str(ROOT / "calculations" / "radial"))

import mgga_fmix as M  # noqa: E402
import gp93r  # noqa: E402
import run_radial as R  # noqa: E402

VARIANTS = json.loads((ROOT / "inputs" / "radial_variants.json").read_text(encoding="utf-8"))
if int(VARIANTS.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/radial_variants.json schema_version")
OUTPUT_DIR = ROOT / "results" / "radial"
OUTPUTS = ("radial_variants_summary.csv", "radial_variants_levels.csv",
           "radial_variants_metadata.json")
AX = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)


def f_cap(s):
    """CAP enhancement factor and its derivative (mu = mu_PBE)."""

    mu = M.MU_PBE
    c = abs(AX) * mu / M.C1
    lg = np.log1p(s)
    den = 1.0 + c * lg
    f = 1.0 + mu * s * lg / den
    df = mu * ((lg + s / (1.0 + s)) * den - s * lg * c / (1.0 + s)) / den**2
    return f, df


def f_pbe(s):
    return M._pbe_enhancement(s), M._pbe_derivative(s)


TAIL_FIT = (1e-16, 1e-20)


def exponential_tail(log_n, r):
    """Replace the density below 1e-20 by its exponential continuation.

    The tail of a numerically computed orbital density is noise below about
    1e-30, and a GGA whose potential depends on grad ln n (as CAP does at large
    s) amplifies that noise.  ln n is fitted linearly in r where the density
    lies between 1e-16 and 1e-20 outside its maximum, and continued linearly
    beyond.
    """

    hi, lo = np.log(TAIL_FIT[0]), np.log(TAIL_FIT[1])
    peak = int(np.argmax(log_n))
    outside = np.arange(len(r)) > peak
    fit = outside & (log_n < hi) & (log_n > lo)
    if fit.sum() < 5:
        return log_n
    slope, intercept = np.polyfit(r[fit], log_n[fit], 1)
    start = np.nonzero(outside & (log_n <= lo))[0]
    if len(start) == 0:
        return log_n
    out = log_n.copy()
    out[start[0]:] = slope * r[start[0]:] + intercept
    return out


def gga_potential(n_sigma, factor, x, r):
    """Exchange potential of one spin channel for a GGA factor F(s).

    With m = 2 n_sigma and s = |grad m| / (2 C1 m^(4/3)), the energy density of
    the channel is e = A_x m^(4/3) F(s) / 2, so that
      v_rho   = (4/3) A_x m^(1/3) (F - s F'),
      v_sigma = A_x m^(4/3) F' s / (4 sigma),  sigma = |grad n_sigma|^2,
    and v_x = v_rho - r^-2 d/dr (2 r^2 v_sigma dn_sigma/dr).
    With g = d ln n_sigma / d ln r these reduce to closed forms that stay finite
    in the far tail:
      s = |g| exp(-ln n_sigma / 3) / (2^(4/3) C1 r),   m^(1/3) s = |g| / (2 C1 r),
      v_sigma dn_sigma/dr = A_x F' sign(g) / (4 C1).
    """

    log_n = exponential_tail(np.log(np.maximum(n_sigma, 1e-300)), r)
    g = CubicSpline(x, log_n)(x, 1)
    s = np.abs(g) * np.exp(-log_n / 3.0) / (2.0 ** (4.0 / 3.0) * M.C1 * r)
    f, df = factor(s)
    m13 = 2.0 ** (1.0 / 3.0) * np.exp(log_n / 3.0)
    vrho = (4.0 / 3.0) * AX * (m13 * f - np.abs(g) / (2.0 * M.C1 * r) * df)
    vsig_dn = AX * df * np.sign(g) / (4.0 * M.C1)
    flux = CubicSpline(x, r**2 * 2.0 * vsig_dn)
    return vrho - flux(x, 1) / r / r**2


_ORIGINAL = R.exchange_potential


def patched_exchange_potential(n_sigma, kind, x, r):
    if kind == "cap":
        return gga_potential(n_sigma, f_cap, x, r)
    return _ORIGINAL(n_sigma, kind, x, r)


class variant:
    """Set the canonical module to one GP93R ablation variant and restore it."""

    def __init__(self, kind):
        self.kind = kind

    def __enter__(self):
        self.saved = (M._FMIX_SPLINE, M.DENSITY_WINDOW_LOW, M.DENSITY_WINDOW_HIGH,
                      M.DENSITY_CUTOFF)
        if self.kind == "gp93_curvature_only":
            M._FMIX_SPLINE = CubicSpline(np.log(M._S_GRID), gp93r.make_fr_values())
        elif self.kind == "gp93_window_only":
            M.DENSITY_WINDOW_LOW = gp93r.WINDOW_LOW
            M.DENSITY_WINDOW_HIGH = gp93r.WINDOW_HIGH
            M.DENSITY_CUTOFF = gp93r.WINDOW_LOW
        return self

    def __exit__(self, *exc):
        (M._FMIX_SPLINE, M.DENSITY_WINDOW_LOW, M.DENSITY_WINDOW_HIGH,
         M.DENSITY_CUTOFF) = self.saved
        return False


def gga_self_check():
    settings = R.INPUT["radial_grid"]
    x, r, _dx = R.grid(settings["r_min_bohr"], settings["r_max_bohr"], settings["points"])
    n_sigma = np.exp(-2.0 * r) / np.pi
    reference = _ORIGINAL(n_sigma, "pbe", x, r)
    analytic = gga_potential(n_sigma, f_pbe, x, r)
    lo, hi = VARIANTS["gga_self_check"]["r_range_bohr"]
    mask = (r > lo) & (r < hi)
    return float(np.max(np.abs(analytic[mask] - reference[mask])))


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
    R.exchange_potential = patched_exchange_potential

    check = gga_self_check()
    print(f"GGA self-check (PBE, analytic vs canonical path): max |dv| = {check:.2e} Ha", flush=True)

    base = {"r_min": R.INPUT["radial_grid"]["r_min_bohr"],
            "r_max": R.INPUT["radial_grid"]["r_max_bohr"],
            "points": R.INPUT["radial_grid"]["points"]}
    grid_check = dict(base, r_max=R.INPUT["grid_check"]["r_max_bohr"],
                      points=R.INPUT["grid_check"]["points"])
    summary, levels = [], []
    for item in VARIANTS["variants"]:
        kind = item["kind"]
        for system in R.INPUT["systems"]:
            mix_main, mix_check = item["mixing"][system["name"]]

            def run(mixing, settings):
                if kind == "cap":
                    return R.scf(system, "cap", mixing, settings)
                with variant(kind):
                    return R.scf(system, "gp93", mixing, settings)

            main_run = run(mix_main, base)
            mixing_run = run(mix_check, base)
            grid_run = run(mix_main, grid_check)
            summary.append({
                "system": system["name"], "functional": item["name"],
                "minus_homo_ev": -main_run["homo"] * R.HARTREE_TO_EV,
                "homo_ha": main_run["homo"],
                "scf_cycles": main_run["cycles"],
                "potential_residual_converged": main_run["converged"],
                "final_potential_residual": main_run["final_residual"],
                "residual_location_bohr": main_run["residual_location_bohr"],
                "mixing": mix_main, "mixing_check": mix_check,
                "max_level_change_mixing_ha": R.max_level_change(main_run, mixing_run),
                "max_level_change_grid_ha": R.max_level_change(main_run, grid_run),
                "n_bound_levels": R.count(main_run, False),
                "n_bound_orbitals_one_spin": R.count(main_run, True),
            })
            for l, values in main_run["virtual"].items():
                for k, value in enumerate(values):
                    levels.append({
                        "system": system["name"], "functional": item["name"],
                        "l": l, "label": f"{l + (2 if l == 0 else 1) + k}{R.L_SYMBOL[l]}",
                        "energy_ha": value, "n_eff": 1.0 / np.sqrt(-2.0 * value),
                        "above_reported_threshold": value > float(R.INPUT["reported_threshold_ha"]),
                    })
            print(f"{system['name']:2s} {item['name']:13s} "
                  f"-eps_HOMO {summary[-1]['minus_homo_ev']:.4f} eV  "
                  f"bound {summary[-1]['n_bound_levels']}  "
                  f"dmix {summary[-1]['max_level_change_mixing_ha']:.1e}  "
                  f"dgrid {summary[-1]['max_level_change_grid_ha']:.1e}", flush=True)
    write_csv("radial_variants_summary.csv", summary)
    write_csv("radial_variants_levels.csv", levels)
    (OUTPUT_DIR / "radial_variants_metadata.json").write_text(json.dumps({
        "input_file": "inputs/radial_variants.json", "input_settings": VARIANTS,
        "radial_settings_file": "inputs/radial.json",
        "selected_parameters": {"s0": M._CURRENT_S0, "p": M._CURRENT_P},
        "gp93r": {"s1": gp93r.S1_DEFAULT, "D": gp93r.D_DEFAULT,
                  "window": [gp93r.WINDOW_LOW, gp93r.WINDOW_HIGH]},
        "gga_self_check_max_abs_dv_ha": check,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
