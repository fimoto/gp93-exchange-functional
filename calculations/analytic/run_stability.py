#!/usr/bin/env python3
"""Local stability thresholds of GGA exchange and the GP93R comparison data.

In a one-orbital-per-spin region the kinetic energy is the von Weizsaecker
functional, and the integrand g^2/(8 n_s) + e_x(n_s, g), g = |grad n_s|, is
rank-one convex (strict Legendre-Hadamard condition) iff

    m > m_c(s) = [ |A_x| max(F''(s), F'(s)/s, 0) / C1^2 ]^3,   m = 2 n_s.

Outputs (results/analytic/):
  stability_curves.csv    m_c(s) for PBE, GP93 (F_mix) and GP93R (F_R)
  stability_summary.csv   maxima, thresholds, the hydrogen-exactness bound,
                          and a finite-difference check of d2e/dg2
  gp93r_hydrogen_potential.csv  -r v_x on the exact H density (GP93, GP93R, PBE)
"""

from __future__ import annotations

import csv
import json
import sys
from math import pi
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "functional"))

import mgga_fmix as M  # noqa: E402
import gp93r  # noqa: E402

OUTPUT_DIR = ROOT / "results" / "analytic"
OUTPUTS = ("stability_curves.csv", "stability_summary.csv",
           "gp93r_hydrogen_potential.csv", "stability_metadata.json")
C1 = M.C1
ABS_AX = -M.A_X
S_H = (6.0 * pi) ** (-1.0 / 3.0)


def derivatives(spline, s):
    u = np.log(s)
    f = spline(u)
    f1 = spline(u, 1) / s
    f2 = (spline(u, 2) - spline(u, 1)) / s**2
    return f, f1, f2


def threshold(f1, f2, s):
    k = np.maximum(np.maximum(f2, f1 / s), 0.0)
    return (ABS_AX * k / C1**2) ** 3


def write_csv(name, rows):
    with (OUTPUT_DIR / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def h_potential(kind):
    """-r v_x on the exact spin-polarized H density (alpha = 0)."""

    x = np.linspace(np.log(1e-4), np.log(40.0), 6000)
    r = np.exp(x)
    n = np.exp(-2.0 * r) / pi
    dn = -2.0 * n
    sigma = dn * dn
    tau = sigma / (8.0 * n)
    if kind == "pbe":
        tau = tau + 1e3 * M.CF * (2.0 * n) ** (5 / 3) / 2
    _e, vrho, vsigma, _vt = M._channel(n, sigma, tau)
    flux = CubicSpline(x, r**2 * 2.0 * vsigma * dn)
    v = vrho - flux(x, 1) / r / r**2
    return r, -r * v


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    M.CORR = ""
    M.set_fmix_parameters()
    canonical = M._FMIX_SPLINE
    fr_spline = CubicSpline(np.log(M._S_GRID), gp93r.make_fr_values())

    s = np.logspace(-2, np.log10(3.0), 4000)
    f_pbe = M._pbe_enhancement(s)
    f1_pbe = M._pbe_derivative(s)
    f2_pbe = np.gradient(f1_pbe, s)
    _f, f1_c, f2_c = derivatives(canonical, s)
    _f, f1_r, f2_r = derivatives(fr_spline, s)
    curves = [{"s": a, "m_c_pbe": b, "m_c_gp93": c, "m_c_gp93r": d,
               "F_pbe": e, "F_gp93": float(canonical(np.log(a))),
               "F_gp93r": float(fr_spline(np.log(a)))}
              for a, b, c, d, e in zip(
                  s, threshold(f1_pbe, f2_pbe, s), threshold(f1_c, f2_c, s),
                  threshold(f1_r, f2_r, s), f_pbe)]
    write_csv("stability_curves.csv", curves)

    # Summary on a fine grid that excludes the numerical spike at the lower
    # end (z = -6, s = 0.0507) of the tabulated w(z).
    fine = np.linspace(0.06, 1.0, 20001)
    summary = []
    for label, spline in (("GP93", canonical), ("GP93R", fr_spline)):
        _f, f1, f2 = derivatives(spline, fine)
        i2, i1 = int(np.argmax(f2)), int(np.argmax(f1 / fine))
        summary.append({
            "functional": label, "max_F2": f2[i2], "s_at_max_F2": fine[i2],
            "max_F1_over_s": (f1 / fine)[i1], "s_at_max_F1_over_s": fine[i1],
            "m_c_longitudinal": (ABS_AX * f2[i2] / C1**2) ** 3,
            "m_c_transverse": (ABS_AX * (f1 / fine)[i1] / C1**2) ** 3,
        })
    pbe_m_c = (ABS_AX * 2.0 * M.MU_PBE / C1**2) ** 3
    summary.append({"functional": "PBE", "max_F2": 2.0 * M.MU_PBE, "s_at_max_F2": 0.0,
                    "max_F1_over_s": 2.0 * M.MU_PBE, "s_at_max_F1_over_s": 0.0,
                    "m_c_longitudinal": pbe_m_c, "m_c_transverse": pbe_m_c})
    write_csv("stability_summary.csv", summary)

    # Hydrogen-exactness bound: F(s_H) - 1 <= s_H^2/2 max(F'/s).
    c0 = float(M.PARAMETERS["gp93_c0"])
    f_sh = 8.0 * pi / 3.0 * S_H * c0
    bound = 2.0 * (f_sh - 1.0) / S_H**2
    # Finite-difference check of d2e/dg2 = A_x F'' / (2 C1^2 m^(4/3)).
    n_sigma, checks = 0.05, []
    m = 2.0 * n_sigma
    for s_value in (0.143, 0.3, 1.0):
        g = s_value * C1 * m ** (4 / 3)
        h = g * 1e-4

        def e_of(gg):
            return M._channel(np.array([n_sigma]), np.array([gg * gg]),
                              np.array([gg * gg / (8 * n_sigma)]))[0][0]

        fd = (e_of(g + h) - 2 * e_of(g) + e_of(g - h)) / h**2
        _f, _f1, f2 = derivatives(canonical, np.array([s_value]))
        analytic = M.A_X * f2[0] / (2 * C1**2 * m ** (4 / 3))
        checks.append(abs(fd - analytic) / abs(analytic))
    meta = {
        "s_H": S_H, "F_GP93_at_s_H": f_sh, "c0": c0,
        "hydrogen_exact_bound_max_F1_over_s": bound,
        "hydrogen_exact_bound_m_c": (ABS_AX * bound / C1**2) ** 3,
        "one_minus_chi_at_s_H_exponent": (S_H / M._CURRENT_S0) ** M._CURRENT_P,
        "fd_check_d2e_dg2_max_relative_error": max(checks),
    }
    (OUTPUT_DIR / "stability_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    r, p_c = h_potential("fmix")
    _r, p_p = h_potential("pbe")
    with gp93r.use_gp93r():
        _r, p_r = h_potential("fmix")
    x = np.log(r)
    n = np.exp(-2.0 * r) / pi
    inner = cumulative_trapezoid(4 * pi * r**3 * n, x, initial=0.0)
    outer = cumulative_trapezoid(4 * pi * r**2 * n, x, initial=0.0)
    r_vh = inner + r * (outer[-1] - outer)
    keep = r >= 0.01
    write_csv("gp93r_hydrogen_potential.csv", [
        {"r_bohr": a, "r_vH": b, "minus_r_vx_gp93": c, "minus_r_vx_gp93r": d,
         "minus_r_vx_pbe": e}
        for a, b, c, d, e in zip(r[keep], r_vh[keep], p_c[keep], p_r[keep], p_p[keep])])
    for row in summary:
        print(row)
    print(meta)


if __name__ == "__main__":
    main()
