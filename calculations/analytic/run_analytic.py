#!/usr/bin/env python3
"""Generate GP93 and hydrogen-regularization numerical data.

This script evaluates the GP93 analytic source and imports the canonical
regularized meta-GGA channel directly.  Descriptor clips, the broad
log-density window, analytic derivatives, and the low-density zero are
verified against finite differences.

Outputs are written to ``results/analytic``:

* gp93_building_block.csv
* hydrogen_regularization.csv
* derivative_checks.csv
* analytic_metadata.json
* DERIVATION_NOTES.md

Figures are deliberately generated in a separate step by
``calculations/plot_results.py`` from these saved data files.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# Keep the functional modules free of runtime bytecode files.
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "results" / "analytic"
SOURCE = ROOT / "functional"
INPUT_PATH = ROOT / "inputs" / "analytic.json"
INPUT = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
if int(INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/analytic.json schema_version")
if not SOURCE.is_dir():
    raise FileNotFoundError(f"Functional source directory not found: {SOURCE}")
sys.path.insert(0, str(SOURCE))

import fmix as FM  # noqa: E402
import gp93_w as G  # noqa: E402
import hydrogen_exact as HX  # noqa: E402
import mgga_fmix as MG  # noqa: E402


PI = math.pi
GAMMA13 = math.gamma(1.0 / 3.0)
NUMERICAL = MG.PARAMETERS["numerical_implementation"]
CM = G.IM(float(NUMERICAL["cm_evaluation_endpoint"]))
K_GP = 8.0 * PI / 3.0
L6P = math.log(6.0 * PI)
AX = -(3.0 / 4.0) * (3.0 / PI) ** (1.0 / 3.0)
C1 = (3.0 * PI * PI) ** (1.0 / 3.0)
CF = (3.0 / 10.0) * (3.0 * PI * PI) ** (2.0 / 3.0)
MUP = 0.21951
KP = 0.804
S0 = float(MG.PARAMETERS["selected_s0"])
P_SWITCH = int(MG.PARAMETERS["selected_p"])
WA = MG.ALPHA_WIDTH
N_WINDOW_HIGH = MG.DENSITY_WINDOW_HIGH
N_WINDOW_LOW = MG.DENSITY_WINDOW_LOW
N_HARD = MG.DENSITY_CUTOFF
S_MIN = MG.S_MIN
S_MAX = MG.S_MAX


def scalar_series(z: float) -> float:
    """Submitted Frobenius continuation, converted to a scalar."""

    return float(np.asarray(G.w_series(float(z), FM.c0)).reshape(-1)[0])


def w_asymptotic(z: np.ndarray | float) -> np.ndarray:
    """Large-z form after the exponentially small tail integral is dropped."""

    z_arr = np.asarray(z, dtype=float)
    return 0.5 - (GAMMA13 / 4.0) * CM * np.asarray(G.Uf(z_arr), dtype=float)


def w_gp93_analytic(
    z: np.ndarray | float,
    exact_to: float = float(INPUT["gp93_analytic_exact_to_z"]),
) -> np.ndarray:
    """Bare GP93 continuation, with a numerically stable large-z evaluation.

    For z <= 0 this is the canonical Frobenius series.  For 0 < z <= 12 it
    evaluates the Green-function expression.  Beyond z=12 it uses
    that expression's asymptotic form; its absolute error at z=12 is about
    6.2e-7 and falls exponentially thereafter.
    """

    z_arr = np.asarray(z, dtype=float)
    flat = z_arr.reshape(-1)
    out = np.empty_like(flat)
    for i, zi in enumerate(flat):
        if zi <= 0.0:
            out[i] = scalar_series(zi)
        elif zi <= exact_to:
            out[i] = G.w_exact(float(zi))
        else:
            out[i] = float(w_asymptotic(zi))
    return out.reshape(z_arr.shape)


def f_gp93_bare_analytic(s: np.ndarray | float) -> np.ndarray:
    """Bare, un-clipped analytic GP93 enhancement factor."""

    s_arr = np.asarray(s, dtype=float)
    z = 3.0 * np.log(s_arr) + L6P
    return K_GP * s_arr * w_gp93_analytic(z)


def f_gp93_source_with_derivative(
    s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The GP93 branch and dF/ds as represented in canonical ``fmix.py``.

    The z clip is differentiated as a mathematical clip: dz/ds is zero when
    z is outside [-6, 30].  This branch is used for the bare-H potential below;
    over r=0.2..15, z=2r lies inside the upper boundary except at the endpoint.
    """

    s_arr = np.asarray(s, dtype=float)
    z_raw = (
        3.0
        * np.log(
            np.maximum(s_arr, float(NUMERICAL["log_input_floor"]))
        )
        + L6P
    )
    z = np.clip(z_raw, FM.Z_MIN, FM.Z_MAX)
    w = FM.w_sp(z)
    wz = FM.w_sp(z, 1)
    active = (z_raw > FM.Z_MIN) & (z_raw < FM.Z_MAX)
    f = K_GP * s_arr * w
    fp = K_GP * (w + np.where(active, 3.0 * wz, 0.0))
    return f, fp, z_raw, z


def f_pbe_with_derivative(
    s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    s_arr = np.asarray(s, dtype=float)
    denominator = 1.0 + MUP * s_arr * s_arr / KP
    f = 1.0 + KP - KP / denominator
    fp = 2.0 * MUP * s_arr / denominator**2
    return f, fp


def f_mix_direct_with_derivative(
    s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Selected ``fmix.make_Fmix`` and its code-consistent derivative."""

    s_arr = np.asarray(s, dtype=float)
    f_pbe, fp_pbe = f_pbe_with_derivative(s_arr)
    f_gp, fp_gp, _, _ = f_gp93_source_with_derivative(s_arr)
    q_raw = (s_arr / S0) ** P_SWITCH
    switch_cap = float(NUMERICAL["switch_exponent_cap"])
    q = np.clip(q_raw, 0.0, switch_cap)
    expmq = np.exp(-q)
    chi = 1.0 - expmq
    chip = np.where(
        q_raw < switch_cap,
        expmq * P_SWITCH * q_raw / np.maximum(s_arr, 1.0e-300),
        0.0,
    )
    f = (1.0 - chi) * f_pbe + chi * f_gp
    fp = (1.0 - chi) * fp_pbe + chi * fp_gp + chip * (f_gp - f_pbe)
    return f, fp, chi


MG.set_fmix_parameters(S0, P_SWITCH)
HX.set_parameters(S0, P_SWITCH)
_FMIX_SPLINE = MG._FMIX_SPLINE


def fd_nonuniform(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Nonuniform three-point radial derivative."""

    f = np.asarray(values, dtype=float)
    r = np.asarray(coordinates, dtype=float)
    df = np.zeros_like(f)
    h1 = r[1:-1] - r[:-2]
    h2 = r[2:] - r[1:-1]
    df[1:-1] = (
        h1 * h1 * f[2:]
        - h2 * h2 * f[:-2]
        + (h2 * h2 - h1 * h1) * f[1:-1]
    ) / (h1 * h2 * (h1 + h2))
    df[0] = (f[1] - f[0]) / (r[1] - r[0])
    df[-1] = (f[-1] - f[-2]) / (r[-1] - r[-2])
    return df


def radial_potential(
    vr: np.ndarray,
    vs: np.ndarray,
    nprime: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """v_x = v_rho - 2 r^-2 d[r^2 v_sigma n']/dr."""

    flux = r * r * vs * nprime
    return vr - 2.0 * fd_nonuniform(flux, r) / (r * r)


def exact_hydrogen(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.exp(-2.0 * r) / PI
    nprime = -2.0 * n
    tau = 0.5 * n
    return n, nprime, tau


def bare_gp93_h_channel(
    r: np.ndarray,
) -> dict[str, np.ndarray]:
    """Bare GP93 branch on exact H, without chi, gates, or descriptor clips.

    The canonical ``w_sp`` representation and its analytic spline derivative
    are used.  A caller may provide a few guard points just beyond r=15; the
    spline's local extrapolation then permits a centered radial derivative at
    the requested r=15 endpoint without introducing the coded z=30 clip kink.
    """

    n, nprime, _ = exact_hydrogen(r)
    m = 2.0 * n
    sig = nprime * nprime
    sigm = 4.0 * sig
    s = np.sqrt(sigm) / (2.0 * C1 * m ** (4.0 / 3.0))
    z_raw = 3.0 * np.log(s) + L6P
    w = FM.w_sp(z_raw)
    wz = FM.w_sp(z_raw, 1)
    f = K_GP * s * w
    fp = K_GP * (w + 3.0 * wz)
    vr = (4.0 / 3.0) * AX * m ** (1.0 / 3.0) * (f - s * fp)
    vs = AX * m ** (4.0 / 3.0) * fp * s / sigm
    vx = radial_potential(vr, vs, nprime, r)
    return {
        "n": n,
        "nprime": nprime,
        "s_raw": s,
        "s_code": s,
        "z_raw": z_raw,
        "z_code": z_raw,
        "F": f,
        "Fp": fp,
        "vr": vr,
        "vs": vs,
        "vx": vx,
    }


def unregularized_h_exact_channel(
    r: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the released unregularized H-exact crossover on exact H."""

    n, nprime, _ = exact_hydrogen(r)
    sigma = nprime * nprime
    energy_density, vr, vs = HX._channel(n, sigma)
    vx = radial_potential(vr, vs, nprime, r)
    return {
        "n": n,
        "nprime": nprime,
        "energy_density": energy_density,
        "vr": vr,
        "vs": vs,
        "vx": vx,
    }


def implemented_h_channel(
    r: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the canonical regularized channel on the exact H density."""

    n, nprime, tau = exact_hydrogen(r)
    sigma = nprime * nprime
    # Enforce the exact iso-orbital identity before calling the implementation.
    tau_w = sigma / (8.0 * n)
    energy_density, vr, vs, vt = MG._channel(n, sigma, tau_w)
    m = np.maximum(2.0 * n, MG.SPIN_DENSITY_FLOOR)
    sigma_m = np.maximum(4.0 * sigma, MG.SIGMA_FLOOR)
    s_raw = np.sqrt(sigma_m) / (2.0 * C1 * m ** (4.0 / 3.0))
    s = np.clip(s_raw, S_MIN, S_MAX)
    x_n, _ = MG._density_window(n)
    vx = radial_potential(vr, vs, nprime, r)

    return {
        "n": n,
        "nprime": nprime,
        "tau": tau,
        "s_raw": s_raw,
        "s_code": s,
        "z_raw": 3.0 * np.log(s_raw) + L6P,
        "z_code": np.clip(3.0 * np.log(s) + L6P, -6.0, 30.0),
        "alpha": np.zeros_like(n),
        "gate_alpha": np.ones_like(n),
        "gate_density": x_n,
        "energy_density": energy_density,
        "vr": vr,
        "vs": vs,
        "vt": vt,
        "hard_zero": n < N_HARD,
        "vx": vx,
    }


def implemented_energy_alpha0(
    n: np.ndarray | float,
    sig: np.ndarray | float,
) -> np.ndarray:
    """Pointwise canonical exchange energy at the iso-orbital limit."""

    n_arr = np.atleast_1d(np.asarray(n, dtype=float))
    sig_arr = np.atleast_1d(np.asarray(sig, dtype=float))
    tau_w = sig_arr / (8.0 * np.maximum(n_arr, MG.DENSITY_FLOOR))
    energy, _, _, _ = MG._channel(n_arr, sig_arr, tau_w)
    return energy if np.ndim(n) else energy[0]


def implemented_energy(
    n: np.ndarray | float,
    sig: np.ndarray | float,
    tau: np.ndarray | float,
) -> np.ndarray:
    """Pointwise canonical exchange energy for independent MGGA descriptors."""

    n_arr = np.atleast_1d(np.asarray(n, dtype=float))
    sig_arr = np.atleast_1d(np.asarray(sig, dtype=float))
    tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
    energy, _, _, _ = MG._channel(n_arr, sig_arr, tau_arr)
    return energy if np.ndim(n) else energy[0]


def exact_minus_r_vx(r: np.ndarray) -> np.ndarray:
    """For one-electron H, v_x=-v_H and -r v_x=r v_H."""

    return 1.0 - np.exp(-2.0 * r) * (1.0 + r)


def write_wf_csv() -> dict[str, np.ndarray]:
    grid = INPUT["enhancement_grid"]
    s = np.logspace(
        float(grid["log10_s_min"]),
        float(grid["log10_s_max"]),
        int(grid["points"]),
    )
    z_raw = 3.0 * np.log(s) + L6P
    f_pbe = FM.F_PBE(s)
    f_bare = f_gp93_bare_analytic(s)
    f_gp_code = FM.F_GP93(s)
    f_mix_direct = FM.make_Fmix(S0, P_SWITCH)(s)
    f_mix_reconstructed, _, chi = f_mix_direct_with_derivative(s)
    s_ch = np.clip(s, S_MIN, S_MAX)
    f_ch_sampled = _FMIX_SPLINE(np.log(s_ch))
    data = {
        "s": s,
        "z_raw": z_raw,
        "z_in_fmix_py": np.clip(z_raw, FM.Z_MIN, FM.Z_MAX),
        "F_PBE": f_pbe,
        "F_GP93_bare_analytic": f_bare,
        "F_GP93_fmix_py_zclipped": f_gp_code,
        "Fmix_direct_selected_parameters": f_mix_direct,
        "Fmix_direct_reconstructed": f_mix_reconstructed,
        "chi": chi,
        "s_after_mgga_clip": s_ch,
        "Fmix_sampled_by_mgga_ch": f_ch_sampled,
    }
    np.savetxt(
        HERE / "gp93_building_block.csv",
        np.column_stack(list(data.values())),
        delimiter=",",
        header=",".join(data.keys()),
        comments="",
        fmt="%.16e",
    )
    return data


def write_vx_csv(
    r: np.ndarray,
    bare: dict[str, np.ndarray],
    unregularized: dict[str, np.ndarray],
    impl: dict[str, np.ndarray],
) -> None:
    exact = exact_minus_r_vx(r)
    data = {
        "r_bohr": r,
        "n_H": impl["n"],
        "dn_dr": impl["nprime"],
        "tau_H": impl["tau"],
        "s_raw": impl["s_raw"],
        "s_after_mgga_clip": impl["s_code"],
        "z_raw": impl["z_raw"],
        "z_after_fmix_clip": impl["z_code"],
        "alpha": impl["alpha"],
        "G_alpha": impl["gate_alpha"],
        "X_n": impl["gate_density"],
        "hard_zero_below_density_window": impl["hard_zero"].astype(float),
        "minus_r_vx_exact": exact,
        "minus_r_vx_bare_GP93": -r * bare["vx"],
        "minus_r_vx_unregularized_H_exact": -r * unregularized["vx"],
        "minus_r_vx_implemented": -r * impl["vx"],
        "vrho_implemented": impl["vr"],
        "vsigma_implemented": impl["vs"],
        "vtau_implemented": impl["vt"],
    }
    np.savetxt(
        HERE / "hydrogen_regularization.csv",
        np.column_stack(list(data.values())),
        delimiter=",",
        header=",".join(data.keys()),
        comments="",
        fmt="%.16e",
    )


def append_check(
    rows: list[dict[str, str | float]],
    check_type: str,
    point: str | float,
    code_or_analytic: float,
    finite_difference: float,
    note: str,
) -> None:
    abs_error = abs(code_or_analytic - finite_difference)
    scale = max(abs(code_or_analytic), abs(finite_difference), 1.0e-300)
    rows.append(
        {
            "check_type": check_type,
            "point": point,
            "code_or_analytic": code_or_analytic,
            "finite_difference_or_reference": finite_difference,
            "abs_error": abs_error,
            "rel_error": abs_error / scale,
            "note": note,
        }
    )


def derivative_checks() -> tuple[list[dict[str, str | float]], dict[str, float]]:
    rows: list[dict[str, str | float]] = []
    check_input = INPUT["derivative_checks"]
    exact_h_samples = tuple(
        float(value) for value in check_input["exact_h_radius_bohr"]
    )
    radial_samples = tuple(
        float(value)
        for value in check_input["radial_potential_sample_r_bohr"]
    )

    # Direct Fmix derivative against the canonical value function.
    f_source = FM.make_Fmix(S0, P_SWITCH)
    direct_samples = sorted(
        {
            *(
                float(value)
                for value in check_input[
                    "direct_fmix_s_excluding_selected_s0"
                ]
            ),
            S0,
        }
    )
    for s in direct_samples:
        _, fp, _ = f_mix_direct_with_derivative(np.array([s]))
        ds = max(
            float(check_input["direct_fmix_absolute_step_min"]),
            float(check_input["direct_fmix_relative_step"]) * s,
        )
        fd_value = float((f_source(s + ds) - f_source(s - ds)) / (2.0 * ds))
        append_check(
            rows,
            "dFmix_direct/ds",
            s,
            float(fp[0]),
            fd_value,
            "includes chi derivative and mathematical derivative of z clip",
        )

    # The spline derivative used by the canonical meta-GGA channel.
    spline_samples = sorted(
        {
            *(
                float(value)
                for value in check_input[
                    "fmix_spline_s_excluding_selected_s0"
                ]
            ),
            S0,
        }
    )
    spline_input = NUMERICAL["fmix_log_s_spline"]
    for s in spline_samples:
        fp = float(_FMIX_SPLINE(math.log(s), 1) / s)
        ds = min(1.0e-6 * max(s, 1.0), 0.25 * s)
        lo = max(s - ds, float(spline_input["s_min"]))
        hi = min(s + ds, float(spline_input["s_max"]))
        fd_value = float(
            (_FMIX_SPLINE(math.log(hi)) - _FMIX_SPLINE(math.log(lo)))
            / (hi - lo)
        )
        append_check(
            rows,
            "dFmix_spline/ds",
            s,
            fp,
            fd_value,
            "same log-s CubicSpline and chain rule as mgga_fmix.py",
        )

    # The canonical implementation differentiates the clipped composite.
    for s_raw in (
        float(value) for value in check_input["descriptor_clip_s"]
    ):
        code_fp = 0.0
        ds = 1.0e-6 * s_raw

        def clipped_value(x: float) -> float:
            return float(
                _FMIX_SPLINE(math.log(float(np.clip(x, S_MIN, S_MAX))))
            )

        fd_value = (clipped_value(s_raw + ds) - clipped_value(s_raw - ds)) / (
            2.0 * ds
        )
        append_check(
            rows,
            "descriptor_clip_derivative",
            s_raw,
            code_fp,
            fd_value,
            "canonical implementation sets the derivative of a clipped "
            "descriptor to zero outside the active interval",
        )

    # Broad log-density-window derivative.
    for n in (
        float(value)
        for value in check_input["density_window_spin_density"]
    ):
        _, derivative = MG._density_window(np.array([n]))
        analytic = float(n * derivative[0])
        dlogn = float(check_input["density_window_log_step"])

        def xgate(y: float) -> float:
            return float(MG._density_window(np.array([y]))[0][0])

        fd_value = (
            xgate(n * math.exp(dlogn))
            - xgate(n * math.exp(-dlogn))
        ) / (2.0 * dlogn)
        append_check(
            rows,
            "d_density_window/dlogn",
            n,
            analytic,
            fd_value,
            "cubic smoothstep in log density; dX/dlog(n) vanishes at endpoints",
        )

    # Full local energy derivative checks in the exact-H alpha=0 sector.  At
    # these points s is inside its clip interval, so the source derivative
    # convention and the derivative of the value function should coincide.
    guard = float(check_input["local_radius_guard_bohr"])
    relative_step = float(check_input["relative_finite_difference_step"])
    for sample_r in exact_h_samples:
        local_r = np.array([sample_r - guard, sample_r, sample_r + guard])
        local = implemented_h_channel(local_r)
        n = float(local["n"][1])
        sig = float(local["nprime"][1] ** 2)
        dn = relative_step * n
        dsig = relative_step * sig
        fd_n = float(
            (
                implemented_energy_alpha0(n + dn, sig)
                - implemented_energy_alpha0(n - dn, sig)
            )
            / (2.0 * dn)
        )
        fd_sig = float(
            (
                implemented_energy_alpha0(n, sig + dsig)
                - implemented_energy_alpha0(n, sig - dsig)
            )
            / (2.0 * dsig)
        )
        append_check(
            rows,
            "implemented_vrho_vs_energy_FD",
            sample_r,
            float(local["vr"][1]),
            fd_n,
            "partial derivative with respect to input spin density at fixed sigma; alpha=0",
        )
        append_check(
            rows,
            "implemented_vsigma_vs_energy_FD",
            sample_r,
            float(local["vs"][1]),
            fd_sig,
            "partial derivative with respect to input sigma at fixed density; alpha=0",
        )

    # Independent n, sigma, and tau derivatives at nonzero alpha.  These
    # points exercise the iso-orbital gate, including vtau, rather than only
    # the exact-H alpha=0 sector.
    nonzero_relative_step = float(
        check_input["nonzero_alpha_relative_finite_difference_step"]
    )
    for point_index, sample in enumerate(
        check_input["nonzero_alpha_samples"], start=1
    ):
        n = float(sample["spin_density"])
        target_s = float(sample["reduced_gradient"])
        target_alpha = float(sample["alpha"])
        m = 2.0 * n
        sigma = (target_s * MG.C1 * m ** (4.0 / 3.0)) ** 2
        tau_w_m = (4.0 * sigma) / (8.0 * m)
        tau_uniform_m = MG.CF * m ** (5.0 / 3.0)
        tau = 0.5 * (tau_w_m + target_alpha * tau_uniform_m)
        _energy, vrho, vsigma, vtau = MG._channel(
            np.array([n]), np.array([sigma]), np.array([tau])
        )
        dn = nonzero_relative_step * n
        dsigma = nonzero_relative_step * sigma
        dtau = nonzero_relative_step * tau
        finite_differences = {
            "implemented_nonzero_alpha_vrho_vs_energy_FD": (
                float(vrho[0]),
                float(
                    (
                        implemented_energy(n + dn, sigma, tau)
                        - implemented_energy(n - dn, sigma, tau)
                    )
                    / (2.0 * dn)
                ),
                "n",
            ),
            "implemented_nonzero_alpha_vsigma_vs_energy_FD": (
                float(vsigma[0]),
                float(
                    (
                        implemented_energy(n, sigma + dsigma, tau)
                        - implemented_energy(n, sigma - dsigma, tau)
                    )
                    / (2.0 * dsigma)
                ),
                "sigma",
            ),
            "implemented_nonzero_alpha_vtau_vs_energy_FD": (
                float(vtau[0]),
                float(
                    (
                        implemented_energy(n, sigma, tau + dtau)
                        - implemented_energy(n, sigma, tau - dtau)
                    )
                    / (2.0 * dtau)
                ),
                "tau",
            ),
        }
        for check_type, (analytic, finite_difference, descriptor) in (
            finite_differences.items()
        ):
            append_check(
                rows,
                check_type,
                point_index,
                analytic,
                finite_difference,
                (
                    f"partial derivative with respect to {descriptor} at fixed "
                    f"other descriptors; alpha={target_alpha:g}, n={n:.0e}"
                ),
            )

    # Radial-grid convergence.  h=0.001 is the numerical reference for the
    # implemented curve; bare GP93 can additionally be compared with exact H.
    step_results: dict[float, tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    potential_grid = INPUT["hydrogen_potential_grid"]
    convergence_steps = tuple(
        float(value) for value in INPUT["radial_convergence_steps_bohr"]
    )
    error_start, error_stop = (
        float(value) for value in INPUT["bare_potential_error_range_bohr"]
    )
    for h in convergence_steps:
        r = np.arange(
            float(potential_grid["evaluation_start_bohr"]),
            float(potential_grid["reported_stop_bohr"]) + 0.5 * h,
            h,
        )
        bare = bare_gp93_h_channel(r)
        impl = implemented_h_channel(r)
        step_results[h] = (r, bare, impl)
        mask = (r >= error_start) & (r <= error_stop)
        max_error = float(
            np.max(
                np.abs(
                    (-r * bare["vx"])[mask]
                    - exact_minus_r_vx(r[mask])
                )
            )
        )
        append_check(
            rows,
            "bare_potential_max_abs_error",
            f"h={h:g}",
            max_error,
            0.0,
            (
                f"max over r={error_start:g}..{error_stop:g} versus "
                "exact -r v_x=r v_H"
            ),
        )

    reference_step = min(convergence_steps)
    r_ref, _, impl_ref = step_results[reference_step]
    minus_ref = -r_ref * impl_ref["vx"]
    for h, (r, _, impl) in step_results.items():
        minus_impl = -r * impl["vx"]
        for sample_r in radial_samples:
            value = float(np.interp(sample_r, r, minus_impl))
            reference = float(np.interp(sample_r, r_ref, minus_ref))
            append_check(
                rows,
                "implemented_potential_grid_convergence",
                f"r={sample_r:g}; h={h:g}",
                value,
                reference,
                (
                    f"reference is the h={reference_step:g} "
                    "radial-divergence evaluation"
                ),
            )

    summary = {
        "r_window_high": 0.5
        * math.log(1.0 / (PI * N_WINDOW_HIGH)),
        "r_window_low": 0.5
        * math.log(1.0 / (PI * N_WINDOW_LOW)),
        "r_hard": 0.5 * math.log(1.0 / (PI * N_HARD)),
        "r_sclip": 1.5
        * math.log(S_MAX * (6.0 * PI) ** (1.0 / 3.0)),
        "s_zmax": math.exp((FM.Z_MAX - L6P) / 3.0),
        "asymptotic_coefficient": (
            K_GP * GAMMA13 * CM / 4.0 * 3.0 ** (2.0 / 3.0)
        ),
        "cm": CM,
    }
    return rows, summary


def write_checks(rows: list[dict[str, str | float]]) -> None:
    fieldnames = [
        "check_type",
        "point",
        "code_or_analytic",
        "finite_difference_or_reference",
        "abs_error",
        "rel_error",
        "note",
    ]
    with (HERE / "derivative_checks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_notes(
    summary: dict[str, float],
    r: np.ndarray,
    bare: dict[str, np.ndarray],
    unregularized: dict[str, np.ndarray],
    impl: dict[str, np.ndarray],
) -> None:
    sample_lines = []
    for sample_r in (6.0, 8.0, 10.0, 12.0, 12.2, 15.0):
        i = int(np.argmin(np.abs(r - sample_r)))
        sample_lines.append(
            "| "
            f"{r[i]:.3f} | {impl['n'][i]:.6e} | {impl['s_raw'][i]:.6g} | "
            f"{impl['gate_density'][i]:.6e} | "
            f"{(-r[i] * bare['vx'][i]):.8f} | "
            f"{(-r[i] * unregularized['vx'][i]):.8f} | "
            f"{(-r[i] * impl['vx'][i]):.8f} |"
        )
    samples = "\n".join(sample_lines)

    notes = f"""# GP93 building block and regularized exact-H diagnostic

Run this calculation from the repository root with
`python calculations/analytic/run_analytic.py`.  The canonical implementation
is imported directly; its equations are not copied into this script.
The numerical grids and convergence-study spacings are read from
`{INPUT_PATH.relative_to(ROOT)}`.

## Source files

- `functional/gp93_w.py`
- `functional/fmix.py`
- `functional/mgga_fmix.py`
- `functional/hydrogen_exact.py`
- `functional/parameters.json`
- `{INPUT_PATH.relative_to(ROOT)}`

## GP93 large-gradient branch

The variables obey

\\[
z=3\\ln s+\\ln(6\\pi),\\qquad
F_{{\\rm GP93}}(s)=\\frac{{8\\pi}}3s\\,w(z).
\\]

Because \\(I_M\\to C_M={summary['cm']:.15g}\\) and
\\(U(-2/3,2,z)\\sim z^{{2/3}}\\),

\\[
F_{{\\rm GP93}}(s)\\sim
-{summary['asymptotic_coefficient']:.10g}\\,
s(\\ln s)^{{2/3}} .
\\]

The bare GP93 building block is hydrogen exact.  The practical functional is
different by design: its broad log-density window keeps the GP93 contribution
in the intermediate region and turns it off before the numerical far tail.

## Canonical density window

For a spin density \\(n_\\sigma\\), define

\\[
t=\\frac{{\\ln(n_\\sigma/n_{{\\rm low}})}}
        {{\\ln(n_{{\\rm high}}/n_{{\\rm low}})}},\\qquad
X_n=3t^2-2t^3
\\]

for \\(0<t<1\\), with \\(X_n=1\\) above
\\(n_{{\\rm high}}={N_WINDOW_HIGH:.0e}\\) and \\(X_n=0\\) below
\\(n_{{\\rm low}}={N_WINDOW_LOW:.0e}\\).  The first derivative vanishes at
both endpoints.  On the exact H density the transition begins at
\\(r={summary['r_window_high']:.6f}\\) bohr and ends at
\\(r={summary['r_window_low']:.6f}\\) bohr.  Thus strict \\(-1/r\\)
asymptotics belongs to the building block, whereas the final functional
targets the radial window controlling low-lying diffuse states.

For the exact H density, \\(\\tau=\\tau^W\\) is imposed analytically before the
canonical `_channel` call, so \\(\\alpha=0\\).  The radial potential is formed
from the returned derivatives as

\\[
v_x(r)=v_\\rho(r)-\\frac2{{r^2}}\\frac{{d}}{{dr}}
\\left[r^2v_\\sigma(r)n'_H(r)\\right].
\\]

| r (bohr) | n | raw s | X_n | bare -r v_x | unregularized -r v_x | regularized -r v_x |
|---:|---:|---:|---:|---:|---:|---:|
{samples}

## Verification

`derivative_checks.csv` records:

- analytic derivatives of the selected mixed enhancement against finite
  differences;
- derivatives of the canonical log-s spline;
- zero derivatives outside the descriptor-clip interval;
- analytic derivatives of the cubic log-density window;
- canonical `vrho` and `vsigma` against finite differences of `_channel`;
- radial-grid convergence of the bare and regularized H potentials.
"""
    (HERE / "DERIVATION_NOTES.md").write_text(notes, encoding="utf-8")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    owned_outputs = (
        "gp93_building_block.csv",
        "hydrogen_regularization.csv",
        "derivative_checks.csv",
        "DERIVATION_NOTES.md",
        "analytic_metadata.json",
    )
    for name in owned_outputs:
        (HERE / name).unlink(missing_ok=True)
    write_wf_csv()

    # Start below the requested plotting range so the first visible radial
    # point does not use a one-sided derivative.  h=0.002 is converged at the
    # visual scale; h=0.001 is included in the numerical check CSV.
    # Guard points beyond r=15 give the visible endpoint a centered radial
    # derivative.  They are never written to the CSV or shown in the figure.
    potential_grid = INPUT["hydrogen_potential_grid"]
    radial_step = float(potential_grid["step_bohr"])
    r_full = np.arange(
        float(potential_grid["evaluation_start_bohr"]),
        float(potential_grid["evaluation_stop_bohr"]) + 0.5 * radial_step,
        radial_step,
    )
    bare_full = bare_gp93_h_channel(r_full)
    unregularized_full = unregularized_h_exact_channel(r_full)
    impl_full = implemented_h_channel(r_full)
    visible = (
        r_full >= float(potential_grid["reported_start_bohr"]) - 1.0e-12
    ) & (
        r_full <= float(potential_grid["reported_stop_bohr"]) + 1.0e-12
    )
    r = r_full[visible]
    bare = {key: value[visible] for key, value in bare_full.items()}
    unregularized = {
        key: value[visible] for key, value in unregularized_full.items()
    }
    impl = {key: value[visible] for key, value in impl_full.items()}
    write_vx_csv(r, bare, unregularized, impl)

    rows, summary = derivative_checks()
    write_checks(rows)
    write_notes(summary, r, bare, unregularized, impl)
    metadata = {
        "input_file": str(INPUT_PATH.relative_to(ROOT)),
        "input_settings": INPUT,
        "functional_parameters_file": "functional/parameters.json",
        "functional_parameters": MG.PARAMETERS,
        "selected_parameters": {"s0": S0, "p": P_SWITCH},
        "outputs": list(owned_outputs[:-1]),
    }
    (HERE / "analytic_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Wrote GP93 CSV data, checks, and derivation notes:")
    for name in (
        "gp93_building_block.csv",
        "hydrogen_regularization.csv",
        "derivative_checks.csv",
        "DERIVATION_NOTES.md",
        "analytic_metadata.json",
    ):
        print(f"  {name}")
    print(
        "Plot separately with:\n"
        "  python calculations/plot_results.py fig-gp93-building-block\n"
        "  python calculations/plot_results.py fig-hydrogen-regularization"
    )


if __name__ == "__main__":
    main()
