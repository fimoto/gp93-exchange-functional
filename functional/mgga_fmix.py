"""Regularized GP93/PBE exchange for PySCF.

This is the implementation used for every self-consistent result in the
repository.  It deliberately distinguishes the hydrogen-exact GP93 building
block from the final, numerically regularized meta-GGA:

    F_mix = F_PBE + X_n(n_sigma) G(alpha) [F_cross - F_PBE].

The density window ``X_n`` and the low-density cutoff remove the strict
``r -> infinity`` Coulomb tail.  Their purpose is to retain the GP93
contribution over the intermediate radial range relevant to low-lying diffuse
states while avoiding an unrestricted divergent branch on numerical grids.
The window is a cubic smoothstep in ``log(n_sigma)``.  It is exactly one above
``DENSITY_WINDOW_HIGH``, exactly zero below ``DENSITY_WINDOW_LOW``, and has a
zero first derivative at both endpoints.  The broad transition keeps the
regularized potential smooth across the window.

The derivatives below include the derivatives of the density window and set the
derivative of clipped descriptors to zero outside the active interval.  Both
RKS and UKS return shapes follow PySCF's ``define_xc_`` convention.
"""

from __future__ import annotations

import json
from math import pi
from pathlib import Path

import numpy as np
from pyscf import dft
from pyscf.dft import libxc
from scipy.interpolate import CubicSpline

import fmix


CORR = ""

PARAMETERS = json.loads(
    (Path(__file__).with_name("parameters.json")).read_text(encoding="utf-8")
)
NUMERICAL = PARAMETERS["numerical_implementation"]
C1 = (3.0 * pi * pi) ** (1.0 / 3.0)
CF = (3.0 / 10.0) * (3.0 * pi * pi) ** (2.0 / 3.0)
A_X = -(3.0 / 4.0) * (3.0 / pi) ** (1.0 / 3.0)
MU_PBE = 0.21951
KAPPA_PBE = 0.804

ALPHA_WIDTH = float(PARAMETERS["alpha_width"])
DENSITY_WINDOW_HIGH = float(PARAMETERS["density_window_high"])
DENSITY_WINDOW_LOW = float(PARAMETERS["density_window_low"])
DENSITY_CUTOFF = DENSITY_WINDOW_LOW
DENSITY_FLOOR = float(NUMERICAL["density_floor"])
SPIN_DENSITY_FLOOR = float(PARAMETERS["spin_density_floor"])
SIGMA_FLOOR = float(PARAMETERS["sigma_floor"])
S_MIN = float(PARAMETERS["s_min"])
S_MAX = float(PARAMETERS["s_max"])

FMIX_SPLINE_INPUT = NUMERICAL["fmix_log_s_spline"]
_S_GRID = np.logspace(
    np.log10(float(FMIX_SPLINE_INPUT["s_min"])),
    np.log10(float(FMIX_SPLINE_INPUT["s_max"])),
    int(FMIX_SPLINE_INPUT["points"]),
)
_FMIX_SPLINE: CubicSpline
_CURRENT_S0 = float(PARAMETERS["selected_s0"])
_CURRENT_P = int(PARAMETERS["selected_p"])


def set_fmix_parameters(
    s0: float = float(PARAMETERS["selected_s0"]),
    p: int = int(PARAMETERS["selected_p"]),
) -> None:
    """Set the outer-switch parameters and rebuild the interpolation spline."""

    global _FMIX_SPLINE, _CURRENT_S0, _CURRENT_P
    _CURRENT_S0 = float(s0)
    _CURRENT_P = int(p)
    values = fmix.make_Fmix(_CURRENT_S0, _CURRENT_P)(_S_GRID)
    _FMIX_SPLINE = CubicSpline(np.log(_S_GRID), values)


def _Fmix_override(s0: float, p: int = 8) -> None:
    """Backward-compatible alias used by the calculation scripts."""

    set_fmix_parameters(s0=s0, p=p)


set_fmix_parameters()


def _pbe_enhancement(s: np.ndarray) -> np.ndarray:
    denominator = 1.0 + MU_PBE * s * s / KAPPA_PBE
    return 1.0 + KAPPA_PBE - KAPPA_PBE / denominator


def _pbe_derivative(s: np.ndarray) -> np.ndarray:
    denominator = 1.0 + MU_PBE * s * s / KAPPA_PBE
    return 2.0 * MU_PBE * s / denominator**2


def _density_window(
    n_sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the broad log-density window and ``dX_n/dn_sigma``."""

    n_safe = np.maximum(np.asarray(n_sigma, dtype=float), DENSITY_FLOOR)
    log_width = np.log(DENSITY_WINDOW_HIGH / DENSITY_WINDOW_LOW)
    coordinate_raw = np.log(n_safe / DENSITY_WINDOW_LOW) / log_width
    coordinate = np.clip(coordinate_raw, 0.0, 1.0)
    value = coordinate * coordinate * (3.0 - 2.0 * coordinate)
    active = (
        (n_sigma > DENSITY_WINDOW_LOW)
        & (n_sigma < DENSITY_WINDOW_HIGH)
    )
    derivative = (
        6.0
        * coordinate
        * (1.0 - coordinate)
        / (log_width * n_safe)
        * active
    )
    return value, derivative


def _channel(
    n_sigma: np.ndarray,
    sigma_sigma: np.ndarray,
    tau_sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return energy density and derivatives for one spin channel.

    The returned derivatives are with respect to ``n_sigma``,
    ``sigma_sigma = |grad n_sigma|^2``, and ``tau_sigma``.
    """

    n_sigma = np.asarray(n_sigma, dtype=float)
    sigma_sigma = np.asarray(sigma_sigma, dtype=float)
    tau_sigma = np.asarray(tau_sigma, dtype=float)

    # Exact spin scaling evaluates the unpolarized parent at 2 n_sigma.
    m_raw = 2.0 * n_sigma
    sigma_m_raw = 4.0 * sigma_sigma
    tau_m = np.maximum(2.0 * tau_sigma, 0.0)
    m = np.maximum(m_raw, SPIN_DENSITY_FLOOR)
    sigma_m = np.maximum(sigma_m_raw, SIGMA_FLOOR)

    m43 = m ** (4.0 / 3.0)
    s_raw = np.sqrt(sigma_m) / (2.0 * C1 * m43)
    s = np.clip(s_raw, S_MIN, S_MAX)
    s_active = (s_raw > S_MIN) & (s_raw < S_MAX)
    sigma_active = sigma_m_raw > SIGMA_FLOOR
    density_active = m_raw > SPIN_DENSITY_FLOOR

    log_s = np.log(s)
    f_mix = _FMIX_SPLINE(log_s)
    df_mix_ds = _FMIX_SPLINE(log_s, 1) / s
    f_pbe = _pbe_enhancement(s)
    df_pbe_ds = _pbe_derivative(s)
    delta_f = f_mix - f_pbe
    delta_df = df_mix_ds - df_pbe_ds

    tau_w = sigma_m / (8.0 * m)
    tau_uniform = CF * m ** (5.0 / 3.0)
    alpha_raw = (tau_m - tau_w) / tau_uniform
    alpha = np.maximum(alpha_raw, 0.0)
    alpha_active = alpha_raw > 0.0

    alpha_argument = (alpha / ALPHA_WIDTH) ** 2
    switch_cap = float(NUMERICAL["switch_exponent_cap"])
    alpha_argument_clipped = np.clip(alpha_argument, 0.0, switch_cap)
    g_alpha = np.exp(-alpha_argument_clipped)
    dg_dalpha = (
        -2.0
        * alpha
        / ALPHA_WIDTH**2
        * g_alpha
        * (alpha_argument < switch_cap)
    )

    x_n, dxn_dn = _density_window(n_sigma)
    combined_gate = g_alpha * x_n

    phi = A_X * m43
    enhancement = f_pbe + combined_gate * delta_f
    energy_density = 0.5 * phi * enhancement

    ds_dm = (
        -(4.0 / 3.0)
        * s_raw
        / m
        * s_active
        * density_active
    )
    ds_dsigma_m = (
        s_raw
        / (2.0 * sigma_m)
        * s_active
        * sigma_active
    )

    dalpha_dm = (
        ((tau_w / m) / tau_uniform - (5.0 / 3.0) * alpha_raw / m)
        * alpha_active
        * density_active
    )
    dalpha_dsigma_m = (
        -1.0 / (8.0 * m * tau_uniform)
        * alpha_active
        * sigma_active
        * density_active
    )
    dalpha_dtau_m = (
        1.0 / tau_uniform * alpha_active * density_active
    )

    denergy_dm = 0.5 * (
        (4.0 / 3.0) * (phi / m) * enhancement * density_active
        + phi * (df_pbe_ds + combined_gate * delta_df) * ds_dm
        + phi * delta_f * x_n * dg_dalpha * dalpha_dm
    )
    denergy_dsigma_m = 0.5 * (
        phi
        * (df_pbe_ds + combined_gate * delta_df)
        * ds_dsigma_m
        + phi * delta_f * x_n * dg_dalpha * dalpha_dsigma_m
    )
    denergy_dtau_m = (
        0.5 * phi * delta_f * x_n * dg_dalpha * dalpha_dtau_m
    )
    denergy_dn_gate = 0.5 * phi * delta_f * g_alpha * dxn_dn

    # Chain rule: m=2n_sigma, sigma_m=4sigma_sigma, tau_m=2tau_sigma.
    vrho = 2.0 * denergy_dm + denergy_dn_gate
    vsigma = 4.0 * denergy_dsigma_m
    vtau = 2.0 * denergy_dtau_m

    hard_zero = n_sigma < DENSITY_CUTOFF
    for array in (energy_density, vrho, vsigma, vtau):
        array[hard_zero] = 0.0
        array[~np.isfinite(array)] = 0.0
    return energy_density, vrho, vsigma, vtau


def _correlation_rks(rho: np.ndarray):
    if not CORR:
        ngrid = rho.shape[-1]
        return (
            np.zeros(ngrid),
            np.zeros(ngrid),
            np.zeros(ngrid),
        )
    exc, vxc = libxc.eval_xc(CORR, rho[:4], spin=0, relativity=0, deriv=1)[:2]
    return exc, np.asarray(vxc[0]), np.asarray(vxc[1])


def _correlation_uks(rho_a: np.ndarray, rho_b: np.ndarray):
    ngrid = rho_a.shape[-1]
    if not CORR:
        return (
            np.zeros(ngrid),
            np.zeros((ngrid, 2)),
            np.zeros((ngrid, 3)),
        )
    exc, vxc = libxc.eval_xc(
        CORR, (rho_a[:4], rho_b[:4]), spin=1, relativity=0, deriv=1
    )[:2]
    return exc, np.asarray(vxc[0]), np.asarray(vxc[1])


def eval_xc(
    xc_code,
    rho,
    spin=0,
    relativity=0,
    deriv=1,
    omega=None,
    verbose=None,
):
    """PySCF ``define_xc_`` callback for RKS or UKS."""

    if deriv != 1:
        raise NotImplementedError("The custom functional provides first derivatives only")

    if spin == 0:
        rho_total = np.asarray(rho, dtype=float)
        rho_spin = 0.5 * rho_total
        sigma_spin = 0.25 * np.sum(rho_total[1:4] ** 2, axis=0)
        e_spin, vr_spin, vs_spin, vt_spin = _channel(
            rho_spin[0], sigma_spin, rho_spin[-1]
        )
        n_total = np.maximum(rho_total[0], 1.0e-250)
        corr_exc, corr_vrho, corr_vsigma = _correlation_rks(rho_total)
        exc = 2.0 * e_spin / n_total + corr_exc
        vrho = vr_spin + corr_vrho
        vsigma = 0.5 * vs_spin + corr_vsigma
        vtau = vt_spin
        return exc, (vrho, vsigma, None, vtau), None, None

    rho_a, rho_b = (np.asarray(channel, dtype=float) for channel in rho)
    e_a, vr_a, vs_a, vt_a = _channel(
        rho_a[0], np.sum(rho_a[1:4] ** 2, axis=0), rho_a[-1]
    )
    e_b, vr_b, vs_b, vt_b = _channel(
        rho_b[0], np.sum(rho_b[1:4] ** 2, axis=0), rho_b[-1]
    )
    n_total = np.maximum(rho_a[0] + rho_b[0], 1.0e-250)
    corr_exc, corr_vrho, corr_vsigma = _correlation_uks(rho_a, rho_b)
    exc = (e_a + e_b) / n_total + corr_exc
    vrho = np.column_stack((vr_a, vr_b)) + corr_vrho
    vsigma = np.column_stack(
        (vs_a, np.zeros_like(vs_a), vs_b)
    ) + corr_vsigma
    vtau = np.column_stack((vt_a, vt_b))
    return exc, (vrho, vsigma, None, vtau), None, None


def GKS(mol, grid: int = 5, restricted: bool | None = None):
    """Return a PySCF KS object configured with this implementation."""

    if restricted is None:
        restricted = mol.spin == 0
    mf = dft.RKS(mol) if restricted else dft.UKS(mol)
    mf.xc = ""
    mf.define_xc_(eval_xc, xctype="MGGA")
    mf.grids.level = grid
    mf.conv_tol = 1.0e-8
    mf.max_cycle = 150
    return mf
