"""Unregularized hydrogen-exact GP93/PBE GGA building block.

This module is intentionally separate from :mod:`mgga_fmix`.  It contains no
density window and no upper large-gradient cutoff.  The GP93 branch is evaluated
from the regular Frobenius continuation up to ``z=0``, the Green-function
solution for moderate positive ``z``, and its exponentially accurate
large-``z`` continuation.  It is used only for the formal hydrogen and
hydrogenic-ion benchmarks and as the building block of the final regularized
functional.
"""

from __future__ import annotations

import json
from math import gamma, pi
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

import gp93_w


A_X = -(3.0 / 4.0) * (3.0 / pi) ** (1.0 / 3.0)
C1 = (3.0 * pi * pi) ** (1.0 / 3.0)
K_GP93 = 8.0 * pi / 3.0
LOG_6PI = np.log(6.0 * pi)
PARAMETERS = json.loads(
    (Path(__file__).with_name("parameters.json")).read_text(encoding="utf-8")
)
NUMERICAL = PARAMETERS["numerical_implementation"]
MU_PBE = 0.21951
KAPPA_PBE = 0.804
CM = gp93_w.IM(float(NUMERICAL["cm_evaluation_endpoint"]))
GAMMA_13 = gamma(1.0 / 3.0)
DEFAULT_S0 = float(PARAMETERS["selected_s0"])
DEFAULT_P = int(PARAMETERS["selected_p"])
CURRENT_S0 = DEFAULT_S0
CURRENT_P = DEFAULT_P
C0 = float(PARAMETERS["gp93_c0"])


def set_parameters(s0: float, p: int) -> None:
    """Set the crossover parameters used by the PySCF callback.

    Stage drivers call this explicitly with the calibrated parameters.  The
    default remains the canonical selection in ``functional/parameters.json``.
    """

    global CURRENT_S0, CURRENT_P
    CURRENT_S0 = float(s0)
    CURRENT_P = int(p)


def _series_scalar(z: float) -> float:
    return float(np.asarray(gp93_w.w_series(float(z), C0)).reshape(-1)[0])


def _w_moderate(z: float) -> float:
    if z <= 0.0:
        return _series_scalar(z)
    return float(gp93_w.w_exact(float(z)))


W_SPLINE_INPUT = NUMERICAL["w_spline"]
_Z_SPLINE_GRID = np.linspace(
    float(W_SPLINE_INPUT["z_min"]),
    float(W_SPLINE_INPUT["z_max"]),
    int(W_SPLINE_INPUT["points"]),
)
_W_SPLINE = CubicSpline(
    _Z_SPLINE_GRID,
    np.array([_w_moderate(value) for value in _Z_SPLINE_GRID]),
)


def w_and_derivative(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the continued GP93 ``w(z)`` and ``dw/dz``."""

    z = np.asarray(z, dtype=float)
    value = np.empty_like(z)
    derivative = np.empty_like(z)

    low = z < _Z_SPLINE_GRID[0]
    middle = (z >= _Z_SPLINE_GRID[0]) & (z <= _Z_SPLINE_GRID[-1])
    high = z > _Z_SPLINE_GRID[-1]

    # The GP93 branch is irrelevant at very small s because the PBE switch is
    # closed.  Holding the regular endpoint avoids evaluating the Frobenius
    # polynomial far outside its useful interval.
    value[low] = _W_SPLINE(_Z_SPLINE_GRID[0])
    derivative[low] = 0.0
    value[middle] = _W_SPLINE(z[middle])
    derivative[middle] = _W_SPLINE(z[middle], 1)

    if np.any(high):
        zh = z[high]
        # Algebraic large-z expansion of U(-2/3,2,z).  This avoids thousands
        # of costly scalar special-function evaluations on DFT grids.
        coefficients = (1.0, -10.0 / 9.0, -10.0 / 81.0, 40.0 / 2187.0, -280.0 / 19683.0)
        u = np.zeros_like(zh)
        du_dz = np.zeros_like(zh)
        for order, coefficient in enumerate(coefficients):
            exponent = 2.0 / 3.0 - order
            u += coefficient * zh**exponent
            du_dz += coefficient * exponent * zh ** (exponent - 1.0)
        prefactor = GAMMA_13 * CM / 4.0
        value[high] = 0.5 - prefactor * u
        derivative[high] = -prefactor * du_dz
    return value, derivative


def pbe_with_derivative(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = 1.0 + MU_PBE * s * s / KAPPA_PBE
        value = 1.0 + KAPPA_PBE - KAPPA_PBE / denominator
        derivative = 2.0 * MU_PBE * s / denominator**2
    derivative[~np.isfinite(derivative)] = 0.0
    return value, derivative


def enhancement_with_derivative(
    s: np.ndarray,
    s0: float | None = None,
    p: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the H-exact mixed enhancement and ``dF/ds``."""

    if s0 is None:
        s0 = CURRENT_S0
    if p is None:
        p = CURRENT_P
    s = np.maximum(
        np.asarray(s, dtype=float),
        float(NUMERICAL["hydrogen_exact_s_floor"]),
    )
    z = 3.0 * np.log(s) + LOG_6PI
    w, dw_dz = w_and_derivative(z)
    f_gp93 = K_GP93 * s * w
    df_gp93 = K_GP93 * (w + 3.0 * dw_dz)
    f_pbe, df_pbe = pbe_with_derivative(s)

    with np.errstate(over="ignore", invalid="ignore"):
        switch_argument_raw = (s / float(s0)) ** int(p)
    switch_cap = float(NUMERICAL["switch_exponent_cap"])
    switch_argument = np.clip(switch_argument_raw, 0.0, switch_cap)
    exp_argument = np.exp(-switch_argument)
    chi = 1.0 - exp_argument
    dchi_ds = np.where(
        switch_argument_raw < switch_cap,
        exp_argument * int(p) * switch_argument_raw / s,
        0.0,
    )
    value = (1.0 - chi) * f_pbe + chi * f_gp93
    derivative = (
        (1.0 - chi) * df_pbe
        + chi * df_gp93
        + dchi_ds * (f_gp93 - f_pbe)
    )
    return value, derivative


def _channel(
    n_sigma: np.ndarray,
    sigma_sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sigma = np.asarray(n_sigma, dtype=float)
    sigma_sigma = np.asarray(sigma_sigma, dtype=float)
    # A numerical floor is needed because m**(4/3) underflows before a
    # Gaussian quadrature grid stops reporting formal tail points.  It is far
    # below any density carrying measurable energy or potential weight.
    m = np.maximum(
        2.0 * n_sigma,
        float(NUMERICAL["hydrogen_exact_density_floor"]),
    )
    sigma_m = np.maximum(
        4.0 * sigma_sigma,
        float(NUMERICAL["hydrogen_exact_sigma_floor"]),
    )
    s = np.sqrt(sigma_m) / (2.0 * C1 * m ** (4.0 / 3.0))
    f, df_ds = enhancement_with_derivative(s)
    phi = A_X * m ** (4.0 / 3.0)
    energy_density = 0.5 * phi * f
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ds_dm = -(4.0 / 3.0) * s / m
        ds_dsigma_m = s / (2.0 * sigma_m)
        denergy_dm = 0.5 * (
            (4.0 / 3.0) * (phi / m) * f + phi * df_ds * ds_dm
        )
        denergy_dsigma_m = 0.5 * phi * df_ds * ds_dsigma_m
        vrho = 2.0 * denergy_dm
        vsigma = 4.0 * denergy_dsigma_m
    for array in (energy_density, vrho, vsigma):
        array[~np.isfinite(array)] = 0.0
    return energy_density, vrho, vsigma


def eval_xc(
    xc_code,
    rho,
    spin=0,
    relativity=0,
    deriv=1,
    omega=None,
    verbose=None,
):
    """PySCF GGA callback for the unregularized building block."""

    if deriv != 1:
        raise NotImplementedError("The custom functional provides first derivatives only")
    if spin == 0:
        rho_total = np.asarray(rho, dtype=float)
        rho_spin = 0.5 * rho_total
        sigma_spin = 0.25 * np.sum(rho_total[1:4] ** 2, axis=0)
        energy_spin, vr_spin, vs_spin = _channel(
            rho_spin[0], sigma_spin
        )
        n_total = np.maximum(rho_total[0], 1.0e-250)
        return (
            2.0 * energy_spin / n_total,
            (vr_spin, 0.5 * vs_spin, None, None),
            None,
            None,
        )

    rho_a, rho_b = (np.asarray(channel, dtype=float) for channel in rho)
    e_a, vr_a, vs_a = _channel(
        rho_a[0], np.sum(rho_a[1:4] ** 2, axis=0)
    )
    e_b, vr_b, vs_b = _channel(
        rho_b[0], np.sum(rho_b[1:4] ** 2, axis=0)
    )
    n_total = np.maximum(rho_a[0] + rho_b[0], 1.0e-250)
    vrho = np.column_stack((vr_a, vr_b))
    vsigma = np.column_stack((vs_a, np.zeros_like(vs_a), vs_b))
    return (e_a + e_b) / n_total, (vrho, vsigma, None, None), None, None
