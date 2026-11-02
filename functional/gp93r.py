"""Real-space variant GP93R of the regularized GP93 exchange.

GP93R is the form used for self-consistent calculations on real-space grids
(RSDFT/OFDFT).  It differs from the canonical functional in ``mgga_fmix`` in
two ways that matter for the one-orbital-per-spin calculations of this
repository:

* ``F_R(s) = F_mix(s)`` for ``s >= s1``.  For ``s < s1`` the factor is replaced
  by a C2 piecewise cubic with ``F_R(0) = 1``, ``F_R'(0) = 0``,
  ``F_R''(0) = 2 mu_PBE``, joined C2 to ``F_mix`` at ``s1``.  ``F_R''`` is
  piecewise linear on ``K`` uniform intervals, ``|F_R'''| <= D``, and
  ``max F_R''`` is minimized by linear programming.  Defaults
  ``s1 = 0.5``, ``D = 800``, ``K = 400``.
* The log-density window is ``1e-6 .. 1e-4`` instead of ``1e-12 .. 1e-6``.

The third and fourth real-space changes (a frozen gate and the gauge-invariant
orbital von Weizsaecker density) do not change anything for one orbital per
spin, where ``alpha = 0`` exactly; they are not represented here.

The same construction is used by ``tools/gp93/make_gp93r_table.py`` in
RSDFT/OFDFT.  The minimax problem has a degenerate optimum (a plateau of
``F_R''``); different LP solver versions can return tables that differ at the
1e-6 level.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linprog

import mgga_fmix as M


S1_DEFAULT = 0.5
D_DEFAULT = 800.0
K_DEFAULT = 400
WINDOW_LOW = 1.0e-6
WINDOW_HIGH = 1.0e-4


def _fmix_derivatives(s1: float) -> tuple[float, float, float]:
    """Return F_mix, dF/ds, and d2F/ds2 of the canonical spline at s1."""

    u = np.log(s1)
    f0 = float(M._FMIX_SPLINE(u))
    f1 = float(M._FMIX_SPLINE(u, 1))
    f2 = float(M._FMIX_SPLINE(u, 2))
    return f0, f1 / s1, (f2 - f1) / s1**2


def small_s_curve(
    s1: float = S1_DEFAULT,
    D: float = D_DEFAULT,
    K: int = K_DEFAULT,
):
    """Return (F, F'') callables of the minimax curve on [0, s1] and max F''."""

    f0, f1, f2 = _fmix_derivatives(s1)
    nodes = np.linspace(0.0, s1, K + 1)
    h = s1 / K
    # Exact moments of the hat functions: int phi_k ds and int (s1-s) phi_k ds.
    m0 = np.full(K + 1, h)
    m0[0] = m0[-1] = h / 2.0
    m1 = (s1 - nodes) * m0
    m1[0] -= h * h / 6.0
    m1[-1] += h * h / 6.0
    a_eq = np.array(
        [
            np.r_[m0, 0.0],
            np.r_[m1, 0.0],
            np.r_[np.eye(K + 1)[0], 0.0],
            np.r_[np.eye(K + 1)[-1], 0.0],
        ]
    )
    b_eq = [f1, f0 - 1.0, 2.0 * M.MU_PBE, f2]
    difference = (np.eye(K + 1, k=1) - np.eye(K + 1))[:-1]
    a_ub = np.vstack(
        [
            np.c_[np.eye(K + 1), -np.ones(K + 1)],
            np.c_[difference, np.zeros(K)],
            np.c_[-difference, np.zeros(K)],
        ]
    )
    b_ub = np.r_[np.zeros(K + 1), np.full(2 * K, D * h)]
    result = linprog(
        np.r_[np.zeros(K + 1), 1.0],
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * (K + 1) + [(0.0, None)],
    )
    if not result.success:
        raise RuntimeError(f"GP93R minimax LP failed for s1={s1}, D={D}")
    f2n = result.x[:-1]
    f1n = np.zeros(K + 1)
    f0n = np.ones(K + 1)
    for k in range(K):
        f1n[k + 1] = f1n[k] + h * (f2n[k] + f2n[k + 1]) / 2.0
        f0n[k + 1] = (
            f0n[k] + h * f1n[k] + h * h * (2.0 * f2n[k] + f2n[k + 1]) / 6.0
        )

    def value(s):
        s = np.asarray(s, dtype=float)
        k = np.minimum((s / h).astype(int), K - 1)
        x = s - nodes[k]
        slope = (f2n[k + 1] - f2n[k]) / h
        return f0n[k] + f1n[k] * x + f2n[k] * x * x / 2.0 + slope * x**3 / 6.0

    def second(s):
        return np.interp(s, nodes, f2n)

    return value, second, float(result.x[-1])


def make_fr_values(
    s1: float = S1_DEFAULT,
    D: float = D_DEFAULT,
    K: int = K_DEFAULT,
) -> np.ndarray:
    """Return F_R on the canonical ln-s grid of ``mgga_fmix``."""

    grid = M._S_GRID
    values = M._FMIX_SPLINE(np.log(grid)).copy()
    value, _second, _max_f2 = small_s_curve(s1, D, K)
    small = grid < s1
    values[small] = value(grid[small])
    return values


class use_gp93r:
    """Context manager that switches ``mgga_fmix`` to GP93R and restores it."""

    def __init__(self, s1: float = S1_DEFAULT, D: float = D_DEFAULT):
        self.s1 = s1
        self.D = D

    def __enter__(self):
        self._saved = (
            M._FMIX_SPLINE,
            M.DENSITY_WINDOW_LOW,
            M.DENSITY_WINDOW_HIGH,
            M.DENSITY_CUTOFF,
        )
        values = make_fr_values(self.s1, self.D)
        M._FMIX_SPLINE = CubicSpline(np.log(M._S_GRID), values)
        M.DENSITY_WINDOW_LOW = WINDOW_LOW
        M.DENSITY_WINDOW_HIGH = WINDOW_HIGH
        M.DENSITY_CUTOFF = WINDOW_LOW
        return self

    def __exit__(self, *exc):
        (
            M._FMIX_SPLINE,
            M.DENSITY_WINDOW_LOW,
            M.DENSITY_WINDOW_HIGH,
            M.DENSITY_CUTOFF,
        ) = self._saved
        return False
