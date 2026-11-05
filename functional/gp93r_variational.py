"""Fully variational and frozen-gate tau-dependent GP93R exchange.

Transcription of F. Imoto, "Fully Variational tau-Dependent GP93R:
Spin-Resolved Exchange Derivatives and Generalized Kohn-Sham Equations"
(working note, 2026-11-05; calculations/gp93r_variational/
GP93R_variational_GKS_note.tex, also docs/notes/ of RSOFDFT_DEVELOP):

    parent variables  m = 2 n_s,  q = 4 gamma_s,  t = 2 tau_s
    s = sqrt(q) / (2 C1 m^{4/3}),  t_W = q / (8 m)  (density based),
    t_U = C_F m^{5/3},  alpha = (t - t_W) / t_U,  G = exp[-(alpha/w)^2]
    f_s = B [P(s) + X(n_s) G(alpha) D(s)],  B = A_x m^{4/3} / 2,  D = F_R - P
    v_n, v_gamma, v_tau of the note's Eqs. (vn)-(vt)

The terms are written out separately (B, H, H_s, H_alpha, s_m, s_q, alpha_m,
alpha_q, alpha_t) so that this module checks ``mgga_fmix._channel`` under
``gp93r.use_gp93r`` independently.

Gate modes
    "variational" : every derivative of the note; with the same cutoff this is
                    the canonical callback under ``use_gp93r``
    "frozen"      : G at its value, the H_alpha terms and v_tau omitted (the
                    first variation omitted by a frozen gate, note Sec. 4); the
                    potential is then not the derivative of the energy
    "one"         : G = 1

Hard zero of a whole spin channel
    "window" (default): n_s < 1e-6, as ``gp93r.use_gp93r`` sets DENSITY_CUTOFF
    "rsdft"           : n_s < 1e-12, as the RSDFT/OFDFT implementation
The note's energy (eq:energy) has no hard cutoff; below the window the two
conventions differ by the PBE exchange of the low-density tail.

Floors and clips are those of ``mgga_fmix`` (parameters.json); the clipped
energy is differentiated piecewise (note Sec. 6, item 2).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

import gp93r
import mgga_fmix as M

A_X = M.A_X
C1 = M.C1
C_F = M.CF


class GP93RModel:
    """GP93R channel with a chosen gate treatment and hard-zero convention."""

    def __init__(self, gate: str = "variational", hard_zero: str = "window",
                 s1: float = gp93r.S1_DEFAULT, D: float = gp93r.D_DEFAULT):
        if gate not in ("variational", "frozen", "one"):
            raise ValueError(f"unknown gate mode {gate!r}")
        if hard_zero not in ("window", "rsdft"):
            raise ValueError(f"unknown hard-zero convention {hard_zero!r}")
        self.gate = gate
        self.hard_zero = hard_zero
        self.n_low = gp93r.WINDOW_LOW
        self.n_high = gp93r.WINDOW_HIGH
        self.cutoff = self.n_low if hard_zero == "window" else M.DENSITY_WINDOW_LOW
        self.w = M.ALPHA_WIDTH
        self.cap = float(M.NUMERICAL["switch_exponent_cap"])
        self.values = gp93r.make_fr_values(s1, D)
        self._spl = CubicSpline(np.log(M._S_GRID), self.values)

    # enhancement factors -----------------------------------------------------
    def R(self, s):
        x = np.log(s)
        return self._spl(x), self._spl(x, 1) / s

    @staticmethod
    def P(s):
        d = 1.0 + M.MU_PBE * s * s / M.KAPPA_PBE
        return 1.0 + M.KAPPA_PBE - M.KAPPA_PBE / d, 2.0 * M.MU_PBE * s / d**2

    def X(self, n):
        L = np.log(self.n_high / self.n_low)
        u = np.clip(np.log(np.maximum(n, M.DENSITY_FLOOR) / self.n_low) / L, 0.0, 1.0)
        inside = (n > self.n_low) & (n < self.n_high)
        return (u * u * (3.0 - 2.0 * u),
                6.0 * u * (1.0 - u) / (L * np.maximum(n, M.DENSITY_FLOOR)) * inside)

    # one spin channel --------------------------------------------------------
    def channel(self, n, gamma, tau):
        """f_s (per volume) and v_n, v_gamma, v_tau of one spin channel."""
        n = np.asarray(n, float)
        gamma = np.asarray(gamma, float)
        tau = np.asarray(tau, float)
        m_raw, q_raw, t = 2.0 * n, 4.0 * gamma, np.maximum(2.0 * tau, 0.0)
        m = np.maximum(m_raw, M.SPIN_DENSITY_FLOOR)
        q = np.maximum(q_raw, M.SIGMA_FLOOR)
        act_m = m_raw > M.SPIN_DENSITY_FLOOR
        act_q = q_raw > M.SIGMA_FLOOR
        s_raw = np.sqrt(q) / (2.0 * C1 * m ** (4.0 / 3.0))
        s = np.clip(s_raw, M.S_MIN, M.S_MAX)
        act_s = (s_raw > M.S_MIN) & (s_raw < M.S_MAX)

        P, dP = self.P(s)
        R, dR = self.R(s)
        D, dD = R - P, dR - dP

        tW = q / (8.0 * m)
        tU = C_F * m ** (5.0 / 3.0)
        a_raw = (t - tW) / tU
        alpha = np.maximum(a_raw, 0.0)
        act_a = a_raw > 0.0
        arg = (alpha / self.w) ** 2
        G = np.exp(-np.minimum(arg, self.cap))
        dG = -2.0 * alpha / self.w**2 * G * (arg < self.cap)
        if self.gate == "one":
            G, dG = np.ones_like(G), np.zeros_like(G)
        X, dX = self.X(n)

        B = 0.5 * A_X * m ** (4.0 / 3.0)
        H = P + X * G * D
        f = B * H
        H_s = dP + X * G * dD
        H_a = X * dG * D
        if self.gate == "frozen":
            H_a = np.zeros_like(H_a)
        s_m = -4.0 * s / (3.0 * m) * act_s * act_m
        s_q = s / (2.0 * q) * act_s * act_q
        a_m = (tW / (m * tU) - 5.0 * alpha / (3.0 * m)) * act_a * act_m
        a_q = -1.0 / (8.0 * m * tU) * act_a * act_q * act_m
        a_t = 1.0 / tU * act_a * act_m

        vn = 2.0 * (4.0 * B / (3.0 * m) * H * act_m + B * (H_s * s_m + H_a * a_m)) + B * D * G * dX
        vg = 4.0 * B * (H_s * s_q + H_a * a_q)
        vt = 2.0 * B * H_a * a_t
        zero = n < self.cutoff
        return tuple(np.where(zero | ~np.isfinite(a), 0.0, a) for a in (f, vn, vg, vt))

    # PySCF -------------------------------------------------------------------
    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        """PySCF ``define_xc_`` callback (exchange only)."""
        if deriv > 1:
            raise NotImplementedError("first derivatives only")
        if spin == 0:
            r = np.asarray(rho, float)
            f, vn, vg, vt = self.channel(0.5 * r[0], 0.25 * np.sum(r[1:4] ** 2, axis=0), 0.5 * r[-1])
            ntot = np.maximum(r[0], 1e-250)
            return 2.0 * f / ntot, (vn, 0.5 * vg, None, vt), None, None
        ra, rb = (np.asarray(c, float) for c in rho)
        fa, vna, vga, vta = self.channel(ra[0], np.sum(ra[1:4] ** 2, axis=0), ra[-1])
        fb, vnb, vgb, vtb = self.channel(rb[0], np.sum(rb[1:4] ** 2, axis=0), rb[-1])
        ntot = np.maximum(ra[0] + rb[0], 1e-250)
        return ((fa + fb) / ntot,
                (np.column_stack((vna, vnb)), np.column_stack((vga, np.zeros_like(vga), vgb)),
                 None, np.column_stack((vta, vtb))), None, None)

    def gks(self, mol, grid: int = 5, restricted: bool | None = None):
        from pyscf import dft
        if restricted is None:
            restricted = mol.spin == 0
        mf = dft.RKS(mol) if restricted else dft.UKS(mol)
        mf.xc = ""
        mf.define_xc_(self.eval_xc, xctype="MGGA")
        mf.grids.level = grid
        return mf

    def numint(self):
        from pyscf import dft
        ni = dft.numint.NumInt()
        return ni.libxc.define_xc_(ni, self.eval_xc, xctype="MGGA")
