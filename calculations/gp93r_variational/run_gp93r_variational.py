#!/usr/bin/env python3
"""Fully variational tau-dependent GP93R: derivative, GKS-matrix and SCF checks.

Scientific question: does differentiating the iso-orbital gate of GP93R (the
standard density-based tau_W meta-GGA of the variational GKS note) give a
usable self-consistent functional, and what does a frozen gate omit?

Three parts, all exchange only (functional/gp93r_variational.py):
  1. point checks: the note's analytic v_n, v_gamma, v_tau against centred
     differences of the same energy density, and against the canonical
     callback mgga_fmix._channel under gp93r.use_gp93r;
  2. GKS matrix: for an occupied-virtual rotation C(theta) = C exp(theta K) of
     PBE orbitals, [E_x(theta) - E_x(-theta)]/(2 theta) against Tr[V_x dD/dtheta];
  3. matched SCF: frozen gate and fully variational gate with the same F_R,
     window, w, basis, grid and SCF settings, plus HF in the same basis, and the
     region where the GKS principal coefficient 1 + v_tau is negative.

Outputs (results/gp93r_variational/):
  point_checks.csv, gks_directional.csv, scf_frozen_vs_variational.csv,
  vtau_negative_regions.csv, gp93r_variational_metadata.json
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "functional"))

import gp93r  # noqa: E402
import gp93r_variational as GV  # noqa: E402
import mgga_fmix as M  # noqa: E402

INPUT = json.loads((ROOT / "inputs" / "gp93r_variational.json").read_text(encoding="utf-8"))
if int(INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/gp93r_variational.json schema_version")
OUTPUT_DIR = ROOT / "results" / "gp93r_variational"
HARTREE_EV = 27.211386245988


def write_csv(name, rows):
    with open(OUTPUT_DIR / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- part 1
def point_classes(cfg):
    rng = np.random.default_rng(cfg["seed"])
    k = cfg["points_per_class"]
    spec = [  # label, log10 n_s range, log10 s range, alpha range
        ("iso-orbital, window open", (-4, 0.5), (-2, 1.5), (0.0, 0.25)),
        ("alpha != 0 (gate edge)", (-4, 0.5), (-2, 1.5), (0.02, 0.35)),
        ("alpha large (PBE)", (-4, 0.5), (-2, 1.5), (0.5, 3.0)),
        ("inside density window", (-5.95, -4.05), (-1, 1.5), (0.0, 0.3)),
        ("join s1 = 0.5", (-3, 0.5), (np.log10(0.45), np.log10(0.55)), (0.0, 0.3)),
        ("small q (s near s_min)", (-3, 0.5), (np.log10(1.3e-4), -2), (0.0, 0.3)),
        ("large s (negative branch)", (-4, -1), (1, 2.5), (0.0, 0.3)),
        ("tau < tau_W (alpha clipped)", (-4, 0.5), (-2, 1.5), (-0.5, -0.01)),
        ("below window (PBE only)", (-11.5, -6.2), (-1, 1.0), (0.0, 0.3)),
    ]
    out = []
    for label, ln, ls, la in spec:
        n = 10.0 ** rng.uniform(*ln, k)
        s = 10.0 ** rng.uniform(*ls, k)
        al = rng.uniform(*la, k)
        m = 2 * n
        q = (2 * GV.C1 * m ** (4 / 3) * s) ** 2
        t = np.maximum(q / (8 * m) + al * GV.C_F * m ** (5 / 3), 0.0)
        out.append((label, n, q / 4, t / 2, s))
    return out


def point_checks(cfg):
    var = GV.GP93RModel("variational", "window")
    rsd = GV.GP93RModel("variational", "rsdft")
    rows, worst_fd, worst_can = [], 0.0, 0.0
    for label, n, g, t, s in point_classes(cfg):
        f, vn, vg, vt = var.channel(n, g, t)
        with gp93r.use_gp93r():
            fc, vnc, vgc, vtc = M._channel(n, g, t)
        can = max(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300 + 1e-12 * np.abs(b).max()))
                  if np.any(b != 0) else np.max(np.abs(a))
                  for a, b in ((f, fc), (vn, vnc), (vg, vgc), (vt, vtc)))
        worst_can = max(worst_can, can)
        # centred 4th-order differences; steps move alpha by <= fd_alpha_step_over_w * w
        m = 2 * n
        h = np.minimum(cfg["fd_relative_step"],
                       cfg["fd_alpha_step_over_w"] * var.w / np.maximum(2 * t / (GV.C_F * m ** (5 / 3)), 1e-300))

        def fd(fun, x, dx):
            d1 = (fun(x + dx) - fun(x - dx)) / (2 * dx)
            d2 = (fun(x + dx / 2) - fun(x - dx / 2)) / dx
            return (4 * d2 - d1) / 3
        E = lambda a, b, c: var.channel(a, b, c)[0]  # noqa: E731
        fdn = fd(lambda x: E(x, g, t), n, h * n)
        fdg = fd(lambda x: E(n, x, t), g, h * g)
        fdt = fd(lambda x: E(n, g, x), t, h * np.maximum(t, 1e-300))
        Bm = 0.5 * abs(GV.A_X) * m ** (4 / 3)
        scale = Bm * (1.0 + np.abs(f) / Bm)
        errs = [np.abs(a - b) * x / scale for a, b, x in ((vn, fdn, n), (vg, fdg, g), (vt, fdt, t))]
        sel = s <= cfg["fd_pass_s_max"]
        e_sel = [float(np.max(e[sel])) if sel.any() else float("nan") for e in errs]
        e_all = [float(np.max(e)) for e in errs]
        if sel.any():
            worst_fd = max(worst_fd, *e_sel)
        f_r = rsd.channel(n, g, t)[0]
        rows.append({
            "class": label, "points": int(n.size),
            "max_rel_diff_vs_canonical": f"{can:.3e}",
            "fd_err_vn_s_le_5": f"{e_sel[0]:.3e}", "fd_err_vgamma_s_le_5": f"{e_sel[1]:.3e}",
            "fd_err_vtau_s_le_5": f"{e_sel[2]:.3e}", "fd_err_max_all_s": f"{max(e_all):.3e}",
            "points_zeroed_window_not_rsdft": int(np.sum((f == 0) & (f_r != 0))),
        })
    write_csv("point_checks.csv", rows)
    return dict(worst_fd_s_le_5=worst_fd, worst_rel_vs_canonical=worst_can,
                pass_fd=worst_fd < cfg["fd_tolerance"], pass_canonical=worst_can < cfg["canonical_tolerance"])


# ---------------------------------------------------------------- parts 2, 3
def molecule(name, basis):
    from pyscf import gto
    g = INPUT["geometries_angstrom"][name]
    return gto.M(atom=g["atom"], basis=basis, spin=g["spin"], verbose=0)


def gks_directional(cfg):
    from pyscf import dft
    rows, worst = [], 0.0
    h1, h2 = cfg["theta"]
    for name in cfg["systems"]:
        mol = molecule(name, cfg["basis"])
        uks = mol.spin != 0
        mf = (dft.UKS(mol) if uks else dft.RKS(mol)).set(xc="PBE")
        mf.grids.level = cfg["grid_level"]
        mf.kernel()
        Cs = list(mf.mo_coeff) if uks else [mf.mo_coeff]
        fs = list(mf.mo_occ) if uks else [mf.mo_occ]
        for mode in ("r2scan", "variational", "frozen"):
            rng = np.random.default_rng(cfg["rotation_seed"])
            Ks = []
            for C, f in zip(Cs, fs):
                K = np.zeros((C.shape[1],) * 2)
                K[np.ix_(f > 0, f == 0)] = rng.normal(size=(int((f > 0).sum()), int((f == 0).sum())))
                K -= K.T
                nk = np.linalg.norm(K)
                Ks.append(K / nk if nk > 0 else K)
            if mode == "r2scan":
                ni, xc = dft.numint.NumInt(), "MGGA_X_R2SCAN"
            else:
                ni, xc = GV.GP93RModel(mode).numint(), ""

            def dm(th):
                d = [C @ expm(th * K) @ np.diag(f) @ expm(th * K).T @ C.T for C, K, f in zip(Cs, Ks, fs)]
                return np.array(d) if uks else d[0]

            def ev(d):
                return (ni.nr_uks if uks else ni.nr_rks)(mol, mf.grids, xc, d)

            _, e0, v0 = ev(dm(0.0))
            dD = [C @ (K @ np.diag(f) - np.diag(f) @ K) @ C.T for C, K, f in zip(Cs, Ks, fs)]
            gks = (sum(np.einsum("ij,ji", v, d) for v, d in zip(v0, dD)) if uks
                   else np.einsum("ij,ji", v0, dD[0]))
            d1 = (ev(dm(h1))[1] - ev(dm(-h1))[1]) / (2 * h1)
            d2 = (ev(dm(h2))[1] - ev(dm(-h2))[1]) / (2 * h2)
            fd = (4 * d2 - d1) / 3
            rel = abs(fd - gks) / max(abs(fd), 1e-14)
            if mode != "frozen":
                worst = max(worst, rel)
            rows.append({"system": name, "exchange": "r2SCAN-x" if mode == "r2scan" else f"GP93R-x {mode}",
                         "E_x_Ha": f"{e0:.10f}", "dE_dtheta_fd": f"{fd:.10e}",
                         "dE_dtheta_gks_matrix": f"{gks:.10e}", "rel_diff": f"{rel:.3e}"})
    write_csv("gks_directional.csv", rows)
    return dict(worst_rel_variational_and_control=worst, pass_gks=worst < cfg["tolerance"])


def homo(mf):
    e, f = mf.mo_energy, mf.mo_occ
    if np.ndim(e) == 2:
        return max(np.max(ei[fi > 0]) for ei, fi in zip(e, f) if np.any(fi > 0))
    return float(np.max(e[f > 0]))


def spin_channels(mf):
    from pyscf import dft
    ni = dft.numint.NumInt()
    ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
    dm = mf.make_rdm1()
    dms = list(dm) if np.ndim(dm) == 3 else [0.5 * dm, 0.5 * dm]
    for d in dms:
        r = ni.eval_rho(mf.mol, ao, d, xctype="MGGA")
        yield r[0], np.sum(r[1:4] ** 2, axis=0), r[-1]


def scf(cfg):
    from pyscf import dft, scf as pscf
    rows, regions = [], []
    var = GV.GP93RModel("variational")
    for name in cfg["systems"]:
        mol = molecule(name, cfg["basis"])
        uks = mol.spin != 0
        hf = (pscf.UHF(mol) if uks else pscf.RHF(mol)).run(conv_tol=1e-10)
        dmh, vk = hf.make_rdm1(), hf.get_k()
        ex_hf = (-0.5 * sum(np.einsum("ij,ji", k, d) for k, d in zip(vk, dmh)) if uks
                 else -0.25 * np.einsum("ij,ji", vk, dmh))
        start = (dft.UKS(mol) if uks else dft.RKS(mol)).set(xc="PBE,", conv_tol=1e-10)
        start.grids.level = cfg["grid_level"]
        start.kernel()
        for gate in cfg["gates"]:
            model = GV.GP93RModel(gate)
            mf = model.gks(mol, cfg["grid_level"])
            mf.conv_tol, mf.max_cycle = cfg["conv_tol"], cfg["max_cycle"]
            e = mf.kernel(dm0=start.make_rdm1())
            ex = (mf._numint.nr_uks if uks else mf._numint.nr_rks)(mol, mf.grids, "", mf.make_rdm1())[1]
            vmin = 1.0
            neg = dict(points=0, electrons=0.0)
            ranges = []
            w = mf.grids.weights
            for n, g, t in spin_channels(mf):
                f, _, _, vt = var.channel(n, g, t)
                sel = n > gp93r.WINDOW_LOW
                if not sel.any():
                    continue
                vmin = min(vmin, float(np.min(1 + vt[sel])))
                bad = sel & (1 + vt < 0)
                if bad.any():
                    m = 2 * n[bad]
                    s = np.sqrt(4 * g[bad]) / (2 * GV.C1 * m ** (4 / 3))
                    al = np.maximum((2 * t[bad] - 4 * g[bad] / (8 * m)) / (GV.C_F * m ** (5 / 3)), 0)
                    Rv, _ = var.R(np.clip(s, M.S_MIN, M.S_MAX))
                    Pv, _ = var.P(np.clip(s, M.S_MIN, M.S_MAX))
                    neg["points"] += int(bad.sum())
                    neg["electrons"] += float(np.sum(w[bad] * n[bad]))
                    ranges.append((n[bad].min(), n[bad].max(), s.min(), s.max(), al.min(), al.max(),
                                   bool(np.all(Rv - Pv < 0))))
            rows.append({"system": name, "gate": gate, "converged": bool(mf.converged), "cycles": int(mf.cycles),
                         "E_tot_Ha": f"{e:.8f}", "E_x_Ha": f"{ex:.8f}", "E_x_HF_Ha": f"{ex_hf:.8f}",
                         "E_x_minus_E_x_HF_Ha": f"{ex - ex_hf:.6f}", "E_tot_HF_Ha": f"{hf.e_tot:.8f}",
                         "minus_eps_HOMO_eV": f"{-homo(mf) * HARTREE_EV:.4f}",
                         "minus_eps_HOMO_HF_eV": f"{-homo(hf) * HARTREE_EV:.4f}",
                         "min_1_plus_vtau_variational": f"{vmin:.4e}"})
            if gate == "frozen":
                if ranges:
                    r = np.array([x[:6] for x in ranges])
                    regions.append({"system": name, "density": "frozen-gate SCF",
                                    "points": neg["points"], "electrons": f"{neg['electrons']:.4f}",
                                    "n_s_min": f"{r[:, 0].min():.2e}", "n_s_max": f"{r[:, 1].max():.2e}",
                                    "s_min": f"{r[:, 2].min():.3f}", "s_max": f"{r[:, 3].max():.3f}",
                                    "alpha_min": f"{r[:, 4].min():.4f}", "alpha_max": f"{r[:, 5].max():.4f}",
                                    "all_on_D_negative": all(x[6] for x in ranges),
                                    "min_1_plus_vtau": f"{vmin:.4e}"})
                else:
                    regions.append({"system": name, "density": "frozen-gate SCF", "points": 0,
                                    "electrons": "0.0000", "n_s_min": "", "n_s_max": "", "s_min": "",
                                    "s_max": "", "alpha_min": "", "alpha_max": "",
                                    "all_on_D_negative": "", "min_1_plus_vtau": f"{vmin:.4e}"})
            print(f"  {name:4s} {gate:11s} converged={mf.converged} cycles={mf.cycles} E={e:.8f}", flush=True)
    write_csv("scf_frozen_vs_variational.csv", rows)
    write_csv("vtau_negative_regions.csv", regions)


def main():
    import pyscf
    import scipy
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("part 1: point checks", flush=True)
    p1 = point_checks(INPUT["point_checks"])
    print(f"  {p1}", flush=True)
    print("part 2: GKS matrix under orbital rotations", flush=True)
    p2 = gks_directional(INPUT["gks_directional"])
    print(f"  {p2}", flush=True)
    print("part 3: matched frozen / variational SCF", flush=True)
    scf(INPUT["scf"])
    meta = {
        "input": "inputs/gp93r_variational.json",
        "input_settings": INPUT,
        "specification": "calculations/gp93r_variational/GP93R_variational_GKS_note.tex",
        "functional": "functional/gp93r_variational.py (GP93R of functional/gp93r.py)",
        "hard_zero_convention": "window (n_s < 1e-6, as gp93r.use_gp93r); RSDFT uses 1e-12",
        "point_checks": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in p1.items()},
        "gks_directional": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in p2.items()},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pyscf": pyscf.__version__},
    }
    (OUTPUT_DIR / "gp93r_variational_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    ok = p1["pass_fd"] and p1["pass_canonical"] and p2["pass_gks"]
    print("checks passed" if ok else "CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
