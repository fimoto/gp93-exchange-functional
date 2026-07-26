#!/usr/bin/env python3
"""Radial exchange-potential diagnostics for H, He, and Na.

The H panel separates the unclipped hydrogen-exact GP93 building block from
the regularized functional used self-consistently.  The He and Na panels
evaluate both regularized Fmix and PBE exchange on one common HF density, so
their distances from the self-interaction-free Hartree reference are directly
comparable.  These panels test the intermediate radial range, not asymptotic
exactness of the practical functional.
"""

from __future__ import annotations

import csv
import json
import sys
from math import pi
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "functional"
OUTPUT_DIR = ROOT / "results" / "potentials"
INPUT_PATH = ROOT / "inputs" / "potentials.json"
INPUT = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
if int(INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/potentials.json schema_version")
if not (SOURCE / "mgga_fmix.py").exists():
    raise FileNotFoundError(f"Functional source not found under {SOURCE}")
sys.path.insert(0, str(SOURCE))

from pyscf import gto, lib, scf  # noqa: E402
from pyscf.dft import numint  # noqa: E402

import mgga_fmix as G  # noqa: E402

FMIX_S0 = float(G.PARAMETERS["selected_s0"])
FMIX_P = int(G.PARAMETERS["selected_p"])
SYSTEM_INPUTS = INPUT["systems"]


def _sandbox_memory_workaround() -> None:
    try:
        Path("/proc/self/statm").read_text(encoding="ascii")
    except OSError:
        lib.current_memory = lambda *args, **kwargs: (0, 0)


_sandbox_memory_workaround()
lib.num_threads(1)
NI = numint.NumInt()
K_F_SPIN = 2.0 * (3.0 * pi * pi) ** (1.0 / 3.0)
A_X = -(3.0 / 4.0) * (3.0 / pi) ** (1.0 / 3.0)


def grid_from_input(spec: dict[str, float | int]) -> np.ndarray:
    return np.linspace(
        float(spec["start"]),
        float(spec["stop"]),
        int(spec["points"]),
    )


def derivative(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Differentiate a smooth radial quantity without amplifying grid noise."""

    return CubicSpline(grid, values).derivative()(grid)


def density_on_axis(mol, dm, r: np.ndarray) -> np.ndarray:
    ao = NI.eval_ao(mol, np.column_stack((r, np.zeros_like(r), np.zeros_like(r))))
    dm_array = np.asarray(dm)
    if dm_array.ndim == 3:
        dm_array = dm_array[0] + dm_array[1]
    return np.einsum("pi,ij,pj->p", ao, dm_array, ao)


def density_and_gradient_on_axis(mol, dm, r: np.ndarray):
    """Return spherical atomic density and analytic radial derivative."""

    ao = NI.eval_ao(
        mol,
        np.column_stack((r, np.zeros_like(r), np.zeros_like(r))),
        deriv=1,
    )
    dm_array = np.asarray(dm)
    if dm_array.ndim == 3:
        dm_array = dm_array[0] + dm_array[1]
    value = np.einsum("pi,ij,pj->p", ao[0], dm_array, ao[0])
    grad = 2.0 * np.einsum("pi,ij,pj->p", ao[1], dm_array, ao[0])
    return value, grad


def radial_divergence(coefficient: np.ndarray, grad_n: np.ndarray, r: np.ndarray):
    return derivative(r * r * coefficient * grad_n, r) / (r * r)


def implemented_channel_vx(
    n: np.ndarray, grad_n: np.ndarray, r: np.ndarray
) -> np.ndarray:
    sigma = grad_n * grad_n
    tau = sigma / np.maximum(8.0 * n, 1.0e-300)
    _, vrho, vsigma, vtau = G._channel(n, sigma, tau)
    if (
        np.nanmax(np.abs(vtau))
        > float(INPUT["maximum_abs_iso_orbital_vtau"])
    ):
        raise RuntimeError("Expected iso-orbital v_tau to vanish in this diagnostic")
    return vrho - 2.0 * radial_divergence(vsigma, grad_n, r)


def hydrogen_exact_gp93_vx(
    n: np.ndarray, grad_n: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Analytic GP93 identity on the exact spin-up hydrogen density."""

    return -(1.0 - np.exp(-2.0 * r) * (1.0 + r)) / r


def pbe_channel_vx(
    n: np.ndarray, grad_n: np.ndarray, r: np.ndarray
) -> np.ndarray:
    sigma = np.maximum(grad_n * grad_n, 1.0e-280)
    m = np.maximum(2.0 * n, 1.0e-13)
    sigma_m = np.maximum(4.0 * sigma, 1.0e-280)
    s_raw = np.sqrt(sigma_m) / (2.0 * G.C1 * m ** (4.0 / 3.0))
    s = np.clip(s_raw, G.S_MIN, G.S_MAX)
    active = (s_raw > G.S_MIN) & (s_raw < G.S_MAX)
    f = G._pbe_enhancement(s)
    fp = G._pbe_derivative(s)
    phi = G.A_X * m ** (4.0 / 3.0)
    ds_dm = -(4.0 / 3.0) * s_raw / m * active
    ds_dsigma_m = s_raw / (2.0 * sigma_m) * active
    de_dm = 0.5 * (
        (4.0 / 3.0) * (phi / m) * f + phi * fp * ds_dm
    )
    de_dsigma_m = 0.5 * phi * fp * ds_dsigma_m
    vrho = 2.0 * de_dm
    vsigma = 4.0 * de_dsigma_m
    return vrho - 2.0 * radial_divergence(vsigma, grad_n, r)


def hartree_potential_spherical(n: np.ndarray, r: np.ndarray) -> np.ndarray:
    shell_q = 4.0 * pi * r * r * n
    q_inside = np.zeros_like(r)
    q_inside[1:] = np.cumsum(
        0.5 * (shell_q[1:] + shell_q[:-1]) * np.diff(r)
    )
    shell_outer = 4.0 * pi * r * n
    primitive = np.zeros_like(r)
    primitive[1:] = np.cumsum(
        0.5 * (shell_outer[1:] + shell_outer[:-1]) * np.diff(r)
    )
    outside = primitive[-1] - primitive
    return q_inside / r + outside


def append_rows(
    rows,
    system,
    r,
    n,
    vx_impl,
    vx_pbe,
    reference,
    mask,
    vx_hydrogen_exact_gp93=None,
):
    if vx_hydrogen_exact_gp93 is None:
        vx_hydrogen_exact_gp93 = np.full_like(r, np.nan)
    for values in zip(
        r, n, vx_impl, vx_pbe, reference, mask, vx_hydrogen_exact_gp93
    ):
        radius, dens, vi, vp, ref, trusted, vh = values
        rows.append(
            {
                "system": system,
                "r_bohr": float(radius),
                "density": float(dens),
                "minus_r_vx_hydrogen_exact_gp93": (
                    float(-radius * vh) if np.isfinite(vh) else ""
                ),
                "minus_r_vx_implemented": float(-radius * vi),
                "minus_r_vx_pbe_x": float(-radius * vp),
                "minus_r_vx_self_interaction_free_reference": float(ref),
                "density_threshold_mask": bool(trusted),
            }
        )


def main() -> None:
    outdir = OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    for name in (
        "radial_potentials.csv",
        "radial_potentials_metadata.json",
    ):
        (outdir / name).unlink(missing_ok=True)
    G._Fmix_override(FMIX_S0, FMIX_P)
    G.CORR = ""
    rows = []
    metadata: dict[str, object] = {
        "input_file": str(INPUT_PATH.relative_to(ROOT)),
        "input_settings": INPUT,
        "selected_parameters": {"s0": FMIX_S0, "p": FMIX_P},
        "functional": "regularized mgga_fmix implementation, exchange-only",
        "hydrogen_design_reference": (
            "analytic hydrogen-exact GP93 identity on the exact H density"
        ),
        "includes_log_density_window_Xn": True,
        "includes_s_and_z_clipping": True,
        "s0": FMIX_S0,
        "p": FMIX_P,
        "comparison": "PBE exchange-only",
        "systems": {},
    }

    # H: exact analytic density, extended far enough to show cutoff collapse.
    # Differentiate on an extended grid, then crop the plot.  This prevents
    # the one-sided spline derivative at the plotting boundary from appearing
    # as a spurious tail feature.
    h_input = SYSTEM_INPUTS["H"]
    r_h_eval = grid_from_input(h_input["evaluation_grid_bohr"])
    n_h_eval = np.exp(-2.0 * r_h_eval) / pi
    grad_h_eval = -2.0 * n_h_eval
    vx_h_design_eval = hydrogen_exact_gp93_vx(
        n_h_eval, grad_h_eval, r_h_eval
    )
    vx_h_eval = implemented_channel_vx(n_h_eval, grad_h_eval, r_h_eval)
    vx_h_pbe_eval = pbe_channel_vx(n_h_eval, grad_h_eval, r_h_eval)
    shown_h = r_h_eval <= float(h_input["reported_stop_bohr"])
    r_h = r_h_eval[shown_h]
    n_h = n_h_eval[shown_h]
    vx_h_design = vx_h_design_eval[shown_h]
    vx_h = vx_h_eval[shown_h]
    vx_h_pbe = vx_h_pbe_eval[shown_h]
    ref_h = 1.0 - np.exp(-2.0 * r_h) * (1.0 + r_h)
    mask_h = np.ones_like(r_h, dtype=bool)
    append_rows(
        rows,
        "H_exact_density",
        r_h,
        n_h,
        vx_h,
        vx_h_pbe,
        ref_h,
        mask_h,
        vx_h_design,
    )
    # He: evaluate both approximations on one common RHF density.  For any
    # closed-shell two-electron density, the self-interaction-free exchange
    # reference is -v_H/2.
    he_input = SYSTEM_INPUTS["He"]
    mol_he = gto.M(
        atom=he_input["atom"],
        basis=he_input["basis"],
        charge=int(he_input["charge"]),
        spin=int(he_input["spin_2S"]),
        verbose=0,
    )
    if he_input["method"] != "RHF":
        raise ValueError("The He common-density diagnostic requires RHF")
    hf_he = scf.RHF(mol_he)
    hf_he.conv_tol = float(INPUT["reference_scf_conv_tol"])
    hf_he.max_cycle = int(INPUT["reference_scf_max_cycle"])
    hf_he.kernel()
    if not hf_he.converged:
        raise RuntimeError("He RHF reference density did not converge")
    r_he_ext = grid_from_input(he_input["hartree_grid_bohr"])
    n_he_ext = density_on_axis(mol_he, hf_he.make_rdm1(), r_he_ext)
    vh_he_ext = hartree_potential_spherical(n_he_ext, r_he_ext)
    r_he = grid_from_input(he_input["reported_grid_bohr"])
    n_he, grad_he = density_and_gradient_on_axis(
        mol_he, hf_he.make_rdm1(), r_he
    )
    vx_he = implemented_channel_vx(0.5 * n_he, 0.5 * grad_he, r_he)
    vx_he_pbe = pbe_channel_vx(0.5 * n_he, 0.5 * grad_he, r_he)
    ref_he = 0.5 * r_he * np.interp(r_he, r_he_ext, vh_he_ext)
    mask_he = n_he > float(he_input["density_threshold"])
    append_rows(
        rows,
        "He_RHF_aug_cc_pV5Z",
        r_he,
        n_he,
        vx_he,
        vx_he_pbe,
        ref_he,
        mask_he,
    )
    # Na ECP: evaluate both approximations on one common one-electron UHF
    # density, for which the self-interaction-free reference is -v_H.
    na_input = SYSTEM_INPUTS["Na"]
    mol_na = gto.M(
        atom=na_input["atom"],
        basis=na_input["basis"],
        ecp=na_input["ecp"],
        charge=int(na_input["charge"]),
        spin=int(na_input["spin_2S"]),
        verbose=0,
    )
    if na_input["method"] != "UHF":
        raise ValueError("The Na common-density diagnostic requires UHF")
    hf_na = scf.UHF(mol_na)
    hf_na.conv_tol = float(INPUT["reference_scf_conv_tol"])
    hf_na.max_cycle = int(INPUT["reference_scf_max_cycle"])
    hf_na.kernel()
    if not hf_na.converged:
        raise RuntimeError("Na UHF reference density did not converge")
    r_na_ext = grid_from_input(na_input["hartree_grid_bohr"])
    n_na_ext = density_on_axis(mol_na, hf_na.make_rdm1(), r_na_ext)
    vh_na_ext = hartree_potential_spherical(n_na_ext, r_na_ext)
    r_na = grid_from_input(na_input["reported_grid_bohr"])
    n_na, grad_na = density_and_gradient_on_axis(
        mol_na, hf_na.make_rdm1(), r_na
    )
    vx_na = implemented_channel_vx(n_na, grad_na, r_na)
    vx_na_pbe = pbe_channel_vx(n_na, grad_na, r_na)
    ref_na = r_na * np.interp(r_na, r_na_ext, vh_na_ext)
    mask_na = n_na > float(na_input["density_threshold"])
    append_rows(
        rows,
        "Na_UHF_SBKJC_3s",
        r_na,
        n_na,
        vx_na,
        vx_na_pbe,
        ref_na,
        mask_na,
    )
    with (outdir / "radial_potentials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata["systems"] = {
        "H": {
            "density": "exact",
            "range_bohr": [
                float(h_input["evaluation_grid_bohr"]["start"]),
                float(h_input["reported_stop_bohr"]),
            ],
            "maximum_abs_error_regularized_through_6_bohr": float(
                np.max(
                    np.abs(
                        (-r_h * vx_h)[
                            r_h <= float(h_input["regularized_error_stop_bohr"])
                        ]
                        - ref_h[
                            r_h <= float(h_input["regularized_error_stop_bohr"])
                        ]
                    )
                )
            ),
        },
        "He": {
            "basis": he_input["basis"],
            "common_density": he_input["method"],
            "range_bohr": [
                float(he_input["reported_grid_bohr"]["start"]),
                float(he_input["reported_grid_bohr"]["stop"]),
            ],
            "reference_converged": bool(hf_he.converged),
            "reference_energy_Ha": float(hf_he.e_tot),
            "mean_abs_error_minus_r_vx": {
                "regularized_Fmix": float(
                    np.mean(np.abs((-r_he * vx_he)[mask_he] - ref_he[mask_he]))
                ),
                "PBE_X": float(
                    np.mean(
                        np.abs((-r_he * vx_he_pbe)[mask_he] - ref_he[mask_he])
                    )
                ),
            },
        },
        "Na": {
            "basis_ecp": f"{na_input['basis']}/{na_input['ecp']}",
            "common_density": f"{na_input['method']} one-electron 3s",
            "range_bohr": [
                float(na_input["reported_grid_bohr"]["start"]),
                float(na_input["reported_grid_bohr"]["stop"]),
            ],
            "reference_converged": bool(hf_na.converged),
            "reference_energy_Ha": float(hf_na.e_tot),
            "mean_abs_error_minus_r_vx": {
                "regularized_Fmix": float(
                    np.mean(np.abs((-r_na * vx_na)[mask_na] - ref_na[mask_na]))
                ),
                "PBE_X": float(
                    np.mean(
                        np.abs((-r_na * vx_na_pbe)[mask_na] - ref_na[mask_na])
                    )
                ),
            },
        },
    }
    (outdir / "radial_potentials_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(
        "Plot separately with: "
        "python calculations/plot_results.py fig-radial-potentials"
    )


if __name__ == "__main__":
    main()
