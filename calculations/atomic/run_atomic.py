#!/usr/bin/env python3
"""Recompute the atomic results and negative-virtual-state data.

Closed-shell He is treated with RKS and its spatial orbitals are counted once.
Every one-electron atom (H and the ECP alkalis) is solved by the same
symmetry-preserving direct orbital minimization.  This avoids method-dependent
root switching and gives a common convergence criterion across the complete
atomic and calibration data sets.

Outputs are written to ``results/atomic``.  Figures are generated separately
from these saved results by ``calculations/plot_results.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
from importlib import metadata as importlib_metadata
import json
import logging
import math
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.optimize import minimize
from pyscf import __version__ as pyscf_version
from pyscf import dft, gto, lib as pyscf_lib, scf


HARTREE_TO_EV = 27.211386245988
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FUNCTIONAL_DIR = REPOSITORY_ROOT / "functional"
OUTPUT_DIR = REPOSITORY_ROOT / "results" / "atomic"
ATOMIC_INPUT_PATH = REPOSITORY_ROOT / "inputs" / "atomic.json"
ATOMIC_INPUT = json.loads(ATOMIC_INPUT_PATH.read_text(encoding="utf-8"))
if int(ATOMIC_INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/atomic.json schema_version")
EXPERIMENTAL_IP_PATH = (
    REPOSITORY_ROOT / ATOMIC_INPUT["experimental_data_file"]
)
FUNCTIONAL_PARAMETERS = json.loads(
    (FUNCTIONAL_DIR / "parameters.json").read_text(encoding="utf-8")
)
FMIX_S0 = float(FUNCTIONAL_PARAMETERS["selected_s0"])
FMIX_P = int(FUNCTIONAL_PARAMETERS["selected_p"])
GRID_LEVEL = int(ATOMIC_INPUT["grid_level"])
CONV_TOL = float(ATOMIC_INPUT["final_conv_tol"])
NEGATIVE_THRESHOLD_HA = float(ATOMIC_INPUT["negative_virtual_threshold_ha"])
DEGENERACY_TOL_HA = float(ATOMIC_INPUT["degeneracy_tolerance_ha"])
OCCUPATION_THRESHOLD = float(ATOMIC_INPUT["occupation_threshold"])
INITIAL_HF = ATOMIC_INPUT["initial_hf"]
ONE_ELECTRON_STAGES = tuple(ATOMIC_INPUT["one_electron_initial_stage"])
MANY_ELECTRON_STAGES = tuple(ATOMIC_INPUT["many_electron_scf_stages"])
NEWTON_FALLBACK = ATOMIC_INPUT["newton_fallback"]
ORBITAL_OPTIMIZER = ATOMIC_INPUT["one_electron_optimizer"]


def load_experimental_ips() -> dict[str, float]:
    with EXPERIMENTAL_IP_PATH.open(newline="", encoding="utf-8") as handle:
        return {
            row["system"]: float(row["ionization_energy_eV"])
            for row in csv.DictReader(handle)
        }


EXPERIMENTAL_IP = load_experimental_ips()

SYSTEMS = tuple(
    {
        **spec,
        "spin": int(spec["spin_2S"]),
        "experimental_ip_ev": EXPERIMENTAL_IP[spec["name"]],
    }
    for spec in ATOMIC_INPUT["systems"]
)

FUNCTIONALS = tuple(ATOMIC_INPUT["functionals"])

L_SYMBOLS = {
    0: "s",
    1: "p",
    2: "d",
    3: "f",
    4: "g",
    5: "h",
    6: "i",
}


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("gp93_atomic")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def apply_container_memory_compatibility(logger: logging.Logger) -> bool:
    """Handle containers where the process PID is absent from mounted /proc.

    PySCF uses /proc/<pid>/statm only to select integral block sizes. Returning
    zero current memory is the same compatibility shim used for the independent
    reproducibility runs and does not alter the electronic-structure equations.
    """
    try:
        pyscf_lib.current_memory()
        return False
    except FileNotFoundError:
        pyscf_lib.current_memory = lambda: (0, 0)
        logger.warning(
            "Applied container-only PySCF current_memory=(0,0) compatibility shim"
        )
        return True


def resolve_original_source(
    workspace_root: Path, source_dir_arg: str | None, logger: logging.Logger
) -> tuple[Path, None]:
    """Locate the functional implementation directory."""
    if source_dir_arg:
        source_dir = Path(source_dir_arg).expanduser().resolve()
        if not (source_dir / "mgga_fmix.py").exists():
            raise FileNotFoundError(f"mgga_fmix.py not found under {source_dir}")
        return source_dir, None

    source_candidates = (FUNCTIONAL_DIR,)
    for extracted in source_candidates:
        if (extracted / "mgga_fmix.py").exists():
            return extracted, None

    raise FileNotFoundError(
        f"Could not find the functional implementation under {FUNCTIONAL_DIR}"
    )


def import_fmix(source_dir: Path):
    sys.path.insert(0, str(source_dir))
    fg = importlib.import_module("mgga_fmix")
    fmix = importlib.import_module("fmix")
    fg.CORR = ""
    fg._Fmix_override(FMIX_S0, FMIX_P)
    return fg, fmix


def build_molecule(spec: dict, basis_override=None) -> gto.Mole:
    kwargs = {
        "atom": spec["atom"],
        "basis": basis_override if basis_override is not None else spec["basis"],
        "charge": spec["charge"],
        "spin": spec["spin"],
        "verbose": 0,
        "unit": ATOMIC_INPUT["coordinate_unit"],
    }
    if spec["ecp"]:
        kwargs["ecp"] = spec["ecp"]
    return gto.M(**kwargs)


def initial_density(mol: gto.Mole, restricted: bool):
    reference = scf.RHF(mol) if restricted else scf.UHF(mol)
    reference.verbose = 0
    reference.chkfile = None
    reference.conv_tol = float(INITIAL_HF["conv_tol"])
    reference.max_cycle = int(INITIAL_HF["max_cycle"])
    reference.kernel()
    if not reference.converged:
        raise RuntimeError("HF initial-density calculation did not converge")
    return reference.make_rdm1()


def make_dft_solver(mol: gto.Mole, restricted: bool, functional: dict, fg):
    mf = dft.RKS(mol) if restricted else dft.UKS(mol)
    mf.verbose = 0
    mf.chkfile = None
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = CONV_TOL
    mf.max_cycle = int(ATOMIC_INPUT["dft_initial_max_cycle"])
    if functional["kind"] == "fmix":
        fg.CORR = ""
        fg._Fmix_override(FMIX_S0, FMIX_P)
        mf.xc = ""
        mf.define_xc_(fg.eval_xc, xctype="MGGA")
    else:
        mf.xc = functional["xc_code"]
    return mf


def make_fmix_rks_adapter(fg):
    """Adapt the supplied spin-channel evaluator to PySCF's RKS shapes.

    The supplied evaluator internally applies exact spin scaling and returns
    UKS-shaped derivatives even on its nominal spin=0 branch. For an
    unpolarized RKS density, n_alpha=n_beta=n/2. The chain rule gives

      v_n     = (v_alpha + v_beta)/2,
      v_sigma = (v_aa + v_ab + v_bb)/4,
      v_tau   = (v_tau,alpha + v_tau,beta)/2.

    This adapter changes only the PySCF interface shape, not the functional.
    """

    def eval_xc(
        xc_code,
        rho,
        spin=0,
        relativity=0,
        deriv=1,
        omega=None,
        verbose=None,
    ):
        rho_array = np.asarray(rho)
        half_density = 0.5 * rho_array
        exc, vxc, fxc, kxc = fg.eval_xc(
            xc_code,
            (half_density, half_density),
            spin=1,
            relativity=relativity,
            deriv=deriv,
            omega=omega,
            verbose=verbose,
        )
        vrho_spin, vsigma_spin, _vlapl, vtau_spin = vxc
        vrho = 0.5 * (vrho_spin[:, 0] + vrho_spin[:, 1])
        vsigma = 0.25 * (
            vsigma_spin[:, 0] + vsigma_spin[:, 1] + vsigma_spin[:, 2]
        )
        vtau = 0.5 * (vtau_spin[:, 0] + vtau_spin[:, 1])
        return exc, (vrho, vsigma, None, vtau), fxc, kxc

    return eval_xc


def run_final_unshifted_scf(
    mol: gto.Mole,
    restricted: bool,
    functional: dict,
    fg,
    logger: logging.Logger,
):
    """Use staged stabilization, then require an unshifted final SCF."""
    dm0 = initial_density(mol, restricted)
    mf = make_dft_solver(mol, restricted, functional, fg)
    current_dm = dm0
    converged_seed_coeff = None
    one_electron = mol.nelectron == 1 and not restricted
    stages = ONE_ELECTRON_STAGES if one_electron else MANY_ELECTRON_STAGES
    for stage in stages:
        mf.damp = float(stage["damping"])
        mf.level_shift = float(stage["level_shift_ha"])
        mf.conv_tol = float(stage["conv_tol"])
        mf.max_cycle = int(stage["max_cycle"])
        try:
            mf.kernel(dm0=current_dm)
        except (AttributeError, np.linalg.LinAlgError) as error:
            # PySCF 2.13's DIIS exception path references the removed
            # numpy.linalg.linalg alias under NumPy 2.x. Preserve the latest
            # density and proceed to the documented zero-shift Newton fallback.
            logger.warning(
                "%s/%s DIIS stage failed (%s); trying zero-shift Newton",
                mol.atom_symbol(0),
                functional["name"],
                type(error).__name__,
            )
            mf.converged = False
            break
        current_dm = mf.make_rdm1()
        if mf.converged and not restricted and mol.nelectron == 1:
            converged_seed_coeff = np.asarray(mf.mo_coeff)[0][
                :, np.asarray(mf.mo_occ)[0] > 0
            ][:, 0].copy()

    # All one-electron atoms use one common, symmetry-preserving variational
    # solution after the zero-shift stage.  This removes functional-dependent
    # root switching from both the atomic comparison and the calibration.
    if mol.nelectron == 1 and not restricted:
        mf = optimize_one_electron_orbital(
            mol,
            mf,
            converged_seed_coeff,
            logger,
            functional_name=functional["name"],
        )
    # Use a second-order solver only as a zero-shift fallback. Any returned
    # value is accepted solely if the fallback itself reports convergence.
    elif not mf.converged and functional["name"] != "Fmix_X":
        fallback = mf.newton()
        fallback.verbose = 0
        fallback.chkfile = None
        fallback.conv_tol = float(NEWTON_FALLBACK["conv_tol"])
        fallback.max_cycle = int(NEWTON_FALLBACK["max_cycle"])
        fallback.level_shift = float(NEWTON_FALLBACK["level_shift_ha"])
        try:
            fallback.kernel(dm0=current_dm)
            mf = fallback
        except Exception as error:
            logger.warning(
                "%s/%s Newton fallback unavailable (%s); retaining missing result",
                mol.atom_symbol(0),
                functional["name"],
                type(error).__name__,
            )
            mf.converged = False
    elif not mf.converged:
        # The custom evaluator provides first derivatives only;
        # PySCF's Newton response requests fxc and cannot be used.
        logger.warning(
            "%s/%s remains unconverged; Newton fallback is inapplicable "
            "because the custom evaluator has no second derivatives",
            mol.atom_symbol(0),
            functional["name"],
        )

    logger.info(
        "%-3s %-7s %-3s converged=%s E=%.10f",
        mol.atom_symbol(0),
        functional["name"],
        "RKS" if restricted else "UKS",
        mf.converged,
        mf.e_tot,
    )
    return mf


def optimize_one_electron_orbital(
    mol: gto.Mole,
    mf,
    seed_coeff,
    logger: logging.Logger,
    functional_name: str = "custom",
):
    """Variational fallback for a one-electron UKS problem.

    The occupied alpha orbital is optimized on the overlap-normalized sphere.
    This avoids SCF root switching and uses the ordinary unshifted Fock
    gradient.  It does not alter the functional or constrain the final
    orbital beyond normalization and the physical atomic s irreducible
    subspace.  Approximate functionals can reorder empty p-like and occupied
    s-like eigenvalues; this does not invalidate the stationary s state.
    """

    if mol.nelectron != 1 or mol.spin != 1:
        raise ValueError("One-electron optimizer requires nelectron=1, spin=1")

    overlap = mol.intor_symmetric("int1e_ovlp")
    # All open-shell systems in the atomic set have an ns ground state.
    # Preserve the physical atomic s irrep; an unconstrained approximate
    # functional can otherwise lower its energy by a spurious s-p
    # symmetry-breaking mixture.
    active = np.where(ao_angular_momenta(mol) == 0)[0]
    overlap_active = overlap[np.ix_(active, active)]
    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(
        overlap_active
    )
    x_orth = (
        overlap_eigenvectors * overlap_eigenvalues ** (-0.5)
    ) @ overlap_eigenvectors.T
    s_half = (
        overlap_eigenvectors * overlap_eigenvalues ** 0.5
    ) @ overlap_eigenvectors.T
    hcore = mf.get_hcore()

    if seed_coeff is None:
        reference_coeff = np.asarray(
            scf.UHF(mol).run(
                conv_tol=float(INITIAL_HF["conv_tol"]), verbose=0
            ).mo_coeff
        )[0, :, 0]
    else:
        reference_coeff = np.asarray(seed_coeff)
    y0 = s_half @ reference_coeff[active]
    y0 /= np.linalg.norm(y0)

    def energy_and_gradient(y):
        norm = np.linalg.norm(y)
        normalized = y / norm
        coeff = np.zeros(mol.nao_nr())
        coeff[active] = x_orth @ normalized
        dm_alpha = np.outer(coeff, coeff)
        density_matrix = np.stack(
            (dm_alpha, np.zeros_like(dm_alpha)),
            axis=0,
        )
        veff = mf.get_veff(mol, density_matrix)
        energy = float(
            mf.energy_tot(dm=density_matrix, h1e=hcore, vhf=veff)
        )
        fock_alpha = hcore + np.asarray(veff)[0]
        fock_orth = (
            x_orth.T @ fock_alpha[np.ix_(active, active)] @ x_orth
        )
        raw_gradient = 2.0 * (fock_orth @ normalized)
        projected_gradient = (
            raw_gradient
            - normalized * np.dot(normalized, raw_gradient)
        ) / norm
        return energy, projected_gradient

    result = minimize(
        energy_and_gradient,
        y0,
        jac=True,
        method=ORBITAL_OPTIMIZER["method"],
        options={
            "gtol": float(ORBITAL_OPTIMIZER["gtol"]),
            "ftol": float(ORBITAL_OPTIMIZER["ftol"]),
            "maxiter": int(ORBITAL_OPTIMIZER["maxiter"]),
            "maxls": int(ORBITAL_OPTIMIZER["maxls"]),
        },
    )
    normalized = result.x / np.linalg.norm(result.x)
    occupied_coeff = np.zeros(mol.nao_nr())
    occupied_coeff[active] = x_orth @ normalized
    dm_alpha = np.outer(occupied_coeff, occupied_coeff)
    density_matrix = np.stack((dm_alpha, np.zeros_like(dm_alpha)), axis=0)
    veff = mf.get_veff(mol, density_matrix)
    fock_alpha = hcore + np.asarray(veff)[0]
    fock_beta = hcore + np.asarray(veff)[1]
    energy_alpha, coeff_alpha = eigh(fock_alpha, overlap)
    energy_beta, coeff_beta = eigh(fock_beta, overlap)
    overlap_with_optimized = np.abs(
        coeff_alpha.T @ overlap @ occupied_coeff
    )
    occupied_index = int(np.argmax(overlap_with_optimized))
    occupation_alpha = np.zeros_like(energy_alpha)
    occupation_beta = np.zeros_like(energy_beta)
    occupation_alpha[occupied_index] = 1.0
    eigenvalue = float(occupied_coeff @ fock_alpha @ occupied_coeff)
    residual = (
        fock_alpha @ occupied_coeff
        - eigenvalue * overlap @ occupied_coeff
    )
    residual_norm = float(np.linalg.norm(residual))
    converged = bool(
        result.success
        and residual_norm < float(ORBITAL_OPTIMIZER["residual_tolerance"])
        and overlap_with_optimized[occupied_index]
        > 1.0 - float(ORBITAL_OPTIMIZER["occupied_overlap_tolerance"])
    )

    mf.mo_energy = np.stack((energy_alpha, energy_beta), axis=0)
    mf.mo_coeff = np.stack((coeff_alpha, coeff_beta), axis=0)
    mf.mo_occ = np.stack((occupation_alpha, occupation_beta), axis=0)
    mf.e_tot = float(result.fun)
    mf.converged = converged
    mf._audit_convergence_method = "direct_one_electron_orbital_minimization"
    mf._audit_orbital_residual_norm = residual_norm
    mf._audit_optimizer_iterations = int(result.nit)
    mf._audit_orbital_symmetry = "s AO subspace"
    mf._audit_occupied_full_fock_rank = occupied_index
    logger.info(
        "%s/%s direct one-electron minimization: converged=%s "
        "E=%.10f residual=%.3e iterations=%d",
        mol.atom_symbol(0),
        functional_name,
        converged,
        mf.e_tot,
        residual_norm,
        result.nit,
    )
    return mf


def ao_angular_momenta(mol: gto.Mole) -> np.ndarray:
    ao_loc = mol.ao_loc_nr()
    labels = np.empty(mol.nao_nr(), dtype=int)
    for shell in range(mol.nbas):
        labels[ao_loc[shell] : ao_loc[shell + 1]] = mol.bas_angular(shell)
    return labels


def orbital_l_weights(
    coeff: np.ndarray, overlap: np.ndarray, ao_l: np.ndarray
) -> dict[int, float]:
    weights: dict[int, float] = {}
    for angular_momentum in sorted(set(int(value) for value in ao_l)):
        idx = np.flatnonzero(ao_l == angular_momentum)
        block = overlap[np.ix_(idx, idx)]
        value = float(np.real(coeff[idx].conj() @ block @ coeff[idx]))
        weights[angular_momentum] = max(0.0, value)
    norm = sum(weights.values())
    if norm > 0:
        weights = {key: value / norm for key, value in weights.items()}
    return weights


def group_degenerate_states(states: list[dict]) -> list[dict]:
    groups: list[list[dict]] = []
    for state in sorted(states, key=lambda item: item["energy_ha"]):
        if (
            not groups
            or abs(state["energy_ha"] - groups[-1][0]["energy_ha"])
            > DEGENERACY_TOL_HA
        ):
            groups.append([state])
        else:
            groups[-1].append(state)

    output: list[dict] = []
    for group_index, members in enumerate(groups, start=1):
        energy = float(np.mean([member["energy_ha"] for member in members]))
        aggregate: dict[int, float] = {}
        for member in members:
            for angular_momentum, weight in member["l_weights"].items():
                aggregate[int(angular_momentum)] = (
                    aggregate.get(int(angular_momentum), 0.0) + float(weight)
                )
        if aggregate:
            l_est = max(aggregate, key=aggregate.get)
            l_confidence = aggregate[l_est] / len(members)
        else:
            l_est = -1
            l_confidence = float("nan")
        output.append(
            {
                "group_index": group_index,
                "energy_ha": energy,
                "energy_ev": energy * HARTREE_TO_EV,
                "degeneracy": len(members),
                "l_est": l_est,
                "l_symbol": L_SYMBOLS.get(l_est, f"l={l_est}"),
                "l_confidence": l_confidence,
                "n_eff": 1.0 / math.sqrt(-2.0 * energy),
                "r_rms_bohr": float(
                    np.mean([member["r_rms_bohr"] for member in members])
                ),
                "orbital_indices_zero_based": [
                    int(member["orbital_index"]) for member in members
                ],
            }
        )
    return output


def analyze_channel(
    channel: str,
    energies: np.ndarray,
    occupations: np.ndarray,
    coefficients: np.ndarray,
    overlap: np.ndarray,
    r2_matrix: np.ndarray,
    ao_l: np.ndarray,
) -> dict:
    occupied_mask = np.asarray(occupations) > OCCUPATION_THRESHOLD
    virtual_mask = ~occupied_mask
    negative_virtual_mask = virtual_mask & (
        np.asarray(energies) < NEGATIVE_THRESHOLD_HA
    )
    negative_states: list[dict] = []
    for orbital_index in np.flatnonzero(negative_virtual_mask):
        weights = orbital_l_weights(
            coefficients[:, orbital_index], overlap=overlap, ao_l=ao_l
        )
        l_est = max(weights, key=weights.get) if weights else -1
        energy = float(energies[orbital_index])
        coefficient = coefficients[:, orbital_index]
        r2_expectation = float(
            np.real(coefficient.conj() @ r2_matrix @ coefficient)
        )
        negative_states.append(
            {
                "orbital_index": int(orbital_index),
                "energy_ha": energy,
                "energy_ev": energy * HARTREE_TO_EV,
                "occupation": float(occupations[orbital_index]),
                "l_est": int(l_est),
                "l_symbol": L_SYMBOLS.get(l_est, f"l={l_est}"),
                "l_weights": {str(key): value for key, value in weights.items()},
                "n_eff": 1.0 / math.sqrt(-2.0 * energy),
                "r_rms_bohr": math.sqrt(max(r2_expectation, 0.0)),
            }
        )
    occupied_energies = np.asarray(energies)[occupied_mask]
    return {
        "channel": channel,
        "n_occupied": int(np.count_nonzero(occupied_mask)),
        "homo_ha": (
            float(np.max(occupied_energies)) if occupied_energies.size else None
        ),
        "negative_virtual_count": int(np.count_nonzero(negative_virtual_mask)),
        "negative_virtual_states": negative_states,
        "negative_virtual_groups": group_degenerate_states(negative_states),
        "mo_energy_ha": [float(value) for value in np.asarray(energies)],
        "mo_occupation": [float(value) for value in np.asarray(occupations)],
    }


def analyze_scf(mol: gto.Mole, mf, restricted: bool) -> dict:
    overlap = mol.intor_symmetric("int1e_ovlp")
    r2_matrix = mol.intor_symmetric("int1e_r2")
    ao_l = ao_angular_momenta(mol)
    channels: list[dict] = []
    if restricted:
        channels.append(
            analyze_channel(
                "spatial",
                np.asarray(mf.mo_energy),
                np.asarray(mf.mo_occ),
                np.asarray(mf.mo_coeff),
                overlap,
                r2_matrix,
                ao_l,
            )
        )
        count_kind = "RKS spatial orbitals (counted once)"
    else:
        for index, name in enumerate(("alpha", "beta")):
            channels.append(
                analyze_channel(
                    name,
                    np.asarray(mf.mo_energy[index]),
                    np.asarray(mf.mo_occ[index]),
                    np.asarray(mf.mo_coeff[index]),
                    overlap,
                    r2_matrix,
                    ao_l,
                )
            )
        count_kind = "UKS spin-orbital diagnostic (alpha/beta separate)"

    occupied_homos = [
        channel["homo_ha"]
        for channel in channels
        if channel["homo_ha"] is not None
    ]
    global_homo = max(occupied_homos)
    return {
        "converged": bool(mf.converged),
        "convergence_method": getattr(
            mf, "_audit_convergence_method", "ordinary_scf"
        ),
        "orbital_residual_norm": getattr(
            mf, "_audit_orbital_residual_norm", None
        ),
        "optimizer_iterations": getattr(
            mf, "_audit_optimizer_iterations", None
        ),
        "orbital_symmetry_constraint": getattr(
            mf, "_audit_orbital_symmetry", None
        ),
        "e_tot_ha": float(mf.e_tot),
        "global_homo_ha": float(global_homo),
        "minus_global_homo_ev": float(-global_homo * HARTREE_TO_EV),
        "count_kind": count_kind,
        "channels": channels,
    }


def generated_et1_basis(base_name: str, element: str) -> tuple[list, dict]:
    """Add one even-tempered diffuse primitive for every l in a basis.

    This is a transparent sensitivity test, not an official d-aug basis.
    """
    basis = copy.deepcopy(gto.basis.load(base_name, element))
    exponents: dict[int, list[float]] = {}
    for shell in basis:
        angular_momentum = int(shell[0])
        for primitive in shell[1:]:
            exponents.setdefault(angular_momentum, []).append(float(primitive[0]))

    generated: dict[str, float] = {}
    for angular_momentum, values in sorted(exponents.items()):
        unique = sorted(set(values))
        if len(unique) < 2:
            continue
        most_diffuse, next_diffuse = unique[0], unique[1]
        new_exponent = most_diffuse * most_diffuse / next_diffuse
        basis.append([angular_momentum, [new_exponent, 1.0]])
        generated[L_SYMBOLS.get(angular_momentum, str(angular_momentum))] = (
            new_exponent
        )
    return basis, generated


def atomic_summary_row(run: dict) -> dict:
    if not run["analysis"]["converged"]:
        return {
            "system": run["system"],
            "functional": run["functional"],
            "functional_label": run["functional_label"],
            "method": run["method"],
            "basis": run["basis"],
            "ecp": run["ecp"] or "",
            "converged": False,
            "e_tot_ha": "",
            "homo_ha": "",
            "minus_homo_ev": "",
            "experimental_ip_ev": run["experimental_ip_ev"],
            "negative_virtual_spatial_count": "",
            "negative_virtual_alpha_count": "",
            "negative_virtual_beta_count": "",
            "count_display": "not converged",
            "count_definition": run["analysis"]["count_kind"],
        }
    channel_map = {item["channel"]: item for item in run["analysis"]["channels"]}
    if run["method"] == "RKS":
        spatial = channel_map["spatial"]["negative_virtual_count"]
        alpha = ""
        beta = ""
        display = f"spatial {spatial}"
    else:
        spatial = ""
        alpha = channel_map["alpha"]["negative_virtual_count"]
        beta = channel_map["beta"]["negative_virtual_count"]
        display = f"alpha {alpha} / beta {beta}"
    return {
        "system": run["system"],
        "functional": run["functional"],
        "functional_label": run["functional_label"],
        "method": run["method"],
        "basis": run["basis"],
        "ecp": run["ecp"] or "",
        "converged": run["analysis"]["converged"],
        "e_tot_ha": run["analysis"]["e_tot_ha"],
        "homo_ha": run["analysis"]["global_homo_ha"],
        "minus_homo_ev": run["analysis"]["minus_global_homo_ev"],
        "experimental_ip_ev": run["experimental_ip_ev"],
        "negative_virtual_spatial_count": spatial,
        "negative_virtual_alpha_count": alpha,
        "negative_virtual_beta_count": beta,
        "count_display": display,
        "count_definition": run["analysis"]["count_kind"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_state_groups_csv(path: Path, runs: list[dict]) -> None:
    rows: list[dict] = []
    for run in runs:
        if not run["analysis"]["converged"]:
            continue
        for channel in run["analysis"]["channels"]:
            for group in channel["negative_virtual_groups"]:
                rows.append(
                    {
                        "system": run["system"],
                        "functional": run["functional"],
                        "method": run["method"],
                        "channel": channel["channel"],
                        "group_index": group["group_index"],
                        "energy_ha": group["energy_ha"],
                        "energy_ev": group["energy_ev"],
                        "degeneracy_within_channel": group["degeneracy"],
                        "l_est": group["l_est"],
                        "l_symbol": group["l_symbol"],
                        "l_confidence": group["l_confidence"],
                        "n_eff": group["n_eff"],
                    }
                )
    write_csv(path, rows)


def run_main_calculations(fg, logger: logging.Logger) -> list[dict]:
    runs: list[dict] = []
    for system in SYSTEMS:
        restricted = system["method"] == "RKS"
        for functional in FUNCTIONALS:
            mol = build_molecule(system)
            mf = run_final_unshifted_scf(
                mol=mol,
                restricted=restricted,
                functional=functional,
                fg=fg,
                logger=logger,
            )
            analysis = analyze_scf(mol, mf, restricted=restricted)
            run = {
                "system": system["name"],
                "atom": system["atom"],
                "basis": system["basis"],
                "ecp": system["ecp"],
                "spin_2S": system["spin"],
                "method": system["method"],
                "experimental_ip_ev": system["experimental_ip_ev"],
                "functional": functional["name"],
                "functional_label": functional["label"],
                "functional_code": functional["xc_code"],
                "analysis": analysis,
            }
            runs.append(run)
            if not analysis["converged"]:
                logger.error(
                    "Unconverged required calculation: %s %s",
                    system["name"],
                    functional["name"],
                )
    return runs


def run_he_basis_sensitivity(
    fg, main_runs: list[dict], logger: logging.Logger
) -> tuple[list[dict], dict]:
    he_spec = next(spec for spec in SYSTEMS if spec["name"] == "He")
    fmix_functional = next(
        functional for functional in FUNCTIONALS if functional["name"] == "Fmix_X"
    )
    sensitivity_input = ATOMIC_INPUT["he_basis_sensitivity"]
    parent_basis = sensitivity_input["even_tempered_parent_basis"]
    et1_basis, generated_exponents = generated_et1_basis(parent_basis, "He")
    basis_specs = tuple(
        (
            entry["label"],
            et1_basis if entry["add_even_tempered"] else entry["basis"],
            bool(entry["add_even_tempered"]),
        )
        for entry in sensitivity_input["entries"]
    )
    output: list[dict] = []
    main_v5z = next(
        run
        for run in main_runs
        if run["system"] == "He" and run["functional"] == "Fmix_X"
    )
    for label, basis, generated in basis_specs:
        if label == "aug-cc-pV5Z" and basis == he_spec["basis"]:
            analysis = main_v5z["analysis"]
            nao = gto.M(
                atom=he_spec["atom"],
                basis="aug-cc-pv5z",
                spin=0,
                verbose=0,
            ).nao_nr()
        else:
            mol = build_molecule(he_spec, basis_override=basis)
            mf = run_final_unshifted_scf(
                mol=mol,
                restricted=True,
                functional=fmix_functional,
                fg=fg,
                logger=logger,
            )
            analysis = analyze_scf(mol, mf, restricted=True)
            nao = mol.nao_nr()
        spatial = analysis["channels"][0]
        output.append(
            {
                "basis": label,
                "generated_even_tempered": generated,
                "nao": int(nao),
                "converged": analysis["converged"],
                "minus_homo_ev": analysis["minus_global_homo_ev"],
                "negative_virtual_spatial_count": spatial[
                    "negative_virtual_count"
                ],
                "unique_negative_groups": len(
                    spatial["negative_virtual_groups"]
                ),
                "group_energies_ha": [
                    group["energy_ha"]
                    for group in spatial["negative_virtual_groups"]
                ],
                "group_degeneracies": [
                    group["degeneracy"]
                    for group in spatial["negative_virtual_groups"]
                ],
                "group_l_symbols": [
                    group["l_symbol"]
                    for group in spatial["negative_virtual_groups"]
                ],
                "group_r_rms_bohr": [
                    group["r_rms_bohr"]
                    for group in spatial["negative_virtual_groups"]
                ],
            }
        )
    return output, generated_exponents


def basis_rows_for_csv(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        output.append(
            {
                **{key: value for key, value in row.items() if not isinstance(value, list)},
                "group_energies_ha": ";".join(
                    f"{value:.10f}" for value in row["group_energies_ha"]
                ),
                "group_degeneracies": ";".join(
                    str(value) for value in row["group_degeneracies"]
                ),
                "group_l_symbols": ";".join(row["group_l_symbols"]),
                "group_r_rms_bohr": ";".join(
                    f"{value:.8f}" for value in row["group_r_rms_bohr"]
                ),
            }
        )
    return output


def he_group_rows(he_fmix_run: dict) -> list[dict]:
    output: list[dict] = []
    for group in he_fmix_run["analysis"]["channels"][0][
        "negative_virtual_groups"
    ]:
        output.append(
            {
                "system": "He",
                "functional": "Fmix_X",
                "method": "RKS",
                "count_kind": "spatial orbitals counted once",
                "group_index": group["group_index"],
                "energy_ha": group["energy_ha"],
                "energy_ev": group["energy_ev"],
                "spatial_degeneracy": group["degeneracy"],
                "l_est": group["l_est"],
                "l_symbol": group["l_symbol"],
                "l_confidence": group["l_confidence"],
                "n_eff": group["n_eff"],
                "r_rms_bohr": group["r_rms_bohr"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Directory containing an alternative mgga_fmix.py implementation",
    )
    parser.add_argument(
        "--skip-basis-sensitivity",
        action="store_true",
        help="Skip the optional He diffuse-basis sensitivity calculations",
    )
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    pyscf_lib.num_threads(1)
    owned_outputs = (
        "atomic_systematics.csv",
        "state_groups.csv",
        "he_negative_virtual_levels.csv",
        "he_basis_sensitivity.csv",
        "results.json",
    )
    for name in owned_outputs:
        (output_dir / name).unlink(missing_ok=True)
    workspace_root = REPOSITORY_ROOT
    logger = configure_logger(output_dir)
    logger.info(
        "Environment: Python %s; PySCF %s; NumPy %s; SciPy %s",
        platform.python_version(),
        pyscf_version,
        np.__version__,
        scipy.__version__,
    )
    if pyscf_version != "2.14.0":
        raise RuntimeError(
            f"This reproduction requires PySCF 2.14.0, found {pyscf_version}"
        )
    apply_container_memory_compatibility(logger)

    source_dir, temporary_source = resolve_original_source(
        workspace_root=workspace_root,
        source_dir_arg=args.source_dir,
        logger=logger,
    )
    logger.info("Using supplied implementation from %s", source_dir)
    fg, _fmix = import_fmix(source_dir)

    try:
        main_runs = run_main_calculations(fg=fg, logger=logger)
        he_fmix_run = next(
            run
            for run in main_runs
            if run["system"] == "He" and run["functional"] == "Fmix_X"
        )

        basis_results: list[dict] = []
        generated_exponents: dict = {}
        if not args.skip_basis_sensitivity:
            basis_results, generated_exponents = run_he_basis_sensitivity(
                fg=fg, main_runs=main_runs, logger=logger
            )

        metadata = {
            "input_file": str(
                ATOMIC_INPUT_PATH.relative_to(REPOSITORY_ROOT)
            ),
            "input_settings": ATOMIC_INPUT,
            "environment": {
                "python": platform.python_version(),
                "pyscf": pyscf_version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "matplotlib": importlib_metadata.version("matplotlib"),
            },
            "settings": {
                "grid_level": GRID_LEVEL,
                "conv_tol": CONV_TOL,
                "negative_virtual_threshold_ha": NEGATIVE_THRESHOLD_HA,
                "degeneracy_tolerance_ha": DEGENERACY_TOL_HA,
                "fmix_s0": FMIX_S0,
                "fmix_p": FMIX_P,
                "final_damping": 0.0,
                "final_level_shift": 0.0,
            },
            "definitions": {
                "closed_shell": "RKS spatial orbitals counted once",
                "open_shell": (
                    "UKS alpha and beta occupation masks analyzed separately; "
                    "counts are spin-orbital diagnostics"
                ),
                "n_eff": "1/sqrt(-2*epsilon_ha)",
                "l_estimate": (
                    "dominant angular-momentum weight of the MO in the AO "
                    "overlap metric"
                ),
            },
            "supplied_source_dir_used": (
                source_dir.relative_to(REPOSITORY_ROOT).as_posix()
                if source_dir.is_relative_to(REPOSITORY_ROOT)
                else str(source_dir)
            ),
            "unconverged_main_runs": [
                {
                    "system": run["system"],
                    "functional": run["functional"],
                }
                for run in main_runs
                if not run["analysis"]["converged"]
            ],
            "main_runs": main_runs,
            "he_basis_sensitivity": basis_results,
            "generated_et1_exponents": generated_exponents,
        }
        with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=False)

        write_csv(
            output_dir / "atomic_systematics.csv",
            [atomic_summary_row(run) for run in main_runs],
        )
        write_state_groups_csv(output_dir / "state_groups.csv", main_runs)
        write_csv(
            output_dir / "he_negative_virtual_levels.csv",
            he_group_rows(he_fmix_run),
        )
        if basis_results:
            write_csv(
                output_dir / "he_basis_sensitivity.csv",
                basis_rows_for_csv(basis_results),
            )

        unconverged = [
            f"{run['system']}/{run['functional']}"
            for run in main_runs
            if not run["analysis"]["converged"]
        ]
        logger.info(
            "Canonical numerical outputs saved; unconverged calculations: %s",
            ", ".join(unconverged) if unconverged else "none",
        )
        logger.info(
            "Plot separately with: "
            "python calculations/plot_results.py fig-atomic-systematics; "
            "python calculations/plot_results.py fig-he-levels; "
            "python calculations/plot_results.py fig-he-comparison"
        )
        if unconverged:
            raise RuntimeError(
                "Required atomic calculations did not converge: "
                + ", ".join(unconverged)
            )
        unconverged_basis = [
            row["basis"]
            for row in basis_results
            if not bool(row["converged"])
        ]
        if unconverged_basis:
            raise RuntimeError(
                "Required He basis-sensitivity calculations did not converge: "
                + ", ".join(unconverged_basis)
            )
    finally:
        if temporary_source is not None:
            temporary_source.cleanup()


if __name__ == "__main__":
    main()
