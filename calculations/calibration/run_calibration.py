#!/usr/bin/env python3
"""Recompute the five-system crossover calibration.

A five-system MAE is valid only when every system converges under the common
protocol.  Every scan point, convergence flag, and per-system result is
retained in machine-readable output.
"""

from __future__ import annotations

import csv
from importlib import metadata as importlib_metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPOSITORY_ROOT / "results" / "calibration"
WORKSPACE_ROOT = REPOSITORY_ROOT
SOURCE_DIR = REPOSITORY_ROOT / "functional"
CALIBRATION_INPUT_PATH = REPOSITORY_ROOT / "inputs" / "calibration.json"
CALIBRATION_INPUT = json.loads(
    CALIBRATION_INPUT_PATH.read_text(encoding="utf-8")
)
if int(CALIBRATION_INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/calibration.json schema_version")
CALIBRATION_ATOMIC_INPUT_PATH = (
    REPOSITORY_ROOT / CALIBRATION_INPUT["atomic_settings_file"]
)
CALIBRATION_ATOMIC_INPUT = json.loads(
    CALIBRATION_ATOMIC_INPUT_PATH.read_text(encoding="utf-8")
)
if int(CALIBRATION_ATOMIC_INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported referenced atomic input schema_version")

# Do not write __pycache__ files into the supplied source tree.
sys.dont_write_bytecode = True

if not (SOURCE_DIR / "mgga_fmix.py").exists():
    raise FileNotFoundError(
        f"Functional source directory not found: {SOURCE_DIR}"
    )
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(REPOSITORY_ROOT / "calculations" / "atomic"))

import numpy as np
import scipy
import pyscf
from pyscf import dft, gto, lib, scf
from pyscf.dft import libxc

import mgga_fmix as FG
import run_atomic as ATOMIC

if CALIBRATION_ATOMIC_INPUT != ATOMIC.ATOMIC_INPUT:
    raise RuntimeError(
        "Calibration atomic settings do not match the settings loaded by "
        "calculations/atomic/run_atomic.py"
    )

P_FIXED = int(CALIBRATION_INPUT["p_fixed"])
S0_VALUES = tuple(float(value) for value in CALIBRATION_INPUT["s0_values"])
HARTREE_TO_EV = 27.211386245988
EXPERIMENTAL_IP_PATH = (
    REPOSITORY_ROOT / CALIBRATION_INPUT["experimental_data_file"]
)


def load_experimental_ips() -> dict[str, float]:
    with EXPERIMENTAL_IP_PATH.open(newline="", encoding="utf-8") as handle:
        return {
            row["system"]: float(row["ionization_energy_eV"])
            for row in csv.DictReader(handle)
        }


EXPERIMENTAL_IP = load_experimental_ips()

ATOMIC_SYSTEMS_BY_NAME = {
    system["name"]: system for system in ATOMIC.SYSTEMS
}
SYSTEMS = tuple(
    {
        "symbol": name,
        "basis": ATOMIC_SYSTEMS_BY_NAME[name]["basis"],
        "ecp": ATOMIC_SYSTEMS_BY_NAME[name]["ecp"],
        "spin": ATOMIC_SYSTEMS_BY_NAME[name]["spin"],
        "experimental_ip_eV": EXPERIMENTAL_IP[name],
    }
    for name in CALIBRATION_INPUT["systems"]
)

RESULTS_CSV = SCRIPT_DIR / "calibration_results.csv"
SUMMARY_CSV = SCRIPT_DIR / "calibration_summary.csv"
RESULTS_JSON = SCRIPT_DIR / "calibration_results.json"
SELECTED_JSON = SCRIPT_DIR / "selected_parameters.json"
RUN_LOG = SCRIPT_DIR / "calibration_run.log"


class RunLogger:
    def __init__(self, path: Path, start: float):
        self.path = path
        self.start = start
        self.handle = path.open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{time.perf_counter() - self.start:9.2f} s] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def install_current_memory_workaround_if_required() -> bool:
    """Patch only sandbox runtimes whose own /proc/<pid>/statm is absent."""

    statm = Path("/proc") / str(os.getpid()) / "statm"
    if statm.exists():
        return False

    # PySCF normally reads statm only to estimate available working memory.
    # Returning zero current use preserves its normal max_memory allowance.
    lib.current_memory = lambda: (0, 0)
    return True


def environment_metadata(_workaround_active: bool) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": importlib_metadata.version("matplotlib"),
        "pyscf": pyscf.__version__,
        "libxc": libxc.libxc_version(),
        "working_directory": ".",
        "script": "calculations/calibration/run_calibration.py",
        "source_directory": "functional",
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "source_files": [
            "functional/mgga_fmix.py",
            "functional/fmix.py",
            "functional/gp93_w.py",
            "functional/parameters.json",
            "inputs/calibration.json",
            "inputs/atomic.json",
        ],
    }


def prepare_system(system: dict[str, Any]) -> tuple[Any, np.ndarray, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "atom": f"{system['symbol']} 0 0 0",
        "basis": system["basis"],
        "spin": system["spin"],
        "verbose": 0,
    }
    if system["ecp"]:
        kwargs["ecp"] = system["ecp"]

    mol = gto.M(**kwargs)
    reference = scf.UHF(mol) if system["spin"] else scf.RHF(mol)
    reference.conv_tol = float(ATOMIC.INITIAL_HF["conv_tol"])
    reference.max_cycle = int(ATOMIC.INITIAL_HF["max_cycle"])
    reference.chkfile = None
    reference.verbose = 0
    reference.kernel()
    metadata = {
        "converged": bool(reference.converged),
        "total_energy_Ha": finite_or_none(reference.e_tot),
    }
    if not reference.converged:
        raise RuntimeError(f"Reference HF did not converge for {system['symbol']}")
    return mol, reference.make_rdm1(), metadata


def run_kernel_attempt(mf: Any, dm0: np.ndarray, label: str) -> dict[str, Any]:
    cycles = {"count": 0}

    def callback(envs: dict[str, Any]) -> None:
        cycle = envs.get("cycle")
        if cycle is not None:
            cycles["count"] = max(cycles["count"], int(cycle) + 1)

    mf.callback = callback
    mf.kernel(dm0=dm0)
    return {
        "label": label,
        "converged": bool(mf.converged),
        "cycles": cycles["count"],
        "total_energy_Ha": finite_or_none(mf.e_tot),
    }


def solve(mol: Any, dm0: np.ndarray, s0: float) -> tuple[Any, list[dict[str, Any]]]:
    """Run a mandatory zero-shift solution and the common one-electron solver."""

    FG._Fmix_override(s0, P_FIXED)
    FG.CORR = ""

    restricted = mol.spin == 0
    mf = dft.RKS(mol) if restricted else dft.UKS(mol)
    mf.xc = ""
    mf.define_xc_(FG.eval_xc, xctype="MGGA")
    mf.verbose = 0
    mf.chkfile = None
    mf.grids.level = ATOMIC.GRID_LEVEL
    attempts: list[dict[str, Any]] = []
    current_dm = dm0
    seed_coeff = None
    stages = (
        ATOMIC.ONE_ELECTRON_STAGES
        if mol.nelectron == 1 and not restricted
        else ATOMIC.MANY_ELECTRON_STAGES
    )
    for stage in stages:
        damping = float(stage["damping"])
        level_shift = float(stage["level_shift_ha"])
        tolerance = float(stage["conv_tol"])
        max_cycle = int(stage["max_cycle"])
        mf.damp = damping
        mf.level_shift = level_shift
        mf.conv_tol = tolerance
        mf.max_cycle = max_cycle
        label = (
            f"damp={damping}, level_shift={level_shift}, "
            f"conv_tol={tolerance}, max_cycle={max_cycle}"
        )
        try:
            attempts.append(run_kernel_attempt(mf, current_dm, label))
        except (AttributeError, np.linalg.LinAlgError) as error:
            attempts.append(
                {
                "label": label,
                "converged": False,
                "cycles": None,
                "total_energy_Ha": finite_or_none(mf.e_tot),
                    "exception": f"{type(error).__name__}: {error}",
                }
            )
            mf.converged = False
            break
        current_dm = mf.make_rdm1()
        if mol.nelectron == 1 and not restricted and mf.converged:
            seed_coeff = np.asarray(mf.mo_coeff)[0][
                :, np.asarray(mf.mo_occ)[0] > 0
            ][:, 0].copy()

    if mol.nelectron == 1 and not restricted:
        optimizer_logger = logging.getLogger("calibration.one_electron")
        if not optimizer_logger.handlers:
            optimizer_logger.addHandler(logging.NullHandler())
        mf = ATOMIC.optimize_one_electron_orbital(
            mol=mol,
            mf=mf,
            seed_coeff=seed_coeff,
            logger=optimizer_logger,
            functional_name=f"Fmix(s0={s0:.2f})",
        )
        attempts.append(
            {
                "label": "s-symmetric direct one-electron minimization",
                "converged": bool(mf.converged),
                "cycles": getattr(mf, "_audit_optimizer_iterations", None),
                "total_energy_Ha": finite_or_none(mf.e_tot),
                "orbital_residual_norm": getattr(
                    mf, "_audit_orbital_residual_norm", None
                ),
            }
        )
    return mf, attempts


def occupied_homo_ip_eV(mf: Any) -> float:
    energies = np.asarray(mf.mo_energy)
    occupations = np.asarray(mf.mo_occ)
    occupied = energies[occupations > 0]
    if occupied.size == 0:
        raise RuntimeError("No occupied orbital found")
    return -float(np.max(occupied)) * HARTREE_TO_EV


def negative_virtual_spin_orbitals(mf: Any) -> int:
    energies = np.asarray(mf.mo_energy)
    occupations = np.asarray(mf.mo_occ)
    return int(np.sum((occupations <= 0) & (energies < 0)))


def write_outputs(document: dict[str, Any]) -> None:
    with RESULTS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    result_fields = (
        "s0",
        "p",
        "system",
        "basis",
        "ecp",
        "spin",
        "experimental_ip_eV",
        "calculated_ip_eV",
        "absolute_error_eV",
        "converged",
        "negative_virtual_spin_orbitals",
        "attempt_count",
        "exception",
    )
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=result_fields, lineterminator="\n"
        )
        writer.writeheader()
        for point in document["scan"]:
            for result in point["systems"]:
                writer.writerow(
                    {
                        "s0": point["s0"],
                        "p": point["p"],
                        "system": result["symbol"],
                        "basis": result["basis"],
                        "ecp": result["ecp"] or "",
                        "spin": result["spin"],
                        "experimental_ip_eV": result["experimental_ip_eV"],
                        "calculated_ip_eV": (
                            "" if result["calculated_ip_eV"] is None else result["calculated_ip_eV"]
                        ),
                        "absolute_error_eV": (
                            "" if result["absolute_error_eV"] is None else result["absolute_error_eV"]
                        ),
                        "converged": result["converged"],
                        "negative_virtual_spin_orbitals": (
                            ""
                            if result["negative_virtual_spin_orbitals"] is None
                            else result["negative_virtual_spin_orbitals"]
                        ),
                        "attempt_count": len(result["attempts"]),
                        "exception": result["exception"] or "",
                    }
                )

    summary_fields = (
        "s0",
        "p",
        "n_converged",
        "all_five_converged",
        "mae_5system_eV",
        "failed_systems",
    )
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=summary_fields, lineterminator="\n"
        )
        writer.writeheader()
        for point in document["scan"]:
            writer.writerow(
                {
                    "s0": point["s0"],
                    "p": point["p"],
                    "n_converged": point["n_converged"],
                    "all_five_converged": point["all_five_converged"],
                    "mae_5system_eV": (
                        "" if point["mae_5system_eV"] is None else point["mae_5system_eV"]
                    ),
                    "failed_systems": ";".join(point["failed_systems"]),
                }
            )


def main() -> int:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    lib.num_threads(1)
    for path in (
        RESULTS_CSV,
        SUMMARY_CSV,
        RESULTS_JSON,
        SELECTED_JSON,
    ):
        path.unlink(missing_ok=True)
    started = time.perf_counter()
    logger = RunLogger(RUN_LOG, started)
    try:
        workaround = install_current_memory_workaround_if_required()
        environment = environment_metadata(workaround)
        logger.write(
            "Environment: "
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}, "
            f"PySCF {pyscf.__version__}, NumPy {np.__version__}, "
            f"SciPy {scipy.__version__}, LibXC {environment['libxc']}"
        )
        logger.write(f"Source: {SOURCE_DIR}")
        logger.write(f"current_memory workaround active: {workaround}")
        if pyscf.__version__ != "2.14.0":
            logger.write(
                f"WARNING: requested PySCF 2.14.0 but running {pyscf.__version__}; "
                "results may not match the locked reproduction environment"
            )

        prepared: dict[str, tuple[Any, np.ndarray]] = {}
        reference_metadata: dict[str, Any] = {}
        for system in SYSTEMS:
            symbol = system["symbol"]
            logger.write(
                f"Preparing {symbol}: basis={system['basis']}, "
                f"ecp={system['ecp'] or 'none'}, spin={system['spin']}"
            )
            mol, dm0, refmeta = prepare_system(system)
            prepared[symbol] = (mol, dm0)
            reference_metadata[symbol] = refmeta
            logger.write(
                f"Reference {symbol}: converged={refmeta['converged']}, "
                f"E={refmeta['total_energy_Ha']:.12f} Ha"
            )

        document: dict[str, Any] = {
            "schema_version": 1,
            "input_file": str(
                CALIBRATION_INPUT_PATH.relative_to(REPOSITORY_ROOT)
            ),
            "input_settings": CALIBRATION_INPUT,
            "atomic_input_settings": CALIBRATION_ATOMIC_INPUT,
            "environment": environment,
            "method": {
                "input_file": str(
                    CALIBRATION_INPUT_PATH.relative_to(REPOSITORY_ROOT)
                ),
                "atomic_settings_file": CALIBRATION_INPUT[
                    "atomic_settings_file"
                ],
                "p": P_FIXED,
                "p_role": "fixed design choice; not fitted in this scan",
                "fitted_parameter": "s0",
                "s0_values": list(S0_VALUES),
                "hartree_to_eV": HARTREE_TO_EV,
                "grid_level": ATOMIC.GRID_LEVEL,
                "conv_tol": ATOMIC.CONV_TOL,
                "aggregation_rule": (
                    "mae_5system_eV is present only if all H, He, Li, Na, and K "
                    "calculations converged; no failed system is excluded"
                ),
                "homo_definition": (
                    "maximum eigenvalue among orbitals with mo_occ > 0, preserving "
                    "the occupation mask rather than flatten-and-index counting"
                ),
                "one_electron_initial_stage": list(
                    ATOMIC.ONE_ELECTRON_STAGES
                ),
                "many_electron_scf_stages": list(
                    ATOMIC.MANY_ELECTRON_STAGES
                ),
                "one_electron_optimizer": dict(
                    ATOMIC.ORBITAL_OPTIMIZER
                ),
                "one_electron_protocol": (
                    "H, Li/CRENBL, Na/SBKJC, and K/SBKJC use the same "
                    "s-symmetric direct occupied-orbital minimization after "
                    "the zero-level-shift stage"
                ),
            },
            "systems": [dict(system) for system in SYSTEMS],
            "reference_hf": reference_metadata,
            "scan": [],
            "summary": {},
        }

        for s0 in S0_VALUES:
            logger.write(f"Starting s0={s0:.2f}, p={P_FIXED}")
            point_results: list[dict[str, Any]] = []
            for system in SYSTEMS:
                symbol = system["symbol"]
                mol, dm0 = prepared[symbol]
                system_started = time.perf_counter()
                result: dict[str, Any] = {
                    **system,
                    "calculated_ip_eV": None,
                    "absolute_error_eV": None,
                    "converged": False,
                    "negative_virtual_spin_orbitals": None,
                    "attempts": [],
                    "exception": None,
                }
                try:
                    mf, attempts = solve(mol, dm0, s0)
                    raw_ip = occupied_homo_ip_eV(mf)
                    converged = bool(mf.converged) and math.isfinite(raw_ip)
                    result["calculated_ip_eV"] = finite_or_none(raw_ip)
                    result["converged"] = converged
                    result["negative_virtual_spin_orbitals"] = negative_virtual_spin_orbitals(mf)
                    result["attempts"] = attempts
                    if converged:
                        result["absolute_error_eV"] = abs(
                            raw_ip - system["experimental_ip_eV"]
                        )
                except Exception as exc:
                    result["exception"] = f"{type(exc).__name__}: {exc}"
                    result["traceback"] = traceback.format_exc()
                system_duration = time.perf_counter() - system_started
                point_results.append(result)
                logger.write(
                    f"s0={s0:.2f} {symbol}: converged={result['converged']}, "
                    f"IP={result['calculated_ip_eV']}, "
                    f"error={result['absolute_error_eV']}, "
                    f"attempts={len(result['attempts'])}, "
                    f"time={system_duration:.2f} s"
                    + (f", exception={result['exception']}" if result["exception"] else "")
                )

            failed = [result["symbol"] for result in point_results if not result["converged"]]
            all_converged = not failed and len(point_results) == len(SYSTEMS)
            mae = (
                float(np.mean([result["absolute_error_eV"] for result in point_results]))
                if all_converged
                else None
            )
            point = {
                "s0": s0,
                "p": P_FIXED,
                "n_converged": sum(result["converged"] for result in point_results),
                "all_five_converged": all_converged,
                "mae_5system_eV": mae,
                "failed_systems": failed,
                "systems": point_results,
            }
            document["scan"].append(point)
            logger.write(
                f"Finished s0={s0:.2f}: n_converged={point['n_converged']}/5, "
                f"MAE={mae if mae is not None else 'MISSING'}, "
                f"failed={failed or 'none'}"
            )
            # Keep a complete, human-readable checkpoint after each s0.
            document["summary"] = {
                "status": "incomplete",
                "n_completed_points": len(document["scan"]),
            }
            write_outputs(document)

        valid = [point for point in document["scan"] if point["mae_5system_eV"] is not None]
        minimum = min(valid, key=lambda point: point["mae_5system_eV"]) if valid else None
        document["summary"] = {
            "status": "complete",
            "n_valid_points": len(valid),
            "n_missing_points": len(document["scan"]) - len(valid),
            "minimum_valid_point": (
                {
                    "s0": minimum["s0"],
                    "p": minimum["p"],
                    "mae_5system_eV": minimum["mae_5system_eV"],
                }
                if minimum
                else None
            ),
        }
        write_outputs(document)
        with SELECTED_JSON.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "s0": minimum["s0"],
                    "p": minimum["p"],
                    "mae_5system_eV": minimum["mae_5system_eV"],
                    "p_role": "fixed design choice",
                    "fitted_parameter": "s0",
                    "selection_rule": (
                        "with p fixed, minimum five-system MAE over s0 points "
                        "for which every calculation converged under the "
                        "common protocol"
                    ),
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        logger.write(f"Wrote {RESULTS_CSV.name}, {SUMMARY_CSV.name}, {RESULTS_JSON.name}")
        logger.write(
            "Plot separately with: "
            "python calculations/plot_results.py fig-calibration"
        )
        logger.write(
            "Minimum valid point: "
            + (
                f"s0={minimum['s0']:.2f}, MAE={minimum['mae_5system_eV']:.6f} eV"
                if minimum
                else "none"
            )
        )
        logger.write(f"Total duration: {time.perf_counter() - started:.2f} s")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
