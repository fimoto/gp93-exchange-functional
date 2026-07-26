#!/usr/bin/env python3
"""Self-consistent H calculation with the unregularized H-exact functional."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib, scf


ROOT = Path(__file__).resolve().parents[2]
FUNCTIONAL_DIR = ROOT / "functional"
OUTPUT_DIR = ROOT / "results" / "atomic"
INPUT_PATH = ROOT / "inputs" / "hydrogen_exact_scf.json"
INPUT = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
if int(INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/hydrogen_exact_scf.json schema_version")
IP_PATH = ROOT / INPUT["experimental_data_file"]
sys.path.insert(0, str(FUNCTIONAL_DIR))

import hydrogen_exact as H_EXACT  # noqa: E402


HARTREE_TO_EV = 27.211386245988
FUNCTIONAL_PARAMETERS = json.loads(
    (FUNCTIONAL_DIR / "parameters.json").read_text(encoding="utf-8")
)
SELECTED_S0 = float(FUNCTIONAL_PARAMETERS["selected_s0"])
SELECTED_P = int(FUNCTIONAL_PARAMETERS["selected_p"])
BASIS = INPUT["basis"]
GRID_LEVEL = int(INPUT["grid_level"])
STAGES = tuple(INPUT["scf_stages"])


def experimental_ip() -> float:
    with IP_PATH.open(newline="", encoding="utf-8") as handle:
        row = next(row for row in csv.DictReader(handle) if row["system"] == "H")
    return float(row["ionization_energy_eV"])


def memory_workaround() -> None:
    try:
        lib.current_memory()
    except FileNotFoundError:
        lib.current_memory = lambda: (0, 0)


def main() -> None:
    memory_workaround()
    lib.num_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("hydrogen_exact_scf.csv", "hydrogen_exact_scf.json"):
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    H_EXACT.set_parameters(SELECTED_S0, SELECTED_P)

    mol = gto.M(
        atom=INPUT["atom"],
        basis=BASIS,
        charge=int(INPUT["charge"]),
        spin=int(INPUT["spin_2S"]),
        unit=INPUT["coordinate_unit"],
        verbose=0,
    )
    reference = scf.UHF(mol)
    reference.verbose = 0
    reference.chkfile = None
    reference.conv_tol = float(INPUT["initial_uhf"]["conv_tol"])
    reference.max_cycle = int(INPUT["initial_uhf"]["max_cycle"])
    reference.kernel()
    if not reference.converged:
        raise RuntimeError("H UHF initial-density calculation did not converge")

    mf = dft.UKS(mol)
    mf.verbose = 0
    mf.chkfile = None
    mf.xc = ""
    mf.define_xc_(H_EXACT.eval_xc, xctype="GGA")
    mf.grids.level = GRID_LEVEL
    density = reference.make_rdm1()
    stage_records: list[dict[str, float | int | bool]] = []
    for stage in STAGES:
        damping = float(stage["damping"])
        level_shift = float(stage["level_shift_ha"])
        tolerance = float(stage["conv_tol"])
        max_cycle = int(stage["max_cycle"])
        mf.damp = damping
        mf.level_shift = level_shift
        mf.conv_tol = tolerance
        mf.max_cycle = max_cycle
        mf.kernel(dm0=density)
        density = mf.make_rdm1()
        stage_records.append(
            {
                "damping": damping,
                "level_shift_ha": level_shift,
                "conv_tol": tolerance,
                "max_cycle": max_cycle,
                "converged": bool(mf.converged),
            }
        )

    energies = np.asarray(mf.mo_energy)
    occupations = np.asarray(mf.mo_occ)
    occupation_threshold = float(INPUT["occupation_threshold"])
    negative_threshold = float(INPUT["negative_virtual_threshold_ha"])
    occupied = energies[occupations > occupation_threshold]
    homo_ha = float(np.max(occupied))
    minus_homo_ev = -homo_ha * HARTREE_TO_EV
    reference_ip = experimental_ip()
    row = {
        "system": "H",
        "functional": "unregularized_hydrogen_exact",
        "basis": BASIS,
        "grid_level": GRID_LEVEL,
        "converged": bool(mf.converged),
        "final_damping": float(mf.damp),
        "final_level_shift_ha": float(mf.level_shift),
        "total_energy_ha": float(mf.e_tot),
        "homo_ha": homo_ha,
        "minus_homo_ev": minus_homo_ev,
        "experimental_ip_ev": reference_ip,
        "absolute_error_ev": abs(minus_homo_ev - reference_ip),
        "negative_virtual_alpha_count": int(
            np.sum(
                (occupations[0] <= occupation_threshold)
                & (energies[0] < negative_threshold)
            )
        ),
        "negative_virtual_beta_count": int(
            np.sum(
                (occupations[1] <= occupation_threshold)
                & (energies[1] < negative_threshold)
            )
        ),
    }
    with (OUTPUT_DIR / "hydrogen_exact_scf.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(row), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)

    payload = {
        "definition": (
            "self-consistent unregularized hydrogen-exact GGA; no density "
            "window, iso-orbital gate, or upper large-gradient cutoff"
        ),
        "input_file": str(INPUT_PATH.relative_to(ROOT)),
        "input_settings": INPUT,
        "numerical_floor_only": True,
        "selected_parameters": {
            "s0": SELECTED_S0,
            "p": SELECTED_P,
        },
        "stages": stage_records,
        "result": row,
    }
    (OUTPUT_DIR / "hydrogen_exact_scf.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if not mf.converged or mf.level_shift != 0.0:
        raise RuntimeError("Unregularized H-exact final zero-shift SCF failed")


if __name__ == "__main__":
    main()
