#!/usr/bin/env python3
"""He with standard functionals (with correlation), including global hybrids.

Reports -eps_HOMO, the number of negative virtual orbitals, and the lowest
virtual eigenvalue in aug-cc-pV5Z.  These are the reference numbers quoted in
the discussion of global hybrids: 20-25 % exact exchange raises -eps_HOMO but
binds no virtual level.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto


ROOT = Path(__file__).resolve().parents[2]
INPUT = json.loads((ROOT / "inputs" / "he_hybrids.json").read_text(encoding="utf-8"))
if int(INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/he_hybrids.json schema_version")
OUTPUT_DIR = ROOT / "results" / "atomic"
OUTPUT = OUTPUT_DIR / "he_hybrids.csv"
HARTREE_TO_EV = 27.211386245988


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)
    mol = gto.M(atom=INPUT["atom"], basis=INPUT["basis"],
                unit=INPUT["coordinate_unit"], verbose=0)
    rows = []
    for entry in INPUT["functionals"]:
        mf = dft.RKS(mol)
        mf.xc = entry["xc"]
        mf.grids.level = int(INPUT["grid_level"])
        mf.conv_tol = float(INPUT["conv_tol"])
        mf.max_cycle = int(INPUT["max_cycle"])
        mf.kernel()
        energies, occupations = mf.mo_energy, mf.mo_occ
        homo = float(np.max(energies[occupations > 0]))
        virtual = energies[occupations == 0]
        negative = virtual < float(INPUT["negative_virtual_threshold_ha"])
        rows.append({
            "functional": entry["name"], "xc": entry["xc"],
            "exact_exchange_fraction": entry["exact_exchange_fraction"],
            "converged": bool(mf.converged), "total_energy_ha": float(mf.e_tot),
            "minus_homo_ev": -homo * HARTREE_TO_EV,
            "negative_virtual_spatial_count": int(np.sum(negative)),
            "lowest_virtual_ha": float(np.min(virtual)),
            "pyscf_version": pyscf.__version__,
        })
        print(f"{entry['name']:6s} -eps_HOMO {rows[-1]['minus_homo_ev']:.3f} eV  "
              f"negative virtuals {rows[-1]['negative_virtual_spatial_count']}  "
              f"lowest virtual {rows[-1]['lowest_virtual_ha']:+.4f} Ha", flush=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
