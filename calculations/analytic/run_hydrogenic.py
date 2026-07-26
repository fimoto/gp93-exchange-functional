#!/usr/bin/env python3
"""Evaluate exchange potentials on exact hydrogenic one-electron densities.

The GP93 building block is exact by construction.  The same radial benchmark
also quantifies how the final density regularization changes the strict
hydrogenic limit.  No Gaussian basis or SCF convergence enters this test.
"""

from __future__ import annotations

import csv
import json
import sys
from math import pi
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[2]
FUNCTIONAL_DIR = ROOT / "functional"
OUTPUT_DIR = ROOT / "results" / "analytic"
INPUT_PATH = ROOT / "inputs" / "hydrogenic.json"
INPUT = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
if int(INPUT.get("schema_version", -1)) != 1:
    raise ValueError("Unsupported inputs/hydrogenic.json schema_version")
sys.path.insert(0, str(FUNCTIONAL_DIR))

import mgga_fmix as REG  # noqa: E402
import hydrogen_exact as BARE  # noqa: E402


FMIX_S0 = float(BARE.DEFAULT_S0)
FMIX_P = int(BARE.DEFAULT_P)
HARTREE_TO_EV = 27.211386245988
SYSTEMS = tuple(
    (entry["label"], int(entry["nuclear_charge"]))
    for entry in INPUT["systems"]
)
SCALED_GRID = INPUT["scaled_radial_grid"]


def radial_divergence(
    coefficient: np.ndarray,
    gradient: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    flux = radius * radius * coefficient * gradient
    derivative = CubicSpline(radius, flux).derivative()(radius)
    return derivative / (radius * radius)


def pbe_channel_potential(
    n: np.ndarray,
    gradient: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    m = np.maximum(2.0 * n, 1.0e-13)
    sigma_m = np.maximum(4.0 * gradient * gradient, 1.0e-280)
    s_raw = np.sqrt(sigma_m) / (2.0 * REG.C1 * m ** (4.0 / 3.0))
    s = np.clip(s_raw, REG.S_MIN, REG.S_MAX)
    active = (s_raw > REG.S_MIN) & (s_raw < REG.S_MAX)
    f = REG._pbe_enhancement(s)
    df = REG._pbe_derivative(s)
    phi = REG.A_X * m ** (4.0 / 3.0)
    ds_dm = -(4.0 / 3.0) * s_raw / m * active
    ds_dsigma_m = s_raw / (2.0 * sigma_m) * active
    de_dm = 0.5 * (
        (4.0 / 3.0) * (phi / m) * f + phi * df * ds_dm
    )
    de_dsigma_m = 0.5 * phi * df * ds_dsigma_m
    vrho = 2.0 * de_dm
    vsigma = 4.0 * de_dsigma_m
    return vrho - 2.0 * radial_divergence(vsigma, gradient, radius)


def regularized_potential(
    n: np.ndarray,
    gradient: np.ndarray,
    radius: np.ndarray,
) -> tuple[np.ndarray, float]:
    sigma = gradient * gradient
    tau = sigma / np.maximum(8.0 * n, 1.0e-300)
    _energy, vrho, vsigma, vtau = REG._channel(n, sigma, tau)
    maximum_vtau = float(np.max(np.abs(vtau)))
    potential = vrho - 2.0 * radial_divergence(vsigma, gradient, radius)
    return potential, maximum_vtau


def building_block_potential(
    n: np.ndarray,
    gradient: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """Evaluate the published unregularized functional derivative."""

    _energy, vrho, vsigma = BARE._channel(n, gradient * gradient)
    return vrho - 2.0 * radial_divergence(vsigma, gradient, radius)


def exact_hydrogenic_density(z: int, radius: np.ndarray) -> np.ndarray:
    return z**3 / pi * np.exp(-2.0 * z * radius)


def hartree_potential(z: int, radius: np.ndarray) -> np.ndarray:
    return (
        1.0 - np.exp(-2.0 * z * radius) * (1.0 + z * radius)
    ) / radius


def expectation(
    density: np.ndarray,
    value: np.ndarray,
    radius: np.ndarray,
) -> float:
    return float(simpson(4.0 * pi * radius * radius * density * value, x=radius))


def evaluate(name: str, z: int) -> list[dict]:
    # Uniform in x=Zr gives the same resolution for every scaled density.
    start = float(SCALED_GRID["start"])
    stop = float(SCALED_GRID["stop"])
    step = float(SCALED_GRID["step"])
    x = np.arange(start, stop + step, step)
    radius = x / z
    density = exact_hydrogenic_density(z, radius)
    gradient = -2.0 * z * density
    v_h = hartree_potential(z, radius)

    REG.CORR = ""
    REG.set_fmix_parameters(FMIX_S0, FMIX_P)
    BARE.set_parameters(FMIX_S0, FMIX_P)
    v_regularized, maximum_vtau = regularized_potential(
        density, gradient, radius
    )
    v_pbe = pbe_channel_potential(density, gradient, radius)
    v_building_block = building_block_potential(density, gradient, radius)

    normalization = expectation(density, np.ones_like(radius), radius)
    base_eigenvalue = -0.5 * z * z
    methods = {
        "unregularized_H_exact": v_building_block,
        "regularized_Fmix": v_regularized,
        "PBE_X": v_pbe,
    }
    rows = []
    for method, v_x in methods.items():
        correction = expectation(density, v_h + v_x, radius)
        epsilon = base_eigenvalue + correction
        rows.append(
            {
                "system": name,
                "Z": z,
                "method": method,
                "density": "exact hydrogenic 1s",
                "density_normalization": normalization,
                "epsilon_1s_Ha": epsilon,
                "minus_epsilon_1s_Ha": -epsilon,
                "exact_Z2_over_2_Ha": 0.5 * z * z,
                "signed_error_mHa": 1000.0 * (-epsilon - 0.5 * z * z),
                "absolute_error_mHa": 1000.0 * abs(
                    -epsilon - 0.5 * z * z
                ),
                "minus_epsilon_1s_eV": -epsilon * HARTREE_TO_EV,
                "hartree_plus_exchange_expectation_Ha": correction,
                "maximum_abs_vtau": (
                    maximum_vtau if method == "regularized_Fmix" else 0.0
                ),
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("hydrogenic_ions.csv", "hydrogenic_ions.json"):
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    rows = [
        row
        for name, z in SYSTEMS
        for row in evaluate(name=name, z=z)
    ]
    with (OUTPUT_DIR / "hydrogenic_ions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "definition": {
            "unregularized_H_exact": (
                "functional derivative of the unregularized GP93/PBE "
                "hydrogen-exact crossover in functional/hydrogen_exact.py "
                "on exact hydrogenic densities"
            ),
            "regularized_Fmix": (
                "final alpha- and log-density-windowed implementation evaluated on "
                "the same exact densities"
            ),
            "PBE_X": "PBE exchange-only potential on the same exact densities",
        },
        "selected_parameters": {"s0": FMIX_S0, "p": FMIX_P},
        "input_file": str(INPUT_PATH.relative_to(ROOT)),
        "input_settings": INPUT,
        "radial_grid": {
            "variable": SCALED_GRID["variable"],
            "minimum": float(SCALED_GRID["start"]),
            "maximum": float(SCALED_GRID["stop"]),
            "spacing": float(SCALED_GRID["step"]),
        },
        "rows": rows,
    }
    (OUTPUT_DIR / "hydrogenic_ions.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
