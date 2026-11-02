#!/usr/bin/env python3
"""Regenerate paper figures from saved numerical results only.

This is deliberately a *plot-only* program.  It reads the CSV/JSON files
written by the calculation stages and never imports PySCF, the functional
implementation, or any calculation driver.  It is therefore useful for
inspecting the separation between expensive electronic-structure calculations
and the subsequent presentation of their recorded results.

Examples
--------
Regenerate the calibration figure after inspecting its CSV file::

    python calculations/plot_results.py calibration

Regenerate one figure from its saved inputs::

    python calculations/plot_results.py fig-he-levels

Regenerate all figures from already-computed results::

    python calculations/plot_results.py all

The program fails with a list of missing result files if the requested
calculation stage has not yet been run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

import figstyle


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

STAGE_OUTPUTS = {
    "calibration": ("fig_calibration.pdf", "fig_calibration.png"),
    "analytic": (
        "fig_gp93_building_block.pdf",
        "fig_gp93_building_block.png",
        "fig_hydrogen_regularization.png",
    ),
    "atomic": (
        "fig_atomic_systematics.png",
        "fig_he_levels.png",
        "fig_he_comparison.png",
    ),
    "h2": ("fig_h2.pdf", "fig_h2.png"),
    "potentials": ("fig_radial_potentials.pdf", "fig_radial_potentials.png"),
    "radial": ("fig_spectrum.pdf", "fig_spectrum.png"),
    "stability": ("fig_realspace.pdf", "fig_realspace.png"),
}

DIRECT_OUTPUTS = {
    "fig-calibration": ("fig_calibration.pdf", "fig_calibration.png"),
    "fig-gp93-building-block": ("fig_gp93_building_block.pdf", "fig_gp93_building_block.png"),
    "fig-hydrogen-regularization": ("fig_hydrogen_regularization.png",),
    "fig-atomic-systematics": ("fig_atomic_systematics.png",),
    "fig-he-levels": ("fig_he_levels.png",),
    "fig-he-comparison": ("fig_he_comparison.png",),
    "fig-h2": ("fig_h2.pdf", "fig_h2.png"),
    "fig-radial-potentials": ("fig_radial_potentials.pdf", "fig_radial_potentials.png"),
    "fig-spectrum": ("fig_spectrum.pdf", "fig_spectrum.png"),
    "fig-realspace": ("fig_realspace.pdf", "fig_realspace.png"),
}

PUBLICATION_PLOT_STYLE = {
    "font.size": 8.5,
    "axes.titlesize": 9.3,
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.2,
    "figure.titlesize": 10.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "figure.dpi": 150,
}

SAVEFIG_OPTIONS = {
    "dpi": 300,
    "bbox_inches": "tight",
    "pad_inches": 0.05,
}


def require_files(paths: Iterable[Path], stage: str) -> None:
    """Raise a useful error before attempting to parse incomplete results."""

    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return
    displayed = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
    raise FileNotFoundError(
        f"Cannot plot stage '{stage}'; required saved result files are absent:\n"
        f"{displayed}\n"
        f"Run the corresponding calculation stage first."
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.relative_to(ROOT)}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data rows found in {path.relative_to(ROOT)}")
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value.strip() == "":
        return math.nan
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Column {key!r} contains non-numeric value {value!r}") from exc


def as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected a Boolean value, received {value!r}")


def columns(rows: list[dict[str, str]], *names: str) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([as_float(row, name) for row in rows], dtype=float)
        for name in names
    }



def _analytic_parameters(stage: str) -> tuple[float, int, float, float, float]:
    metadata_path = RESULTS_DIR / "analytic" / "analytic_metadata.json"
    require_files((metadata_path,), stage)
    metadata = read_json(metadata_path)
    selected = metadata["selected_parameters"]
    parameters = metadata["functional_parameters"]
    return (
        float(selected["s0"]),
        int(selected["p"]),
        float(parameters["density_window_high"]),
        float(parameters["density_window_low"]),
        float(parameters["s_max"]),
    )



def plot_hydrogen_regularization(
    rows: list[dict[str, str]],
    n_window_high: float,
    n_window_low: float,
    s_max: float,
) -> Path:
    data = columns(
        rows,
        "r_bohr",
        "n_H",
        "s_raw",
        "X_n",
        "minus_r_vx_exact",
        "minus_r_vx_bare_GP93",
        "minus_r_vx_unregularized_H_exact",
        "minus_r_vx_implemented",
    )
    order = np.argsort(data["r_bohr"])
    for name in data:
        data[name] = data[name][order]
    r = data["r_bohr"]
    r_window_high = 0.5 * math.log(1.0 / (math.pi * n_window_high))
    r_window_low = 0.5 * math.log(1.0 / (math.pi * n_window_low))
    r_sclip = 1.5 * math.log(s_max * (6.0 * math.pi) ** (1.0 / 3.0))

    plt.rcParams.update(PUBLICATION_PLOT_STYLE)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.20, 5.00),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0]},
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(
        r,
        data["minus_r_vx_exact"],
        color="black",
        linestyle=":",
        linewidth=1.4,
        label=r"exact H: $-rv_x=rv_H$",
    )
    ax.plot(
        r,
        data["minus_r_vx_bare_GP93"],
        "-.",
        color="0.5",
        linewidth=1.5,
        label="bare GP93 branch",
    )
    ax.plot(
        r,
        data["minus_r_vx_unregularized_H_exact"],
        "--",
        color="#cc443d",
        linewidth=2.0,
        label="unregularized H-exact crossover",
    )
    ax.plot(
        r,
        data["minus_r_vx_implemented"],
        color="#225ea8",
        linewidth=2.2,
        label=r"regularized: $\chi+G(\alpha)X_n$",
    )
    ax.axhline(1.0, color="0.5", linewidth=0.7)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(r_window_high, color="#7a5195", linestyle="-.", linewidth=1.0)
    ax.axvline(r_window_low, color="#ef8a62", linestyle=":", linewidth=1.2)
    ax.axvspan(r_window_low, r[-1], color="#ef8a62", alpha=0.09)
    ax.text(
        r_window_high + 0.08,
        1.48,
        r"$X_n$ starts",
        color="#7a5195",
        fontsize=7.4,
        rotation=90,
        va="top",
    )
    ax.text(
        r_window_low + 0.08,
        1.48,
        r"$X_n=0$",
        color="#b35836",
        fontsize=7.4,
        rotation=90,
        va="top",
    )
    ax.annotate(
        "intermediate-range shape retained;\nfar tail switched off smoothly",
        xy=(10.0, np.interp(10.0, r, data["minus_r_vx_implemented"])),
        xytext=(6.8, 0.30),
        fontsize=7.6,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#225ea8"},
        color="#225ea8",
    )
    ax.set(
        ylabel=r"$-r\,v_x(r)$",
        xlim=(0.2, 15.0),
        ylim=(-0.08, 1.60),
        title="Exact-H intermediate range and regularized far tail",
    )
    ax.legend(markerfirst=False, loc="upper left", fontsize=7.4, framealpha=0.93)

    ax = axes[1]
    ax.plot(
        r,
        data["X_n"],
        color="#225ea8",
        linewidth=2.0,
        label=r"density window $X_n$",
    )
    ax.axvline(r_window_high, color="#7a5195", linestyle="-.", linewidth=1.0)
    ax.axvline(r_window_low, color="#ef8a62", linestyle=":", linewidth=1.2)
    ax.axvline(
        r_sclip,
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        label=rf"raw $s={s_max:.0f}$ (already hard-zeroed)",
    )
    ax.set(
        xlabel=r"$r$ (bohr)",
        ylabel=r"$X_n$",
        ylim=(-0.03, 1.04),
    )
    ax2 = ax.twinx()
    ax2.plot(
        r,
        data["n_H"],
        color="#238b45",
        linewidth=1.3,
        alpha=0.9,
        label=r"$n_H$",
    )
    ax2.plot(
        r,
        data["s_raw"],
        color="#dd8d29",
        linewidth=1.3,
        alpha=0.9,
        label=r"raw $s$",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$n_H$ and raw $s$ (log scale)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="outside lower center",
        ncol=4,
        fontsize=7.2,
        framealpha=0.93,
    )
    output = FIGURE_DIR / "fig_hydrogen_regularization.png"
    fig.savefig(output, **SAVEFIG_OPTIONS)
    plt.close(fig)
    return output


def plot_analytic() -> tuple[Path, ...]:
    building_path = RESULTS_DIR / "analytic" / "gp93_building_block.csv"
    hydrogen_path = RESULTS_DIR / "analytic" / "hydrogen_regularization.csv"
    require_files((building_path, hydrogen_path), "analytic")
    s0, p_switch, n_high, n_low, s_max = _analytic_parameters("analytic")
    return (
        *plot_gp93_building_block(read_csv(building_path), s0, p_switch),
        plot_hydrogen_regularization(
            read_csv(hydrogen_path),
            n_high,
            n_low,
            s_max,
        ),
    )


def plot_fig_gp93_building_block() -> tuple[Path, ...]:
    result_path = RESULTS_DIR / "analytic" / "gp93_building_block.csv"
    require_files((result_path,), "fig-gp93-building-block")
    s0, p_switch, _, _, _ = _analytic_parameters(
        "fig-gp93-building-block"
    )
    return plot_gp93_building_block(read_csv(result_path), s0, p_switch)


def plot_fig_hydrogen_regularization() -> tuple[Path, ...]:
    result_path = RESULTS_DIR / "analytic" / "hydrogen_regularization.csv"
    require_files((result_path,), "fig-hydrogen-regularization")
    _, _, n_high, n_low, s_max = _analytic_parameters(
        "fig-hydrogen-regularization"
    )
    return (
        plot_hydrogen_regularization(
            read_csv(result_path),
            n_high,
            n_low,
            s_max,
        ),
    )


def _atomic_orders(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    systems = list(dict.fromkeys(row["system"] for row in rows))
    functionals = list(dict.fromkeys(row["functional"] for row in rows))
    label_by_functional = {
        row["functional"]: row["functional_label"] for row in rows
    }
    labels = [label_by_functional[name] for name in functionals]
    return systems, functionals, labels


def plot_atomic_systematics(rows: list[dict[str, str]]) -> Path:
    systems, functionals, labels = _atomic_orders(rows)
    short_labels = {
        "PBE_X": "PBE-X",
        "SCAN_X": "SCAN-X",
        "Fmix_X": r"$F_{\rm mix}$-X",
    }
    display_labels = [
        short_labels.get(functional, label)
        for functional, label in zip(functionals, labels)
    ]
    run_map = {(row["system"], row["functional"]): row for row in rows}
    missing_pairs = [
        (system, functional)
        for system in systems
        for functional in functionals
        if (system, functional) not in run_map
    ]
    if missing_pairs:
        raise ValueError(f"atomic_systematics.csv lacks rows for {missing_pairs}")
    colors = ("#9b9b9b", "#8a84c8", "#2664a8")
    if len(functionals) > len(colors):
        colors = tuple(plt.cm.tab10(np.linspace(0.0, 1.0, len(functionals))))

    plt.rcParams.update(PUBLICATION_PLOT_STYLE)
    fig = plt.figure(figsize=(7.05, 5.35), layout="constrained")
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.35, 1.0),
        hspace=0.16,
    )
    ax_ip = fig.add_subplot(grid[0])
    ax_count = fig.add_subplot(grid[1])
    x = np.arange(len(systems), dtype=float)
    width = min(0.23, 0.75 / max(len(functionals), 1))

    center = 0.5 * (len(functionals) - 1)
    for index, (functional, label, color) in enumerate(
        zip(functionals, labels, colors)
    ):
        values = [
            (
                as_float(run_map[(system, functional)], "minus_homo_ev")
                if as_bool(run_map[(system, functional)]["converged"])
                else np.nan
            )
            for system in systems
        ]
        ax_ip.bar(
            x + (index - center) * width,
            values,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            label=label,
        )
    experimental = [
        as_float(run_map[(system, functionals[0])], "experimental_ip_ev")
        for system in systems
    ]
    ax_ip.scatter(
        x,
        experimental,
        marker="D",
        color="#ce4038",
        edgecolor="white",
        linewidth=0.7,
        s=65,
        label="experimental IP (reference)",
        zorder=4,
    )
    ax_ip.set_xticks(x, systems)
    ax_ip.set_ylabel(r"$-\epsilon_{\rm HOMO}$ (eV)")
    ax_ip.set_title("Atomic exchange-only systematics")
    ax_ip.grid(axis="y", alpha=0.20)
    handles, legend_labels = ax_ip.get_legend_handles_labels()
    handle_by_label = dict(zip(legend_labels, handles))
    source_labels = ["experimental IP (reference)", *labels]
    ax_ip.legend(
        [handle_by_label[label] for label in source_labels],
        ["Expt. IP", *display_labels],
        ncol=2,
        fontsize=8.0,
        loc="upper right",
        framealpha=0.95,
    )

    count_values = np.zeros((len(systems), len(functionals)))
    count_text: list[list[str]] = []
    methods: dict[str, str] = {}
    for row_index, system in enumerate(systems):
        row_text: list[str] = []
        for column_index, functional in enumerate(functionals):
            row = run_map[(system, functional)]
            methods[system] = row["method"]
            if not as_bool(row["converged"]):
                count_values[row_index, column_index] = np.nan
                row_text.append("not\nconverged")
            elif row["method"] == "RKS":
                count = int(as_float(row, "negative_virtual_spatial_count"))
                count_values[row_index, column_index] = count
                row_text.append(f"spatial\n{count}")
            else:
                alpha_count = int(as_float(row, "negative_virtual_alpha_count"))
                beta_count = int(as_float(row, "negative_virtual_beta_count"))
                count_values[row_index, column_index] = alpha_count + beta_count
                row_text.append(rf"$\alpha$ {alpha_count} / $\beta$ {beta_count}")
        count_text.append(row_text)

    count_scale_max = max(1.0, float(np.nanmax(count_values)))
    count_cmap = plt.cm.Blues.copy()
    count_cmap.set_bad("#eeeeee")
    image = ax_count.imshow(
        np.ma.masked_invalid(count_values),
        aspect="auto",
        cmap=count_cmap,
        vmin=0,
        vmax=count_scale_max,
    )
    ax_count.set_xticks(range(len(functionals)), display_labels)
    ax_count.set_yticks(
        range(len(systems)),
        [
            f"{system} (RKS spatial)"
            if methods[system] == "RKS"
            else f"{system} (UKS spin)"
            for system in systems
        ],
    )
    ax_count.set_title(
        "Negative virtual diagnostics\n"
        "RKS: spatial count; UKS: alpha/beta spin counts"
    )
    for row_index in range(len(systems)):
        for column_index in range(len(functionals)):
            value = count_values[row_index, column_index]
            color = (
                "white"
                if np.isfinite(value) and value > 0.55 * count_scale_max
                else "black"
            )
            ax_count.text(
                column_index,
                row_index,
                count_text[row_index][column_index],
                ha="center",
                va="center",
                fontsize=8.2,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=ax_count, pad=0.018, fraction=0.040)
    colorbar.set_label("diagnostic count", fontsize=8.5, labelpad=5)
    colorbar.ax.tick_params(labelsize=7.8)
    output = FIGURE_DIR / "fig_atomic_systematics.png"
    fig.savefig(output, **SAVEFIG_OPTIONS)
    plt.close(fig)
    return output


def plot_he_levels(rows: list[dict[str, str]]) -> Path:
    groups = sorted(rows, key=lambda row: as_float(row, "energy_ha"))
    plt.rcParams.update(PUBLICATION_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(3.40, 2.85), layout="constrained")
    ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
    colors = plt.cm.Blues(np.linspace(0.55, 0.9, max(len(groups), 1)))
    if groups:
        for x_value, (group, color) in enumerate(zip(groups, colors)):
            energy = as_float(group, "energy_ha")
            degeneracy = int(as_float(group, "spatial_degeneracy"))
            offsets = (
                np.linspace(-0.14, 0.14, degeneracy)
                if degeneracy > 1
                else np.asarray([0.0])
            )
            ax.hlines(energy, x_value - 0.25, x_value + 0.25, color=color, lw=2.0)
            ax.scatter(
                x_value + offsets,
                np.full(degeneracy, energy),
                s=72,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
            text = (
                rf"$\epsilon={energy:.5f}$ Ha"
                "\n"
                rf"$l\approx{int(as_float(group, 'l_est'))}$ "
                rf"({group['l_symbol']}), $g_{{\rm spatial}}={degeneracy}$"
                "\n"
                rf"$n_{{\rm eff}}={as_float(group, 'n_eff'):.3f}$"
            )
            ax.annotate(
                text,
                (x_value, energy),
                xytext=(0, 24 if x_value % 2 == 0 else -58),
                textcoords="offset points",
                ha="center",
                va="bottom" if x_value % 2 == 0 else "top",
                fontsize=7.0,
            )
        ax.set_xticks(
            range(len(groups)),
            [
                f"{group['l_symbol']}-like"
                for group in groups
            ],
        )
        ax.set_ylim(min(as_float(group, "energy_ha") for group in groups) - 0.045, 0.025)
        ax.set_xlim(-0.55, len(groups) - 0.45)
    else:
        ax.text(
            0.5,
            0.5,
            "No negative virtual RKS spatial orbitals",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_xticks([])
    ax.set_ylabel("virtual gKS eigenvalue (Ha)")
    ax.set_title(r"He: negative virtual gKS levels (RKS)")
    output = FIGURE_DIR / "fig_he_levels.png"
    fig.savefig(output, **SAVEFIG_OPTIONS)
    plt.close(fig)
    return output


def plot_he_comparison(rows: list[dict[str, str]]) -> Path:
    _, functionals, _ = _atomic_orders(rows)
    he_rows = {
        row["functional"]: row for row in rows if row["system"] == "He"
    }
    missing = [name for name in functionals if name not in he_rows]
    if missing:
        raise ValueError(f"atomic_systematics.csv lacks He rows for {missing}")
    values = [as_float(he_rows[name], "minus_homo_ev") for name in functionals]
    counts = [
        int(as_float(he_rows[name], "negative_virtual_spatial_count"))
        for name in functionals
    ]
    labels_by_name = {
        "PBE_X": "PBE-X",
        "SCAN_X": "SCAN-X",
        "Fmix_X": r"$F_{\mathrm{mix}}$-X",
    }
    labels = [
        labels_by_name.get(name, he_rows[name]["functional_label"])
        for name in functionals
    ]
    colors = ("#9b9b9b", "#8a84c8", "#2664a8")
    if len(values) > len(colors):
        colors = tuple(plt.cm.tab10(np.linspace(0.0, 1.0, len(values))))
    experimental_ip = as_float(he_rows[functionals[0]], "experimental_ip_ev")

    plt.rcParams.update(PUBLICATION_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(3.40, 2.85), layout="constrained")
    x = np.arange(len(values))
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axhline(
        experimental_ip,
        color="#ce4038",
        ls="--",
        lw=1.5,
        label=f"Expt. IP {experimental_ip:.2f} eV",
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.55,
            rf"$N_{{\rm virt}}={count}$",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$-\epsilon_{\rm HOMO}$ (eV)")
    ax.set_ylim(0, max(max(values), experimental_ip) + 4.5)
    ax.set_title("He exchange-only comparison (RKS)")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(markerfirst=False, loc="upper left", fontsize=7.2)
    output = FIGURE_DIR / "fig_he_comparison.png"
    fig.savefig(output, **SAVEFIG_OPTIONS)
    plt.close(fig)
    return output


def plot_atomic() -> tuple[Path, ...]:
    systematics_path = RESULTS_DIR / "atomic" / "atomic_systematics.csv"
    levels_path = RESULTS_DIR / "atomic" / "he_negative_virtual_levels.csv"
    require_files((systematics_path, levels_path), "atomic")
    systematics = read_csv(systematics_path)
    levels = read_csv(levels_path)
    return (
        plot_atomic_systematics(systematics),
        plot_he_levels(levels),
        plot_he_comparison(systematics),
    )


def plot_fig_atomic_systematics() -> tuple[Path, ...]:
    result_path = RESULTS_DIR / "atomic" / "atomic_systematics.csv"
    require_files((result_path,), "fig-atomic-systematics")
    return (plot_atomic_systematics(read_csv(result_path)),)


def plot_fig_he_levels() -> tuple[Path, ...]:
    result_path = RESULTS_DIR / "atomic" / "he_negative_virtual_levels.csv"
    require_files((result_path,), "fig-he-levels")
    return (plot_he_levels(read_csv(result_path)),)


def plot_fig_he_comparison() -> tuple[Path, ...]:
    result_path = RESULTS_DIR / "atomic" / "atomic_systematics.csv"
    require_files((result_path,), "fig-he-comparison")
    return (plot_he_comparison(read_csv(result_path)),)






def plot_calibration() -> tuple[Path, ...]:
    """Five-system MAE versus the outer-switch scale s0 (one column)."""

    result_path = RESULTS_DIR / "calibration" / "calibration_results.json"
    require_files((result_path,), "calibration")
    document = read_json(result_path)
    scan = document.get("scan")
    if not isinstance(scan, list) or not scan:
        raise ValueError("calibration_results.json has no non-empty 'scan' array")
    valid = [point for point in scan if point.get("mae_5system_eV") is not None]
    if len(valid) != len(scan):
        raise ValueError("calibration scan contains a point without an MAE")
    x = np.asarray([float(point["s0"]) for point in valid])
    y = np.asarray([float(point["mae_5system_eV"]) for point in valid])
    minimum = int(np.argmin(y))

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_IN, 2.3), layout="constrained")
    ax.plot(x, y, color=figstyle.BLACK, lw=figstyle.LINE_WIDTH["thin"],
            marker="o", ms=4.0, mfc="white", mec=figstyle.BLACK, mew=0.8)
    ax.plot([x[minimum]], [y[minimum]], ls="none", marker="o", ms=4.0,
            mfc=figstyle.RED, mec=figstyle.RED)
    ax.set_xlabel(r"$s_0$")
    ax.set_ylabel(r"MAE of $-\varepsilon_{\rm HOMO}$ (eV)")
    ax.set_xlim(0.10, 0.37)
    ax.set_ylim(0.75, 1.05)
    ax.minorticks_on()
    outputs = figstyle.save(fig, FIGURE_DIR / "fig_calibration")
    plt.close(fig)
    return outputs


def plot_gp93_building_block(
    rows: list[dict[str, str]],
    s0: float,
    p_switch: int,
) -> tuple[Path, ...]:
    """w(z), the crossover region of F(s), and the large-s branch (full page)."""

    data = columns(rows, "s", "z_raw", "F_PBE", "F_GP93_bare_analytic",
                   "F_GP93_fmix_py_zclipped", "Fmix_direct_selected_parameters")
    order = np.argsort(data["s"])
    for name in data:
        data[name] = data[name][order]
    s = data["s"]
    z = data["z_raw"]
    w_bare = data["F_GP93_bare_analytic"] / (8.0 * math.pi / 3.0 * s)

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.PAGE_IN, 2.35),
                             layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    ax = axes[0]
    series = (z >= -6.0) & (z <= 0.0)
    exact = (z > 0.0) & (z <= 12.0)
    asymptotic = (z >= 12.0) & (z <= 30.0)
    ax.plot(z[series], w_bare[series], color=figstyle.BLUE, ls=(0, (5, 2)),
            lw=figstyle.LINE_WIDTH["main"], label=r"series, $z\leq0$")
    ax.plot(z[exact], w_bare[exact], color=figstyle.RED, ls="-",
            lw=figstyle.LINE_WIDTH["main"], label=r"exact, $0<z\leq12$")
    ax.plot(z[asymptotic], w_bare[asymptotic], color=figstyle.GREEN,
            ls=(0, (1.2, 1.5)), lw=figstyle.LINE_WIDTH["main"] + 0.4,
            label="asymptotic")
    zero = int(np.argmin(np.abs(z)))
    ax.plot([0.0], [w_bare[zero]], ls="none", marker="o", ms=3.5,
            mfc=figstyle.BLACK, mec=figstyle.BLACK)
    ax.axhline(0.0, color=figstyle.BLACK, lw=figstyle.LINE_WIDTH["axis"])
    ax.set_xlim(-6.0, 30.0)
    ax.set_ylim(-2.4, 1.6)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$w(z)$")
    ax.legend(markerfirst=False, loc="upper right")
    figstyle.panel_label(ax, "(a)")

    ax = axes[1]
    detail = (s >= 1.0e-2) & (s <= 4.0)
    ax.plot(s[detail], data["F_PBE"][detail], **figstyle.STYLE["PBE"], label="PBE")
    ax.plot(s[detail], data["F_GP93_bare_analytic"][detail],
            **figstyle.STYLE["bare"], label=r"$F_{\rm GP93}$")
    ax.plot(s[detail], data["Fmix_direct_selected_parameters"][detail],
            **figstyle.STYLE["GP93"], label=r"$F_{\rm mix}$")
    ax.axvline(s0, color=figstyle.GRAY, lw=figstyle.LINE_WIDTH["thin"], ls=":")
    ax.axvline((6.0 * math.pi) ** (-1.0 / 3.0), color=figstyle.BLACK,
               lw=figstyle.LINE_WIDTH["thin"], ls=":")
    ax.axhline(0.0, color=figstyle.BLACK, lw=figstyle.LINE_WIDTH["axis"])
    ax.set_xscale("log")
    ax.set_xlim(1.0e-2, 4.0)
    ax.set_ylim(-8.0, 3.0)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$F(s)$")
    ax.text(s0 * 0.93, -2.0, r"$s_0$", ha="right", fontsize=figstyle.FONT_SIZE["annotation"])
    ax.text((6.0 * math.pi) ** (-1.0 / 3.0) * 1.07, -2.0, r"$s_{\rm H}$", ha="left",
            fontsize=figstyle.FONT_SIZE["annotation"])
    ax.legend(markerfirst=False, loc="lower left")
    figstyle.panel_label(ax, "(b)")

    ax = axes[2]
    wide = (s >= 1.0e-3) & (s <= 1.0e5)
    magnitude = np.abs(data["Fmix_direct_selected_parameters"])
    ax.plot(s[wide], np.abs(data["F_GP93_bare_analytic"])[wide],
            **figstyle.STYLE["bare"], label=r"$|F_{\rm GP93}|$")
    ax.plot(s[wide], magnitude[wide], **figstyle.STYLE["GP93"], label=r"$|F_{\rm mix}|$")
    ax.plot(s[wide], data["F_PBE"][wide], **figstyle.STYLE["PBE"], label="PBE")
    large = np.logspace(1.0, 5.0, 200)
    lead = 4.9014 * large * np.log(large) ** (2.0 / 3.0)
    ax.plot(large, lead, color=figstyle.BLACK, lw=figstyle.LINE_WIDTH["thin"],
            ls=(0, (3, 2)), label=r"$4.90\,s(\ln s)^{2/3}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-3, 1.0e5)
    ax.set_ylim(1.0e-2, 1.0e7)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$|F(s)|$")
    ax.legend(markerfirst=False, loc="upper left", bbox_to_anchor=(0.10, 1.0))
    figstyle.panel_label(ax, "(c)")
    outputs = figstyle.save(fig, FIGURE_DIR / "fig_gp93_building_block")
    plt.close(fig)
    return outputs


def plot_h2() -> tuple[Path, ...]:
    """H2 binding curves of GP93-LYP and FCI (one column)."""

    curve_path = RESULTS_DIR / "h2" / "h2_curve.csv"
    metadata_path = RESULTS_DIR / "h2" / "h2_metadata.json"
    require_files((curve_path, metadata_path), "h2")
    rows = read_csv(curve_path)
    metadata = read_json(metadata_path)
    data = columns(rows, "R_A", "Erel_fmix_eV", "Erel_fci_eV")
    order = np.argsort(data["R_A"])
    r = data["R_A"][order]
    e_fmix = data["Erel_fmix_eV"][order]
    e_fci = data["Erel_fci_eV"][order]
    experimental_re = float(metadata["experimental_reference"]["value"])
    fine = np.linspace(float(r.min()), float(r.max()),
                       int(metadata["plot_interpolation_points"]))

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_IN, 2.45), layout="constrained")
    ax.plot(fine, CubicSpline(r, e_fci)(fine), color=figstyle.BLACK,
            lw=figstyle.LINE_WIDTH["secondary"], label="FCI")
    ax.plot(r, e_fci, ls="none", marker="o", ms=3.2, mfc=figstyle.BLACK,
            mec=figstyle.BLACK)
    ax.plot(fine, CubicSpline(r, e_fmix)(fine), color=figstyle.RED,
            lw=figstyle.LINE_WIDTH["main"], label="GP93-LYP")
    ax.plot(r, e_fmix, ls="none", marker="s", ms=3.2, mfc="white", mec=figstyle.RED,
            mew=0.8)
    ax.axvline(experimental_re, color=figstyle.GRAY, lw=figstyle.LINE_WIDTH["thin"],
               ls=(0, (1.2, 1.5)))
    ax.axhline(0.0, color=figstyle.BLACK, lw=figstyle.LINE_WIDTH["axis"])
    ax.set_xlim(0.45, 1.65)
    ax.set_ylim(-5.2, 2.0)
    ax.set_xlabel("$R$ (Å)")
    ax.set_ylabel(r"$E(R)-2E_{\rm H}$ (eV)")
    ax.minorticks_on()
    ax.legend(markerfirst=False, loc="upper right")
    outputs = figstyle.save(fig, FIGURE_DIR / "fig_h2")
    plt.close(fig)
    return outputs


def plot_potentials() -> tuple[Path, ...]:
    """-r v_x on fixed densities of H, He, and the Na valence electron (full page)."""

    result_path = RESULTS_DIR / "potentials" / "radial_potentials.csv"
    require_files((result_path,), "potentials")
    rows = read_csv(result_path)
    by_system: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_system.setdefault(row["system"], []).append(row)
    expected = ("H_exact_density", "He_RHF_aug_cc_pV5Z", "Na_UHF_SBKJC_3s")
    absent = [system for system in expected if system not in by_system]
    if absent:
        raise ValueError(f"radial_potentials.csv lacks systems {absent}")

    def system_data(name):
        system_rows = by_system[name]
        data = columns(system_rows, "r_bohr", "minus_r_vx_hydrogen_exact_gp93",
                       "minus_r_vx_implemented", "minus_r_vx_pbe_x",
                       "minus_r_vx_self_interaction_free_reference")
        order = np.argsort(data["r_bohr"])
        mask = np.asarray([as_bool(row["density_threshold_mask"])
                           for row in system_rows], dtype=bool)[order]
        return (data["r_bohr"][order], data["minus_r_vx_hydrogen_exact_gp93"][order],
                data["minus_r_vx_implemented"][order], data["minus_r_vx_pbe_x"][order],
                data["minus_r_vx_self_interaction_free_reference"][order], mask)

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.PAGE_IN, 2.3), layout="constrained")
    limits = (((0.0, 15.0), (-0.1, 1.4)), ((0.0, 3.3), (0.0, 1.2)), ((3.2, 5.5), (0.0, 1.2)))
    names = ("H", "He", "Na")
    for k, (system, ax) in enumerate(zip(expected, axes)):
        r, block, impl, pbe, ref, mask = system_data(system)
        ax.plot(r, ref, color=figstyle.BLACK, lw=figstyle.LINE_WIDTH["main"] + 0.6,
                label="reference")
        if k == 0:
            ax.plot(r[mask], block[mask], **figstyle.STYLE["bare"],
                    label="building block")
        ax.plot(r[mask], impl[mask], **figstyle.STYLE["GP93"], label="GP93")
        ax.plot(r[mask], pbe[mask], **figstyle.STYLE["PBE"], label="PBE")
        ax.set_xlim(*limits[k][0])
        ax.set_ylim(*limits[k][1])
        ax.set_xlabel(r"$r$ (bohr)")
        if k == 0:
            ax.set_ylabel(r"$-r\,v_{\rm x}(r)$")
            handles, labels = ax.get_legend_handles_labels()
        ax.minorticks_on()
        figstyle.panel_label(ax, f"({'abc'[k]}) {names[k]}")
    axes[2].legend(handles, labels, markerfirst=False, loc="lower right")
    outputs = figstyle.save(fig, FIGURE_DIR / "fig_radial_potentials")
    plt.close(fig)
    return outputs


def plot_radial() -> tuple[Path, ...]:
    """Bound virtual levels from the basis-free radial solver (full page)."""

    levels_path = RESULTS_DIR / "radial" / "radial_levels.csv"
    exact_path = ROOT / "data" / "he_exact_ks_levels.csv"
    require_files((levels_path, exact_path), "radial")
    rows = read_csv(levels_path)
    he_exact = {0: [], 1: [], 2: []}
    for row in read_csv(exact_path):
        l = "spd".find(row["level"][-1])
        if row["level"] != "1s" and 0 <= l <= 2 and int(row["level"][0]) <= 3:
            he_exact[l].append(as_float(row, "energy_ha"))
    h_exact = {0: [-0.125, -1.0 / 18.0], 1: [-0.125, -1.0 / 18.0], 2: [-1.0 / 18.0]}
    # Solid bars: angular momentum is shown by color and by a fixed slot within
    # each column, labeled s, p, d above the zero line (short bars cannot show a
    # dash pattern legibly).
    level_color = {0: figstyle.RED, 1: figstyle.BLUE, 2: figstyle.GREEN}
    columns_ = (("PBE-x", "PBE_X"), ("GP93-x", "GP93_X"), ("GP93R-x", "GP93R_X"))
    bar_lw = figstyle.LINE_WIDTH["main"] + 0.7

    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.PAGE_IN, 2.5), sharey=True,
                             layout="constrained")
    for panel, (system, exact, exact_label) in enumerate(
            (("H", h_exact, "exact"), ("He", he_exact, "exact KS"))):
        ax = axes[panel]

        def slot(column, l):
            return column + 0.2 + 0.25 * l

        def bar(column, l, energy, marginal=False):
            x0 = slot(column, l)
            if marginal:
                ax.plot([x0 - 0.1, x0 + 0.1], [energy, energy], color=figstyle.GRAY,
                        lw=figstyle.LINE_WIDTH["thin"], ls=":")
            else:
                ax.plot([x0 - 0.1, x0 + 0.1], [energy, energy], color=level_color[l],
                        lw=bar_lw, ls="-", solid_capstyle="butt")

        for i, (_label, key) in enumerate(columns_):
            for row in rows:
                if row["system"] == system and row["functional"] == key:
                    bar(i, int(row["l"]), as_float(row, "energy_ha"),
                        as_bool(row["above_reported_threshold"]))
        for l, values in exact.items():
            for e in values:
                bar(3, l, e)
        for column in range(4):
            for l, name in ((0, "s"), (1, "p"), (2, "d")):
                ax.text(slot(column, l), 0.0035, name, ha="center", va="bottom",
                        color=level_color[l], fontsize=figstyle.FONT_SIZE["annotation"])
            if column > 0:
                ax.axvline(column - 0.05, color=figstyle.GRAY,
                           lw=figstyle.LINE_WIDTH["thin"], ls=(0, (1, 2)))
        ax.set_xticks([i + 0.45 for i in range(4)])
        ax.set_xticklabels([c[0] for c in columns_] + [exact_label])
        ax.tick_params(axis="x", which="both", length=0)
        ax.axhline(0.0, color=figstyle.BLACK, lw=figstyle.LINE_WIDTH["axis"])
        ax.set_xlim(-0.1, 4.0)
        figstyle.panel_label(ax, f"({'ab'[panel]}) {system}", y=0.06)
    axes[0].set_ylabel("virtual eigenvalue (Ha)")
    axes[0].set_ylim(-0.17, 0.016)
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    outputs = figstyle.save(fig, FIGURE_DIR / "fig_spectrum")
    plt.close(fig)
    return outputs


def plot_stability() -> tuple[Path, ...]:
    """F(s), local stability thresholds, and -r v_x on H (full page)."""

    curves_path = RESULTS_DIR / "analytic" / "stability_curves.csv"
    meta_path = RESULTS_DIR / "analytic" / "stability_metadata.json"
    pot_path = RESULTS_DIR / "analytic" / "gp93r_hydrogen_potential.csv"
    require_files((curves_path, meta_path, pot_path), "stability")
    c = columns(read_csv(curves_path), "s", "m_c_pbe", "m_c_gp93", "m_c_gp93r",
                "F_pbe", "F_gp93", "F_gp93r")
    meta = read_json(meta_path)
    p = columns(read_csv(pot_path), "r_bohr", "r_vH", "minus_r_vx_gp93",
                "minus_r_vx_gp93r", "minus_r_vx_pbe")
    s_h = float(meta["s_H"])

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.PAGE_IN, 2.35), layout="constrained")
    ax = axes[0]
    lin = c["s"] <= 1.5
    ax.plot(c["s"][lin], c["F_pbe"][lin], **figstyle.STYLE["PBE"], label="PBE")
    ax.plot(c["s"][lin], c["F_gp93"][lin], **figstyle.STYLE["GP93"], label="GP93")
    ax.plot(c["s"][lin], c["F_gp93r"][lin], **figstyle.STYLE["GP93R"], label="GP93R")
    for x0, text in ((s_h, r"$s_{\rm H}$"), (0.5, r"$s_1$")):
        ax.axvline(x0, color=figstyle.GRAY, lw=figstyle.LINE_WIDTH["thin"], ls=":")
        ax.text(x0 + 0.02, 0.12, text, fontsize=figstyle.FONT_SIZE["annotation"])
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(0.0, 2.5)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$F(s)$")
    ax.legend(markerfirst=False, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    ax.minorticks_on()
    figstyle.panel_label(ax, "(a)")

    ax = axes[1]
    ax.axhspan(1.0e-3, 1.0, color="#e4e4e4", lw=0)
    ax.plot(c["s"], c["m_c_pbe"], **figstyle.STYLE["PBE"], label="PBE")
    ax.plot(c["s"], c["m_c_gp93"], **figstyle.STYLE["GP93"], label="GP93")
    ax.plot(c["s"], c["m_c_gp93r"], **figstyle.STYLE["GP93R"], label="GP93R")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-2, 3.0)
    ax.set_ylim(1.0e-6, 1.0e6)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m_{\rm c}(s)$ (bohr$^{-3}$)")
    ax.legend(markerfirst=False, loc="upper right")
    figstyle.panel_label(ax, "(b)")

    ax = axes[2]
    ax.plot(p["r_bohr"], p["r_vH"], color=figstyle.BLACK,
            lw=figstyle.LINE_WIDTH["main"] + 0.6, label=r"$r\,v_{\rm H}$")
    ax.plot(p["r_bohr"], p["minus_r_vx_gp93"], **figstyle.STYLE["GP93"], label="GP93")
    ax.plot(p["r_bohr"], p["minus_r_vx_gp93r"], **figstyle.STYLE["GP93R"], label="GP93R")
    ax.plot(p["r_bohr"], p["minus_r_vx_pbe"], **figstyle.STYLE["PBE"], label="PBE")
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(-0.1, 1.75)
    ax.set_xlabel(r"$r$ (bohr)")
    ax.set_ylabel(r"$-r\,v_{\rm x}(r)$")
    ax.legend(markerfirst=False, loc="upper right")
    ax.minorticks_on()
    figstyle.panel_label(ax, "(c)")
    outputs = figstyle.save(fig, FIGURE_DIR / "fig_realspace")
    plt.close(fig)
    return outputs


PLOTTERS = {
    "calibration": plot_calibration,
    "analytic": plot_analytic,
    "atomic": plot_atomic,
    "h2": plot_h2,
    "potentials": plot_potentials,
    "radial": plot_radial,
    "stability": plot_stability,
}

DIRECT_PLOTTERS = {
    "fig-calibration": plot_calibration,
    "fig-gp93-building-block": plot_fig_gp93_building_block,
    "fig-hydrogen-regularization": plot_fig_hydrogen_regularization,
    "fig-atomic-systematics": plot_fig_atomic_systematics,
    "fig-he-levels": plot_fig_he_levels,
    "fig-he-comparison": plot_fig_he_comparison,
    "fig-h2": plot_h2,
    "fig-radial-potentials": plot_potentials,
    "fig-spectrum": plot_radial,
    "fig-realspace": plot_stability,
}

TARGET_PLOTTERS = {**PLOTTERS, **DIRECT_PLOTTERS}
TARGET_OUTPUTS = {**STAGE_OUTPUTS, **DIRECT_OUTPUTS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate figures from saved CSV/JSON results without running "
            "electronic-structure calculations."
        )
    )
    parser.add_argument(
        "stage",
        choices=(*PLOTTERS, *DIRECT_PLOTTERS, "all"),
        help="calculation-result group or individual figure to regenerate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    targets = tuple(PLOTTERS) if args.stage == "all" else (args.stage,)
    outputs: list[Path] = []
    try:
        for target in targets:
            target_outputs = TARGET_PLOTTERS[target]()
            outputs.extend(target_outputs)
            print(
                f"{target}: wrote "
                + ", ".join(str(path.relative_to(ROOT)) for path in target_outputs),
                flush=True,
            )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"plot_results.py: {exc}", file=sys.stderr)
        return 2

    expected = {
        FIGURE_DIR / name
        for target in targets
        for name in TARGET_OUTPUTS[target]
    }
    if set(outputs) != expected:
        print(
            "plot_results.py: internal output mapping mismatch",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
