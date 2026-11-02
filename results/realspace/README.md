# Real-space molecular runs (H2 and CH4)

These results come from the finite-difference real-space Kohn-Sham code
RSDFT/OFDFT (development version, commit 76fcb22), which is derived from
RSDFT and is not part of this repository. They are recorded here so that the
numbers quoted in the paper can be checked. They cannot be regenerated from
this repository. Pseudopotential files (Troullier-Martins, `H-PBE-TM.psv`,
`C-PBE-TM.psv`) are not redistributed.

## Settings common to all runs

- Periodic cubic cell, central finite differences of twelfth order
  (`Md = 6`), SCF threshold `1e-10` on the density residual, linear density
  mixing 0.05, start from atomic densities.
- H2 bond length 1.40 bohr along x; CH4 regular tetrahedron with
  C-H = 2.05 bohr.
- `XCtype GP93R` with its default LYP correlation (GP93R-LYP), or with
  `GP93_CORR NONE` (GP93R-x); `XCtype PBE` for PBE.
- `PLANAR_AVG 1` writes `planar_avg.dat`: planar averages over the xy plane
  of the local potential `Vloc` (Ha, including the exchange-correlation
  potential) and of the density, along z.

## `realspace_convergence.csv`

One row per run: molecule, functional, cell side (bohr), grid points per
direction, grid spacing h (bohr), SCF iterations, total energy (Ha, without
the constant term), HOMO and LUMO eigenvalues (Ha), -eps_HOMO (eV), the
local potential and the density on the plane farthest from the molecule
(z = 0), and -eps_HOMO measured from that plane (eV). Eigenvalues are
measured from the average electrostatic potential of the cell.

The H2 GP93R-x rows test the discretization: 12 bohr with 48^3 and 64^3
points (h = 0.25 and 0.1875 bohr) and 14 bohr with 56^3 points (h = 0.25
bohr). In the 12 bohr cell the local potential on the farthest plane still
contains the attractive GP93R exchange potential, so that plane is not a
vacuum reference. In the 14 bohr cell it lies 0.02 eV below the cell average.

## `runs/<name>/`

Input (`fort.1`, `fort.970`) and output (`eigenvalues`, `planar_avg.dat`)
of each run. The directory `h2_x_56_14` is the 14 bohr cell with 56^3
points.
