# External reference data

`experimental_ip.csv` contains the neutral-atom ground-state ionization
energies used for calibration and evaluation.  Values and uncertainties are
from the NIST Atomic Spectra Database (ASD), version 5.12, accessed
2026-07-24:

- https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=H&units=1
- https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=He&units=1
- https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Li&units=1
- https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Na&units=1
- https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=K&units=1

Reference: A. Kramida, Yu. Ralchenko, J. Reader, and NIST ASD Team,
NIST Atomic Spectra Database, version 5.12 (2024),
https://doi.org/10.18434/T4W30F.

`h2_reference.csv` contains the experimental H2 equilibrium bond length used
only as the vertical reference marker in the molecular figure.  The value is
from the NIST Computational Chemistry Comparison and Benchmark Database
(CCCBDB), accessed 2026-07-25:

- https://cccbdb.nist.gov/exp2x.asp?casno=1333740

The electronic-structure calculation itself does not access the network.

`he_exact_ks_levels.csv` contains the exact Kohn--Sham eigenvalues of the He
atom (obtained from the exact Kohn--Sham potential of Umrigar and Gonze) as
tabulated by A. Savin, C. J. Umrigar, and X. Gonze, Chem. Phys. Lett. 288,
391 (1998), Table 3.  They are used only as reference markers in
`fig_spectrum.png` and in the radial-spectrum table.
