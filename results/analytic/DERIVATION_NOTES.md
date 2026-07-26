# GP93 building block and regularized exact-H diagnostic

Run this calculation from the repository root with
`python calculations/analytic/run_analytic.py`.  The canonical implementation
is imported directly; its equations are not copied into this script.
The numerical grids and convergence-study spacings are read from
`inputs/analytic.json`.

## Source files

- `functional/gp93_w.py`
- `functional/fmix.py`
- `functional/mgga_fmix.py`
- `functional/hydrogen_exact.py`
- `functional/parameters.json`
- `inputs/analytic.json`

## GP93 large-gradient branch

The variables obey

\[
z=3\ln s+\ln(6\pi),\qquad
F_{\rm GP93}(s)=\frac{8\pi}3s\,w(z).
\]

Because \(I_M\to C_M=0.419973683298291\) and
\(U(-2/3,2,z)\sim z^{2/3}\),

\[
F_{\rm GP93}(s)\sim
-4.901446613\,
s(\ln s)^{2/3} .
\]

The bare GP93 building block is hydrogen exact.  The practical functional is
different by design: its broad log-density window keeps the GP93 contribution
in the intermediate region and turns it off before the numerical far tail.

## Canonical density window

For a spin density \(n_\sigma\), define

\[
t=\frac{\ln(n_\sigma/n_{\rm low})}
        {\ln(n_{\rm high}/n_{\rm low})},\qquad
X_n=3t^2-2t^3
\]

for \(0<t<1\), with \(X_n=1\) above
\(n_{\rm high}=1e-06\) and \(X_n=0\) below
\(n_{\rm low}=1e-12\).  The first derivative vanishes at
both endpoints.  On the exact H density the transition begins at
\(r=6.335390\) bohr and ends at
\(r=13.243146\) bohr.  Thus strict \(-1/r\)
asymptotics belongs to the building block, whereas the final functional
targets the radial window controlling low-lying diffuse states.

For the exact H density, \(\tau=\tau^W\) is imposed analytically before the
canonical `_channel` call, so \(\alpha=0\).  The radial potential is formed
from the returned derivatives as

\[
v_x(r)=v_\rho(r)-\frac2{r^2}\frac{d}{dr}
\left[r^2v_\sigma(r)n'_H(r)\right].
\]

| r (bohr) | n | raw s | X_n | bare -r v_x | unregularized -r v_x | regularized -r v_x |
|---:|---:|---:|---:|---:|---:|---:|
| 6.000 | 1.955764e-06 | 20.5153 | 1.000000e+00 | 0.99995469 | 0.99995469 | 0.99995452 |
| 8.000 | 3.582106e-08 | 77.8282 | 8.537774e-01 | 0.99999900 | 0.99999900 | 1.15274289 |
| 10.000 | 6.560856e-10 | 295.254 | 4.542969e-01 | 0.99999995 | 0.99999995 | 0.92440411 |
| 12.000 | 1.201663e-11 | 1120.1 | 8.550391e-02 | 0.99999996 | 0.99999996 | 0.39788818 |
| 12.200 | 8.054986e-12 | 1279.86 | 6.152540e-02 | 0.99999996 | 0.99999996 | 0.33603715 |
| 15.000 | 2.978624e-14 | 4148.64 | 0.000000e+00 | 0.99999970 | 1.00000518 | -0.00000000 |

## Verification

`derivative_checks.csv` records:

- analytic derivatives of the selected mixed enhancement against finite
  differences;
- derivatives of the canonical log-s spline;
- zero derivatives outside the descriptor-clip interval;
- analytic derivatives of the cubic log-density window;
- canonical `vrho` and `vsigma` against finite differences of `_channel`;
- radial-grid convergence of the bare and regularized H potentials.
