#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful implementation of the GP93 continuation and crossover formulas.

This module provides (i) the exact representation for ``z > 0`` (Eq. 14),
(ii) the Frobenius series with ``c0`` matching (Eqs. 22--31), (iii) the
tail-integral approximation used in Eqs. 43--44 and reported in Eq. 50, and
(iv) the continuation to ``z < 0``.
"""
import numpy as np
import json
from math import factorial as _fact
from math import pi, gamma
from pathlib import Path
from scipy.integrate import quad
from scipy.special import hyp1f1
try:
    from scipy.special import hyperu as _hyperu
    def Uf(z): return _hyperu(-2/3., 2.0, z)
except Exception:
    import mpmath as mp
    def Uf(z): return float(mp.hyperu(-2/3., 2.0, z))
def Mf(z): return hyp1f1(-2/3., 2.0, z)
G13=gamma(1/3.)
W=lambda t: t*(t+2)*np.exp(-2*t)

PARAMETERS = json.loads(
    Path(__file__).with_name("parameters.json").read_text(encoding="utf-8")
)
NUMERICAL = PARAMETERS["numerical_implementation"]
QUADRATURE_LIMIT = int(NUMERICAL["gp93_quadrature_limit"])
TAIL_ENDPOINT = float(NUMERICAL["gp93_tail_endpoint"])
FROBENIUS_ORDER = int(NUMERICAL["frobenius_order"])

def IM(z): return quad(lambda t: W(t)*Mf(t), 0, z, limit=QUADRATURE_LIMIT)[0]
def IU(z, T=TAIL_ENDPOINT): return quad(lambda t: W(t)*Uf(t), z, T, limit=QUADRATURE_LIMIT)[0]
def w_exact(z):
    return 0.5 - (G13/4.0)*(Uf(z)*IM(z) + Mf(z)*IU(z))

# --- Frobenius series (Eqs. 22--25 and 29) ---
def frobenius_AB(N=FROBENIUS_ORDER):
    A=np.zeros(N+1); B=np.zeros(N+1)
    A[0]=1.0; B[0]=0.0
    # r_m: coefficient of (1/6)[2 - exp(-z)(z + 2)].
    r=np.zeros(N+2)
    for m in range(N+2):
        em=(-1)**m/_fact(m)
        em1=((-1)**(m-1)/_fact(m-1)) if m>=1 else 0.0
        r[m]=(1/6.0)*(2*(m==0) - (2*em + em1))
    A[1]=-1/3.0; B[1]=0.0
    for m in range(1,N):
        A[m+1]=( (m-2/3.0)*A[m] )/((m+1)*(m+2))
        B[m+1]=( r[m] + (m-2/3.0)*B[m] )/((m+1)*(m+2))
    return A,B
A_,B_=frobenius_AB(FROBENIUS_ORDER)
def w_series(z,c0):
    zp=np.power.outer(np.atleast_1d(z),np.arange(len(A_)))
    return (zp@(c0*A_+B_))

if __name__=='__main__':
    # Check c0 matching (Eq. 31) over several z_m values.
    print("c0 matching check (Eq. 31):")
    for zm in (2.0,3.0,4.0,5.0):
        wex=w_exact(zm)
        phi=sum(A_[n]*zm**n for n in range(len(A_)))
        psi=sum(B_[n]*zm**n for n in range(len(B_)))
        c0=(wex-psi)/phi
        print(f"  z_m={zm}: c0={c0:.8f}")
    c0=( w_exact(3.0)-sum(B_[n]*3.0**n for n in range(len(B_))) )/sum(A_[n]*3.0**n for n in range(len(A_)))
    print(f"c0=w(0)={c0:.6f}  [compare with reference value 0.689]")
    # Internal check: series versus exact representation on z in [0.5, 4].
    errs=[abs(w_series(z,c0)[0]-w_exact(z)) for z in (0.5,1.0,2.0,4.0)]
    print("series vs exact |Delta w| (z=0.5,1,2,4):", " ".join(f"{e:.1e}" for e in errs))
    # Accuracy of the tail-integral approximation; supplies Eq. (50).
    print("\nTail-integral I_U(z) approximation errors [Eq. (50)]:")
    for z in (2.0,3.0,4.0,6.0,10.0,20.0):
        ex=IU(z)
        a3=np.exp(-2*z)*z**(8/3.)*(0.5+10/(9*z)-20/(81*z*z))
        a4=np.exp(-2*z)*z**(8/3.)*(0.5+10/(9*z)-20/(81*z*z)-430/(2187*z**3))
        print(
            f"  z={z:5.1f}: 3 terms {abs(a3/ex-1):.2e}  "
            f"4 terms {abs(a4/ex-1):.2e}"
        )
    m3=max(abs((np.exp(-2*z)*z**(8/3.)*(0.5+10/(9*z)-20/(81*z*z)))/IU(z)-1) for z in np.linspace(2,20,37))
    m4=max(abs((np.exp(-2*z)*z**(8/3.)*(0.5+10/(9*z)-20/(81*z*z)-430/(2187*z**3)))/IU(z)-1) for z in np.linspace(2,20,37))
    print(
        f"  max over z in [2,20]: 3 terms {m3:.2e} / "
        f"4 terms {m4:.2e}  [Eq. (50)]"
    )
    # Check the large-z behavior of w (negative divergence creates a wall).
    print(
        "\nw(z) behavior:",
        " ".join(f"w({z})={w_exact(z):+.3f}" for z in (0.5, 2, 4, 6, 8, 12)),
    )
    print(
        "w(z<0) series:",
        " ".join(f"w({z})={w_series(z, c0)[0]:+.3f}" for z in (-0.5, -2, -4)),
    )
