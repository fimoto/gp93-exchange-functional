#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GP93/PBE GGA building block used by the regularized functional.

``F_GP93`` is represented on the finite interpolation interval used by the
self-consistent implementation.  The un-clipped hydrogen-exact continuation
used for the formal H benchmark is evaluated independently in
``calculations/analytic/run_analytic.py``.
"""
import json
from pathlib import Path

import numpy as np
from math import pi, gamma
from scipy.interpolate import CubicSpline
import gp93_w as G

PARAMETERS = json.loads(
    Path(__file__).with_name("parameters.json").read_text(encoding="utf-8")
)
NUMERICAL = PARAMETERS["numerical_implementation"]
W_SPLINE_INPUT = NUMERICAL["w_spline"]
L6P=np.log(6*pi); K=8*pi/3.0
c0=float(PARAMETERS["gp93_c0"])
# Blended w(z) spline: series below zc to exact form above zc, with the
# tanh switch of Eqs. (55)--(56).
zc_in = float(W_SPLINE_INPUT["blend_center"])
dz_in = float(W_SPLINE_INPUT["blend_width"])
Z_MIN = float(W_SPLINE_INPUT["z_min"])
Z_MAX = float(W_SPLINE_INPUT["z_max"])
zg=np.linspace(Z_MIN,Z_MAX,int(W_SPLINE_INPUT["points"]))
w_ser=G.w_series(zg,c0)
w_ex=np.array([
    G.w_exact(z)
    if z > float(W_SPLINE_INPUT["exact_branch_minimum_z"])
    else np.nan
    for z in zg
])
eta=0.5*(1+np.tanh((zg-zc_in)/dz_in))
w_ex_safe=np.where(np.isnan(w_ex),w_ser,w_ex)
w_all=(1-eta)*w_ser+eta*w_ex_safe
w_sp=CubicSpline(zg,w_all)
def F_GP93(s):
    s=np.asarray(s,float)
    z=np.clip(
        3*np.log(np.maximum(s,float(NUMERICAL["log_input_floor"])))+L6P,
        Z_MIN,
        Z_MAX,
    )
    return K*s*w_sp(z)
MUP=0.21951; KP=0.804
def F_PBE(s): return 1+KP-KP/(1+MUP*np.asarray(s,float)**2/KP)
def make_Fmix(s0,p=4):
    def F(s):
        s=np.asarray(s,float)
        chi=1-np.exp(
            -np.clip((s/s0)**p,0,float(NUMERICAL["switch_exponent_cap"]))
        )
        return (1-chi)*F_PBE(s)+chi*F_GP93(s)
    return F
