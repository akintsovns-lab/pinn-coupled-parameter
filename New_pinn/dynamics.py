"""Reduced autonomous rapidity equation (Eq. 29 of the manuscript):

    d theta / d eta = cosh(theta) [ alpha E(eta) tanh(theta)
                                    - eps0 sinh^3(theta) cosh(theta) ],

integrated in the retarded phase eta with DOP853.  This is the author's
integrator of fig02_rd_attractor.py (moved here, same content).

NOT included: the full system, Eqs. (4)-(6): it needs the field components and
the leading term F of Ref. akintsov2025 (ОТСУТСТВУЕТ).  Consequently the test of
the invariant  gamma - P_z = 1  cannot be run here.

Start value: the author's script starts from theta(eta_0) = 1e-3 (not 0).  With
theta = 0 exactly, tanh(theta) = 0 and theta stays 0, so the start value must
be nonzero.  The manuscript text on the start of Eq. (29) is flagged for the
author (README / comparison.md).
"""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import solve_ivp

from . import config

L_PULSE = 5.0 * 2.0 * math.pi        # pulse half-width, 5 wavelengths (Table III)
THETA_START = 1e-3


def envelope(eta, L=L_PULSE):
    return np.exp(-(np.asarray(eta) / L) ** 2)


def rhs(eta, y, alpha, eps0, rr=True, L=L_PULSE):
    th = y[0]
    ch, sh = np.cosh(th), np.sinh(th)
    drive = alpha * envelope(eta, L) * np.tanh(th)
    damp = eps0 * sh ** 3 * ch if rr else 0.0
    return [ch * (drive - damp)]


def log_run(name, rtol, atol, extra=""):
    print(f"[run] {name}: DOP853 rtol={rtol:g} atol={atol:g} {extra}".rstrip())


def integrate(alpha, eps0, rr=True, n=4000, L=L_PULSE,
              rtol=config.RTOL, atol=config.ATOL, name="eq29"):
    span = (-3.0 * L, 3.0 * L)
    grid = np.linspace(*span, n)
    log_run(name, rtol, atol, f"rr={rr} N={n}")
    sol = solve_ivp(rhs, span, [THETA_START], t_eval=grid,
                    args=(alpha, eps0, rr, L), method="DOP853",
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return grid, sol.y[0]
