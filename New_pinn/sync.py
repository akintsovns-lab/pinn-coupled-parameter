"""Synchronization-related relations.

Implemented (closed form): alpha = omega_0 Q and the design-curve shift
H0_opt = 1 + f eps0 <sinh^4 theta> with the author's phenomenological factor
f = 0.55 (fig07_design_curve.py).

ОТСУТСТВУЕТ: Q(a0, H0; eps0) itself.  It is the time average of cos(delta phi)
along the LL-modified trajectory of the FULL system (Eqs. 4-6), which needs the
leading term F of Ref. akintsov2025.  It is not invented here.
"""
from __future__ import annotations
import math
import numpy as np

from .attractor import theta_star

F_AVG = 0.55     # pulse average of sinh^4 relative to its peak (author's value)
ALPHA_RD = 0.36  # alpha / omega used throughout the paper (RD regime)


def alpha_from_Q(H0: float, Q: float) -> float:
    """alpha / omega = (omega_0 / omega) Q, with omega_0/omega = H0 (units m omega c / e)."""
    return H0 * Q


def Q_map(*args, **kwargs):
    raise NotImplementedError(
        "ОТСУТСТВУЕТ: Q(a0,H0;eps0) needs the full trajectory (Eqs. 4-6, "
        "leading term F of Ref. akintsov2025); not part of this deposit.")


def sinh_star(alpha: float, eps0: float) -> float:
    return math.sinh(theta_star(alpha, eps0))


def h0_opt(a0, eps0: float, alpha: float = ALPHA_RD, f: float = F_AVG):
    s = np.minimum(np.asarray(a0, dtype=float), sinh_star(alpha, eps0))
    return 1.0 + f * eps0 * s ** 4
