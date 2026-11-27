"""Peak frequency (Eq. 40) and fractional linewidth (Eq. 41)."""
from __future__ import annotations
import math


def peak_rr(alpha: float, eps0: float) -> float:
    """omega'_peak,RR / omega = cosh^2 th* = sqrt(alpha/eps0)."""
    return math.sqrt(alpha / eps0)


def peak_class(a0: float) -> float:
    """classical estimate omega'_peak / omega ~ a0^2."""
    return a0 * a0


def fractional_linewidth(eps0: float, sinh_th: float) -> float:
    """d omega' / omega'_peak ~ 2 eps0 sinh^2 th tanh th,  tanh = sinh / sqrt(1 + sinh^2)."""
    return 2.0 * eps0 * sinh_th ** 2 * sinh_th / math.sqrt(1.0 + sinh_th ** 2)
