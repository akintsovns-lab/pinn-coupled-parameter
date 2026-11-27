"""Erber function g(chi)."""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import kv


def g_rational(chi):
    """Rational approximation (Eq. 43; used for all numbers in the paper)."""
    chi = np.asarray(chi, dtype=float)
    return (1.0 + 4.8 * (1.0 + chi) * np.log1p(1.7 * chi) + 2.44 * chi ** 2) ** (-2.0 / 3.0)


def g_integral_B1_as_printed(chi: float) -> float:
    """Eq. (B1) exactly as printed in the manuscript:

        g = 9 sqrt(3) / (8 pi chi) * Int_0^inf K_{5/3}(x/chi) x dx / chi .

    With u = x/chi the chi-dependence cancels: the printed expression is a
    CONSTANT (= 3.248), not g(chi).  Reported to the author (README, comparison.md);
    the code does not "fix" it.
    """
    val, _ = quad(lambda x: kv(5.0 / 3.0, x / chi) * x / chi, 0.0, np.inf, limit=200)
    return 9.0 * math.sqrt(3.0) / (8.0 * math.pi * chi) * val
