"""Radiation-damped attractor (Eq. 22, Eq. 33 'Lambda', Eq. 34 'eps_crit')."""
from __future__ import annotations
import math


def theta_star(alpha: float, eps0: float) -> float:
    """sinh(th*) cosh(th*) = sqrt(alpha/eps0)  ->  th* = asinh(2 sqrt(alpha/eps0)) / 2."""
    return 0.5 * math.asinh(2.0 * math.sqrt(alpha / eps0))


def gamma_star(alpha: float, eps0: float) -> float:
    """gamma* = cosh(th*)  (the paper writes ~ (alpha/eps0)^(1/4))."""
    return math.cosh(theta_star(alpha, eps0))


def gamma_star_approx(alpha: float, eps0: float) -> float:
    return (alpha / eps0) ** 0.25


def stability_Lambda(alpha: float, eps0: float) -> float:
    """Lambda = eps0 (3 sinh^2 cosh^2 + sinh^4) - alpha sech^2   at theta*."""
    th = theta_star(alpha, eps0)
    s, c = math.sinh(th), math.cosh(th)
    return eps0 * (3 * s * s * c * c + s ** 4) - alpha / (c * c)


def eps_crit(alpha: float, theta0: float) -> float:
    """eps_crit = alpha sech^2 th0 / (3 sinh^2 th0 cosh^2 th0 + sinh^4 th0)."""
    s, c = math.sinh(theta0), math.cosh(theta0)
    return alpha / (c * c) / (3 * s * s * c * c + s ** 4)
