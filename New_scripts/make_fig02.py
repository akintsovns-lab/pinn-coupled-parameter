"""Article Fig. 2 -- rapidity saturation at the radiation-damped attractor.

Panel (a): rapidity theta as a function of the retarded phase eta/L for the
radiation-dominated parameters (a0 = 857, alpha = 0.36 omega, lambda = 1 um),
with and without the Landau-Lifshitz damping term.
Panel (b): maximum Lorentz factor versus laser intensity for the four regimes.

NOTE ON THE CHOICE OF INTEGRATION VARIABLE.  Evaluating the closed-form solution on a uniform grid in the
laboratory time t and differentiating it numerically; because the natural
variable of the coupled-parameter solution is the retarded phase eta, with
d eta / dt = 1/gamma along the trajectory, the uniform t grid sampled eta with
an abruptly varying density and produced a spurious kink in the classical
curve.  Here the equation is integrated directly in eta with DOP853
(rtol = 1e-11, atol = 1e-13) and both curves are plotted on the common
abscissa eta/L.

NOTE ON THE COEFFICIENT.  The dissipative coefficient
is quadratic in the field amplitude,

    eps_rad = eps0 a0^2,   eps0 = 4 pi r_e / (3 lambda) = 1.180e-8 at 1 um,

and the dissipative term carries the envelope factor E^2 explicitly.  On the
phase-locked orbit sinh(theta) ~= a0 E, so

    d theta / d eta = cosh(theta) [ alpha E(eta) tanh(theta)
                                    - eps0 sinh^3(theta) cosh(theta) ],

whose fixed point is sinh(theta*) cosh(theta*) = sqrt(alpha/eps0), i.e.
gamma* = (alpha/eps0)^{1/4} = 74.3 (theta* = 5.00).  With a coefficient linear
in the amplitude, eps_rad = eps0 a0, the same balance would instead give
cosh(theta*) = sqrt(alpha/(eps0 a0)) = 188.6 at a0 = 857.

Run from the repository root:  python3 scripts/make_fig02.py

(Author's script; the integrator and the constants were moved to rrcp/, content unchanged.)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from rrcp import config, constants
from rrcp.attractor import theta_star as _theta_star
from rrcp.dynamics import L_PULSE as L, envelope, integrate as _integrate

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.3,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

EPS0 = constants.eps0(1e-6)      # 4 pi r_e / 3 lambda, lambda = 1 um (1.1803760836e-08)
ALPHA = 0.36                     # alpha / omega = omega_0 Q / omega
A0_RD = 857.0
REGIMES = [("LT", 1e18, 0.85), ("WNL", 1e20, 8.5),
           ("UR", 1e22, 85.0), ("RD", 1e24, 857.0)]


def theta_star(alpha=ALPHA, eps0=EPS0):
    return _theta_star(alpha, eps0)


def integrate(rr=True, n=4000):
    return _integrate(ALPHA, EPS0, rr=rr, n=n, name="fig2 panel (a)")


def main():
    th_star = theta_star()
    g_star = math.cosh(th_star)

    eta_rr, th_rr = integrate(rr=True)
    eta_cl = np.linspace(-3.0 * L, 3.0 * L, 4000)
    # Classical reference: the instantaneous free-electron ceiling
    # theta_class = arcsinh(a0 E(eta)), peaking at arcsinh(a0) = 7.45.
    th_cl = np.arcsinh(A0_RD * envelope(eta_cl))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.9))

    # adiabatic locked orbit: sinh theta = min(a0 E, sinh theta*(E)) with
    # sinh th* cosh th* = sqrt(alpha E / eps0)
    loc = 0.5 * np.arcsinh(2.0 * np.sqrt(ALPHA * envelope(eta_cl) / EPS0))
    th_ad = np.minimum(th_cl, loc)

    ax1.plot(eta_cl / L, th_cl, ls="--", color="0.35",
             label=r"classical, $\theta\to\mathrm{arcsinh}\,a_0=7.45$")
    ax1.plot(eta_cl / L, th_ad, color="#c1272d",
             label=r"LL damping, adiabatic locked orbit")
    ax1.plot(eta_rr / L, th_rr, color="#c1272d", lw=0.9, ls="-.",
             label=r"LL damping, direct integration")
    ax1.axhline(th_star, color="#c1272d", lw=0.7, ls=":")
    ax1.text(-2.9, th_star + 0.2,
             r"$\theta^*=%.2f\;(\gamma^*=%.1f)$" % (th_star, g_star),
             color="#c1272d", fontsize=8)
    ax1.set_xlabel(r"retarded phase $\eta/L$")
    ax1.set_ylabel(r"rapidity $\theta=\mathrm{arccosh}\,\gamma$")
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 10.8)
    ax1.legend(loc="upper left", frameon=False, fontsize=7)
    ax1.text(0.02, 0.02, "(a)", transform=ax1.transAxes, fontweight="bold")

    inten = np.array([r[1] for r in REGIMES])
    a0s = np.array([r[2] for r in REGIMES])
    g_cl = np.sqrt(1.0 + a0s ** 2)
    g_rr = np.minimum(g_cl, g_star)

    Igrid = np.logspace(17.5, 24.5, 300)
    a0grid = np.sqrt(Igrid / 1.37e18)
    ax2.plot(Igrid, np.sqrt(1.0 + a0grid ** 2), ls="--", color="0.35",
             label=r"classical, $\gamma_{\max}\simeq a_0$")
    ax2.plot(Igrid, np.minimum(np.sqrt(1.0 + a0grid ** 2), g_star),
             color="#c1272d", label=r"with LL damping (cap $\gamma^*$)")
    ax2.plot(inten, g_cl, "o", ms=4, mfc="w", mec="0.35")
    ax2.plot(inten, g_rr, "s", ms=4, color="#c1272d")
    for (tag, I, a0), gc, gr in zip(REGIMES, g_cl, g_rr):
        ax2.annotate(tag, (I, gr), textcoords="offset points",
                     xytext=(7, -3), fontsize=8, color="#c1272d")
    ax2.annotate("", xy=(1e24, g_rr[-1]), xytext=(1e24, g_cl[-1]),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.2"))
    ax2.text(6.0e23, 2.2e2,
             r"$\times%.1f$" % (g_cl[-1] / g_rr[-1]), fontsize=8,
             ha="right")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"intensity $I$ (W/cm$^2$)")
    ax2.set_ylabel(r"$\gamma_{\max}$")
    ax2.set_xlim(3e17, 5e24)
    ax2.set_ylim(0.9, 4e3)
    ax2.legend(loc="upper left", frameon=False)
    ax2.text(0.97, 0.03, "(b)", transform=ax2.transAxes, fontweight="bold", ha="right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig2_trajectories.{ext}")
    np.savetxt("data/fig2_panel_a.csv",
               np.column_stack([eta_rr / L, th_rr,
                                np.arcsinh(A0_RD * envelope(eta_rr))]),
               delimiter=",", header="eta_over_L,theta_LL,theta_classical",
               comments="")
    print(f"theta* = {th_star:.4f}   gamma* = {g_star:.2f}   "
          f"cap = {g_cl[-1] / g_rr[-1]:.2f}x   max theta_LL = {th_rr.max():.4f}")


if __name__ == "__main__":
    main()
