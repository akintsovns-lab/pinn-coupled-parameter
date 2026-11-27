"""Article Fig. 7 -- optimal static magnetic field for autoresonant capture.

Panel (a): dimensionless optimum H0^opt = 1 + B(a0) versus the normalized
amplitude a0 for lambda = 1 um and lambda = 10 um, where the radiative bias is

    B(a0) = f eps0 <sinh^4 theta>,   eps0 = 4 pi r_e / (3 lambda),
    sinh theta_max = min(a0, sinh theta*),  sinh th* cosh th* = sqrt(alpha/eps0),

f = 0.55 being the pulse-average of sinh^4 relative to its peak value.  Because
the radiation-reaction coefficient is quadratic in the amplitude
(eps_rad = eps0 a0^2), the bias saturates at
B -> f alpha tanh^2 theta* ~= 0.20 once a0 exceeds sinh theta* = 74, so the
optimum shift saturates near 20 % instead of growing from 13 % to 20 %.

Panel (b): the same optimum expressed as an absolute field,

    B[T] = 1.071e4 * H0 * (1 um / lambda),

for lambda = 1, 2, 5 and 10 um, with the band of fields already demonstrated in
the laboratory (0.6-1.5 kT, laser-driven capacitor coils) shaded.  Only
long-wavelength (CO2-class) drivers place the optimum inside the accessible
band; at 1 um the required field is ~13 kT.

Run from the repository root:  python3 scripts/make_fig07.py

(Author's script; only labels and the demonstrated-field band (0.6-1.5 kT) were changed, data identical.)
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

R_E_CM = 2.8179403262e-13
ALPHA = 0.36            # alpha / omega
F_AVG = 0.55            # pulse average of sinh^4 relative to the peak
B0_1UM = 1.0710e4       # tesla, field for H0 = 1 at lambda = 1 um
B_RECORD = 1.5e3        # tesla, upper end of demonstrated laser-driven coil fields (0.6-1.5 kT)
B_LOW = 0.6e3           # tesla, lower end of the demonstrated range
FACILITIES = [("APOLLON", 100.0, (-16, -14)), ("ELI-NP / SIOM", 270.0, (26, 7))]


def eps0_of(lam_um):
    return 4.0 * math.pi * R_E_CM / (3.0 * lam_um * 1e-4)


def sinh_star(lam_um):
    e0 = eps0_of(lam_um)
    return math.sinh(0.5 * math.asinh(2.0 * math.sqrt(ALPHA / e0)))


def bias(a0, lam_um):
    e0 = eps0_of(lam_um)
    s = np.minimum(np.asarray(a0, dtype=float), sinh_star(lam_um))
    return F_AVG * e0 * s ** 4


def h0_opt(a0, lam_um):
    return 1.0 + bias(a0, lam_um)


def main():
    a0 = np.logspace(0, 3.5, 600)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.9))

    for lam, col, ls in ((1.0, "#c1272d", "-"), (10.0, "#1f5fa8", "--")):
        ax1.plot(a0, h0_opt(a0, lam), color=col, ls=ls,
                 label=r"$\lambda=%g\ \mu$m ($\sinh\theta^*=%.0f$)"
                       % (lam, sinh_star(lam)))
        ax1.axhline(1.0 + F_AVG * ALPHA * math.tanh(
            math.asinh(sinh_star(lam))) ** 2, color=col, lw=0.6, ls=":")
    ax1.axhline(1.0, color="0.6", lw=0.6)
    ax1.text(1.15, 1.012, r"cold cyclotron resonance $H_0=1$",
             color="0.4", fontsize=7.5)
    ax1.text(1.15, 1.15, r"saturation $1+f\alpha\tanh^2\theta^*=1.20$",
             color="#c1272d", fontsize=7.5, ha="left", va="center")
    for name, av, off in FACILITIES:
        ax1.plot([av], [h0_opt(av, 1.0)], "o", ms=4, color="k", zorder=5)
        ax1.annotate(name, (av, h0_opt(av, 1.0)), textcoords="offset points",
                     xytext=off, fontsize=7.5, ha="center",
                     bbox=dict(boxstyle="square,pad=0.15", fc="white",
                               ec="none", alpha=0.9), zorder=6)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"normalized amplitude $a_0$")
    ax1.set_ylabel(r"$H_0^{\mathrm{opt}}$ (units of $mc\omega/e$)")
    ax1.set_xlim(1, 3e3)
    ax1.set_ylim(0.98, 1.28)
    ax1.legend(loc="upper left", frameon=False)
    ax1.text(0.97, 0.05, "(a)", transform=ax1.transAxes,
             fontweight="bold", ha="right")

    for lam, col, ls in ((1.0, "#c1272d", "-"), (2.0, "#e07b39", "-"),
                         (5.0, "#3f8f4f", "--"), (10.0, "#1f5fa8", "--")):
        ax2.plot(a0, B0_1UM * h0_opt(a0, lam) / lam, color=col, ls=ls,
                 label=r"$\lambda=%g\ \mu$m" % lam)
    ax2.axhspan(B_LOW, B_RECORD, color="0.85", zorder=0)
    ax2.axhline(B_RECORD, color="0.35", lw=0.8, ls="-.")
    ax2.text(1.15, 2.4e2, r"fields demonstrated in the laboratory"
             "\n" r"(0.6--1.5 kT, laser-driven coils)",
             fontsize=7.5, color="0.25", ha="left", va="bottom")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"normalized amplitude $a_0$")
    ax2.set_ylabel(r"required field $B_0^{\mathrm{opt}}$ (T)")
    ax2.set_xlim(1, 3e3)
    ax2.set_ylim(2e2, 3e4)
    ax2.legend(loc="upper right", frameon=False, ncol=2)
    ax2.text(0.97, 0.05, "(b)", transform=ax2.transAxes,
             fontweight="bold", ha="right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig7_design_curve.{ext}")
    np.savetxt("data/fig7_design_curve.csv",
               np.column_stack([a0] + [h0_opt(a0, l) for l in (1, 2, 5, 10)]),
               delimiter=",",
               header="a0,H0opt_1um,H0opt_2um,H0opt_5um,H0opt_10um",
               comments="")
    for lam in (1.0, 2.0, 5.0, 10.0):
        print(f"lambda={lam:5.1f} um  sinh_th*={sinh_star(lam):6.1f}  "
              f"H0opt(857)={h0_opt(857.0, lam):.4f}  "
              f"B={B0_1UM * h0_opt(857.0, lam) / lam:8.1f} T")


if __name__ == "__main__":
    main()
