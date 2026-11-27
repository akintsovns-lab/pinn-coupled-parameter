"""Compares quantities computed by rrcp (or read from the recorded source-run data)
with the numbers quoted in the manuscript.  Writes comparison.md.

The 'paper' column holds EXPECTED values copied from the manuscript text; they are
never used as results.  A discrepancy > 2 % is only reported, nothing is corrected.
Run from the repository root:  python3 scripts/make_comparison.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402

from rrcp import attractor, config, constants, dynamics, erber, spectrum, sync   # noqa: E402

EPS0 = constants.eps0(1e-6)
ALPHA = sync.ALPHA_RD
TH = attractor.theta_star(ALPHA, EPS0)
SH = math.sinh(TH)

rows = []   # (quantity, paper, code, note)


def add(q, paper, code, note=""):
    rows.append((q, paper, code, note))


add("eps0 (1 um)", 1.180e-8, EPS0)
add("a_S = mc^2/(hbar omega)", 4.1e5, constants.a_S(1e-6))
add("B0 (T), H0 = 1, 1 um", 1.071e4, constants.B0_tesla(1e-6))
add("a0_crit = eps0^(-1/3)  (Eq. 14)", 4.4e2, EPS0 ** (-1 / 3))
add("eps_rad at a0 = 857", 8.7e-3, EPS0 * 857.0 ** 2)
add("gamma* (alpha = 0.36)", 74.3, attractor.gamma_star(ALPHA, EPS0))
add("theta*", 5.00, TH)
add("gamma_class / gamma* at a0 = 857", 11.5, 857.0 / attractor.gamma_star(ALPHA, EPS0))
add("Lambda / omega (Eq. 33)", 1.44, attractor.stability_Lambda(ALPHA, EPS0))
add("eps_crit, theta0 = 5.68 (Eq. 34)", 9.1e-15, attractor.eps_crit(ALPHA, 5.68),
    "theta0 = 5.68 is taken from the text; its origin (gamma0 ~ 146) is flagged for the author")
add("omega'_peak,RR / omega (Eq. 40)", 5.5e3, spectrum.peak_rr(ALPHA, EPS0))
add("downshift a0^2 / peak", 133, spectrum.peak_class(857.0) / spectrum.peak_rr(ALPHA, EPS0))
add("linewidth RD (Eq. 41)", 1.3e-4, spectrum.fractional_linewidth(EPS0, SH))
add("linewidth WNL (Eq. 41), a0 = 8.5", 1.7e-6, spectrum.fractional_linewidth(EPS0, 8.5),
    "Table V now reads ~1e-6")
add("linewidth LT (Eq. 41), a0 = 0.85", 1e-8, spectrum.fractional_linewidth(EPS0, 0.85), "Table V: ~1e-8 (order of magnitude)")
for chi, pv in ((0.16, 0.56), (0.5, 0.30), (1.0, 0.18), (2.0, 0.10), (5.0, 0.04)):
    add(f"g({chi}) rational (Eq. 43)", pv, float(erber.g_rational(chi)))
add("g(0.5) from Eq. (B1) as printed", 0.30, erber.g_integral_B1_as_printed(0.5),
    "B1 as printed is chi-independent (3.248): typo/flaw in the manuscript, see README")
add("H0_opt saturation 1 + f alpha tanh^2 th*", 1.20, float(sync.h0_opt(857.0, EPS0)))
add("B_opt (kT) at 1 um, a0 = 857", 12.8,
    constants.B0_tesla(1e-6) * float(sync.h0_opt(857.0, EPS0)) / 1e3)
e10 = constants.eps0(10e-6)
add("B_opt (kT) at 10 um, a0 = 857", 1.28,
    constants.B0_tesla(10e-6) * float(sync.h0_opt(857.0, e10)) / 1e3)

# reduced Eq. (29): direct integration
eta, th = dynamics.integrate(ALPHA, EPS0, rr=True, name="comparison: Eq. 29, RD")
add("max theta from Eq. (29) integration vs theta*", TH, float(th.max()),
    "the damped orbit cannot exceed the attractor; ratio shown in the log")

# recorded source-run results (NOT computed by this code)
t6 = json.load(open(ROOT / "data" / "measured_results.json"))["table6"]
val = lambda k: float(t6[k]["value"] if isinstance(t6[k], dict) else t6[k])
ratio = val("E_inv_L_RD") / val("E_inv_LL_RD")
add("E_inv(L)/E_inv(LL), RD  [recorded Table VI, not computed]", 54, ratio)
print(f"E_inv(L)/E_inv(LL) RD = {ratio:.2f}  [recorded results]")
for r in ("LT", "WNL", "UR", "RD"):
    print(f"  {r}: E_inv(L)={val('E_inv_L_' + r):.2e}  E_inv(LL)={val('E_inv_LL_' + r):.2e}")
print(f"ODE tolerances used in this run: rtol={config.RTOL:g} atol={config.ATOL:g}")
print("Q(a0=857, H0=1.2): ОТСУТСТВУЕТ (cannot be computed here, see README)")
print("Fig. 4 band-width ratio (c/a): ОТСУТСТВУЕТ")

lines = ["| quantity | paper | code | discrepancy % | note |", "|---|---|---|---|---|"]
flag = []
for q, p, c, n in rows:
    d = 100.0 * (c - p) / p
    mark = "  **>2%**" if abs(d) > 2.0 else ""
    if abs(d) > 2.0:
        flag.append((q, p, c, d))
    lines.append(f"| {q} | {p:.4g} | {c:.4g} | {d:+.2f}{mark} | {n} |")
lines += ["| Q(a0 = 857, H0 = 1.2) | 0.30 | ОТСУТСТВУЕТ | n/a | map needs the full system, not in the deposit |",
          "| Fig. 4 band-width ratio c/a | ≈40 % (old text) | ОТСУТСТВУЕТ | n/a | same |",
          "| E_inv(LL) reduction wrt E_inv(L), Fig. 6(b) | ≃54 | recorded only | n/a | no network in the deposit |"]
lines += ["", "## Discrepancies above 2 % (ТРЕБУЕТ АВТОРА, nothing corrected)", ""]
lines += [f"- {q}: paper {p:.4g}, code {c:.4g}, {d:+.1f} %" for q, p, c, d in flag] or ["- none"]
(ROOT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf8")
print("written comparison.md;", len(flag), "discrepancies > 2 %")
