"""Article Fig. 6(b) -- Lorentz-invariant violation of PINN-L and PINN-LL.

Data: Table VI of the manuscript, i.e. the recorded results of the source run
(data/measured_results.json, key "table6", status MEASURED).  The panel is
NOT recomputed from the equations: the trained networks are not part of the
deposit.  The annotation (ratio E_inv(L)/E_inv(LL) in the RD regime) is
computed from the same data, not typed in.

Run from the repository root:  python3 scripts/make_fig06b.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.3,
    "figure.dpi": 200,
})

ROOT = Path(__file__).resolve().parent.parent
DATA = next(d for d in (ROOT / "data", ROOT / "scripts" / "data")
            if (d / "measured_results.json").exists())
OUT = ROOT / "figures"
REGIMES = ["LT", "WNL", "UR", "RD"]


def main():
    t6 = json.load(open(DATA / "measured_results.json"))["table6"]
    def val(k):
        v = t6[k]
        return float(v["value"] if isinstance(v, dict) else v)

    e_l = [val(f"E_inv_L_{r}") for r in REGIMES]
    e_ll = [val(f"E_inv_LL_{r}") for r in REGIMES]
    ratio = e_l[-1] / e_ll[-1]

    print("Fig. 6(b) data (Table VI, measured_results.json):")
    for r, a, b in zip(REGIMES, e_l, e_ll):
        print(f"  {r:4s} E_inv(L) = {a:.2e}   E_inv(LL) = {b:.2e}   L/LL = {a / b:.2f}")
    print(f"  annotation: x{ratio:.0f}  (E_inv(L)/E_inv(LL) in RD = {ratio:.2f})")

    x = list(range(len(REGIMES)))
    fig, ax = plt.subplots(figsize=(3.53, 2.25))
    ax.plot(x, e_l, ls="--", marker="s", ms=5, color="0.35",
            label="PINN-L (classical)")
    ax.plot(x, e_ll, ls="-", marker="o", ms=5, color="#c1272d",
            label="PINN-LL")
    ax.annotate("", xy=(3, e_ll[-1]), xytext=(3, e_l[-1]),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.2"))
    ax.text(2.9, (e_l[-1] * e_ll[-1]) ** 0.5, r"$\times%.0f$" % ratio,
            ha="right", va="center", fontsize=8, color="#c1272d")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlim(-0.3, 3.3)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_xlabel("intensity regime")
    ax.set_ylabel(r"$\mathcal{E}_{\mathrm{inv}}$")
    ax.set_title("(b) Lorentz-invariant violation")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig6b_consistency.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    main()
