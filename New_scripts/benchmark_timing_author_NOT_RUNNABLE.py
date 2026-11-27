"""Absolute timing benchmark: PINN surrogate vs. DOP853 reference integrator.

Produces the absolute numbers requested by referee 2, comment 6:

  * wall-clock cost of one DOP853 solution of the coupled-parameter system at
    the tolerances used for the reference trajectories,
  * wall-clock cost of one forward pass of the trained PINN over the same
    dense output grid, and of a single query point,
  * the resulting speed-up, and
  * the one-off training cost read from the stored optimisation histories.

Run:  python benchmark_timing.py            (CPU)
      CUDA_VISIBLE_DEVICES=0 python benchmark_timing.py   (GPU, if available)
"""

from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import torch

from model import CoupledPINN
from physics import solve_reference

RUNS = json.load(open("checkpoints/runs.json"))
N_DENSE = 4000
N_REPEAT = 20
RTOL, ATOL = 1e-11, 1e-13


def time_it(fn, n_repeat=N_REPEAT, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    return (time.perf_counter() - t0) / n_repeat


def dop853_cost(tag):
    cfg = RUNS[tag]
    fn = lambda: solve_reference(cfg["a0"], cfg["L"], eta0=cfg["eta0"],
                                 t_max=cfg["t_max"], n_dense=N_DENSE,
                                 rtol=RTOL, atol=ATOL)
    return time_it(fn, n_repeat=5)


def pinn_cost(tag, device="cpu"):
    ckpt = f"checkpoints/pinn_{tag}_constrained.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else state
    cfg = RUNS[tag]
    net = CoupledPINN(t_scale=cfg["t_max"]) if "t_scale" in \
        CoupledPINN.__init__.__code__.co_varnames else CoupledPINN()
    try:
        net.load_state_dict(sd)
    except Exception as exc:                                  # pragma: no cover
        print(f"  [{tag}] state_dict mismatch ({exc}); timing the "
              f"architecture with random weights instead")
    net.eval().to(device)

    t_dense = torch.linspace(0.0, cfg["t_max"], N_DENSE,
                             device=device).unsqueeze(1)
    t_one = t_dense[:1]

    with torch.no_grad():
        dense = time_it(lambda: net(t_dense))
        single = time_it(lambda: net(t_one), n_repeat=200)
    n_par = sum(p.numel() for p in net.parameters())
    return dense, single, n_par


def training_cost():
    out = {}
    for f in sorted(glob.glob("checkpoints/history_*.json")):
        hist = json.load(open(f))
        out[os.path.basename(f)[8:-5]] = {
            "epochs": max(r["epoch"] for r in hist) + 1,
            "stages": len({r.get("stage", "-") for r in hist}),
            "wall_s": round(max(r["wall"] for r in hist), 1),
        }
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}   torch {torch.__version__}\n")

    rows = []
    for tag in ("LT", "WNL", "UR"):
        t_ode = dop853_cost(tag)
        t_dense, t_single, n_par = pinn_cost(tag, device)
        rows.append(dict(regime=tag, a0=RUNS[tag]["a0"],
                         dop853_ms=1e3 * t_ode,
                         pinn_dense_ms=1e3 * t_dense,
                         pinn_per_point_us=1e6 * t_dense / N_DENSE,
                         pinn_single_us=1e6 * t_single,
                         speedup=t_ode / t_dense,
                         parameters=n_par))
        r = rows[-1]
        print(f"{tag}: DOP853 {r['dop853_ms']:8.2f} ms | "
              f"PINN {r['pinn_dense_ms']:7.3f} ms "
              f"({r['pinn_per_point_us']:.3f} us/point) | "
              f"speed-up x{r['speedup']:.0f}")

    result = {"device": device, "torch": torch.__version__,
              "n_dense": N_DENSE, "rtol": RTOL, "atol": ATOL,
              "inference": rows, "training": training_cost()}
    with open("data/benchmark_timing.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("\ntraining cost (wall-clock seconds, single device):")
    for k, v in result["training"].items():
        print(f"  {k:24s} {v['epochs']:5d} epochs, "
              f"{v['stages']} stage(s), {v['wall_s']:7.1f} s")
    print("\nwritten to data/benchmark_timing.json")


if __name__ == "__main__":
    main()
