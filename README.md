# pinn-coupled-parameter

Code accompanying the manuscript *Radiation-Reaction-Modified Dynamics of Relativistic
Electrons in Gaussian Laser Fields: Coupled-Parameter Theory Beyond the Classical Limit*
(N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Qing-Hua Qin).

## Radiation-reaction paper (manuscript under review)
This repository is the code deposit of the manuscript above and of no other paper. Files for the figures it supports:
`scripts/make_fig02.py` (Fig. 2), `scripts/make_fig06b.py` (Fig. 6(b)), `scripts/make_fig07.py` (Fig. 7);
closed-form quantities: `rrcp/` and `tests/`. Material that is not in this repository (full-system integrator,
synchronization maps, PINN code and weights, remaining figure scripts) is available from the corresponding author upon
reasonable request, as stated in the manuscript.

**Scope, stated openly.** This release contains only what exists in the authors' files
and what follows from closed-form expressions of the paper. It does **not** contain the
full coupled-parameter integrator (Eqs. 4-6), the synchronization maps Q(a0, H0; eps0),
the PINN-L/PINN-LL code (torch or NumPy), trained networks, or the scripts of Figs. 1, 3, 4, 5
and 6(a). Those parts are marked "не включено — нет файла" / ОТСУТСТВУЕТ below.

## Quick start
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh            # tests, Figs. 2, 6(b), 7, comparison.md
```
Tested with Python 3.12.3, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, pytest 9.2.0 (Linux x86_64).
Fixed seed: `rrcp/config.py` (nothing stochastic is used). ODE tolerances: `rrcp/config.py`
(rtol = 1e-11, atol = 1e-13); every integration prints the tolerances it used.

## Figure / table -> command -> status
| Item | Command | Status |
|---|---|---|
| Fig. 1 (chi_e vs a0) | - | не включено — нет файла |
| Fig. 2 (rapidity saturation, gamma_max cap) | `python scripts/make_fig02.py` | воспроизводится из скриптов автора (panel (a): direct DOP853 integration of the reduced Eq. 29; panel (b): closed-form cap gamma*) |
| Fig. 3 (phase portraits) | - | не включено — нет файла |
| Fig. 4 (synchronization map Q) | - | не включено — нет файла (Q needs the full system) |
| Fig. 5 (spectrum) | - | не включено — нет файла (peak frequencies and linewidths: see tests) |
| Fig. 6(a) (training losses) | - | не включено — нет файла (needs the trained network) |
| Fig. 6(b) (E_inv of PINN-L / PINN-LL) | `python scripts/make_fig06b.py` | воспроизводится из записанных результатов автора (Table VI, `data/measured_results.json`); **not recomputed**, the annotation is computed from the data |
| Fig. 7 (design curve) | `python scripts/make_fig07.py` | воспроизводится из скриптов автора |
| Table VII (timing) | `scripts/benchmark_timing_author_NOT_RUNNABLE.py` | не включено — нет файла (the script needs `model.py`, `physics.py`, `checkpoints/`, absent). Recorded numbers are in `data/measured_results.json` |
| Table VI | - | не включено — нет файла (only the recorded values) |
| eps0, a_S, B0, a0_crit, eps_rad, gamma*, theta*, Lambda, eps_crit, peak frequency, linewidth, g(chi) | `pytest tests`, `python scripts/make_comparison.py` | вычисляется по формуле |
| Light-cone invariant test gamma - P_z = 1 (< 1e-10) | `tests/test_dynamics.py` (skipped) | не включено — нет файла (needs the full system, Eqs. 4-6) |
| PINN-L / PINN-LL (torch and NumPy) | - | не включено — нет файла (`rrcp/pinn/README.md`) |

## What the code does where it differs from the text
* **Eq. (29) start value.** `rrcp/dynamics.py` (the author's integrator) starts from theta(eta_0) = 1e-3,
  not from 0: with theta = 0 exactly, tanh(theta) = 0 and the solution stays at 0.
* **Eq. (B1).** As printed, the integral is independent of chi (it equals 3.248) and therefore is not g(chi).
  `rrcp/erber.py` implements it as printed (`g_integral_B1_as_printed`, an xfail test documents this);
  all quoted g values use the rational approximation, Eq. (43).
* **Design curve.** `H0_opt = 1 + f eps0 <sinh^4 theta>` uses the author's phenomenological pulse-average
  factor f = 0.55 (`rrcp/sync.py`, `scripts/make_fig07.py`).
* **Eqs. (49)-(50), the 13 445 parameters, the RD row of Table VII, the network output:** there is no network
  code in this release, so no statement can be made about them.

## Printed diagnostics
`make_comparison.py` prints: E_inv(L)/E_inv(LL) per regime (recorded values), the tolerances of each
integration, and states that Q(a0 = 857, H0 = 1.2) and the Fig. 4 band-width ratio cannot be computed here.
`comparison.md` (generated) lists quantity | paper | code | discrepancy %.

## Statements of the manuscript and the files that support them
| Statement | File | Status |
|---|---|---|
| "figure scripts ... are part of the accompanying repository" (Sec. IV C) | `scripts/make_fig02.py`, `make_fig06b.py`, `make_fig07.py` | partly (Figs. 1, 3-5, 6(a) missing) |
| "invariant-conservation test" gamma - P_z (Sec. IV C) | - | absent |
| Data Availability: integrator, spectral and design-curve analyses, PINN-LL | `rrcp/dynamics.py` (reduced Eq. 29), `rrcp/spectrum.py`, `rrcp/erber.py`, `scripts/make_fig07.py` | partly; full integrator and PINN-LL absent |
| Appendix C: "dependency-free NumPy reference implementation" of PINN-LL | - | absent |
| Appendix C: "the benchmark script is included" | `scripts/benchmark_timing_author_NOT_RUNNABLE.py` | included, not runnable without the networks |

## Notes for readers of the manuscript
* **gamma_max.** In the manuscript, gamma_max ~ a0 (theta_max = arcsinh a0, Table III) is the classical ceiling of the
  phase-locked captured orbit (H0 close to H0_opt) used in Fig. 2 and Table III. It is a different quantity from the
  free-electron value in a plane wave without a static field (H0 = 0); the latter is not computed in this repository.
* **PINN mass-shell loss.** If the network output is the rapidity theta alone, cosh^2 theta - sinh^2 theta - 1 vanishes
  identically; the manuscript therefore states that the mass-shell loss (Eq. 50) acts on independent outputs
  (gamma, P). No network code is part of this release, so this cannot be checked here.
* **Relation to other repositories.** This release is the code deposit of the radiation-reaction manuscript only.
  It does not describe or contain the code of the earlier classical-limit paper.

## License
Not chosen yet; to be decided by the authors (no `LICENSE` file is included).
