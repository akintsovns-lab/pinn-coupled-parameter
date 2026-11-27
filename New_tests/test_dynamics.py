import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rrcp import constants, attractor, dynamics    # noqa: E402

EPS0 = constants.eps0(1e-6)
ALPHA = 0.36


def test_eq29_regression_against_authors_csv():
    ref = np.loadtxt(ROOT / "data" / "reference" / "fig2_panel_a_author.csv",
                     delimiter=",", skiprows=1)
    eta, th = dynamics.integrate(ALPHA, EPS0, rr=True, n=4000, name="test regression")
    assert np.allclose(eta / dynamics.L_PULSE, ref[:, 0], atol=1e-12)
    assert np.max(np.abs(th - ref[:, 1])) < 1e-8


def test_eq29_peak_below_classical_cap_and_near_attractor():
    eta, th = dynamics.integrate(ALPHA, EPS0, rr=True, name="test cap")
    th_star = attractor.theta_star(ALPHA, EPS0)
    # the damped orbit must stay below the classical ceiling asinh(a0)=7.45
    assert th.max() < np.arcsinh(857.0)
    # and it is limited by the attractor: no overshoot beyond 5 percent
    assert th.max() < 1.05 * th_star


@pytest.mark.skip(reason="ОТСУТСТВУЕТ: the invariant gamma - P_z = 1 needs the full system "
                         "Eqs. (4)-(6) (leading term F of Ref. akintsov2025), not in this deposit")
def test_invariant_gamma_minus_pz():
    pass
