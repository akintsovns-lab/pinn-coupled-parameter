"""Checks of closed-form quantities against the numbers quoted in the manuscript.
The paper numbers are EXPECTED values (tolerances), not results of this code."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrcp import constants, attractor, erber, spectrum, sync   # noqa: E402

EPS0 = constants.eps0(1e-6)
ALPHA = sync.ALPHA_RD


def test_eps0():
    assert EPS0 == pytest.approx(1.180e-8, rel=1e-3)


def test_gamma_star():
    assert attractor.gamma_star(ALPHA, EPS0) == pytest.approx(74.3, rel=2e-3)


def test_theta_star():
    assert attractor.theta_star(ALPHA, EPS0) == pytest.approx(5.00, abs=5e-3)


def test_stability_lambda_is_4alpha():
    assert attractor.stability_Lambda(ALPHA, EPS0) == pytest.approx(4 * ALPHA, rel=2e-3)


def test_eps_crit_rd():
    assert attractor.eps_crit(ALPHA, 5.68) == pytest.approx(9.1e-15, rel=2e-2)


def test_erber_rd():
    assert float(erber.g_rational(0.16)) == pytest.approx(0.56, abs=5e-3)


@pytest.mark.parametrize("chi,expected", [(0.5, 0.30), (1.0, 0.18), (2.0, 0.10), (5.0, 0.04)])
def test_erber_table(chi, expected):
    assert float(erber.g_rational(chi)) == pytest.approx(expected, abs=1.5e-2)


def test_linewidth_wnl():
    # WNL: a0 = 8.5 < sinh(theta*), so sinh(theta) = a0
    assert spectrum.fractional_linewidth(EPS0, 8.5) == pytest.approx(1.7e-6, rel=2e-2)


def test_linewidth_rd():
    s = sync.sinh_star(ALPHA, EPS0)
    assert spectrum.fractional_linewidth(EPS0, s) == pytest.approx(1.3e-4, rel=2e-2)


def test_peak_rr_and_downshift():
    p = spectrum.peak_rr(ALPHA, EPS0)
    assert p == pytest.approx(5.5e3, rel=2e-2)
    assert spectrum.peak_class(857.0) / p == pytest.approx(133, rel=2e-2)


@pytest.mark.xfail(strict=True, reason="Eq. (B1) as printed is chi-independent (=3.248); "
                   "reported to the author (README, comparison.md)")
def test_B1_as_printed_matches_rational():
    assert erber.g_integral_B1_as_printed(0.5) == pytest.approx(0.30, abs=2e-2)


def test_Q_map_is_absent():
    with pytest.raises(NotImplementedError):
        sync.Q_map(857.0, 1.2, EPS0)
