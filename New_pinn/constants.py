"""Physical constants (CODATA 2018) and quantities derived from them."""
import math

R_E = 2.8179403262e-15          # classical electron radius, m
MC2_EV = 0.51099895000e6        # electron rest energy, eV
HBARC_EV_M = 1.973269804e-7     # hbar c, eV m
C_LIGHT = 299792458.0           # m/s


def eps0(lam_m: float = 1e-6) -> float:
    """eps0 = 4 pi r_e / (3 lambda)   (= 1.180e-8 at 1 um)."""
    return 4.0 * math.pi * R_E / (3.0 * lam_m)


def a_S(lam_m: float = 1e-6) -> float:
    """Schwinger amplitude in units of a0: a_S = m c^2 / (hbar omega)."""
    hbar_omega_ev = 2.0 * math.pi * HBARC_EV_M / lam_m
    return MC2_EV / hbar_omega_ev


def B0_tesla(lam_m: float = 1e-6) -> float:
    """B0 = m c omega / e = 2 pi (m c^2/e) / (c lambda)   [T]."""
    return 2.0 * math.pi * MC2_EV / (C_LIGHT * lam_m)


def a0_of_intensity(I_w_cm2: float, lam_um: float = 1.0) -> float:
    """a0 = sqrt(I / I_rel), I_rel = 1.37e18 lambda^-2 W/cm^2 (as in the paper)."""
    return math.sqrt(I_w_cm2 * lam_um ** 2 / 1.37e18)
