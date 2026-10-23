import numpy as np
from scipy.optimize import fsolve, brentq

from thermo import mixture, entropy_func
from CCE.src import compressible
def nozzle(p1, t1, pa, equ, m, cfg, cd, fuel_type):
    """Convergent nozzle with variable cp: gross thrust, ideal/actual velocity, exit area, Mach."""

    error = False
    if p1 < pa:
        error = True
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), error

    p_dummy = 1e5
    psi1 = entropy_func(t1, p1, equ=equ, fuel_type=fuel_type)
    h1, _, _, _, _, _, _, _ = mixture(t1, p1, equivalence_ratio=equ, fuel_type=fuel_type)

    # ideal full expansion to ambient
    psi2_s_id = psi1 - np.log(p1 / pa)

    def find_t2_s(t):
        return psi2_s_id - entropy_func(t, p1, equ=equ, fuel_type=fuel_type)

    try:
        t2_s = brentq(find_t2_s, 100, t1)
    except ValueError:
        error = True
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), error

    h2_s, _, _, _, _, _, _, _ = mixture(t2_s, p_dummy, equivalence_ratio=equ, fuel_type=fuel_type)
    v2_id = np.sqrt(2 * (h1 - h2_s))

    # sonic (choked) state: h1 - h(T*) = a(T*)^2 / 2
    def choke_residual(t):
        h, _, _, _, R, gamma, _, _ = mixture(t, p_dummy, equivalence_ratio=equ, fuel_type=fuel_type)
        return (h1 - h) - 0.5 * gamma * R * t

    t21s = brentq(choke_residual, 0.5 * t1, t1)
    _, _, _, _, R_s, gamma_s, _, _ = mixture(t21s, p_dummy, equivalence_ratio=equ, fuel_type=fuel_type)
    p2s = p1 * np.exp(entropy_func(t21s, p1, equ=equ, fuel_type=fuel_type) - psi1)

    if p2s <= pa:
        # subsonic nozzle: exit static pressure = ambient
        v2 = v2_id
        F = cfg * m * v2
        _, _, _, _, R2, gamma2, _, _ = mixture(t2_s, pa, equivalence_ratio=equ, fuel_type=fuel_type)
        Mach = v2 / np.sqrt(gamma2 * R2 * t2_s)
        A2 = m / (pa / (R2 * t2_s) * v2 * cd) if v2 > 0 else float("nan")
    else:
        # choked nozzle: Mach 1 at exit, under-expanded
        Mach = 1.0
        v2 = np.sqrt(gamma_s * R_s * t21s)
        rho2 = p2s / (R_s * t21s)
        A2 = m / (rho2 * v2 * cd)
        F = cfg * m * v2 + A2 * (p2s - pa)

    return F, v2_id, v2, A2, Mach, error