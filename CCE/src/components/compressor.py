import numpy as np
from scipy.optimize import fsolve, brentq

# from timeit import default_timer as timer

from thermo import entropy_func, mixture


def compressor(t1, p1, m, eta, pr):
    # Right now assumes pure air in the compressors and fans

    psi1 = entropy_func(t1, p1)  # entropy function for initial state
    psi2 = psi1 + np.log(pr) / eta  # entropy function after compression
    p2 = p1 * pr  # pressure after compression

    def find_t2(t):
        residual = psi2 - entropy_func(t, p1)
        return residual

    t2 = brentq(find_t2, 200, 6000)

    h1, _, _, _, _, _, _, _ = mixture(
        t1, p1
    )  # enthalpy before compression
    h2, _, _, _, _, _, _, _ = mixture(
        t2, p2
    )  # enthalpy after compression
    P = m * (h2 - h1)  # power needed for the compression

    return p2, t2, P


def compressor_isentropic(t1, p1, m, eta, pr):

    # Right now assumes pure air in the compressors and fans

    # enthalpy before compression
    h1, _, _, _, _, _, _, _ = mixture(t1, p1)

    # entropy function for initial state
    psi1 = entropy_func(t1, p1)

    # no change in entropy for isentropic compression
    psi2 = psi1

    # pressure after compression
    p2 = p1 * pr

    def find_t2_ideal(t):
        residual = psi2 - entropy_func(t, p2)
        return residual

    # find ideal t2
    t2_ideal = brentq(find_t2_ideal, 200, 6000)

    h2_ideal, _, _, _, _, _, _, _ = mixture(t2_ideal, p2)  # ideal enthalpy

    # Find real h2
    h2 = h1 + (h2_ideal - h1) / eta

    def find_t2(t):
        h2_guess, _, _, _, _, _, _, _ = mixture(t, p2)
        residual = h2 - h2_guess
        return residual

    # find ideal t2
    t2 = brentq(find_t2, 200, 6000)

    # power needed for the compression
    P = m * (h2 - h1)

    return p2, t2, P