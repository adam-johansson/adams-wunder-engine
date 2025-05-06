import numpy as np
from scipy.optimize import fsolve, brentq

# from timeit import default_timer as timer

from thermo import entropy_func, mixture


def compressor(t1, p1, m, eta, pr):

    # For now, this function only has polytropic efficiency functionality
    # Right now assumes pure air in the compressors and fans

    psi1 = entropy_func(t1, p1)  # entropy function for initial state
    psi2 = psi1 + np.log(pr) / eta  # entropy function after compression
    p2 = p1 * pr  # pressure after compression

    def find_t2(t):
        return psi2 - entropy_func(t, p1)

    # start = timer()
    t2 = brentq(find_t2, 200, 6000)
    # end = timer()
    # print(f'time compressor: {end - start}')

    h1, _, _, _, _, _, _, _ = mixture(
        t1, p1, equ=0
    )  # enthalpy before compression
    h2, _, _, _, _, _, _, _ = mixture(
        t2, p2, equ=0
    )  # enthalpy after compression
    P = m * (h2 - h1)  # power needed for the compression

    return p2, t2, P
