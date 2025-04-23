import numpy as np
from scipy.optimize import fsolve, brentq

# from timeit import default_timer as timer

from CCE.src import thermo_outdated


def compressor(t1, p1, m, eta, pr):
    # For now, this function only has polytropic efficiency functionality
    # Right now assumes pure air in the compressors and fans

    psi1 = thermo_outdated.entropy_func(t1, p1)  # entropy function for initial state
    psi2 = psi1 + np.log(pr) / eta  # entropy function after compression
    p2 = p1 * pr  # pressure after compression

    def find_t2(t):
        return psi2 - thermo_outdated.entropy_func(t, p1)

    # start = timer()
    t2 = brentq(find_t2, 200, 6000)
    # end = timer()
    # print(f'time compressor: {end - start}')

    cp, h1, s, M = thermo_outdated.properties(
        t1, p1, equ=0
    )  # enthalpy before compression
    cp, h2, s, M = thermo_outdated.properties(
        t2, p2, equ=0
    )  # enthalpy after compression
    P = m * (h2 - h1)  # power needed for the compression

    return p2, t2, P
