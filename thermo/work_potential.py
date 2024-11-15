from thermo.thermo_computations import mixture
from thermo.entropy_func import entropy_func

from scipy.optimize import fsolve, brentq
import numpy as np


def work_potential(t1, p1, equ1, p0, fuel_type):

    # I think this function calculates the maximum amount of work that can be extracted when expanding the
    # fluid from p1 to p0
    p_dummy = 1e5
    h1, _, _, _, _, _, _ = mixture(t1, p_dummy, equ1, fuel_type)  # get thermo properties for the fluid

    psi1 = entropy_func(t1, p1, equ1, fuel_type)
    psi0_isen = psi1 - np.log(p1/p0)

    def find_t0(t):
        return psi0_isen - entropy_func(t, p1, equ1, fuel_type)  # p1 or p0 here?

    t0_isen = brentq(find_t0, 200, 6000)

    h0_isen, _, _, _, _, _, _ = mixture(t0_isen, p_dummy, equ1, fuel_type)  # get thermo_outdated properties for the fluid

    wp = h1 - h0_isen

    return wp
