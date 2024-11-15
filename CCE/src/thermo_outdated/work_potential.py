from CCE.src.thermo_outdated import properties, entropy_func

from scipy.optimize import fsolve, brentq
import numpy as np


def work_potential(t1, p1, equ1, p0, fuel_type):
    p_dummy = 1e5
    cp1, h1, s1, M1 = properties(t1, p_dummy, equ1, fuel_type)  # get thermo_outdated properties for the fluid

    psi1 = entropy_func(t1, p1, equ1, fuel_type)
    psi0_isen = psi1 - np.log(p1/p0)

    def find_t0(t):
        return psi0_isen - entropy_func(t, p1)  # p1 or p0 here?

    t0_isen = brentq(find_t0, 200, 6000)

    cp0_isen, h0_isen, s0_isen, M0_isen = properties(t0_isen, p_dummy, equ1, fuel_type)  # get thermo_outdated properties for the fluid

    wp = h1 - h0_isen

    return wp
