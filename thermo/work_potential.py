from thermo.thermo_computations import mixture
from thermo.entropy_func import entropy_func

from scipy.optimize import fsolve, brentq
import numpy as np


def work_potential(t1, p1, equ1, p0, fuel_type):

    ##The work potential WP is the extractable specific work
    #when expanding a fluid from a given state with temperature 𝑇1 and pressure 𝑝1 to ambient
    #conditions 𝑝0.

    # specific enthalpy of the fluid at the given state
    h1, _, _, _, _, _, _, _ = mixture(
        t1, p1, equ1, fuel_type
    )

    # entropy function at given state
    psi1 = entropy_func(t1, p1, equ1, fuel_type)

    # entropy function after isentropic expansion to ambient state
    psi0_isen = psi1 - np.log(p1 / p0)


    def find_t0(t):
        return psi0_isen - entropy_func(t, p1, equ1, fuel_type)  # p1 or p0 here?

    t0_isen = brentq(find_t0, 200, 6000)

    h0_isen, _, _, _, _, _, _, _ = mixture(
        t0_isen, p0, equ1, fuel_type
    )  # get thermo_outdated properties for the fluid

    # basically v_id^2 / 2
    wp = h1 - h0_isen

    return wp
