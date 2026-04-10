

import sys
sys.path.append("./../")

import numpy as np

from thermo import entropy_func, mixture

from scipy.optimize import brentq



# from Claude
def isentropic_to_polytropic_eff(eta_is, pr, t1, p1, equ1, fuel_type):
    """
    One-time conversion: extract polytropic efficiency from 
    reference engine isentropic efficiency + known pressure ratio.
    """
    psi1 = entropy_func(t1, p1, equ1, fuel_type)
    p2 = p1 * pr

    def find_t2_s(t):
        psi2 = entropy_func(t, p2, equ1, fuel_type)
        return psi2 - psi1

    t2_s = brentq(find_t2_s, 200, 2500)
    exponent = np.log(t2_s / t1) / np.log(pr)  # effective (gamma-1)/gamma

    eta_p = np.log(1 - eta_is * (1 - pr**exponent)) / (exponent * np.log(pr))
    return eta_p

# from Saravanamutta
def isentropic_to_polytropic_eff2(t1, t2, p1, p2, equ1, fuel_type):

    h, u, cp, cv, R, gamma1, s, M_avg = mixture(t1, p1, equ1, fuel_type=fuel_type)
    h, u, cp, cv, R, gamma2, s, M_avg = mixture(t2, p2, equ1, fuel_type=fuel_type)
    print(gamma1)
    print(gamma2)

    gamma = (gamma1 + gamma2)/2

    eta_p = (gamma/(gamma-1)) * (np.log(t1/t2) /np.log(p1/p2)  )
    return eta_p

# Saravanamutto
def isentropic_to_polytropic_eff3(t1, t2, p1, p2, equ1, fuel_type, eta_isen):

    h, u, cp, cv, R, gamma1, s, M_avg = mixture(t1, p1, equ1, fuel_type=fuel_type)
    h, u, cp, cv, R, gamma2, s, M_avg = mixture(t2, p2, equ1, fuel_type=fuel_type)
    print(gamma1)
    print(gamma2)

    gamma = (gamma1 + gamma2)/2

    eta_p = (gamma/(gamma-1)) * np.log(1 - eta_isen * (1 - (p2/p1)**((gamma-1)/gamma)  )      ) / np.log(p2/p1)
    return eta_p


# Walsh and Fletcher
def isentropic_to_polytropic_eff4(p1, p2, p2_isen):

    eta_p = np.log(p1/p2_isen)/np.log(p1/p2)
    return eta_p

# Walsh and Fletcher
def polytropic_to_isentropic(t1, t2, p1, p2, equ1, fuel_type, eta_isen):

    h, u, cp, cv, R, gamma1, s, M_avg = mixture(t1, p1, equ1, fuel_type=fuel_type)
    h, u, cp, cv, R, gamma2, s, M_avg = mixture(t2, p2, equ1, fuel_type=fuel_type)
    print(gamma1)
    print(gamma2)

    gamma = (gamma1 + gamma2)/2

    eta_p = (gamma/(gamma-1)) * np.log(1 - eta_isen * (1 - (p2/p1)**((gamma-1)/gamma)  )      ) / np.log(p2/p1)
    return eta_s



isentropic_eff = 0.94
p1 = 3.8765*1e5
p2 = 0.4280*1e5

pr = p1/p2
t1 = 1157.2048
t2 = 691.9581

equ1 = 0.0203/0.067
fuel_type = "jetA"

polytropic_eff = isentropic_to_polytropic_eff(isentropic_eff, pr, t1, p1, equ1, fuel_type)

print(polytropic_eff)

polytropic_eff = isentropic_to_polytropic_eff2(t1, t2, p1, p2, equ1, fuel_type)

print(polytropic_eff)


polytropic_eff = isentropic_to_polytropic_eff3(t1, t2, p1, p2, equ1, fuel_type, isentropic_eff)

print(polytropic_eff)
