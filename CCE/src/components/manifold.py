from thermo import mixture
from scipy.optimize import fsolve


def mix(m1, t1, equ1, m2, t2, equ2, fuel_type=False):
    p = 1e5
    m3 = m1 + m2  # outlet mass flow is equal sum of inlet mass flows
    equ3 = (m1 * equ1 + m2 * equ2) / m3  # CHECK IF THIS IS REALLY CORRECT
    h1, _, _, _, _, _, _, _ = mixture(t1, p, equ1, fuel_type=fuel_type)
    h2, _, _, _, _, _, _, _ = mixture(t2, p, equ2, fuel_type=fuel_type)

    h3 = (m1 * h1 + m2 * h2) / m3

    def find_t3(t):
        h3_guess, _, _, _, _, _, _, _ = mixture(
            t[0].astype(float), p, equ3, fuel_type
        )
        return h3 - h3_guess

    t3 = fsolve(find_t3, t1)[0]

    return t3, equ3


# HOW TO MIX PRESSURE?
