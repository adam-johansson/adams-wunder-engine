from CCE.src import thermo
from scipy.optimize import fsolve


def mix(m1, t1, equ1, m2, t2, equ2, fuel_type=False):
    m3 = m1 + m2  # outlet mass flow is equal sum of inlet mass flows
    equ3 = (m1 * equ1 + m2 * equ2)/m3  # CHECK IF THIS IS REALLY CORRECT
    cp1, h1, s1, M1 = thermo.properties(t1, equ1, fuel_type=fuel_type)
    cp2, h2, s2, M2 = thermo.properties(t2, equ2, fuel_type=fuel_type)

    h3 = (m1 * h1 + m2 * h2) / m3

    def find_t3(t):
        cp3, h3_guess, s3, M3 = thermo.properties(t[0].astype(float), equ3, fuel_type=fuel_type)
        return h3 - h3_guess

    t3 = fsolve(find_t3, t1)[0]

    return t3, equ3

# HOW TO MIX PRESSURE?