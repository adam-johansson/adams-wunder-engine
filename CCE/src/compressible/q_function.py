import sys
sys.path.append(".")


import numpy as np
from thermo import mixture


def calc_area_q(m, Mach, p, T, equivalence_ratio=0.0):

    _, _, _, _, R, gamma, _, _ = mixture(T, p, equivalence_ratio=equivalence_ratio)
    A = m / ( np.sqrt(gamma)*Mach*p*(1/np.sqrt(R*T))*(1+0.5*(gamma-1)*Mach**2)**( -0.5*(gamma+1)/(gamma-1) ))

    return A



if __name__ == "__main__":
    

    # fan
    m = 0.24865912973989E+03
    p = 0.35573931327320E+05
    T = 0.25672984967476E+03
    Mach = 0.7

    A = calc_area_q(m, Mach, p, T)
    print(f"Area: {A:.4f} m2")