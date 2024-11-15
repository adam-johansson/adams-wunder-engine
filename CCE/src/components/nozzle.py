import numpy as np
from scipy.optimize import fsolve, brentq
from CCE.src import thermo_outdated, compressible


def nozzle(p1, t1, pa, equ, m, cfg, cd, fuel_type):
    """calculating nozzle stuff"""

    error = False
    if p1 < pa:
        #print(f'Nozzle pressure is {p1} and lower than ambient pressure {pa}')
        F = float('nan')
        v2 = float('nan')
        v2_id = float('nan')
        error = True
        return F, v2_id, v2, error

    # entropy function at nozzle inlet
    psi1 = thermo_outdated.entropy_func(t1, p1, equ=equ, fuel_type=fuel_type)
    # enthalpy nozzle inlet
    cp1, h1, s1, mol1 = thermo_outdated.properties(t1, p1, equ=equ, fuel_type=fuel_type)
    # ideal static entropy function
    psi2_s_id = psi1 - np.log(p1/pa)

    def find_t2(t):
        #if t[0] < 200 or t[0] > 6000:
        #    return 1e9
        return psi2_s_id - thermo_outdated.entropy_func(t, p1)

    # iteratively solve for t2 static
    #t2_s = fsolve(find_t2, t1)[0]
    try:
        t2_s = brentq(find_t2, 200, 1500)
    except ValueError:
        print('Problem with nozzle.')
        error = True
        return float('nan'), float('nan'), float('nan'), error

    p_dummy = 1e5
    cp2, h2_s, s2, mol2 = thermo_outdated.properties(t2_s, p_dummy, equ=equ, fuel_type=fuel_type)

    # ideal exit velocity is difference between total inlet enthalpy and exit static enthalpy
    v2_id = np.sqrt(2 * (h1 - h2_s))

    # critical ambient pressure
    R_uni = 8.3144626  # J mol^-1 K^-1
    R = R_uni / mol1
    cv1 = cp1 - R
    gamma1 = cp1 / cv1
    pc = p1 * (1 - ((gamma1 - 1) / (gamma1 + 1))) ** (gamma1 / (gamma1 - 1))

    if pc < pa:
        # subsonic nozzle
        v2 = v2_id
        # static pressure is equal to ambient so no pressure term
        F = cfg * m * v2
        return F, v2_id, v2, error

    else:
        # choked nozzle
        # get speed of sound in exhaust stream
        t21s = t1 / (1 + 0.5 * (gamma1-1))  # static temperature at nozzle exit for Mach 1
        a2 = compressible.speed_of_sound(t21s, equ=equ, fuel_type=fuel_type)
        v2 = a2

        # static pressure
        p2s = p1 / ((t1 / t21s) ** (gamma1 / (gamma1-1)))

        # nozzle exit area
        rho2 = p2s / (R * t21s)
        A2 = m / (rho2 * v2 * cd)

        # thrust
        F = cfg * m * v2 + A2 * (p2s - pa)

        return F, v2_id, v2, error
