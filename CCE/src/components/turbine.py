import numpy as np
from scipy.optimize import brentq

#from timeit import default_timer as timer

from CCE.src import thermo_outdated


# add fuel type in the input later
def turbine(t1_main, p1_main, m1_main, equ1_main, power_req, eta, fuel_type, cooling,
            t_cool=9999, m1_cool=9999, q_ngv=9999):

    p_dummy = 1e5
    error = False
    cp1_main, h1_main, s1_main, M1_main = thermo_outdated.properties(t1_main, p1_main, equ1_main, fuel_type=fuel_type)
    if not cooling:
        h1 = h1_main
        m1 = m1_main
        equ1 = equ1_main
        t1 = t1_main
    else:
        if t_cool == 9999:
            raise Exception('Cooling is turned on but no cooling properties are given.')

        # assumes pure air for cooling
        # TODO: NOTE THAT PRESSURE IS ONLY USED HERE TO GET ENTROPY. COULD add p cool
        cp1_cool, h1_cool, s1_cool, M1_cool = thermo_outdated.properties(t_cool, p1_main, equ=0)  # cooling air props
        m1 = m1_main + q_ngv * m1_cool  # mass flow after mixing of stator cooling
        h1 = (m1_main * h1_main + q_ngv * m1_cool * h1_cool) / m1  # enthalpy after mixing before rotor
        equ1 = equ1_main * m1_main / m1  # mass average (think this is correct)

        def find_t1(t):
            cp1, h1_guess, s1, M1 = thermo_outdated.properties(t, p_dummy, equ1, fuel_type=fuel_type)
            return h1 - h1_guess

        #start = timer()
        #t1 = fsolve(find_t1, x0=t1_main)[0]
        try:
            t1 = brentq(find_t1, 200, 1500)
        except ValueError:
            #print('trubbel')
            error = True
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),\
                float('nan'), float('nan'), error


    h2 = h1 - (power_req / m1) / eta  # specific enthalpy after expansion (for pressure calculation)
    h2_real = h1 - (power_req / m1)  # specific enthalpy after expansion (real)
    equ2 = equ1  # no cooling air is inserted at the rotor. only before or after

    def find_t2(t):
        cp2, h2_guess, s2, M2 = thermo_outdated.properties(t, p_dummy, equ1, fuel_type=fuel_type)
        return h2 - h2_guess

    def find_t2_real(t):
        cp2, h2_guess, s2, M2 = thermo_outdated.properties(t, p_dummy, equ1, fuel_type=fuel_type)
        return h2_real - h2_guess

    #t2 = fsolve(find_t2, x0=t1_main)[0]  # temperature after rotor (for calculating pressure drop)
    try:
        t2 = brentq(find_t2, 200, 1500)
    except ValueError:
        error = True
        #print('trubbel')
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), \
            float('nan'), float('nan'), error

    #t2_real = fsolve(find_t2_real, x0=t1_main)[0]  # temperature after rotor (real)
    try:
        t2_real = brentq(find_t2_real, 200, 1500)
    except ValueError:
        #print('trubbel')
        error = True
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), \
            float('nan'), float('nan'), error

    psi1 = thermo_outdated.entropy_func(t1, p1_main, equ1, fuel_type)  # before work extraction from rotor
    psi2 = thermo_outdated.entropy_func(t2, p1_main, equ2, fuel_type)  # after rotor
    pr = np.exp(psi2 - psi1)  # pressure ratio over rotor
    p2 = p1_main * pr  # pressure after rotor

    # cooling air inserted after rotor

    if not cooling:
        t3 = t2_real
        m3 = m1
        equ3 = equ2
    else:
        # assumes pure air for cooling
        m3 = m1 + (1 - q_ngv) * m1_cool
        h3 = (m1 * h2_real + (1 - q_ngv) * m1_cool * h1_cool) / m3
        equ3 = equ1 * m1 / m3  # mass average (think this is correct)

        def find_t3(t):
            cp3, h3_guess, s3, M3 = thermo_outdated.properties(t, p_dummy, equ3, fuel_type=fuel_type)
            return h3 - h3_guess

        #t3 = fsolve(find_t3, x0=t1_main)[0]
        try:
            t3 = brentq(find_t3, 200, 1500)
        except ValueError:
            #print('trubbel')
            error = True
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),\
                float('nan'), float('nan'), error

    #end = timer()
    #print(f'time turbine: {end - start}')
    return p2, t1, t2_real, t3, m1, m3, equ3, error
