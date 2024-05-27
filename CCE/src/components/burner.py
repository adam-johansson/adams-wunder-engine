from CCE.src import thermo


def burner(p1, t1, equ1, t2, dp, eta, fuel_type):
    # pressure loss
    p2 = p1 * (1 - dp)

    far_s, LHV = thermo.fuel_props(fuel_type)

    f0 = 0.01  # initial guess
    convergence = False
    while not convergence:
        f_theo = f0
        f = f_theo / eta
        cp1, h1, s1, M1 = thermo.properties(t1, equ=equ1, fuel_type=fuel_type)
        cp2, h2, s2, M2 = thermo.properties(t2, equ=f/far_s + equ1, fuel_type=fuel_type)

        # theoretical fuel air ratio (not taking fuel enthalpy into account)
        f_theo = (h1 - h2) / (h2 - LHV)
        # iterate until convergence
        if abs(f_theo - f0) < 1e-8:
            break
        f0 = f_theo

    # taking into account combustion losses
    f = f_theo / eta

    return p2, t2, f
