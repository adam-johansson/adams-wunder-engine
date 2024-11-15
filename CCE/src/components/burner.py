from CCE.src import thermo_outdated
from piston_engine.src.piston.polynomials_outdated import H2, JETA


def burner(p1, t1, equ1, t2, dp, eta, fuel_type, t_fuel):
    # pressure loss
    p2 = p1 * (1 - dp)

    # stochiometric fuel air ratio
    far_s, LHV = thermo_outdated.fuel_props(fuel_type)

    if fuel_type == "H2":
        cp_f, hf, s_f, M_f = H2(t_fuel, p1)
    elif fuel_type == "jetA":
        cp_f, hf = JETA(t_fuel)

    # note that the fuel air ratio that is sought is ratio of added fuel to pure air in the gas
    f0 = 0.01  # initial guess
    convergence = False
    while not convergence:
        # actual fuel air ratio
        f = f0 / eta
        cp1, h1, s1, M1 = thermo_outdated.properties(t1, p=p1, equ=equ1, fuel_type=fuel_type)
        cp2, h2, s2, M2 = thermo_outdated.properties(t2, p=p2, equ=f / far_s + equ1, fuel_type=fuel_type)

        # theoretical fuel air ratio
        f_theo = ((h2 - h1) / (LHV + hf - h2)) * (1 + equ1 * far_s)
        # iterate until convergence
        if abs(f_theo - f0) < 1e-9:
            break
        f0 = f_theo

    # taking into account combustion losses
    f = f0 / eta

    return p2, t2, f
