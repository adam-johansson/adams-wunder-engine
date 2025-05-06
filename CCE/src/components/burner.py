from thermo import fuel_props, mixture, H2, JETA_L
from scipy.optimize import fsolve, brentq

def burner(p1, t1, equ1, t2, dp, eta, fuel_type, t_fuel):
    # pressure loss
    p2 = p1 * (1 - dp)

    # stochiometric fuel air ratio
    far_s, LHV = fuel_props(fuel_type)

    if fuel_type == "H2":
        cp_f, hf, s_f, _, M_f = H2(t_fuel, p1)
    elif fuel_type == "jetA":
        cp_f, hf, s_f, _, M_f = JETA_L(t_fuel)

    # note that the fuel air ratio that is sought is ratio of added fuel to pure air in the gas

    h1, u, cp, cv, R, gamma, s, M = mixture(t1, p1, equ=equ1, fuel_type=fuel_type)

    # fuel air ratio of fuel already (burned) in the gas from previous burner
    far0 = equ1 * far_s

    def find_far(far):
        h2, u, cp, cv, R, gamma, s, M = mixture(t2, p2, equ=far/far_s, fuel_type=fuel_type)

        residual = h2 * (1 + far0 + far) - h1 * (1 + far0) - far * hf
        return residual


    f = brentq(find_far, 0.0, far_s)

    # taking into account combustion losses
    f_real = f / eta

    return p2, t2, f_real
