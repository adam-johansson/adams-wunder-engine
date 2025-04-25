from thermo import fuel_props, mixture, H2, JETA_L


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
    f0 = 0.01  # initial guess
    convergence = False
    while not convergence:
        # actual fuel air ratio
        f = f0 / eta
        h1, _, _, _, _, _, _, _ = mixture(
            t1, p=p1, equ=equ1, fuel_type=fuel_type
        )
        h2, _, _, _, _, _, _, _ = mixture(
            t2, p=p2, equ=f / far_s + equ1, fuel_type=fuel_type
        )

        # theoretical fuel air ratio
        f_theo = ((h2 - h1) / (LHV + hf - h2)) * (1 + equ1 * far_s)
        # iterate until convergence
        if abs(f_theo - f0) < 1e-9:
            break
        f0 = f_theo

    # taking into account combustion losses
    f = f0 / eta

    return p2, t2, f
