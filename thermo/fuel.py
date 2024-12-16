from thermo.thermo_computations import mixture


def fuel_props(fuel_type):

    N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

    _, _, _, _, _, _, _, M_air = mixture(t=999, p=1e5) # mole mass is constant with temperature

    if fuel_type == 'jetA':
        LHV = 43.0319e6  # from the difference of formation enthalpies
        Mfuel = 167.31e-3  # using c12h23 (JET-A)
        far_s = (Mfuel / (17.75 * M_air * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)))

    elif fuel_type == 'H2':
        LHV = 119.9605e6  # from the difference of formation enthalpies
        Mfuel = 2.01588e-3  # from Coolprop
        far_s = (Mfuel / (0.5 * M_air * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)))

    else:
        raise Exception(f'Unknown fuel type. The fuel type was {fuel_type}.')

    return far_s, LHV