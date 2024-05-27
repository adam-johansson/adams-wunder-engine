from piston_engine.src.piston.polynomials import N2, O2, CO2, H2O, Ar


def properties(t, equ, fuel_type = False):

    """
    Function that return thermodynamic properties of a mixture based on the
    properties of the individual species. This is for a combustion gas of
    JET_A/Diesel (C12H23). Argon is included in the air but not CO2, this will be included in the future.
    Further, combustion of H2 will also be available.

    Parameters
    ----------
    equ : float
        Equivalence ratio of the gas That is fuel air ratio divided by
        stochiometric fuel air ratio.
    t : float
        Temperature of the gas.
    x_N2in : float
        Mole fraction of N2 in the original air gas before combustion.
    x_O2in : float
        Mole fraction of O2 in the original air gas before combustion.
    h_array : array
        Array with the specific enthalpies of N2, O2, CO2 and H2O, in that order.
    cp_array : array
        Array with the specific heat capacities of N2, O2, CO2 and H2O, in that order.

    Returns
    -------
    h : float
        Mass specific enthalpy of the mixture.
    u : float
        Mass specific internal energy of the mixture.
    cp : float
        Mass specific heat capacity at constant pressure of the mixture.
    cv : float
        Mass specific heat capacity at constant volume of the mixture.
    R : float
        Specifc gas constant of the mixture.
    gamma : float
        Isentropic exponent of the mixture.
    s : float
        Specific entropy of the mixture. UNSURE ABOUT THIS ONE

    """
    N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar
    x_CO2_air = 0  # no co2 in air for now

    #print(x_Ar_air, x_N2_air, x_O2_air)

    #x_N2_air = 0.780840  # molar fraction of N2
    #x_O2_air = 0.209476  # molar fraction of O2
    #x_Ar_air = 0.009365  # molar fraction of Ar
    #x_CO2_air = 0.000319  # molar fraction of CO2

    cp_N2, h_N2, s_N2, M_N2 = N2(t)
    cp_O2, h_O2, s_O2, M_O2 = O2(t)
    cp_Ar, h_Ar, s_Ar, M_Ar = Ar(t)
    cp_H2O, h_H2O, s_H2O, M_H2O = H2O(t)
    cp_CO2, h_CO2, s_CO2, M_CO2 = CO2(t)

    M_air = x_N2_air * M_N2 + x_O2_air * M_O2 + x_Ar_air * M_Ar + x_CO2_air * M_CO2
    # air consisting of N2, O2, Ar and CO2

    if equ == 0:
        # if the fluid is pure air
        mu_N2 = x_N2_air * (M_N2 / M_air)  # mass fraction of N2 in the fluid
        mu_O2 = x_O2_air * (M_O2 / M_air)  # mass fraction of O2 in the fluid
        mu_Ar = x_Ar_air * (M_Ar / M_air)  # mass fraction of Ar in the fluid
        mu_CO2 = 0  # mass fraction of CO2 in the fluid
        mu_H2O = 0  # mass fraction of H2O in the fluid

        M = M_air

    else:
        if fuel_type == 'jetA':
            N = 5.75 * equ + 17.75 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas

            f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
            f2 = 17.75 * (1 - equ)  # O2
            f3 = 12 * equ  # CO2
            f4 = 11.5 * equ  # H2O
            f5 = 17.75 * (x_Ar_air / x_O2_air)

            x_N2 = f1 / N  # molar fractions
            x_O2 = f2 / N
            x_CO2 = f3 / N
            x_H2O = f4 / N
            x_Ar = f5 / N

            M = x_N2 * M_N2 + x_O2 * M_O2 + x_CO2 * M_CO2 + x_H2O * M_H2O + x_Ar * M_Ar  # molar mass of the fluid

            mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
            mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
            mu_CO2 = x_CO2 * (M_CO2 / M)  # mass fraction of CO2 in the fluid
            mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
            mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid

        elif fuel_type == 'H2':
            N = 0.5 * equ + 0.5 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas

            f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
            f2 = 0.5 * (1 - equ)  # O2
            f4 = equ  # H2O
            f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar

            x_N2 = f1 / N  # molar fractions
            x_O2 = f2 / N
            x_H2O = f4 / N
            x_Ar = f5 / N

            M = x_N2 * M_N2 + x_O2 * M_O2 + x_H2O * M_H2O + x_Ar * M_Ar  # molar mass of the fluid

            mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
            mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
            mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
            mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid
            mu_CO2 = 0.0  # no CO2 for H2

        else:
            raise Exception('Fuel type must be specified.')

    cp = mu_N2 * cp_N2 + mu_O2 * cp_O2 + mu_CO2 * cp_CO2 + mu_H2O * cp_H2O + mu_Ar * cp_Ar  # heat capacity at constant
    # pressure
    h = mu_N2 * h_N2 + mu_O2 * h_O2 + mu_CO2 * h_CO2 + mu_H2O * h_H2O + mu_Ar * h_Ar  # specific enthalpy
    s = mu_N2 * s_N2 + mu_O2 * s_O2 + mu_CO2 * s_CO2 + mu_H2O * s_H2O + mu_Ar * s_Ar  # specific entropy COULD BE WRONG

    #u = h - R * t  # inner energy
    #cv = cp - R  # heat capacity at constant volume
    #gamma = cp / cv  # heat capacity ratio

    return cp, h, s, M
