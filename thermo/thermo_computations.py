from thermo.polynomials import N2, O2, CO2, H2O, Ar, JETA_G, H2

# from scipy.optimize import fsolve
from numba import njit


@njit()
def mixture(t, p, equ=0, fuel_type=False, pure_fuel=False, fuel_equ_ratio=0.0):
    """
    Function that return thermodynamic properties of a mixture based on the
    properties of the individual species. This is for a combustion gas of
    JET_A/Diesel (C12H23).

    Parameters
    ----------
    equ : float
        Equivalence ratio of the gas That is fuel air ratio divided by
        stochiometric fuel air ratio, weight ratios that is.
    t : float
        Temperature of the gas.
    p : float
        Pressure of the gas mixture.
    fuel_type : string
        Type of fuel. H2 or jetA


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

    # if equ > 1:
    #    #print(f"Equivalence ratio: {equ} is larger than 1")
    #    equ = 1.0

    Runiv = 8.3144626  # J mol^-1 K^-1

    # The amount of moles of the gas before combustion. Could be pure air or air + fuel mixture
    if pure_fuel is True and fuel_type == "jetA":
        # O2 + N2 + Ar + C12H23
        N_air = 1 + 3.7274 + 0.0444 + fuel_equ_ratio * (1 / 17.75)
        x_JETA_air = fuel_equ_ratio * (1 / 17.75) / N_air  # molar fraction of C12H23
        x_H2_air = 0.0  # molar fraction of H2 (0 for JetA)
        _, _, _, _, M_JETA = JETA_G(t, p)
        _, _, _, _, M_H2 = H2(t, p)

    elif pure_fuel is True and fuel_type == "H2":
        # O2 + N2 + Ar + H2
        N_air = 1 + 3.7274 + 0.0444 + fuel_equ_ratio * 2
        x_H2_air = fuel_equ_ratio * 2 / N_air  # molar fraction of H2
        x_JETA_air = 0.0  # molar fraction of JetA (0 for H2)
        _, _, _, _, M_H2 = H2(t, p)
        _, _, _, _, M_JETA = JETA_G(t, p)

    else:
        # O2 + N2 + Ar
        N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air (per 1 mole of O2)

    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

    # retrieve the molar masses
    _, _, _, _, M_N2 = N2(t, p)
    _, _, _, _, M_O2 = O2(t, p)
    _, _, _, _, M_Ar = Ar(t, p)
    _, _, _, _, M_H2O = H2O(t, p)
    _, _, _, _, M_CO2 = CO2(t, p)

    if pure_fuel:
        # fuel in mixture
        M_air = (
            x_N2_air * M_N2
            + x_O2_air * M_O2
            + x_Ar_air * M_Ar
            + x_JETA_air * M_JETA
            + x_H2_air * M_H2
        )
    else:
        # no fuel in mixture
        M_air = x_N2_air * M_N2 + x_O2_air * M_O2 + x_Ar_air * M_Ar

    if equ == 0 and fuel_equ_ratio == 0:

        if pure_fuel:
            mu_JETA = x_JETA_air * (M_JETA / M_air)
            mu_H2 = x_H2_air * (M_H2 / M_air)

        # if the fluid is pure air
        mu_N2 = x_N2_air * (M_N2 / M_air)  # mass fraction of N2 in the fluid
        mu_O2 = x_O2_air * (M_O2 / M_air)  # mass fraction of O2 in the fluid
        mu_Ar = x_Ar_air * (M_Ar / M_air)  # mass fraction of Ar in the fluid
        mu_CO2 = 0  # mass fraction of CO2 in the fluid
        mu_H2O = 0  # mass fraction of H2O in the fluid

        M = M_air

    else:
        # THIS IS FORE THE CASE OF FUEL IN THE AIR
        if pure_fuel:
            if fuel_type == "jetA":

                # (CO2 + H2O - O2 - JETA) * equ + O2 + N2 + Ar + JETA
                N = (
                    4.75 * equ
                    + 17.75 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)
                    + fuel_equ_ratio
                )  # total number of moles in gas

                f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
                f2 = 17.75 * (1 - equ)  # O2
                f3 = 12 * equ  # CO2
                f4 = 11.5 * equ  # H2O
                f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar
                f6 = fuel_equ_ratio - equ  # C12H23

                x_N2 = f1 / N  # molar fractions
                x_O2 = f2 / N
                x_CO2 = f3 / N
                x_H2O = f4 / N
                x_Ar = f5 / N
                x_JETA = f6 / N

                M = (
                    x_N2 * M_N2
                    + x_O2 * M_O2
                    + x_CO2 * M_CO2
                    + x_H2O * M_H2O
                    + x_Ar * M_Ar
                    + x_JETA * M_JETA
                )  # molar mass of the fluid

                mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
                mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
                mu_CO2 = x_CO2 * (M_CO2 / M)  # mass fraction of CO2 in the fluid
                mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
                mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid
                mu_JETA = x_JETA * (M_JETA / M)  # mass fraction of JETA in the fluid
                mu_H2 = 0.0

            elif fuel_type == "H2":
                # (H2O - O2 - H2) * equ + O2 + N2 + Ar + H2
                N = (
                    -0.5 * equ
                    + 0.5 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)
                    + fuel_equ_ratio
                )  # total number of moles in gas

                f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
                f2 = 0.5 * (1 - equ)  # O2
                f4 = equ  # H2O
                f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar
                f6 = fuel_equ_ratio - equ  # H2

                x_N2 = f1 / N  # molar fractions
                x_O2 = f2 / N
                x_H2O = f4 / N
                x_Ar = f5 / N
                x_H2 = f6 / N

                M = (
                    x_N2 * M_N2
                    + x_O2 * M_O2
                    + x_H2O * M_H2O
                    + x_Ar * M_Ar
                    + x_H2 * M_H2
                )  # molar mass of the fluid

                mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
                mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
                mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
                mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid
                mu_CO2 = 0.0  # no CO2 for H2
                mu_H2 = x_H2 * (M_H2 / M)  # mass fraction of H2 in the fluid
                mu_JETA = 0.0

            else:
                raise Exception("Fuel type must be specified.")

        # THIS IS FOR NO FUEL IN THE AIR. ONLY PURE AIR AND COMBUSTION PRODUCTS
        else:
            if fuel_type == "jetA":
                N = 5.75 * equ + 17.75 * (
                    1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air
                )  # total number of moles in gas

                f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
                f2 = 17.75 * (1 - equ)  # O2
                f3 = 12 * equ  # CO2
                f4 = 11.5 * equ  # H2O
                f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar

                x_N2 = f1 / N  # molar fractions
                x_O2 = f2 / N
                x_CO2 = f3 / N
                x_H2O = f4 / N
                x_Ar = f5 / N

                M = (
                    x_N2 * M_N2
                    + x_O2 * M_O2
                    + x_CO2 * M_CO2
                    + x_H2O * M_H2O
                    + x_Ar * M_Ar
                )  # molar mass of the fluid

                mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
                mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
                mu_CO2 = x_CO2 * (M_CO2 / M)  # mass fraction of CO2 in the fluid
                mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
                mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid

            elif fuel_type == "H2":
                N = 0.5 * equ + 0.5 * (
                    1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air
                )  # total number of moles in gas

                f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
                f2 = 0.5 * (1 - equ)  # O2
                f4 = equ  # H2O
                f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar

                x_N2 = f1 / N  # molar fractions
                x_O2 = f2 / N
                x_H2O = f4 / N
                x_Ar = f5 / N

                M = (
                    x_N2 * M_N2 + x_O2 * M_O2 + x_H2O * M_H2O + x_Ar * M_Ar
                )  # molar mass of the fluid

                mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
                mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
                mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
                mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid
                mu_CO2 = 0.0  # no CO2 for H2

            else:
                raise Exception("Fuel type must be specified.")

    # partial pressures
    if pure_fuel:
        p_JETA = mu_JETA * p
        p_H2 = mu_H2 * p

    p_N2 = mu_N2 * p
    p_O2 = mu_O2 * p
    p_Ar = mu_Ar * p
    p_H2O = mu_H2O * p
    p_CO2 = mu_CO2 * p

    # now use the partial pressures to get the correct entropy values
    cp_N2, h_N2, s_N2, _, M_N2 = N2(t, p_N2)
    cp_O2, h_O2, s_O2, _, M_O2 = O2(t, p_O2)
    cp_Ar, h_Ar, s_Ar, _, M_Ar = Ar(t, p_Ar)

    cp_H2O, h_H2O, s_H2O, _, M_H2O = H2O(t, p_H2O)
    cp_CO2, h_CO2, s_CO2, _, M_CO2 = CO2(t, p_CO2)

    if pure_fuel:
        cp_JETA, h_JETA, s_JETA, _, M_JETA = JETA_G(t, p_JETA)
        cp_H2, h_H2, s_H2, _, M_H2 = H2(t, p_H2)

    if pure_fuel:
        cp = (
            mu_N2 * cp_N2
            + mu_O2 * cp_O2
            + mu_CO2 * cp_CO2
            + mu_H2O * cp_H2O
            + mu_Ar * cp_Ar
            + mu_JETA * cp_JETA
            + mu_H2 * cp_H2
        )
        h = (
            mu_N2 * h_N2
            + mu_O2 * h_O2
            + mu_CO2 * h_CO2
            + mu_H2O * h_H2O
            + mu_Ar * h_Ar
            + mu_JETA * h_JETA
            + mu_H2 * h_H2
        )
        s = (
            mu_N2 * s_N2
            + mu_O2 * s_O2
            + mu_CO2 * s_CO2
            + mu_H2O * s_H2O
            + mu_Ar * s_Ar
            + mu_JETA * s_JETA
            + mu_H2 * s_H2
        )
    else:
        cp = (
            mu_N2 * cp_N2
            + mu_O2 * cp_O2
            + mu_CO2 * cp_CO2
            + mu_H2O * cp_H2O
            + mu_Ar * cp_Ar
        )  # heat capacity at constant
        # pressure
        h = (
            mu_N2 * h_N2 + mu_O2 * h_O2 + mu_CO2 * h_CO2 + mu_H2O * h_H2O + mu_Ar * h_Ar
        )  # specific enthalpy

        s = (
            mu_N2 * s_N2 + mu_O2 * s_O2 + mu_CO2 * s_CO2 + mu_H2O * s_H2O + mu_Ar * s_Ar
        )  # specific entropy

    R = Runiv / M  # specific gas constant
    u = h - R * t  # inner energy
    cv = cp - R  # specific heat capacity at constant volume
    gamma = cp / cv  # isentropic exponent

    return h, u, cp, cv, R, gamma, s, M


@njit()
def equivalence_derivative(equ, t, p, fuel_type, pure_fuel, fuel_equ_ratio):
    """
    Function that returns the partial derivative of specific gas constant R
    and specific internal energy u with respect to the equivalence ration.

    Parameters
    ----------
    equ : float
        Equivalence ratio of the gas That is fuel air ratio divided by
        stochiometric fuel air ratio.
    t : float
        Temperature of the gas.
    fuel_type : string
        JetA or H2.


    Returns
    -------
    dellRdellequ : float
        Partial derivative of the specific gas constant R with respect
        to the equivalence ratio.
    delludellequ : float
        Partial derivative of the specific internal energy u with respect
        to the equivalence ratio.

    """

    # THESE CALCULATIONS CAN PROBABLY BE QUICKER IF MU IS TARGETED DIRECTLY
    # if equ > 1:
    #    #print(f"Equivalence ratio: {equ} is larger than 1")
    #    equ = 1.0

    Runiv = 8.3144626  # J mol^-1 K^-1

    # The amount of moles of the gas before combustion. Could be pure air or air + fuel mixture
    if pure_fuel is True and fuel_type == "jetA":
        # O2 + N2 + Ar + C12H23
        N_air = 1 + 3.7274 + 0.0444 + fuel_equ_ratio * (1 / 17.75)
        x_JETA_air = fuel_equ_ratio * (1 / 17.75) / N_air  # molar fraction of C12H23
        x_H2_air = 0.0  # molar fraction of H2 (0 for JetA)
        _, h_JETA, _, _, M_JETA = JETA_G(t, p)
        _, h_H2, _, _, M_H2 = H2(t, p)

    elif pure_fuel is True and fuel_type == "H2":
        # O2 + N2 + Ar + H2
        N_air = 1 + 3.7274 + 0.0444 + fuel_equ_ratio * 2
        x_H2_air = fuel_equ_ratio * 2 / N_air  # molar fraction of H2
        x_JETA_air = 0.0  # molar fraction of JetA (0 for H2)
        _, h_H2, _, _, M_H2 = H2(t, p)
        _, h_JETA, _, _, M_JETA = JETA_G(t, p)

    else:
        # O2 + N2 + Ar
        N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air (per 1 mole of O2)

    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

    # retrieve the molar masses
    _, h_N2, _, _, M_N2 = N2(t, p)
    _, h_O2, _, _, M_O2 = O2(t, p)
    _, h_Ar, _, _, M_Ar = Ar(t, p)
    _, h_H2O, _, _, M_H2O = H2O(t, p)
    _, h_CO2, _, _, M_CO2 = CO2(t, p)

    if pure_fuel:
        if fuel_type == "jetA":

            Ntot = (
                4.75 * equ
                + 17.75 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)
                + fuel_equ_ratio
            )  # total number of moles in gas
            dNtotdequ = 4.75

            f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
            f2 = 17.75 * (1 - equ)  # O2
            f3 = 12 * equ  # CO2
            f4 = 11.5 * equ  # H20
            f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar
            f6 = fuel_equ_ratio - equ

            df1dequ = 0
            df2dequ = -17.75
            df3dequ = 12
            df4dequ = 11.5
            df5dequ = 0
            df6dequ = -1

            x_N2 = f1 / Ntot  # molar fraction
            x_O2 = f2 / Ntot
            x_CO2 = f3 / Ntot
            x_H2O = f4 / Ntot
            x_Ar = f5 / Ntot
            x_JETA = f6 / Ntot

            M = (
                x_N2 * M_N2
                + x_O2 * M_O2
                + x_CO2 * M_CO2
                + x_H2O * M_H2O
                + x_Ar * M_Ar
                + x_JETA * M_JETA
            )

            dellx_N2dellequ = (df1dequ * Ntot - f1 * dNtotdequ) / (Ntot ** 2.0)
            dellx_O2dellequ = (df2dequ * Ntot - f2 * dNtotdequ) / (Ntot ** 2.0)
            dellx_CO2dellequ = (df3dequ * Ntot - f3 * dNtotdequ) / (Ntot ** 2.0)
            dellx_H2Odellequ = (df4dequ * Ntot - f4 * dNtotdequ) / (Ntot ** 2.0)
            dellx_Ardellequ = (df5dequ * Ntot - f5 * dNtotdequ) / (Ntot ** 2.0)
            dellx_JETAdellequ = (df6dequ * Ntot - f6 * dNtotdequ) / (Ntot ** 2.0)

            dellMdellequ = (
                M_N2 * dellx_N2dellequ
                + M_O2 * dellx_O2dellequ
                + M_CO2 * dellx_CO2dellequ
                + M_H2O * dellx_H2Odellequ
                + M_Ar * dellx_Ardellequ
                + M_JETA * dellx_JETAdellequ
            )

            dellRdellequ = -(Runiv / M ** 2.0) * dellMdellequ

            # Analytical shorter expressions
            # Partial derivative of h with resptect to equ
            # k2 = (12*M_CO2*hco2 + 11.5*M_H2O*hh2o - 17.75*M_O2*ho2)*(M_N2*(1 + x_N2in/x_O2in) + M_O2)
            # k3 = (M_N2*hn2*(1+x_N2in/x_O2in) + M_O2*ho2)*(12*M_CO2 + 11.5*M_H2O - 17.75*M_O2)
            # k4 = 17.75*(M_N2*(1 + x_N2in/x_O2in) + M_O2) + equ * (12*M_CO2 + 11.5*M_H2O - 17.75*M_O2)

            # dellhdellequ = 17.75*(k2 - k3)/(k4 ** 2.0)

            # Partial derivative of R with respect to equ
            # k1 = Runiv * 17.75 *(5.75 * (M_N2 * (1 + x_N2in/x_O2in) + M_O2) -
            #                     (1 + x_N2in/x_O2in)*(11.5 * M_H2O + 12* M_CO2 - 17.75*M_O2)  )

            # dellRdellequ =  k1/((17.75 * (M_N2*(1 + x_N2in/x_O2in) + M_O2) + equ*(11.5*M_H2O + 12*M_CO2 - 17.75*M_O2) ) ** 2.0)

            dellmu_N2dellequ = (dellx_N2dellequ * M - x_N2 * dellMdellequ) * (
                M_N2 / M ** 2.0
            )
            dellmu_O2dellequ = (dellx_O2dellequ * M - x_O2 * dellMdellequ) * (
                M_O2 / M ** 2.0
            )
            dellmu_CO2dellequ = (dellx_CO2dellequ * M - x_CO2 * dellMdellequ) * (
                M_CO2 / M ** 2.0
            )
            dellmu_H2Odellequ = (dellx_H2Odellequ * M - x_H2O * dellMdellequ) * (
                M_H2O / M ** 2.0
            )
            dellmu_Ardellequ = (dellx_Ardellequ * M - x_Ar * dellMdellequ) * (
                M_Ar / M ** 2.0
            )
            dellmu_JETAdellequ = (dellx_JETAdellequ * M - x_JETA * dellMdellequ) * (
                M_JETA / M ** 2.0
            )

            dellhdellequ = (
                h_N2 * dellmu_N2dellequ
                + h_O2 * dellmu_O2dellequ
                + h_CO2 * dellmu_CO2dellequ
                + h_H2O * dellmu_H2Odellequ
                + h_Ar * dellmu_Ardellequ
                + h_JETA * dellmu_JETAdellequ
            )

            delludellequ = dellhdellequ - dellRdellequ * t

        elif fuel_type == "H2":
            Ntot = (
                -0.5 * equ
                + 0.5 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)
                + fuel_equ_ratio
            )  # total number of moles in gas
            dNtotdequ = -0.5

            f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
            f2 = 0.5 * (1 - equ)  # O2
            f4 = equ  # H2O
            f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar
            f6 = fuel_equ_ratio - equ  # H2

            df1dequ = 0
            df2dequ = -0.5
            df4dequ = 1.0
            df5dequ = 0
            df6dequ = -1

            x_N2 = f1 / Ntot  # molar fractions
            x_O2 = f2 / Ntot
            x_H2O = f4 / Ntot
            x_Ar = f5 / Ntot
            x_H2 = f6 / Ntot

            M = x_N2 * M_N2 + x_O2 * M_O2 + x_H2O * M_H2O + x_Ar * M_Ar + x_H2 * M_H2

            dellx_N2dellequ = (df1dequ * Ntot - f1 * dNtotdequ) / (Ntot ** 2.0)
            dellx_O2dellequ = (df2dequ * Ntot - f2 * dNtotdequ) / (Ntot ** 2.0)
            dellx_H2Odellequ = (df4dequ * Ntot - f4 * dNtotdequ) / (Ntot ** 2.0)
            dellx_Ardellequ = (df5dequ * Ntot - f5 * dNtotdequ) / (Ntot ** 2.0)
            dellx_H2dellequ = (df6dequ * Ntot - f6 * dNtotdequ) / (Ntot ** 2.0)

            dellMdellequ = (
                M_N2 * dellx_N2dellequ
                + M_O2 * dellx_O2dellequ
                + M_H2O * dellx_H2Odellequ
                + M_Ar * dellx_Ardellequ
                + M_H2 * dellx_H2dellequ
            )

            dellRdellequ = -(Runiv / M ** 2.0) * dellMdellequ

            dellmu_N2dellequ = (dellx_N2dellequ * M - x_N2 * dellMdellequ) * (
                M_N2 / M ** 2.0
            )
            dellmu_O2dellequ = (dellx_O2dellequ * M - x_O2 * dellMdellequ) * (
                M_O2 / M ** 2.0
            )
            dellmu_H2Odellequ = (dellx_H2Odellequ * M - x_H2O * dellMdellequ) * (
                M_H2O / M ** 2.0
            )
            dellmu_Ardellequ = (dellx_Ardellequ * M - x_Ar * dellMdellequ) * (
                M_Ar / M ** 2.0
            )
            dellmu_H2dellequ = (dellx_H2dellequ * M - x_H2 * dellMdellequ) * (
                M_H2 / M ** 2.0
            )

            dellhdellequ = (
                h_N2 * dellmu_N2dellequ
                + h_O2 * dellmu_O2dellequ
                + h_H2O * dellmu_H2Odellequ
                + h_Ar * dellmu_Ardellequ
                + h_H2 * dellmu_H2dellequ
            )

            delludellequ = dellhdellequ - dellRdellequ * t

        else:
            raise Exception("Fuel type must be specified.")

    else:

        # direct injection, no fuel in mixture before combustion
        if fuel_type == "jetA":

            Ntot = 5.75 * equ + 17.75 * (
                1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air
            )  # total number of moles in gas
            dNtotdequ = 5.75

            f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
            f2 = 17.75 * (1 - equ)  # O2
            f3 = 12 * equ  # CO2
            f4 = 11.5 * equ  # H20
            f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar

            df1dequ = 0
            df2dequ = -17.75
            df3dequ = 12
            df4dequ = 11.5
            df5dequ = 0

            x_N2 = f1 / Ntot  # molar fraction
            x_O2 = f2 / Ntot
            x_CO2 = f3 / Ntot
            x_H2O = f4 / Ntot
            x_Ar = f5 / Ntot

            M = x_N2 * M_N2 + x_O2 * M_O2 + x_CO2 * M_CO2 + x_H2O * M_H2O + x_Ar * M_Ar

            dellx_N2dellequ = (df1dequ * Ntot - f1 * dNtotdequ) / (Ntot ** 2.0)
            dellx_O2dellequ = (df2dequ * Ntot - f2 * dNtotdequ) / (Ntot ** 2.0)
            dellx_CO2dellequ = (df3dequ * Ntot - f3 * dNtotdequ) / (Ntot ** 2.0)
            dellx_H2Odellequ = (df4dequ * Ntot - f4 * dNtotdequ) / (Ntot ** 2.0)
            dellx_Ardellequ = (df5dequ * Ntot - f5 * dNtotdequ) / (Ntot ** 2.0)

            dellMdellequ = (
                M_N2 * dellx_N2dellequ
                + M_O2 * dellx_O2dellequ
                + M_CO2 * dellx_CO2dellequ
                + M_H2O * dellx_H2Odellequ
                + M_Ar * dellx_Ardellequ
            )

            dellRdellequ = -(Runiv / M ** 2.0) * dellMdellequ

            # Analytical shorter expressions
            # Partial derivative of h with resptect to equ
            # k2 = (12*M_CO2*hco2 + 11.5*M_H2O*hh2o - 17.75*M_O2*ho2)*(M_N2*(1 + x_N2in/x_O2in) + M_O2)
            # k3 = (M_N2*hn2*(1+x_N2in/x_O2in) + M_O2*ho2)*(12*M_CO2 + 11.5*M_H2O - 17.75*M_O2)
            # k4 = 17.75*(M_N2*(1 + x_N2in/x_O2in) + M_O2) + equ * (12*M_CO2 + 11.5*M_H2O - 17.75*M_O2)

            # dellhdellequ = 17.75*(k2 - k3)/(k4 ** 2.0)

            # Partial derivative of R with respect to equ
            # k1 = Runiv * 17.75 *(5.75 * (M_N2 * (1 + x_N2in/x_O2in) + M_O2) -
            #                     (1 + x_N2in/x_O2in)*(11.5 * M_H2O + 12* M_CO2 - 17.75*M_O2)  )

            # dellRdellequ =  k1/((17.75 * (M_N2*(1 + x_N2in/x_O2in) + M_O2) + equ*(11.5*M_H2O + 12*M_CO2 - 17.75*M_O2) ) ** 2.0)

            dellmu_N2dellequ = (dellx_N2dellequ * M - x_N2 * dellMdellequ) * (
                M_N2 / M ** 2.0
            )
            dellmu_O2dellequ = (dellx_O2dellequ * M - x_O2 * dellMdellequ) * (
                M_O2 / M ** 2.0
            )
            dellmu_CO2dellequ = (dellx_CO2dellequ * M - x_CO2 * dellMdellequ) * (
                M_CO2 / M ** 2.0
            )
            dellmu_H2Odellequ = (dellx_H2Odellequ * M - x_H2O * dellMdellequ) * (
                M_H2O / M ** 2.0
            )
            dellmu_Ardellequ = (dellx_Ardellequ * M - x_Ar * dellMdellequ) * (
                M_Ar / M ** 2.0
            )

            dellhdellequ = (
                h_N2 * dellmu_N2dellequ
                + h_O2 * dellmu_O2dellequ
                + h_CO2 * dellmu_CO2dellequ
                + h_H2O * dellmu_H2Odellequ
                + h_Ar * dellmu_Ardellequ
            )

            delludellequ = dellhdellequ - dellRdellequ * t

        elif fuel_type == "H2":
            Ntot = (1 - 0.5) * equ + 0.5 * (
                1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air
            )  # total number of moles in gas
            dNtotdequ = 0.5

            f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
            f2 = 0.5 * (1 - equ)  # O2
            f4 = equ  # H2O
            f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar

            df1dequ = 0
            df2dequ = -0.5
            df4dequ = 1.0
            df5dequ = 0

            x_N2 = f1 / Ntot  # molar fractions
            x_O2 = f2 / Ntot
            x_H2O = f4 / Ntot
            x_Ar = f5 / Ntot

            M = x_N2 * M_N2 + x_O2 * M_O2 + x_H2O * M_H2O + x_Ar * M_Ar

            dellx_N2dellequ = (df1dequ * Ntot - f1 * dNtotdequ) / (Ntot ** 2.0)
            dellx_O2dellequ = (df2dequ * Ntot - f2 * dNtotdequ) / (Ntot ** 2.0)
            dellx_H2Odellequ = (df4dequ * Ntot - f4 * dNtotdequ) / (Ntot ** 2.0)
            dellx_Ardellequ = (df5dequ * Ntot - f5 * dNtotdequ) / (Ntot ** 2.0)

            dellMdellequ = (
                M_N2 * dellx_N2dellequ
                + M_O2 * dellx_O2dellequ
                + M_H2O * dellx_H2Odellequ
                + M_Ar * dellx_Ardellequ
            )

            dellRdellequ = -(Runiv / M ** 2.0) * dellMdellequ

            dellmu_N2dellequ = (dellx_N2dellequ * M - x_N2 * dellMdellequ) * (
                M_N2 / M ** 2.0
            )
            dellmu_O2dellequ = (dellx_O2dellequ * M - x_O2 * dellMdellequ) * (
                M_O2 / M ** 2.0
            )
            dellmu_H2Odellequ = (dellx_H2Odellequ * M - x_H2O * dellMdellequ) * (
                M_H2O / M ** 2.0
            )
            dellmu_Ardellequ = (dellx_Ardellequ * M - x_Ar * dellMdellequ) * (
                M_Ar / M ** 2.0
            )

            dellhdellequ = (
                h_N2 * dellmu_N2dellequ
                + h_O2 * dellmu_O2dellequ
                + h_H2O * dellmu_H2Odellequ
                + h_Ar * dellmu_Ardellequ
            )

            delludellequ = dellhdellequ - dellRdellequ * t

        else:
            raise Exception("Fuel type must be specified.")

    return dellRdellequ, delludellequ


@njit()
def molar_fractions(equ, fuel_type, pure_fuel=False, fuel_equ_ratio=0.0):
    """
    Function that return thermodynamic properties of a mixture based on the
    properties of the individual species. This is for a combustion gas of
    JET_A/Diesel (C12H23).

    Parameters
    ----------
    equ : float
        Equivalence ratio of the gas That is fuel air ratio divided by
        stochiometric fuel air ratio, weight ratios that is.
    t : float
        Temperature of the gas.
    p : float
        Pressure of the gas mixture.
    fuel_type : string
        Type of fuel. H2 or jetA


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

    # The amount of moles of the gas before combustion. Could be pure air or air + fuel mixture
    if pure_fuel is True and fuel_type == "jetA":
        # O2 + N2 + Ar + C12H23
        N_air = 1 + 3.7274 + 0.0444 + fuel_equ_ratio * (1 / 17.75)
        x_JETA_air = fuel_equ_ratio * (1 / 17.75) / N_air  # molar fraction of C12H23
        x_H2_air = 0.0  # molar fraction of H2 (0 for JetA)

    elif pure_fuel is True and fuel_type == "H2":
        # O2 + N2 + Ar + H2
        N_air = 1 + 3.7274 + 0.0444 + fuel_equ_ratio * 2
        x_H2_air = fuel_equ_ratio * 2 / N_air  # molar fraction of H2
        x_JETA_air = 0.0  # molar fraction of JetA (0 for H2)

    else:
        # O2 + N2 + Ar
        N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air (per 1 mole of O2)

    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

    # THIS IS FORE THE CASE OF FUEL IN THE AIR
    if pure_fuel is True:
        if fuel_type == "jetA":

            # (CO2 + H2O - O2 - JETA) * equ + O2 + N2 + Ar + JETA
            N = (
                4.75 * equ
                + 17.75 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)
                + fuel_equ_ratio
            )  # total number of moles in gas

            f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
            f2 = 17.75 * (1 - equ)  # O2
            f3 = 12 * equ  # CO2
            f4 = 11.5 * equ  # H2O
            f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar
            f6 = fuel_equ_ratio - equ  # C12H23

            x_N2 = f1 / N  # molar fractions
            x_O2 = f2 / N
            x_CO2 = f3 / N
            x_H2O = f4 / N
            x_Ar = f5 / N
            x_JETA = f6 / N

        elif fuel_type == "H2":
            # (H2O - O2 - H2) * equ + O2 + N2 + Ar + H2
            N = (
                -0.5 * equ
                + 0.5 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)
                + fuel_equ_ratio
            )  # total number of moles in gas

            f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
            f2 = 0.5 * (1 - equ)  # O2
            f4 = equ  # H2O
            f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar
            f6 = fuel_equ_ratio - equ  # H2

            x_N2 = f1 / N  # molar fractions
            x_O2 = f2 / N
            x_H2O = f4 / N
            x_Ar = f5 / N
            x_H2 = f6 / N
            x_CO2 = 0.0

        else:
            raise Exception("Fuel type must be specified.")

    # THIS IS FOR NO FUEL IN THE AIR. ONLY PURE AIR AND COMBUSTION PRODUCTS
    else:
        if fuel_type == "jetA":
            N = 5.75 * equ + 17.75 * (
                1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air
            )  # total number of moles in gas

            f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
            f2 = 17.75 * (1 - equ)  # O2
            f3 = 12 * equ  # CO2
            f4 = 11.5 * equ  # H2O
            f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar

            x_N2 = f1 / N  # molar fractions
            x_O2 = f2 / N
            x_CO2 = f3 / N
            x_H2O = f4 / N
            x_Ar = f5 / N

        elif fuel_type == "H2":
            N = 0.5 * equ + 0.5 * (
                1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air
            )  # total number of moles in gas

            f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
            f2 = 0.5 * (1 - equ)  # O2
            f4 = equ  # H2O
            f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar

            x_N2 = f1 / N  # molar fractions
            x_O2 = f2 / N
            x_H2O = f4 / N
            x_Ar = f5 / N
            x_CO2 = 0.0

        else:
            raise Exception("Fuel type must be specified.")

    if pure_fuel and fuel_type == "jetA":
        x_fuel = x_JETA
    elif pure_fuel and fuel_type == "H2":
        x_fuel = x_H2
    else:
        x_fuel = 0.0

    return x_N2, x_O2, x_CO2, x_H2O, x_Ar, x_fuel


@njit()
def mass_fractions(equ, fuel_type, pure_fuel, fuel_equ_ratio):
    """
    Function that return thermodynamic properties of a mixture based on the
    properties of the individual species. This is for a combustion gas of
    JET_A/Diesel (C12H23).

    Parameters
    ----------
    equ : float
        Equivalence ratio of the gas That is fuel air ratio divided by
        stochiometric fuel air ratio, weight ratios that is.
    t : float
        Temperature of the gas.
    p : float
        Pressure of the gas mixture.
    fuel_type : string
        Type of fuel. H2 or jetA


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

    t_dummy = 999
    p_dummy = 1e5

    # The amount of moles of the gas before combustion. Could be pure air or air + fuel mixture
    if pure_fuel is True and fuel_type == "jetA":
        # O2 + N2 + Ar + C12H23
        N_air = 1 + 3.7274 + 0.0444 + fuel_equ_ratio * (1 / 17.75)
        x_JETA_air = fuel_equ_ratio * (1 / 17.75) / N_air  # molar fraction of C12H23
        x_H2_air = 0.0  # molar fraction of H2 (0 for JetA)
        _, _, _, _, M_JETA = JETA_G(t_dummy, p_dummy)
        _, _, _, _, M_H2 = H2(t_dummy, p_dummy)

    elif pure_fuel is True and fuel_type == "H2":
        # O2 + N2 + Ar + H2
        N_air = 1 + 3.7274 + 0.0444 + fuel_equ_ratio * 2
        x_H2_air = fuel_equ_ratio * 2 / N_air  # molar fraction of H2
        x_JETA_air = 0.0  # molar fraction of JetA (0 for H2)
        _, _, _, _, M_H2 = H2(t_dummy, p_dummy)
        _, _, _, _, M_JETA = JETA_G(t_dummy, p_dummy)

    else:
        # O2 + N2 + Ar
        N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air (per 1 mole of O2)

    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

    # retrieve the molar masses
    _, _, _, _, M_N2 = N2(t_dummy, p_dummy)
    _, _, _, _, M_O2 = O2(t_dummy, p_dummy)
    _, _, _, _, M_Ar = Ar(t_dummy, p_dummy)
    _, _, _, _, M_H2O = H2O(t_dummy, p_dummy)
    _, _, _, _, M_CO2 = CO2(t_dummy, p_dummy)

    if pure_fuel:
        # fuel in mixture
        M_air = (
            x_N2_air * M_N2
            + x_O2_air * M_O2
            + x_Ar_air * M_Ar
            + x_JETA_air * M_JETA
            + x_H2_air * M_H2
        )
    else:
        # no fuel in mixture
        M_air = x_N2_air * M_N2 + x_O2_air * M_O2 + x_Ar_air * M_Ar

    if equ == 0:

        if pure_fuel:
            mu_JETA = x_JETA_air * (M_JETA / M_air)
            mu_H2 = x_H2_air * (M_H2 / M_air)

        # if the fluid is pure air
        mu_N2 = x_N2_air * (M_N2 / M_air)  # mass fraction of N2 in the fluid
        mu_O2 = x_O2_air * (M_O2 / M_air)  # mass fraction of O2 in the fluid
        mu_Ar = x_Ar_air * (M_Ar / M_air)  # mass fraction of Ar in the fluid
        mu_CO2 = 0  # mass fraction of CO2 in the fluid
        mu_H2O = 0  # mass fraction of H2O in the fluid

    else:
        # THIS IS FORE THE CASE OF FUEL IN THE AIR
        if pure_fuel is True:
            if fuel_type == "jetA":

                # (CO2 + H2O - O2 - JETA) * equ + O2 + N2 + Ar + JETA
                N = (
                    4.75 * equ
                    + 17.75 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)
                    + fuel_equ_ratio
                )  # total number of moles in gas

                f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
                f2 = 17.75 * (1 - equ)  # O2
                f3 = 12 * equ  # CO2
                f4 = 11.5 * equ  # H2O
                f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar
                f6 = fuel_equ_ratio - equ  # C12H23

                x_N2 = f1 / N  # molar fractions
                x_O2 = f2 / N
                x_CO2 = f3 / N
                x_H2O = f4 / N
                x_Ar = f5 / N
                x_JETA = f6 / N

                M = (
                    x_N2 * M_N2
                    + x_O2 * M_O2
                    + x_CO2 * M_CO2
                    + x_H2O * M_H2O
                    + x_Ar * M_Ar
                    + x_JETA * M_JETA
                )  # molar mass of the fluid

                mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
                mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
                mu_CO2 = x_CO2 * (M_CO2 / M)  # mass fraction of CO2 in the fluid
                mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
                mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid
                mu_JETA = x_JETA * (M_JETA / M)  # mass fraction of JETA in the fluid

            elif fuel_type == "H2":
                # (H2O - O2 - H2) * equ + O2 + N2 + Ar + H2
                N = (
                    -0.5 * equ
                    + 0.5 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)
                    + fuel_equ_ratio
                )  # total number of moles in gas

                f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
                f2 = 0.5 * (1 - equ)  # O2
                f4 = equ  # H2O
                f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar
                f6 = fuel_equ_ratio - equ  # H2

                x_N2 = f1 / N  # molar fractions
                x_O2 = f2 / N
                x_H2O = f4 / N
                x_Ar = f5 / N
                x_H2 = f6 / N

                M = (
                    x_N2 * M_N2
                    + x_O2 * M_O2
                    + x_H2O * M_H2O
                    + x_Ar * M_Ar
                    + x_H2 * M_H2
                )  # molar mass of the fluid

                mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
                mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
                mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
                mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid
                mu_CO2 = 0.0  # no CO2 for H2
                mu_H2 = x_H2 * (M_H2 / M)  # mass fraction of H2 in the fluid

            else:
                raise Exception("Fuel type must be specified.")

        # THIS IS FOR NO FUEL IN THE AIR. ONLY PURE AIR AND COMBUSTION PRODUCTS
        else:
            if fuel_type == "jetA":
                N = 5.75 * equ + 17.75 * (
                    1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air
                )  # total number of moles in gas

                f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
                f2 = 17.75 * (1 - equ)  # O2
                f3 = 12 * equ  # CO2
                f4 = 11.5 * equ  # H2O
                f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar

                x_N2 = f1 / N  # molar fractions
                x_O2 = f2 / N
                x_CO2 = f3 / N
                x_H2O = f4 / N
                x_Ar = f5 / N

                M = (
                    x_N2 * M_N2
                    + x_O2 * M_O2
                    + x_CO2 * M_CO2
                    + x_H2O * M_H2O
                    + x_Ar * M_Ar
                )  # molar mass of the fluid

                mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
                mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
                mu_CO2 = x_CO2 * (M_CO2 / M)  # mass fraction of CO2 in the fluid
                mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
                mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid

            elif fuel_type == "H2":
                N = 0.5 * equ + 0.5 * (
                    1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air
                )  # total number of moles in gas

                f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
                f2 = 0.5 * (1 - equ)  # O2
                f4 = equ  # H2O
                f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar

                x_N2 = f1 / N  # molar fractions
                x_O2 = f2 / N
                x_H2O = f4 / N
                x_Ar = f5 / N

                M = (
                    x_N2 * M_N2 + x_O2 * M_O2 + x_H2O * M_H2O + x_Ar * M_Ar
                )  # molar mass of the fluid

                mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
                mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
                mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
                mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid
                mu_CO2 = 0.0  # no CO2 for H2

            else:
                raise Exception("Fuel type must be specified.")

    if pure_fuel and fuel_type == "jetA":
        mu_fuel = mu_JETA
    elif pure_fuel and fuel_type == "H2":
        mu_fuel = mu_H2
    else:
        mu_fuel = 0.0

    return mu_N2, mu_O2, mu_CO2, mu_H2O, mu_Ar, mu_fuel
