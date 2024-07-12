from .polynomials import N2, O2, CO2, H2O, Ar
from scipy.optimize import fsolve


def mixture(equ, t, p, fuel_type):
    """
    Function that return thermodynamic properties of a mixture based on the 
    properties of the individual species. This is for a combustion gas of
    JET_A/Diesel (C12H23).
    
    Parameters
    ----------
    equ : float
        Equivalence ratio of the gas That is fuel air ratio divided by
        stochiometric fuel air ratio.
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

    Runiv = 8.3144626  # J mol^-1 K^-1

    N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar
    x_CO2_air = 0  # no co2 in air for now

    # retrieve the molar masses
    cp_N2, h_N2, s_N2, M_N2 = N2(t, p)
    cp_O2, h_O2, s_O2, M_O2 = O2(t, p)
    cp_Ar, h_Ar, s_Ar, M_Ar = Ar(t, p)
    cp_H2O, h_H2O, s_H2O, M_H2O = H2O(t, p)
    cp_CO2, h_CO2, s_CO2, M_CO2 = CO2(t, p)

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
            f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar

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

    # partial pressures
    p_N2 = mu_N2 * p
    p_O2 = mu_O2 * p
    p_Ar = mu_Ar * p
    p_H2O = mu_H2O * p
    p_CO2 = mu_CO2 * p

    # now use the partial pressures to get the correct entropy values
    cp_N2, h_N2, s_N2, M_N2 = N2(t, p_N2)
    cp_O2, h_O2, s_O2, M_O2 = O2(t, p_O2)
    cp_Ar, h_Ar, s_Ar, M_Ar = Ar(t, p_Ar)
    if mu_H2O > 0:
        cp_H2O, h_H2O, s_H2O, M_H2O = H2O(t, p_H2O)
    if mu_CO2 > 0:
        cp_CO2, h_CO2, s_CO2, M_CO2 = CO2(t, p_CO2)

    cp = mu_N2 * cp_N2 + mu_O2 * cp_O2 + mu_CO2 * cp_CO2 + mu_H2O * cp_H2O + mu_Ar * cp_Ar  # heat capacity at constant
    # pressure
    h = mu_N2 * h_N2 + mu_O2 * h_O2 + mu_CO2 * h_CO2 + mu_H2O * h_H2O + mu_Ar * h_Ar  # specific enthalpy
    s = mu_N2 * s_N2 + mu_O2 * s_O2 + mu_CO2 * s_CO2 + mu_H2O * s_H2O + mu_Ar * s_Ar  # specific entropy

    R = Runiv / M  # specific gas constant
    u = h - R * t  # inner energy
    cv = cp - R  # specific heat capacity at constant volume
    gamma = cp/cv  # isentropic exponent

    return h, u, cp, cv, R, gamma, s


def equivalence_derivative(equ, t, p, fuel_type):
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

    Runiv = 8.3144626  # J mol^-1 K^-1

    cp_N2, h_N2, s_N2, M_N2 = N2(t, p)
    cp_O2, h_O2, s_O2, M_O2 = O2(t, p)
    cp_Ar, h_Ar, s_Ar, M_Ar = Ar(t, p)
    cp_CO2, h_CO2, s_CO2, M_CO2 = CO2(t, p)
    cp_H2O, h_H2O, s_H2O, M_H2O = H2O(t, p)

    N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar
    x_CO2_air = 0  # no co2 in air for now
    
    if fuel_type == 'jetA':

        Ntot = 5.75 * equ + 17.75 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas
        dNtotdequ = 5.75

        f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
        f2 = 17.75 * (1-equ)  # O2
        f3 = 12 * equ  # CO2
        f4 = 11.5 * equ  # H20
        f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar

        df1dequ = 0
        df2dequ = -17.75
        df3dequ = 12
        df4dequ = 11.5
        df5dequ = 0

        x_N2 = f1/Ntot  # molar fraction
        x_O2 = f2/Ntot
        x_CO2 = f3/Ntot
        x_H2O = f4/Ntot
        x_Ar = f5/Ntot

        M = x_N2*M_N2 + x_O2*M_O2 + x_CO2*M_CO2 + x_H2O*M_H2O + x_Ar * M_Ar

        dellx_N2dellequ = (df1dequ * Ntot - f1 * dNtotdequ) / (Ntot**2)
        dellx_O2dellequ = (df2dequ * Ntot - f2 * dNtotdequ) / (Ntot**2)
        dellx_CO2dellequ = (df3dequ * Ntot - f3 * dNtotdequ) / (Ntot**2)
        dellx_H2Odellequ = (df4dequ * Ntot - f4 * dNtotdequ) / (Ntot**2)
        dellx_Ardellequ = (df5dequ * Ntot - f5 * dNtotdequ) / (Ntot**2)

        dellMdellequ = M_N2 * dellx_N2dellequ + M_O2 * dellx_O2dellequ + \
            M_CO2 * dellx_CO2dellequ + M_H2O * dellx_H2Odellequ + \
            M_Ar * dellx_Ardellequ

        dellRdellequ = - (Runiv / M**2) * dellMdellequ

        # Analytical shorter expressions
        # Partial derivative of h with resptect to equ
        #k2 = (12*M_CO2*hco2 + 11.5*M_H2O*hh2o - 17.75*M_O2*ho2)*(M_N2*(1 + x_N2in/x_O2in) + M_O2)
        #k3 = (M_N2*hn2*(1+x_N2in/x_O2in) + M_O2*ho2)*(12*M_CO2 + 11.5*M_H2O - 17.75*M_O2)
        #k4 = 17.75*(M_N2*(1 + x_N2in/x_O2in) + M_O2) + equ * (12*M_CO2 + 11.5*M_H2O - 17.75*M_O2)

        #dellhdellequ = 17.75*(k2 - k3)/(k4**2)

        # Partial derivative of R with respect to equ
        #k1 = Runiv * 17.75 *(5.75 * (M_N2 * (1 + x_N2in/x_O2in) + M_O2) -
        #                     (1 + x_N2in/x_O2in)*(11.5 * M_H2O + 12* M_CO2 - 17.75*M_O2)  )

        #dellRdellequ =  k1/((17.75 * (M_N2*(1 + x_N2in/x_O2in) + M_O2) + equ*(11.5*M_H2O + 12*M_CO2 - 17.75*M_O2) )**2)

        dellmu_N2dellequ = (dellx_N2dellequ * M - x_N2 * dellMdellequ)*(M_N2/M**2)
        dellmu_O2dellequ = (dellx_O2dellequ * M - x_O2 * dellMdellequ)*(M_O2/M**2)
        dellmu_CO2dellequ = (dellx_CO2dellequ * M - x_CO2 * dellMdellequ)*(M_CO2/M**2)
        dellmu_H2Odellequ = (dellx_H2Odellequ * M - x_H2O * dellMdellequ)*(M_H2O/M**2)
        dellmu_Ardellequ = (dellx_Ardellequ * M - x_Ar * dellMdellequ)*(M_Ar/M**2)

        dellhdellequ = h_N2*dellmu_N2dellequ + h_O2*dellmu_O2dellequ + \
            h_CO2*dellmu_CO2dellequ + h_H2O*dellmu_H2Odellequ + \
            h_Ar * dellmu_Ardellequ

        delludellequ = dellhdellequ - dellRdellequ * t

    elif fuel_type == 'H2':
        Ntot = (1 - 0.5) * equ + 0.5 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas
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

        M = x_N2*M_N2 + x_O2*M_O2 + x_H2O*M_H2O + x_Ar * M_Ar

        dellx_N2dellequ = (df1dequ * Ntot - f1 * dNtotdequ) / (Ntot ** 2)
        dellx_O2dellequ = (df2dequ * Ntot - f2 * dNtotdequ) / (Ntot ** 2)
        dellx_H2Odellequ = (df4dequ * Ntot - f4 * dNtotdequ) / (Ntot ** 2)
        dellx_Ardellequ = (df5dequ * Ntot - f5 * dNtotdequ) / (Ntot ** 2)

        dellMdellequ = M_N2 * dellx_N2dellequ + M_O2 * dellx_O2dellequ + \
            M_H2O * dellx_H2Odellequ + M_Ar * dellx_Ardellequ

        dellRdellequ = - (Runiv / M ** 2) * dellMdellequ

        dellmu_N2dellequ = (dellx_N2dellequ * M - x_N2 * dellMdellequ)*(M_N2/M**2)
        dellmu_O2dellequ = (dellx_O2dellequ * M - x_O2 * dellMdellequ)*(M_O2/M**2)
        dellmu_H2Odellequ = (dellx_H2Odellequ * M - x_H2O * dellMdellequ)*(M_H2O/M**2)
        dellmu_Ardellequ = (dellx_Ardellequ * M - x_Ar * dellMdellequ)*(M_Ar/M**2)

        dellhdellequ = h_N2*dellmu_N2dellequ + h_O2*dellmu_O2dellequ + \
            h_H2O*dellmu_H2Odellequ + h_Ar * dellmu_Ardellequ

        delludellequ = dellhdellequ - dellRdellequ * t

    else:
        raise Exception('Fuel type must be specified.')

    return dellRdellequ, delludellequ

def isen(T0, p0, p_final, gamma):
    
    """
    def find_isen(T,p0,equ0,p_final,x_N2in, x_O2in):
        cp_N2, h_N2, sn2 = polynomials.N2(T)
        cp_O2, h_O2, so2 = polynomials.O2(T)
        cp_CO2, h_CO2, sCO2 = polynomials.CO2(T)
        cp_H2O, h_H2O, sH2O = polynomials.H2O(T)
    
        h_array = [h_N2, h_O2, h_CO2, h_H2O]
        cp_array = [cp_N2, cp_O2, cp_CO2, cp_H2O]
    
        h, u, cp, cv, R, gamma = mixture(equ0, T0, x_N2in, x_O2in, h_array, cp_array)
        
        
        #basically
        
        # s(T_final,p_final) - s(T0,p0) = 0
        # return s(T,p_final) - s(T0,p0)
    """
    T_final = T0*(p_final/p0)**((gamma-1)/gamma)


    #T_final = fsolve(find_isen,T0)
    
    
    
    return T_final



