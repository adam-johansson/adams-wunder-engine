from thermo import polynomials, molar_fractions
from scipy.optimize import fsolve, least_squares
import math
import numpy as np

from thermo.newton_raphson import newton_method
from numba import jit


# THIS IS FOR GRI_MECH, Unimolecular and recombination reactions
# k = A T^m exp(-E/RT)
# conentrations in mol/cm^3
# A = [1/s, cm^3/mol/s,cm^6/mol^2/s] first, second and third order reactions respectively
# T is in Kelvin
# E is [cal/mol]


# KOEFFICIENTS                              A            n            E_A
# 2H+M<=>H2+M                              1.000E+18   -1.000        .00

# Equilibrium constant. Ratio between forward and reverse reactions
# K_c = k_f / k_r

# Kc is koncentration equilibrium constant. Moles per liter
# Kp is partial pressure equilibrium constant

### ÖPPEN FRÅGA: KAN/BÖR DENNA FUNKTION GÅ ATT KÖRA MED VEKTORER?


def equilibrium_OHC(T, equ, p, x0):
    """
    Function that calculates the chemical equilibrium of oxygen, hydrogen and carbon (OHC). This is the zone 2,
    behind the flame front, of hydrocarbon combustion.

    These reactions proceed much faster than the reactions of the Zeldovich mechanism, (at the temperature of the
    cylinder), so we assume that we always have partial equilibrium.

    We have 5 chemical reaction equations:

    (1) H2 = 2 H
    (2) O2 = 2 O
    (3) H2O = (1/2) H2 + OH
    (4) H2O = (1/2) O2 + H2
    (5) CO2 = CO + (1/2) O2

    Further we have mass conservation of the atoms O, H and C.

    Combined with partial pressures must add up to the total pressure of the system.



    :param t:
    :param equ:
    :param p:
    :return:
    """

    # Universal gas constant from NASA polynomials pdf
    R = 8.314510  # J mol^-1 K^-1

    # standard state pressure
    p_std = 1e5

    # thermodynamic properties, mass bases
    # Gibbs free energy is for standard state, meaning no pressure dependence
    _, _, _, g_H2, M_H2 = polynomials.H2(T, p_std)
    _, _, _, g_H, M_H = polynomials.H(T, p_std)
    _, _, _, g_O2, M_O2 = polynomials.O2(T, p_std)
    _, _, _, g_O, M_O = polynomials.O(T, p_std)
    _, _, _, g_H2O, M_H2O = polynomials.H2O(T, p_std)
    _, _, _, g_OH, M_OH = polynomials.OH(T, p_std)
    _, _, _, g_CO2, M_CO2 = polynomials.CO2(T, p_std)
    _, _, _, g_CO, M_CO = polynomials.CO(T, p_std)

    # convert to molar based free energy (J / mol)
    g_H2 = g_H2 * M_H2
    g_H = g_H * M_H
    g_O2 = g_O2 * M_O2
    g_O = g_O * M_O
    g_H2O = g_H2O * M_H2O
    g_OH = g_OH * M_OH
    g_CO2 = g_CO2 * M_CO2
    g_CO = g_CO * M_CO

    # The stoichiometric coefficients for the five reactions
    #(1) H2 = 2 H
    #(2) O2 = 2 O
    #(3) H2O = (1/2) H2 + OH
    #(4) H2O = (1/2) O2 + H2
    #(5) CO2 = CO + (1/2) O2

    # Stoichiometric coefficients
    nu_1_H2, nu_1_H = -1, 2
    nu_2_O2, nu_2_O = -1, 2
    nu_3_H2O, nu_3_H2, nu_3_OH = -1, 0.5, 1
    nu_4_H2O, nu_4_O2, nu_4_H2 = -1, 0.5, 1
    nu_5_CO2, nu_5_CO, nu_5_O2 = -1, 1, 0.5

    # sum of stoichiometric coefficients
    sum_nu_1 = nu_1_H2 + nu_1_H
    sum_nu_2 = nu_2_O2 + nu_2_O
    sum_nu_3 = nu_3_H2O + nu_3_H2 + nu_3_OH
    sum_nu_4 = nu_4_H2O + nu_4_O2 + nu_4_H2
    sum_nu_5 = nu_5_CO2 + nu_5_CO + nu_5_O2

    # calculate the free molar reactions enthalpy of the reactions (energy after - energy before)
    Delta_g_R_1 = nu_1_H2 * g_H2 + nu_1_H * g_H
    Delta_g_R_2 = nu_2_O2 * g_O2 + nu_2_O * g_O
    Delta_g_R_3 = nu_3_H2O * g_H2O + nu_3_H2 * g_H2 + nu_3_OH * g_OH
    Delta_g_R_4 = nu_4_H2O * g_H2O + nu_4_O2 * g_O2 + nu_4_H2 * g_H2
    Delta_g_R_5 = nu_5_CO2 * g_CO2 + nu_5_CO * g_CO + nu_5_O2 * g_O2

    # partial pressure equilibrium constants for all the reactions
    Kp_1 = math.exp(-Delta_g_R_1 / (R * T))
    Kp_2 = math.exp(-Delta_g_R_2 / (R * T))
    Kp_3 = math.exp(-Delta_g_R_3 / (R * T))
    Kp_4 = math.exp(-Delta_g_R_4 / (R * T))
    Kp_5 = math.exp(-Delta_g_R_5 / (R * T))

    # Assume stoichiometric combustion for now
    t_dummy = 298
    p_dummy = 1e5

    fuel_type = "jetA"
    # retrieve the initial molar fractions before chemical equilibrium
    R, M, x_N2, x_O2, x_CO2, x_H2O, x_Ar = molar_fractions(t_dummy, p_dummy, equ=equ, fuel_type=fuel_type)

    #H_O_ratio = (2 * x_H2O) / (2 * x_O2 + 2 * x_CO2 + x_H2O)
    O_C_ratio = (2 * x_O2 + 2 * x_CO2 + x_H2O) / x_CO2
    H_C_ratio = (2 * x_H2O) / (x_CO2)
    N_C_ratio = (2 * x_N2) / (x_CO2)
    Ar_C_ratio = (x_Ar) / (x_CO2)

    #@jit(nopython=True, cache=True)
    def OHC_system(partial_pressures):
        # fsolve
        #p_H2, p_H, p_O2, p_O, p_H2O, p_OH, p_CO2, p_CO, p_N2, p_Ar = partial_pressures
        # newton

        p_H2 = partial_pressures[0][0]
        p_H = partial_pressures[1][0]
        p_O2 = partial_pressures[2][0]
        p_O = partial_pressures[3][0]
        p_H2O = partial_pressures[4][0]
        p_OH = partial_pressures[5][0]
        p_CO2 = partial_pressures[6][0]
        p_CO = partial_pressures[7][0]
        p_N2 = partial_pressures[8][0]
        p_Ar = partial_pressures[9][0]

        # The five chemical equilibrium equations
        # Equation 1: H2 = 2 H
        eq1 = (p_H ** nu_1_H) * (p_H2 ** nu_1_H2) * (p_std ** (-sum_nu_1)) - Kp_1

        # Equation 2: O2 = 2 O
        eq2 = (p_O ** nu_2_O) * (p_O2 ** nu_2_O2) * (p_std ** (-sum_nu_2)) - Kp_2

        # Equation 3: H2O =  1/2 H2 + OH
        eq3 = (p_H2 ** nu_3_H2) * (p_OH ** nu_3_OH) * (p_H2O ** nu_3_H2O) * (p_std ** (-sum_nu_3)) - Kp_3

        # Equation 4: H20 = 1/2 O2 + H2
        eq4 = (p_O2 ** nu_4_O2) * (p_H2 ** nu_4_H2) * (p_H2O ** nu_4_H2O) * (p_std ** (-sum_nu_4)) - Kp_4

        # Equation 5: CO2 = CO + 1/2 O2
        eq5 = (p_CO ** nu_5_CO) * (p_O2 ** nu_5_O2) * (p_CO2 ** nu_5_CO2) * (p_std ** (-sum_nu_5)) - Kp_5

        # Equation 6: Sum of partial pressures equals pressure of gas mixture
        eq6 = p_H2 + p_H + p_O2 + p_O + p_H2O + p_OH + p_CO2 + p_CO + p_N2 + p_Ar - p

        # Equation 7: Fraction of H and O atoms stays constant (add lambda here later)
        #eq7 = ((2*p_H2O + 2*p_H2 + p_H + p_OH) / (p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH)) - H_O_ratio

        # Equation 7: Fraction of O and C atoms stays constant
        eq7 = (p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH) / (p_CO2 + p_CO) - O_C_ratio

        # Equation 8: Fraction of H and C atoms stays constant
        eq8 = (2*p_H2O + 2*p_H2 + p_H + p_OH) / (p_CO2 + p_CO) - H_C_ratio

        # Equation 9: Fraction of N and C atoms stays constant
        eq9 = ((2 * p_N2) / (p_CO2 + p_CO)) - N_C_ratio

        # Equation 10: Fraction of Ar and C atoms stays constant
        eq10 = (p_Ar / (p_CO2 + p_CO)) - Ar_C_ratio

        # for newton

        residual = np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10], dtype=np.float64)

        residual = np.atleast_2d(residual).T

        # adding scaling factors (best with errors in same range)
        #residual = np.array([eq1 * scaling[1], eq2 * scaling[1], eq3 * scaling[2], eq4 * scaling[3], eq5 * scaling[4],
        #                      eq6 * scaling[5], eq7 * scaling[6], eq8 * scaling[7], eq9 * scaling[8], eq10 * scaling[9]], dtype=np.float64)

        #residual = np.atleast_2d(residual).T

        # for fsolve
        #error = np.array([eq1, eq2, eq3, eq4 * 1e-3, eq5, eq6, eq7, eq8, eq9, eq10])

        return residual

    #@jit(nopython=True, cache=True)
    def OHC_system_jacobian(partial_pressures):
        #p_H2, p_H, p_O2, p_O, p_H2O, p_OH, p_CO2, p_CO, p_N2, p_Ar = partial_pressures

        p_H2 = partial_pressures[0][0]
        p_H = partial_pressures[1][0]
        p_O2 = partial_pressures[2][0]
        p_O = partial_pressures[3][0]
        p_H2O = partial_pressures[4][0]
        p_OH = partial_pressures[5][0]
        p_CO2 = partial_pressures[6][0]
        p_CO = partial_pressures[7][0]
        p_N2 = partial_pressures[8][0]
        p_Ar = partial_pressures[9][0]


        # The five chemical equilibrium equations
        # Equation 1: H2 = 2 H
        df1dph2 = nu_1_H2 * (p_H2 ** (nu_1_H2 - 1) ) * (p_H ** nu_1_H) * (p_std ** (-sum_nu_1))
        df1dph = nu_1_H * (p_H ** (nu_1_H - 1) ) * (p_H2 ** nu_1_H2) * (p_std ** (-sum_nu_1))
        J1 = np.array([df1dph2, df1dph, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Equation 2: O2 = 2 O
        df2dpo2 = (p_O ** nu_2_O) * nu_2_O2 * (p_O2 ** (nu_2_O2 - 1)) * (p_std ** (-sum_nu_2))
        df2dpo = nu_2_O * (p_O ** (nu_2_O - 1)) * (p_O2 ** nu_2_O2) * (p_std ** (-sum_nu_2))
        J2 = np.array([0.0, 0.0, df2dpo2, df2dpo, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Equation 3: H2O =  1/2 H2 + OH
        df3dph2o = (p_H2 ** nu_3_H2) * (p_OH ** nu_3_OH) * nu_3_H2O * (p_H2O ** (nu_3_H2O - 1)) * (p_std ** (-sum_nu_3))
        df3dpoh = (p_H2 ** nu_3_H2) * (p_H2O ** nu_3_H2O) * (p_std ** (-sum_nu_3))
        df3dph2 = nu_3_H2 * (p_H2 ** (nu_3_H2 - 1) ) * (p_OH ** nu_3_OH) * (p_H2O ** nu_3_H2O) * (p_std ** (-sum_nu_3))

        J3 = np.array([df3dph2, 0.0, 0.0, 0.0, df3dph2o, df3dpoh, 0.0, 0.0, 0.0, 0.0])

        # Equation 4: H20 = 1/2 O2 + H2
        df4dph2o = (p_O2 ** nu_4_O2) * (p_H2 ** nu_4_H2) * nu_4_H2O * (p_H2O ** (nu_4_H2O - 1)) * (p_std ** (-sum_nu_4))
        df4dph2 = (p_O2 ** nu_4_O2) * (p_H2O ** nu_4_H2O) * (p_std ** (-sum_nu_4))
        df4dpo2 = nu_4_O2 * (p_O2 ** (nu_4_O2 - 1)) * (p_H2 ** nu_4_H2) * (p_H2O ** nu_4_H2O) * (p_std ** (-sum_nu_4))

        J4 = np.array([df4dph2, 0.0, df4dpo2, 0.0, df4dph2o, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Equation 5: CO2 = CO + 1/2 O2
        df5dpco2 = (p_CO ** nu_5_CO) * (p_O2 ** nu_5_O2) * nu_5_CO2 * (p_CO2 ** (nu_5_CO2 - 1)) * (p_std ** (-sum_nu_5))
        df5dpo2 = (p_CO ** nu_5_CO) * nu_5_O2 * (p_O2 ** (nu_5_O2 - 1)) * (p_CO2 ** nu_5_CO2) * (p_std ** (-sum_nu_5))
        df5dpco = (p_O2 ** nu_5_O2) * (p_CO2 ** nu_5_CO2) * (p_std ** (-sum_nu_5))
        J5 = np.array([0.0, 0.0, df5dpo2, 0.0, 0.0, 0.0, df5dpco2, df5dpco, 0.0, 0.0])

        # Equation 6: Sum of partial pressures equals pressure of gas mixture
        J6 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Equation 7: Fraction of H and O atoms stays constant (add lambda here later)
        #df7dph2o = (( 2 * (2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH) - (2*p_H2 + p_H + p_OH))
        #            / ((p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH)**2))
        #df7dph2 = 2 / (p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH)
        #df7dph = 1 / (p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH)
        #df7dpoh = (( (p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O) - (2*p_H2O + 2*p_H2 + p_H))
        #           / ((p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH)**2))
        #df7dpco2 = - 2 * (2*p_H2O + 2*p_H2 + p_H + p_OH) / ( (p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH) ** 2 )
        #df7dpco = - (2*p_H2O + 2*p_H2 + p_H + p_OH) / ( (p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH) ** 2 )
        #df7dpo = - (2*p_H2O + 2*p_H2 + p_H + p_OH) / ( (p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH) ** 2 )
        #df7dpo2 = - 2 * (2*p_H2O + 2*p_H2 + p_H + p_OH) / ( (p_H2O + 2*p_CO2 + p_CO + 2*p_O2 + p_O + p_OH) ** 2 )
        #J7 = np.array([df7dph2, df7dph, df7dpo2, df7dpo, df7dph2o, df7dpoh, df7dpco2, df7dpco, 0.0, 0.0])

        # Equation 7: Fraction of O and C atoms stays constant
        df7dpo2 = 2 / (p_CO2 + p_CO)
        df7dpo = 1 / (p_CO2 + p_CO)
        df7dph2o = 1 / (p_CO2 + p_CO)
        df7dpoh = 1 / (p_CO2 + p_CO)
        df7dpco2 = (p_CO - (p_H2O + 2*p_O2 + p_O + p_OH) ) / ((p_CO2 + p_CO)**2)
        df7dpco = (- (p_H2O + p_CO2 + 2*p_O2 + p_O + p_OH)) / ((p_CO2 + p_CO) ** 2)

        J7 = np.array([0.0, 0.0, df7dpo2, df7dpo, df7dph2o, df7dpoh, df7dpco2, df7dpco, 0.0, 0.0])

        # Equation 8: Fraction of H and C atoms stays constant
        df8dph2 = 2 / (p_CO2 + p_CO)
        df8dph = 1 / (p_CO2 + p_CO)
        df8dph2o = 2 / (p_CO2 + p_CO)
        df8dpoh = 1 / (p_CO2 + p_CO)

        df8dpco2 = - (2 * p_H2O + 2 * p_H2 + p_H + p_OH) / ((p_CO2 + p_CO) ** 2)
        df8dpco = - (2 * p_H2O + 2 * p_H2 + p_H + p_OH) / ((p_CO2 + p_CO) ** 2)


        #df8dpco2 = 1 / (2*p_H2O + 2*p_H2 + p_H + p_OH)
        #df8dpco = 1 / (2*p_H2O + 2*p_H2 + p_H + p_OH)
        #df8dph2o = - 2 * (p_CO2 + p_CO) / ((2*p_H2O + 2*p_H2 + p_H + p_OH) ** 2)
        #df8dph2 = - 2 * (p_CO2 + p_CO) / ((2*p_H2O + 2*p_H2 + p_H + p_OH) ** 2)
        #df8dph = - (p_CO2 + p_CO) / ((2*p_H2O + 2*p_H2 + p_H + p_OH) ** 2)
        #df8dpoh = - (p_CO2 + p_CO) / ((2*p_H2O + 2*p_H2 + p_H + p_OH) ** 2)

        J8 = np.array([df8dph2, df8dph, 0.0, 0.0, df8dph2o, df8dpoh, df8dpco2, df8dpco, 0.0, 0.0])



        # Equation 9: Fraction of N and C atoms stays constant
        df9dpn2 = 2 / (p_CO2 + p_CO)
        df9dpco2 = - (2 * p_N2) / ((p_CO2 + p_CO)**2)
        df9dpco = - (2 * p_N2) / ((p_CO2 + p_CO)**2)

        J9 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, df9dpco2, df9dpco, df9dpn2, 0.0])


        # Equation 10: Fraction of Ar and C atoms stays constant
        df10dpar = 1 / (p_CO2 + p_CO)
        df10dpco2 = - (p_Ar) / ((p_CO2 + p_CO)**2)
        df10dpco = - (p_Ar) / ((p_CO2 + p_CO)**2)

        J10 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, df10dpco2, df10dpco, 0.0, df10dpar])

        #J = np.vstack((J1 * scaling[0], J2 * scaling[1], J3 * scaling[2], J4 * scaling[3], J5 * scaling[4],
        #               J6 * scaling[5], J7 * scaling[6], J8 * scaling[7], J9 * scaling[8], J10 * scaling[9]))

        J = np.vstack((J1, J2, J3, J4, J5, J6, J7, J8, J9, J10))

        return J

    # initial guess for partial pressures
    # p_H2, p_H, p_O2, p_O, p_H2O, p_OH, p_CO2, p_CO, p_N2, p_Ar
    #print(0.0049914 + 0.0011238 + 0.0155 + 0.001178 + 0.114522 + 0.0075 + 0.1 + 0.03164 + x_N2 + x_Ar)
    #x0 = np.array([8.41136963e-02, 5.17338953e-02, 1.05300227e-01, 3.69680352e-05, 4.77035178e-04,
    #       6.19899484e-06, 1.46035345e-02, 1.00659887e-01, 6.35498642e-01, 7.56992534e-03]) * p

    #x0 = np.array([0.0049914, 0.0011238, 0.0107067, 0.001178, 0.114522, 0.0085485, 0.098127, 0.03164, x_N2, x_Ar]) * p

    # scaling factors
    #scale = [1,1,1,1,1,1,1,1,1,1]
    #partial_pressures = fsolve(OHC_system, x0, factor=0.1, fprime=OHC_system_jacobian, diag=scale)
    #partial_pressures = fsolve(OHC_system, x0, factor=0.01)
    partial_pressures = newton_method(OHC_system, x0, OHC_system_jacobian)

    #errors = OHC_system(x0)

    # Solve the system with bounds using least_squares
    #result = least_squares(OHC_system, x0, bounds=(0, np.inf), jac=OHC_system_jacobian)

    # Extract the solution
    #partial_pressures = result.x

    mol_fractions = partial_pressures / p

    return mol_fractions
    #return errors
