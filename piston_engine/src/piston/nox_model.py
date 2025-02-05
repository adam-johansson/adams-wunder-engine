import math

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from thermo import mixture, molar_fractions, equilibrium_OHC, polynomials, euler


def nox_calculations(temperatures, masses, pressures, volumes, fuel_type, lambda_z1, phi, rpm, m_tot, equ_tot):

    """
    This model calculates the nox produced in the reaction zone of the two-zone model. That is the burned zone.

    It uses the extended Zeldovich mechanics. Ignoring NOx creation in flame front. No prompt NOx.

    How it is done comes from the text book by Merker 2005, Simulating Combustion.

    m_tot is the total mass that flows out of the engine during one cycle, and equ_tot is the equivalence ratio of
    the outflow

    :return:
    """

    # Universal gas constant from NASA polynomials pdf
    R = 8.314510  # J mol^-1 K^-1

    # standard state pressure
    p_std = 1e5

    equ = 1 / lambda_z1  # Equivalence ratio in the burned zone

    # initial guess for the OHC-system molar fractions
    x0 = np.array(
        [0.0049914, 0.0011238, 0.0107067, 0.001178, 0.114522, 0.0085485, 0.098127, 0.03164, 0.7, 0.01]) * pressures[0]


    # convert crank angles to time
    rps = rpm / 60
    radians_per_s = rps * 2 * np.pi

    #times = phi[1:] / radians_per_s
    times = phi / radians_per_s

    def dNOdt_fun(t, NO, x0):

        c_NO = NO

        i = np.nonzero(times==t)

        T = temperatures[i][0]
        p = pressures[i][0]

        p_i = equilibrium_OHC(T, equ, p, fuel_type, x0)
        # results are the partial_pressures

        # mol fractions
        mol = p_i / p

        # initial guess for next equilibrium
        x0 = mol * p

        # convert partial pressure to molar concentrations (mol / m^3)
        c_i = p_i.T / (R * T)

        # extract concentrations needed for Zeldovich mechanism
        c_H = c_i[1,0]
        c_O2 = c_i[2,0]
        c_O = c_i[3,0]
        c_OH = c_i[5,0]
        c_N2 = c_i[8,0]


        # thermodynamic properties, mass bases
        # Gibbs free energy is for standard state, meaning no pressure dependence
        _, _, _, g_N2, M_N2 = polynomials.N2(T, p_std)
        _, _, _, g_O, M_O = polynomials.O(T, p_std)
        _, _, _, g_NO, M_NO = polynomials.NO(T, p_std)
        _, _, _, g_N, M_N = polynomials.N(T, p_std)
        _, _, _, g_O2, M_O2 = polynomials.O2(T, p_std)
        _, _, _, g_OH, M_OH = polynomials.OH(T, p_std)
        _, _, _, g_H, M_H = polynomials.H(T, p_std)

        # convert to molar based free energy (J / mol)
        g_N2 = g_N2 * M_N2
        g_O = g_O * M_O
        g_NO = g_NO * M_NO
        g_N = g_N * M_N
        g_O2 = g_O2 * M_O2
        g_OH = g_OH * M_OH
        g_H = g_H * M_H

        # The stoichiometric coefficients for the three reactions
        # (1) N2 + O = NO + N
        # (2) N + O2 = NO + O
        # (3) N + OH = NO + H

        # Stoichiometric coefficients
        nu_1_N2, nu_1_O, nu_1_NO, nu_1_N = -1, -1, 1, 1
        nu_2_N, nu_2_O2, nu_2_NO, nu_2_O = -1, -1, 1, 1
        nu_3_N, nu_3_OH, nu_3_NO, nu_3_H = -1, -1, 1, 1

        # sum of stoichiometric coefficients
        #sum_nu_1 = nu_1_N2 + nu_1_O + nu_1_NO + nu_1_N
        #sum_nu_2 = nu_2_N + nu_2_O2 + nu_2_NO + nu_2_O
        #sum_nu_3 = nu_3_N + nu_3_OH + nu_3_NO + nu_3_H

        # calculate the free molar reactions enthalpy of the reactions (energy after - energy before)
        Delta_g_R_1 = nu_1_N2 * g_N2 + nu_1_O * g_O + nu_1_NO * g_NO + nu_1_N * g_N
        Delta_g_R_2 = nu_2_N * g_N + nu_2_O2 * g_O2 + nu_2_NO * g_NO + nu_2_O * g_O
        Delta_g_R_3 = nu_3_N * g_N + nu_3_OH * g_OH + nu_3_NO * g_NO + nu_3_H * g_H


        # partial pressure equilibrium constants for all the reactions
        Kp_1 = math.exp(-Delta_g_R_1 / (R * T))
        Kp_2 = math.exp(-Delta_g_R_2 / (R * T))
        Kp_3 = math.exp(-Delta_g_R_3 / (R * T))

        # concentration equilibrium constants are equal to partial pressure constants since all reactions have sums
        # of stoichiometric coefficients equal to 0. I.e. same amount of molecules before and after reaction
        Kc_1 = Kp_1
        Kc_2 = Kp_2
        Kc_3 = Kp_3


        # forward reaction coefficents (from GRI_MECH 3.0 from the simulating combustion book)
        # Units (cm^3 / (mol s)
        k1_f = 0.544e14 * T ** 0.1 * math.exp(-38020/T)
        k2_f = 9.0e9 * T * math.exp(-3280 / T)
        k3_f = 3.36e13 * math.exp(-195 / T)

        # reverse reaction coefficients (not that convention is either f (forward) and r (reverse) OR r (right) and l (left)
        k1_r = k1_f / Kc_1
        k2_r = k2_f / Kc_2
        k3_r = k3_f / Kc_3

        # assuming the concentration to be quasi-steady (dNdT = 0) (MAYBE NEED TO CHANGE UNITS OF CONCENTRATIONS)
        c_N = (k1_f * c_O * c_N2 + k2_r * c_NO * c_O + k3_r * c_NO * c_H) / (k1_r * c_NO + k2_f * c_O2 + k3_f * c_OH)

        # time derivative of NO concentration
        dc_NOdt = 2 * k1_f * c_O * c_N2 - 2 * k1_r * c_NO * c_N


        return dc_NOdt, x0

    #fun_args = (x0)
    #sol = solve_ivp(dNOdt, args=fun_args, t_span=(min(times), max(times)), method='RK45', y0=np.array([0.0]), t_eval=times)

    NO_0 = 0.0
    c_NO, dNOdt = euler(dNOdt_fun, NO_0, times, x0)

    # plot temperatures and pressure
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    lns1 = ax1.plot(times, c_NO, color='red', label="NO concentration")
    lns2 = ax2.plot(times, dNOdt, label="dNOdt")

    # Set y limits and grid visibility
    #for ax, ylim in zip([ax1, ax2], [16, 4]):
    #    ax.set_ylim(0, ylim)
    #    ax.grid(True)

    #ax1.set_xlim(1500, 3000)

    # set which axis to which side
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="lower left")
    ax1.set_title("NO production")
    ax1.set_ylabel("NO concentration [mol/m^3]")

    plt.show()

    # total NOx
    NO_tot = c_NO[-1]
    return NO_tot
