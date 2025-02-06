import math

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

import cantera as ct

from numba import jit

from thermo import mixture, molar_fractions, equilibrium_OHC, polynomials, euler_cantera


def nox_calculations(
    temperatures,
    masses,
    pressures,
    volumes,
    fuel_type,
    lambda_z1,
    phi,
    rpm,
    m_tot,
    mf_tot,
    equ_global,
    m_global,
):
    """
    This model calculates the nox produced in the reaction zone of the two-zone model. That is the burned zone.

    It uses the extended Zeldovich mechanics. Ignoring NOx creation in flame front. No prompt NOx.

    How it is done comes from the text book by Merker 2005, Simulating Combustion.

    m_tot is the total mass that flows out of the engine during one cycle, and equ_tot is the equivalence ratio of
    the outflow

    :return:
    """

    # Universal gas constant from NASA polynomials pdf
    R_univ = 8.314510  # J mol^-1 K^-1

    # standard state pressure
    p_std = 1e5

    gas = ct.Solution("gri30.yaml")

    equ = 1 / lambda_z1  # Equivalence ratio in the burned zone

    # convert crank angles to time
    rps = rpm / 60
    radians_per_s = rps * 2 * np.pi

    # times = phi[1:] / radians_per_s
    times = np.ndarray.flatten(phi / radians_per_s)

    dVdt_s = np.gradient(np.ndarray.flatten(volumes), times)

    t_dummy = 1000
    p_dummy = 1e5
    R_mixture, M_mixture, xi_N2_0, xi_O2_0, xi_CO2_0, xi_H2O_0, xi_Ar_0 = molar_fractions(
        t_dummy, p_dummy, equ=equ, fuel_type=fuel_type
    )

    # replace N2 with argon
    xi_Ar_0 = xi_Ar_0 + xi_N2_0

    # skip the first data point where the volume is 0
    # times = times[1:]
    # temperatures = temperatures[1:]
    # masses = masses[1:]
    # pressures = pressures[1:]
    # volumes = volumes[1:]
    # dVdt_s = dVdt_s[1:]

    def dNOdt_fun(t, var):

        c_NO = var[0]
        #c_N = var[1]

        # i = np.nonzero(times==t)
        idx = (np.abs(times - t)).argmin()

        T = temperatures[idx][0]
        p = pressures[idx][0]
        V = volumes[idx][0]
        dVdt = dVdt_s[idx]

        # since we want to look at the OHC system isolated, we replace all N2 with Ar
        gas.TPX = T, p, f"CO2:{xi_CO2_0}, H2O:{xi_H2O_0}, O2:{xi_O2_0}, Ar:{xi_Ar_0}"

        gas.equilibrate("TP")

        mixture = gas.mole_fraction_dict(threshold=1e-20)

        # OHC system (FAST) assuming equilibirum
        # mol fractions
        # maybe the concentrations should be effeceted by the NO production
        xi_O2 = mixture["O2"]
        xi_OH = mixture["OH"]
        xi_O = mixture["O"]
        xi_H = mixture["H"]

        # extract concentrations needed for Zeldovich mechanism
        c_H = (xi_H * p) / (R_univ * T)
        c_O2 = (xi_O2 * p) / (R_univ * T)
        c_O = (xi_O * p) / (R_univ * T)
        c_OH = (xi_OH * p) / (R_univ * T)

        # N2 should be effected?
        c_N2 = (xi_N2_0 * p) / (R_univ * T)

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
        # sum_nu_1 = nu_1_N2 + nu_1_O + nu_1_NO + nu_1_N
        # sum_nu_2 = nu_2_N + nu_2_O2 + nu_2_NO + nu_2_O
        # sum_nu_3 = nu_3_N + nu_3_OH + nu_3_NO + nu_3_H

        # calculate the free molar reactions enthalpy of the reactions (energy after - energy before)
        Delta_g_R_1 = nu_1_N2 * g_N2 + nu_1_O * g_O + nu_1_NO * g_NO + nu_1_N * g_N
        Delta_g_R_2 = nu_2_N * g_N + nu_2_O2 * g_O2 + nu_2_NO * g_NO + nu_2_O * g_O
        Delta_g_R_3 = nu_3_N * g_N + nu_3_OH * g_OH + nu_3_NO * g_NO + nu_3_H * g_H


        # partial pressure equilibrium constants for all the reactions
        Kp_1 = math.exp(-Delta_g_R_1 / (R_univ * T))
        Kp_2 = math.exp(-Delta_g_R_2 / (R_univ * T))
        Kp_3 = math.exp(-Delta_g_R_3 / (R_univ * T))

        # concentration equilibrium constants are equal to partial pressure constants since all reactions have sums
        # of stoichiometric coefficients equal to 0. I.e. same amount of molecules before and after reaction
        Kc_1 = Kp_1
        Kc_2 = Kp_2
        Kc_3 = Kp_3

        # forward reaction coefficents (from GRI_MECH 3.0 from the simulating combustion book)
        # Units (cm^3 / (mol s)
        k1_f = 0.544e14 * (T**0.1) * math.exp(-38020 / T)
        k2_f = 9.0e9 * T * math.exp(-3280 / T)
        k3_f = 3.36e13 * math.exp(-195 / T)

        # convert to m^3 from cm^3
        k1_f = k1_f * 1e-6
        k2_f = k2_f * 1e-6
        k3_f = k3_f * 1e-6

        # reverse reaction coefficients (note that convention is either f (forward) and r (reverse) OR r (right) and l (left)
        k1_r = k1_f / Kc_1
        k2_r = k2_f / Kc_2
        k3_r = k3_f / Kc_3



        """
        if V > 0:
            dc_Ndt = (
                k1_f * c_O * c_N2
                - k2_f * c_N * c_O2
                - k3_f * c_N * c_OH
                - k1_r * c_NO * c_N
                + k2_r * c_NO * c_O
                + k3_r * c_NO * c_H
                - (c_N / V) * dVdt
            )
            dc_NOdt = (
                k1_f * c_O * c_N2
                + k2_f * c_N * c_O2
                + k3_f * c_N * c_OH
                - k1_r * c_NO * c_N
                - k2_r * c_NO * c_O
                - k3_r * c_NO * c_H
                - (c_NO / V) * dVdt
            )
        else:
            dc_Ndt = (
                k1_f * c_O * c_N2
                - k2_f * c_N * c_O2
                - k3_f * c_N * c_OH
                - k1_r * c_NO * c_N
                + k2_r * c_NO * c_O
                + k3_r * c_NO * c_H
            )
            dc_NOdt = (
                k1_f * c_O * c_N2
                + k2_f * c_N * c_O2
                + k3_f * c_N * c_OH
                - k1_r * c_NO * c_N
                - k2_r * c_NO * c_O
                - k3_r * c_NO * c_H
            )
        """
        # time derivative of NO concentration (accounting for volume change)
        # d[A]dt = R - [A] / V * dVdt (R is reaction rate)
        if V > 0:

            # assuming the concentration to be quasi-steady (dNdT = 0)
            c_N = (k1_f * c_O * c_N2 + k2_r * c_NO * c_O + k3_r * c_NO * c_H) / (
                        k1_r * c_NO + k2_f * c_O2 + k3_f * c_OH + dVdt / V)

            dc_NOdt = 2 * k1_f * c_O * c_N2 - 2 * k1_r * c_NO * c_N - (c_NO / V ) * dVdt - (c_N / V ) * dVdt

        else:
            # assuming the concentration to be quasi-steady (dNdT = 0)
            c_N = (k1_f * c_O * c_N2 + k2_r * c_NO * c_O + k3_r * c_NO * c_H) / (
                        k1_r * c_NO + k2_f * c_O2 + k3_f * c_OH)

            dc_NOdt = 2 * k1_f * c_O * c_N2 - 2 * k1_r * c_NO * c_N

        # I DONT THINK THIS IS CORRECT BUT IT MATCHES VALIDATION BETTER
        #dc_NOdt = 2 * k1_f * c_O * c_N2 - 2 * k1_r * c_NO * c_N

        # I think N2 concentration should decrease with increasing NO (can probably ignore that)
        # it wont effect much I think. This is probably a later problem.
        # print(f"N2: {c_N2}, N: {c_N}, NO: {c_NO}")

        #return dc_NOdt, dc_Ndt
        return dc_NOdt

    # All methods except DOP853 seems to be equal fast
    # sol = solve_ivp(dNOdt_fun, t_span=(min(times), max(times)), method='RK45', y0=np.array([0.0]), t_eval=times)
    sol = solve_ivp(
        dNOdt_fun,
        t_span=(min(times), max(times)),
        method="RK45",
        y0=np.array([0.0]),
        t_eval=times,
    )

    NO_concentration = np.ndarray.flatten(sol.y[0])
    dNOdt_concentration = np.gradient(NO_concentration, times)

    # absolute amount of NO molecules (mol)
    NO_mol = NO_concentration * np.ndarray.flatten(volumes)
    dNOdt_mol = np.gradient(NO_mol, times)

    # NO_mol = NO_concentration

    # now we want ppm. Question is if it is mass based or volume based?

    # mass of NO
    # (kg/mol molar mass)
    _, _, _, g_NO, M_NO = polynomials.NO(t_dummy, p_dummy)

    # mass of NO in the cylinder
    m_NO = NO_mol * M_NO

    # NOx concentration mass of NO divided by total mass of gas leaving cylinder
    # should we instead have mass of cylinder gasses at each point in time?
    #no_concentration_mass = m_NO[1:] / np.ndarray.flatten(masses[1:])
    no_concentration_mass = m_NO / m_tot

    #
    _, _, _, _, _, _, _, M_global = mixture(t_dummy, p_dummy, equ_global, fuel_type)

    # total number of moles of molecules in entire cylinder
    mol_global = m_global / M_global

    # number of moles of NO divided by total number of moles in cylinder (ignoring any chemical reactions)
    no_concentration_volume = NO_mol / mol_global

    # base it only on zone two (NOT CORRECT)
    # nox_concentration = m_NO / masses[-1][0]

    # convert to ppm
    no_concentration_mass = no_concentration_mass * 1e6
    no_concentration_volume = no_concentration_volume * 1e6

    # Emission (g/kg)
    EI_nox = (m_NO / mf_tot) * 1e3

    print(f"NOx concentration in exhaust volume {no_concentration_volume[-1]} PPM")
    print(f"NOx concentration in exhaust mass {no_concentration_mass[-1]} PPM")
    print(f"Emission index (g NO per kg fueL) {EI_nox[-1]} g/kg")

    """
    # plot temperatures and pressure
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    lns1 = ax1.plot(times * 1000, NO_concentration, color='red', label="NO concentration")
    lns2 = ax2.plot(times * 1000, dNOdt_concentration, label="dNOdt")

    #ax1.set_xlim(1500, 3000)

    # set which axis to which side
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="lower left")
    ax1.set_title("NO production")
    ax1.set_ylabel(" NO concentration [mol / m^3]")
    ax2.set_ylabel("NO production [mol/ m^3 s]")
    ax1.set_xlabel("Time [ms]")

    # plot temperatures and pressure
    fig, ax3 = plt.subplots()

    ax4 = ax3.twinx()

    lns1 = ax3.plot(times * 1000, volumes, color='red', label="V")
    lns2 = ax4.plot(times * 1000, masses * 1000, label="m")



    # set which axis to which side
    ax3.yaxis.tick_left()
    ax4.yaxis.tick_right()

    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc="lower left")
    ax3.set_title("m and V burned zone")
    ax3.set_ylabel("V [m^3]")
    ax4.set_ylabel("m [g]")
    ax3.set_xlabel("Time [ms]")

    # plot NO mol amount
    fig, ax5 = plt.subplots()

    ax6 = ax5.twinx()

    lns1 = ax5.plot(times * 1000, NO_mol, color='red', label="NO")
    lns2 = ax6.plot(times * 1000, dNOdt_mol, label="dNOdt")



    # set which axis to which side
    ax5.yaxis.tick_left()
    ax6.yaxis.tick_right()

    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc="lower left")
    ax5.set_title("Mol NO")
    ax5.set_ylabel("NO [mol]")
    ax6.set_ylabel("dNOdt [mol / s]")
    ax5.set_xlabel("Time [ms]")

    plt.show()
    """

    return no_concentration_mass, NO_mol, dNOdt_mol, times
