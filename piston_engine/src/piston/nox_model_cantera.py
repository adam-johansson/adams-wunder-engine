import math

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

import cantera as ct

from numba import jit

from thermo import mixture, molar_fractions, equilibrium_OHC, polynomials, euler_cantera, molar_fractions_combustion


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
    xi_N2_0, xi_O2_0, xi_CO2_0, xi_H2O_0, xi_Ar_0 = molar_fractions(
        equ=equ, fuel_type=fuel_type
    )

    # Cantera
    T0 = temperatures[0][0]
    p0 = pressures[0][0]


    #xi_N2_0, xi_O2_0, xi_CO2_0, xi_H2O_0, xi_Ar_0 = molar_fractions_combustion(T0, p0, equ_sc=0.0, equ_combustion=equ)


    # replace N2 with argon
    xi_Ar_0 = xi_Ar_0 + xi_N2_0


    def dNOdt_fun(t, var):

        idx = (np.abs(times - t)).argmin()

        T = temperatures[idx][0]
        p = pressures[idx][0]
        V = volumes[idx][0]
        dVdt = dVdt_s[idx]

        # concentration
        if V > 0:
            c_NO = var[0] / V
        else:
            c_NO = 0.0

        # since we want to look at the OHC system isolated, we replace all N2 with Ar
        gas.TPX = T, p, f"CO2:{xi_CO2_0}, H2O:{xi_H2O_0}, O2:{xi_O2_0}, Ar:{xi_Ar_0}"

        gas.equilibrate("TP")

        fractions = gas.mole_fraction_dict(threshold=1e-20)

        # OHC system (FAST) assuming equilibirum
        # mol fractions
        # maybe the concentrations should be effeceted by the NO production
        xi_O2 = fractions["O2"]
        xi_OH = fractions["OH"]
        try:
            xi_O = fractions["O"]
            xi_H = fractions["H"]
        except:
            xi_O = 0.0
            xi_H = 0.0

        # extract concentrations needed for Zeldovich mechanism
        c_H = (xi_H * p) / (R_univ * T)
        c_O2 = (xi_O2 * p) / (R_univ * T)
        c_O = (xi_O * p) / (R_univ * T)
        c_OH = (xi_OH * p) / (R_univ * T)

        # N2 should be effected?
        c_N2 = (xi_N2_0 * p) / (R_univ * T)

        # First step for reducing O2 and O and N2
        #c_O2old = c_O2
        #c_N2old = c_N2

        #c_O2 = c_O2 - 0.5 * c_NO
        #c_N2 = c_N2 - 0.5 * c_NO

        #print(f" O2 old: {c_O2old} O2 :{c_O2}")
        #print(f" N2 old: {c_N2old} N2 :{c_N2}")
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



        coefficients = "grimech"

        if coefficients == "book":

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

        else:
            # from GRI_MECH 3.0 (the same as above??)
            # they are on the form: k = A T ^ m exp(-E / RT)
            # concentrations: mol/cm3. A are 1/s, cm3/mol/s, cm6/mol2/s for first, second, and third order reactions, respectively;
            # T is in K; and E is in cal/mol.

            # (1) N+NO<=>N2+O                              2.700E+13     .000     355.00
            # (2) N+O2<=>NO+O                              9.000E+09    1.000    6500.00
            # (3) N+OH<=>NO+H                              3.360E+13     .000     385.00

            # The stoichiometric coefficients for the three reactions
            # (1) N2 + O = NO + N
            # (2) N + O2 = NO + O
            # (3) N + OH = NO + H

            A1 = 2.70e13
            E1 = 355.0
            A2 = 9.0e9
            E2 = 6500.0
            A3 = 3.360e13
            E3 = 385.0

            # Universal gas constant in cal / (K * mol)
            R_univ_cal = 1.98720425864083

            k1_r = A1 * math.exp(- E1 / (R_univ_cal * T))
            k2_f = A2 * T * math.exp(- E2 / (R_univ_cal * T))
            k3_f = A3 * math.exp(- E3 / (R_univ_cal * T))


            # convert to m^3 from cm^3
            k1_r = k1_r * 1e-6
            k2_f = k2_f * 1e-6
            k3_f = k3_f * 1e-6

            # k1_f is different because the gri mech reaction was defined backwards
            k1_f = k1_r * Kc_1
            k2_r = k2_f / Kc_2
            k3_r = k3_f / Kc_3


        # assuming the concentration of N to be quasi-steady (dNdT = 0)

        if V > 0:
            c_N = (k1_f * c_O * c_N2 + k2_r * c_NO * c_O + k3_r * c_NO * c_H) / (
                    k1_r * c_NO + k2_f * c_O2 + k3_f * c_OH + dVdt / V)

            dc_NOdt = 2 * k1_f * c_O * c_N2 - 2 * k1_r * c_NO * c_N - (c_NO / V) * dVdt - (c_N / V ) * dVdt
        else:
            c_N = (k1_f * c_O * c_N2 + k2_r * c_NO * c_O + k3_r * c_NO * c_H) / (
                    k1_r * c_NO + k2_f * c_O2 + k3_f * c_OH)

            dc_NOdt = 2 * k1_f * c_O * c_N2 - 2 * k1_r * c_NO * c_N

        dNOdt = dc_NOdt * V + c_NO * dVdt

        return dNOdt

    # All methods except DOP853 seems to be equal fast
    # sol = solve_ivp(dNOdt_fun, t_span=(min(times), max(times)), method='RK45', y0=np.array([0.0]), t_eval=times)
    sol = solve_ivp(
        dNOdt_fun,
        t_span=(min(times), max(times)),
        method="RK45",
        y0=np.array([0.0]),
        t_eval=times,
        max_step=1e-4,
    )

    NO_mol = np.ndarray.flatten(sol.y[0])
    dNOdt_mol = np.gradient(NO_mol, times)

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

    # convert to ppm
    no_concentration_mass = no_concentration_mass * 1e6
    no_concentration_volume = no_concentration_volume * 1e6

    # Emission (g/kg)
    EI_nox = (m_NO[-1] / mf_tot) * 1e3

    print(f"NOx concentration in exhaust volume {no_concentration_volume[-1]} PPM")
    print(f"NOx concentration in exhaust mass {no_concentration_mass[-1]} PPM")
    print(f"Emission index (g NO per kg fueL) {EI_nox} g/kg")

    return no_concentration_mass, dNOdt_mol, times
