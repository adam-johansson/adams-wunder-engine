import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from thermo import mixture, molar_fractions


def nox_calculations(T, m, p, V, fuel_type, lambda_z1, phi, rpm, m_tot, equ_tot):

    """
    This model calculates the nox produced in the reaction zone of the two-zone model. That is the burned zone.

    It uses the extended Zeldovich mechanics. Ignoring NOx creation in flame front. No prompt NOx.

    How it is done comes from the text book by Merker 2005, Simulating Combustion.

    m_tot is the total mass that flows out of the engine during one cycle, and equ_tot is the equivalence ratio of
    the outflow

    :return:
    """

    # rate of change of NO (from textbook). Simplification.

    # species concentrations are in mol/dm^3 maybe. At least mol / volume

    # question is, where do I get N2 and O2 from? they are supposed to be in the burned zone, but there is no oxygen
    # there?   maybe I get them from zone 1?

    # temperature and pressure does not affect molar mass
    t_dummy = 1000
    p_dummy = 1e5

    # molar mass of gas in reaction zone and molar fractions of nitrogen and oxygen
    R, M, x_N2, x_O2 = molar_fractions(t_dummy, p_dummy, equ=1/lambda_z1, fuel_type=fuel_type)

    # total number of moles in the reaction zone (kg divided by kg / mol)
    # skip first value since that is 0
    n = m[1:] / M

    # total number of moles of nitrogen and oxygen
    n_N2 = x_N2 * n
    n_O2 = x_O2 * n

    # get molar concentrations of nitrogen and oxygen in reaction zone (mole per m^3?)
    N2 = n_N2 / V[1:]
    O2 = n_O2 / V[1:]

    # convert to mole / cm^3 since that is the unit of the reaction coefficients
    N2 = N2 * 1e-6
    O2 = O2 * 1e-6

    # get reaction rate (mole per second)??
    dNOdt = 4.7 * 1e13 * N2 * np.sqrt(O2) * np.exp(-67837/T[1:])

    # convert crank angles to time
    rps = rpm / 60
    radians_per_s = rps * 2 * np.pi

    t = phi[1:] / radians_per_s


    # integrate reaction rate
    # nox is in mole??
    nox = integrate.cumulative_simpson(dNOdt, x=t, axis=0, initial=0.0)

    # number of NOx molecules per total number of molecules
    # (of total number of molecules in the cylinder gases, not only reaction zone)

    # molar mass of outflow gas
    _, M_tot, _, _ = molar_fractions(t_dummy, p_dummy, equ=equ_tot, fuel_type=fuel_type)

    # total number of moles of molecules leaving the cylinder during one cycle
    n_tot = m_tot / M_tot

    # NOx concentration (total number of moles NOx created divided by total number of moles leaving cylinder)
    nox_concentration = nox / n_tot

    # convert to ppm
    nox_concentration = nox_concentration * 1e6


    # should get less or around 2000 ppm NOx

    # plot temperatures and pressure
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(t, dNOdt, 'm', label="dNOdt")
    ax2.plot(t, nox_concentration, 'r', label="NO")


    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('dNOdt [mole/s]', color='m')
    ax2.set_ylabel('NOx [ppm]', color='r')

    #ax2.legend()
    ax1.legend()

    plt.show()

    return nox
