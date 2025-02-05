import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from thermo import flame_temp_inhouse, flame_temp_cea, mixture


def twozone(phi, P, T, V, m, mf, evo, sc, lhv, far_s, equ, fuel_type):

    """
    Divides the cylinder volume into two zones, for more accurate NOx calculations.

    Zone 1: reaction zone, burned zone (this is after the flame front)
    Zone 2: unburned zone, outside the flame front.

    :param phi:
    :param P:
    :param T:
    :param V:
    :param m:
    :param qf:
    :param evo:
    :param sc:
    :param lhv:
    :param far_s:
    :param equ:
    :param fuel_type:
    :return:
    """

    # model assumes both zones have same pressure
    # p1 = p2 = p

    # high pressure crank angles
    phi_hp = np.array(phi[np.argwhere((phi > sc) & (phi < evo))])

    # high pressure pressure curve
    P_hp = np.array(P[np.argwhere((phi > sc) & (phi < evo))])

    # high pressure temperature (only between start of combustion and exhaust valve opening)
    T_hp = np.array(T[np.argwhere((phi > sc) & (phi < evo))])

    # high pressure volume(only between start of combustion and exhaust valve opening)
    V_hp = np.array(V[np.argwhere((phi > sc) & (phi < evo))])

    # high pressure heat addition (only between start of combustion and exhaust valve opening)
    mf_hp = np.array(mf[np.argwhere((phi > sc) & (phi < evo))])

    # high pressure heat addition (only between start of combustion and exhaust valve opening)
    qf_hp = mf_hp * lhv

    # high pressure cylinder mass (only between start of combustion and exhaust valve opening)
    m_hp = np.array(m[np.argwhere((phi > sc) & (phi < evo))])

    # high pressure equivalence ratio (only between start of combustion and exhaust valve opening)
    equ_hp = np.array(equ[np.argwhere((phi > sc) & (phi < evo))])

    # calculate fuel mass
    m_fuel = integrate.cumulative_simpson(mf_hp, x=phi_hp, axis=0, initial=0.0) # + fuel_burned from last cycle

    # lambda_0 is the air-fuel-ratio in the reaction zone, assumed to be constant.
    # for small to medium sized diesel engines with intake swirl lambda_0 = 1.0

    # NOTE THAT FOR spark ignition (hydrogen??) then we use lambda_0 = lambda_global
    lambda_0 = 1.0

    # L_min is minmal air requirement (kg of air per kg of fuel)
    # this is the stochiometric air requirements + taken the residual exhaust gases into account

    # far_sc is the fuel air ratio (of burned gases) in the cylinder before fuel injection
    equ_sc = equ[np.argwhere(phi < sc)][-1][0]
    far_sc = equ_sc * far_s

    # air amount (not pure air)
    L_min = 1 / (far_s - far_sc)

    # mass in zone 1 (reaction zone, burned zone, kärt barn har många namn)
    m1 = (lambda_0 * L_min + 1) * m_fuel

    # mass in zone 2 (unburned gas)
    m2 = m_hp - m1

    # gas constants
    # pressure and temperature doesnt effect R
    t_dummy = 1000
    p_dummy = 1e5
    _, _, _, _, R1, _, _, _ = mixture(t_dummy, p_dummy, equ=1 / lambda_0, fuel_type=fuel_type)
    _, _, _, _, R2, _, _, _ = mixture(t_dummy, p_dummy, equ=equ_sc, fuel_type=fuel_type)



    # motoring pressure (polytropic compression from start of combustion)
    # polytropic exponent (how to calculate this??? based on last 10 degrees crank angle before combustion)

    # Pressure at start of combustion
    Psc = P[np.argwhere(phi > sc)[0]][0]
    # Volume at start of combustion
    Vsc = V[np.argwhere(phi > sc)[0]][0]

    # determine the polytrope exponent for the last 10 or so CA before start of combustion
    # number of degrees to avg
    ca_avg = 10

    # get pressure and volume for those angles
    P_poly = np.array(P[np.argwhere((phi > sc - ca_avg * np.pi / 180) & (phi < sc))])
    V_poly = np.array(V[np.argwhere((phi > sc - ca_avg * np.pi / 180) & (phi < sc))])

    # polytrope exponent
    # take first and last values to calculate the exponent
    #n = (np.log(P_poly[-1]) - np.log(P_poly[0]) ) / (np.log(V_poly[0]) - np.log(V_poly[-1]))

    # calculate exponent between each steps and take average
    n_poly = np.mean( ( np.log(P_poly[1:-1]) - np.log(P_poly[0:-2]) ) / ( np.log(V_poly[0:-2]) - np.log(V_poly[1:-1]) ) )


    # Pmotor is the theoretical motoring pressure if the engine were to run without fuel. between start of combution
    # to exhaust valve opening
    Pmotor = Psc * (Vsc / V_hp) ** n_poly


    # difference cylinder pressure and motoring pressure
    Pdiff = P_hp - Pmotor

    # the integrand from the Heider 1996 paper
    integrand = Pdiff * m1

    # denominator is constant
    denominator = integrate.simpson(integrand, x=phi_hp, axis=0)

    # nominator is function of angle
    nominator = integrate.cumulative_simpson(integrand, x=phi_hp, axis=0, initial=0.0)

    # B funktion
    B = 1 - nominator / denominator

    # A function
    # temperature at start of combustion
    t_soc = T[np.argwhere(phi > sc)[0]][0]

    # adiabatic flame temperature
    # thoughts: adiabatic flame temperature gets lower when we are not using pure air. this needs to be adjusted for.
    # otherwise we can't investigate EGR

    # use my own flame temp function (switch to cantera later when I implement it for NOx maybe) (gave 3000K flame temp)
    t_flame = flame_temp_inhouse(t_soc, equ_sc, fuel_type)

    # this is the cea program
    #t_flame = flame_temp_cea(t_soc, equ_sc, fuel_type, Psc)


    #t_flame = 2093 + 273.15 # wikipedia for kerosene (but this is for 20C air) need to adjust for temperature
    # 2254 + 273.15 # wikipedia for hydrogen and air


    # Kaiser used a factor here. Could be used to fit model to experimental data
    # he used 0.9 when validating. look at his thesis
    factor = 0.85

    # for validation we want A = 1595 K
    A = (t_flame - t_soc) * factor

    # adjust accorind to Heider
    # C = 0.15 for 4 valves and central injection
    C = 0.15

    # global air-fuel equivalence ratio (when all fuel is injected)
    lambda_gl = 1 / equ_hp[-1]

    # test with and without this when we get some numbers for nox (Astar gives slightly higher temp)
    #Astar = A * (1.2 + (lambda_gl[0] - 1.2)**C) / (2.2 * lambda_0)
    Astar = A
    print(f"Twozone factor Astar: {Astar}")



    # solve for temperature in zone 1, T1
    T1 = (P_hp * V_hp + m2 * R2 * Astar * B) / ( m1 * R1 + m2 * R2)

    # zone 2 temperature
    T2 = T1 - Astar * B

    # zone 1 volume
    V1 = m1 * R1 * T1 / P_hp

    return T1, m1, P_hp, V1, lambda_0, phi_hp, equ_hp, T2, m2, T_hp

