import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate


def twozone(phi, P, T, V, m, qf, evo, sc, lhv, far_s, equ):

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
    qf_hp = np.array(qf[np.argwhere((phi > sc) & (phi < evo))])

    # high pressure cylinder mass (only between start of combustion and exhaust valve opening)
    m_hp = np.array(m[np.argwhere((phi > sc) & (phi < evo))])

    # calculate fuel mass
    m_fuel = (1/lhv) * integrate.cumulative_simpson(qf_hp, x=phi_hp, axis=0, initial=0.0) # + fuel_burned from last cycle

    # lambda_0 is the air-fuel-ratio in the reaction zone, assumed to be constant.
    # for small to medium sized diesel engines with intake swirl lambda_0 = 1.0
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

    # gas constants (change later)
    R1 = 287
    R2 = 287


    # motoring pressure (polytropic compression from start of combustion)
    # polytropic exponent (how to calculate this??? based on last 10 degrees crank angle before combustion)
    gamma = 1.4

    # Pressure at start of combustion
    Psc = P[np.argwhere(phi > sc)[0]][0]
    # Volume at start of combustion
    Vsc = V[np.argwhere(phi > sc)[0]][0]

    # Pmotor is the theoretical motoring pressure if the engine were to run without fuel. between start of combution
    # to exhaust valve opening
    Pmotor = Psc * (Vsc / V_hp) ** gamma


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
    t_flame = 2093 + 273.15 # wikipedia for kerosene (but this is for 20C air) need to adjust for temperature
    # 2254 + 273.15 # wikipedia for hydrogen and air

    # Kaiser used a factor here. Could be used to fit model to experimental data
    A = t_flame - t_soc

    # solve for temperature in zone 1, T1
    T1 = (P_hp * V_hp + m2 * R2 * A * B) / ( m1 * R1 + m2 * R2)

    # zone 2 temperature
    T2 = T1 - A * B

    # plot temperatures and pressure
    fig, ax1 = plt.subplots()

    #ax2 = ax1.twinx()
    #ax2.plot(phi_hp * 180 / np.pi, P_hp*1e-5, label="Cylinder pressure")
    ax1.plot(phi_hp * 180 / np.pi, T1, label="Zone 1")
    ax1.plot(phi_hp * 180 / np.pi, T2, label="Zone 2")
    ax1.plot(phi_hp * 180 / np.pi, T_hp, label="Single zone")

    ax1.set_xlabel('Crank angle [deg]')
    #ax1.set_ylabel('Cylinder pressure [bar]', color='g')
    ax1.set_ylabel('Temperature [K]', color='b')

    #ax2.legend()
    ax1.legend()
    plt.show()

    return T2, T1

