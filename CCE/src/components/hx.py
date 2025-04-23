from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve


def hx(pc_i, tc_i, heating_power, m_c, th_i):

    # assuming pressure losses
    # hot is oil and cold is air
    dp_h = 0.06
    dp_c = 0.057

    # hot oil
    ph_i = 10e5
    m_h = 6.6
    cp_h = PropsSI("Cpmass", "T", th_i, "P", ph_i, "INCOMP::PNF2")

    # cold bypass air
    hc_i = PropsSI("Hmass", "T", tc_i, "P", pc_i, "Air")

    # assuming constant cp for the oil
    C_oil = m_h * cp_h

    # assuming pressure losses (dont care about oil for now)
    pc_o = pc_i * (1 - dp_c)

    # we know the amount of heat that has to be dispensed into the bypass air
    hc_o = hc_i + heating_power / m_c

    # finding output temperature of air
    def find_tc_o(tc_o):
        # guessing outlet air enthalpy
        hc_o_guess = PropsSI("Hmass", "T", tc_o, "P", pc_o, "Air")

        return hc_o_guess - hc_o

    tc_o = fsolve(find_tc_o, x0=350)[0]

    def find_eps(eps):

        # calculate average cp of air
        cp_c = (hc_o - hc_i) / (tc_o - tc_i)

        # get cmin
        Cmin = min(cp_c * m_c, C_oil)

        q = eps * Cmin * (th_i - tc_i)

        return q - heating_power

    eps = fsolve(find_eps, x0=0.8)[0]

    # returning total heat rejected and outlet oil temp
    return pc_o, tc_o
