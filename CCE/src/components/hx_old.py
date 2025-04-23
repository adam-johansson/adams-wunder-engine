from CCE.src import thermo_outdated
from scipy.optimize import brentq


def hx_NASA(pc_i, tc_i, heating_power, m_c, th_i):

    # assuming pressure losses
    dp_c = 0.057

    p_dummy = 1e5

    # cold bypass air
    cp, hc_i, s, M = thermo_outdated.properties(tc_i, p_dummy, 0)

    # assuming pressure losses (dont care about oil for now)
    pc_o = pc_i * (1 - dp_c)

    # we know the amount of heat that has to be dispensed into the bypass air
    hc_o = hc_i + heating_power / m_c

    # finding output temperature of air
    def find_tc_o(tc_o):
        if tc_o < 200:
            return 1e9
        # guessing outlet air enthalpy
        cp, hc_o_guess, s, M = thermo_outdated.properties(tc_o, p_dummy, 0)

        return hc_o_guess - hc_o

    tc_o = brentq(find_tc_o, 200, 6000)

    # returning total heat rejected and outlet oil temp
    return pc_o, tc_o
