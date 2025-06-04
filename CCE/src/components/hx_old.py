from thermo import mixture
from scipy.optimize import brentq


def hx_NASA(pc_i, tc_i, heating_power, th_i):

    # input:
    # pressure cold in
    # temp cold in
    # temp hot side in

    # assuming pressure losses on cold side
    dp_c = 0.057

    # cold bypass air
    hc_i, _, _, _, _, _, _, _ = mixture(tc_i, pc_i, 0)

    # assuming pressure losses (dont care about oil for now)
    pc_o = pc_i * (1 - dp_c)

    # we assume that the air will be heated to 90% of the oil temperature
    tc_o = tc_i + (th_i-tc_i) * 0.9

    hc_o, _, _, _, _, _, _, _ = mixture(tc_o, pc_o, 0)

    # we know the amount of heat that has to be dispensed into the bypass air
    # calculate mass flow of cooling air
    m_c = heating_power / (hc_o - hc_i)

    # calculate needed oil mass flow
    # assume oil reaches 90% of delta T
    th_o = th_i - (th_i - tc_i) * 0.9

    # from engineering toolbox "engine oil"
    cp_oil = 2000  # J / kg * K

    # oil mass flow
    m_oil = heating_power / ((th_i - th_o) * cp_oil)

    # return air outlet pressure and temperature and massflow
    return pc_o, tc_o, m_c, m_oil
