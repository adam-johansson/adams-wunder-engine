from thermo import mixture
from scipy.optimize import brentq


def hx_NASA(pc_i, tc_i, heating_power, th_i):

    # input:
    # pressure cold in
    # temp cold in
    # temp hot side in

    # assuming pressure losses on cold side
    dp_c = 0.057 # from Kaisers thesis and previouslys Kaisers synergy paper

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

    # IF WE WOULD HAVE USED WATER ETHYLENE GLYCOL (60%)

    # we assume that the air will be heated to 90% of the water ethylene glycole temp
    th_i_water = 373 # 10 K under the boiling boint at 1 atm (probably higher at higher pressure)
    tc_o_water = tc_i + (th_i_water - tc_i) * 0.9

    hc_o_water, _, _, _, _, _, _, _ = mixture(tc_o_water, pc_o, 0)

    # we know the amount of heat that has to be dispensed into the bypass air
    # calculate mass flow of cooling air
    m_c_water = heating_power / (hc_o_water - hc_i)

    # calculate needed oil mass flow
    # assume water reaches 90% of delta T
    th_o_water = th_i - (th_i - tc_i) * 0.9

    # from engineering toolbox "ethyl glycole"
    cp_water = 3528  # J / kg * K

    # water mass flow
    m_water = heating_power / ((th_i - th_o_water) * cp_water)

    #print(f"Mass flow of water: {m_water} kg/s air mass flow: {m_c_water} kg/s")
    #print(f"temperature out water: {th_o_water} air temperature out: {tc_o_water}")



    #print(f"Oil temperature in: {th_i} air temperature in: {tc_i}")
    #print(f"Oil temperature out: {th_o} air temperature out: {tc_o}")
    #print(f"Oil mass flow: {m_oil} air mass flow: {m_c}")





    #print(f"Oil temperature in: {th_i} air temperature in: {tc_i}")
    #print(f"Oil temperature out: {th_o} air temperature out: {tc_o}")
    #print(f"Oil mass flow: {m_oil} air mass flow: {m_c}")
    #print(f"cp*mdot oil: {cp_oil * m_oil} cp*mdot air: {m_c * cp_air}")

    # return air outlet pressure and temperature and massflow
    return pc_o, tc_o, m_c, m_oil
