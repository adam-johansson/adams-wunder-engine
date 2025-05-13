import numpy as np

import pandas as pd

from thermo import H2, JETA_L, mixture



def power_balance(
    induced_power_tot,
    friction_loss_tot,
    aux_loss_tot,
    fuel_pump,
    pe_cirumv,
    turbine_cooling,
    hpc_gear,
    hpc,
    power_lpt,
    lp_spool,
    ipc,
    fan_gear,
    fan,
    pe_heat,
    h2_heat,
    bypass_heat,
):
    """
    csv of power balance of the two spools
    """
    headers = ("component", "P [kW]")
    components = (
        "PE",
        "PE friction",
        "aux",
        "fuel_pump",
        "PE circumv compressor",
        "CA compressor",
        "gearbox",
        "HPC",
        "sum HP",
        "LPT",
        "LP spool",
        "IPC",
        "fan gear",
        "fan",
        "sum LP",
        "heat_PE",
        "friction_heat",
        "cooling_H2",
        "cooling_bypass",
        "sum_heat",
    )
    components = np.atleast_2d(components).T

    sum_HP = (
        induced_power_tot
        - friction_loss_tot
        - aux_loss_tot
        - fuel_pump
        - pe_cirumv
        - turbine_cooling
        - hpc_gear
        - hpc
    )
    sum_LP = power_lpt - lp_spool - ipc - fan_gear - fan
    sum_heat = pe_heat + friction_loss_tot - h2_heat - bypass_heat

    powers = (
        np.array(
            [
                induced_power_tot,
                -friction_loss_tot,
                -aux_loss_tot,
                -fuel_pump,
                -pe_cirumv,
                -turbine_cooling,
                -hpc_gear,
                -hpc,
                sum_HP,
                power_lpt,
                -lp_spool,
                -ipc,
                -fan_gear,
                -fan,
                sum_LP,
                pe_heat,
                friction_loss_tot,
                h2_heat,
                bypass_heat,
                sum_heat,
            ]
        )
        * 1e-3
    )

    data = np.concatenate((components, np.atleast_2d(powers).T), axis=1)
    data = np.vstack([headers, data])
    np.savetxt("simulation_data/power_balance.csv", data, delimiter=",", fmt="%s")

    return


def energy_flow_fuel(
    lhv,
    m_f_p,
    m_f_b,
    t_tank,
    fuel_type,
    t_fuel,
    P_indicated,
    piston_heatloss,
    t_p_in,
    t_p_out,
    m_p_in,
    m_p_out,
    equ_p_out,
    t_b_in,
    equ_b_in,
    m_b_in,
    t_b_out,
    equ_b_out,
    m_b_out,
):
    """
    keeping track of the fuel energy
    """

    # pressure needed as input to thermo_outdated
    p_dummy = 1e5
    # total fuel flow (piston + burner)
    m_f = m_f_p + m_f_b

    # fuel enthalpy from tank
    if fuel_type == "H2":
        _, hf_tank, _, _, _ = H2(t_tank, p=p_dummy)
    elif fuel_type == "jetA":
        _, hf_tank, _, _ = JETA_L(t_tank)

    # fuel energy incoming per second from the fuel tank (lower heating value + enthalpy)
    P_f_t = m_f * (lhv + hf_tank)

    # added enthalpy to fuel from oil-fuel heat exchanger
    if fuel_type == "H2":
        _, hf, _, _, _ = H2(t_fuel, p_dummy)
    elif fuel_type == "jetA":
        _, hf,_, _ = JETA_L(t_fuel)

    P_f_hx = m_f * (hf - hf_tank)

    # total fuel energy flow
    P_f = P_f_t + P_f_hx

    # piston engine fuel energy
    P_f_p = m_f_p * (lhv + hf)

    # piston engine indicated power and its fraction of fuel energy
    if P_f_p > 0:
        fraction_indicated = P_indicated / P_f_p

        # piston engine wall heat losses
        fraction_wall = piston_heatloss / P_f_p

        # piston engine exhaust energy

        # enthalpy into the piston
        h_p_in, _, _, _, _, _, _, _ = mixture(t_p_in, p_dummy, 0)

        # enthalpy out of the piston
        h_p_out, _, _, _, _, _, _, _ = mixture(t_p_out, p_dummy, equ_p_out, fuel_type)

        # gas energy increase
        P_p_gas = h_p_out * m_p_out - h_p_in * m_p_in
        fraction_gas = P_p_gas / P_f_p

        # exhaust gas energy from piston
        P_p_gas2 = P_f_p - P_indicated - piston_heatloss
        fraction_2 = P_p_gas2 / P_f_p

    else:
        fraction_indicated = 0.0
        fraction_wall = 0.0
        fraction_gas = 0.0
        fraction_2 = 0.0
        P_p_gas2 = 0.0
        P_p_gas = 0.0


    # BURNER
    # burner fuel energy
    P_f_b = m_f_b * (lhv + hf)

    # enthalpy in burner
    h_b_in, _, _, _, _, _, _, _ = mixture(t_b_in, p_dummy, equ_b_in, fuel_type)

    # enthalpy out burner
    h_b_out, _, _, _, _, _, _, _ = mixture(t_b_out, p_dummy, equ_b_out, fuel_type)

    # increase in gas energy burner
    P_b_gas = h_b_out * m_b_out - h_b_in * m_b_in

    # fraction of fuel going to gas energy (hopefully 99.99%)
    fraction_gas_burner = P_b_gas / P_f_b

    # mass balance piston
    m_balance_p = m_p_out - m_p_in - m_f_p

    # mass balance burner
    m_balance_b = m_b_out - m_b_in - m_f_b

    # calculate energy for hpc

    # energy for ipc

    # energy for fan

    # energy for outer fan

    # heat energy from piston to bypass stream

    # increase in thrust due to said energy
    station1 = ("fuel tank", "oil-fuel hx gain", "total fuel power")
    station2 = (
        "fuel power piston",
        "indicated power piston",
        "piston wall heat loss",
        "energy through exhaust (diff)",
        "gas energy increase piston",
    )

    station3 = ("fuel power burner", "gas energy increase")

    station4 = (
        "fuel tank",
        "fuel flow into piston",
        "air flow into piston",
        "mass flow out of piston",
        "mass flow balance",
        "mass flow into the burner",
        "fuel flow into burner",
        "mass flow out of burner",
        "mass balance burner",
    )

    masses = (
        np.array(
            [
                m_f,
                m_f_p,
                m_p_in,
                m_p_out,
                m_balance_p,
                m_b_in,
                m_f_b,
                m_b_out,
                m_balance_b,
            ]
        )
        * 1e3
    )

    powers_fuel_tank = np.array([P_f_t, P_f_hx, P_f]) * 1e-3
    perc_fuel_tank = np.array([P_f_t / P_f, P_f_hx / P_f, P_f / P_f]) * 100
    power_piston = (
        np.array([P_f_p, P_indicated, piston_heatloss, P_p_gas2, P_p_gas]) * 1e-3
    )
    perc_piston = (
        np.array(
            [1.0, fraction_indicated, fraction_wall, fraction_2, fraction_gas]
        )
        * 100
    )
    power_burner = np.array([P_f_b, P_b_gas]) * 1e-3
    perc_burner = np.array([P_f_b / P_f_b, fraction_gas_burner]) * 100

    data = pd.DataFrame(
        {
            "Station for total fuel": pd.Series(station1),
            "Power [kW]": pd.Series(powers_fuel_tank),
            "Percentage of LHV": pd.Series(perc_fuel_tank),
            "Station for piston fuel": pd.Series(station2),
            "Power2 [kW]": pd.Series(power_piston),
            "% of injected fuel energy": pd.Series(perc_piston),
            "Station for burner fuel": pd.Series(station3),
            "Power3 [kW]": pd.Series(power_burner),
            "% of fuel energy": pd.Series(perc_burner),
            "Station for mass flow": pd.Series(station4),
            "Mass flow [g/s]": pd.Series(masses),
        }
    )

    data.to_csv("simulation_data/energy_flow_fuel.csv")
    return
