import thermo


def calc_efficiencies(F, mdot_fuel, m_cold, v_cold_id, m_cold_hx, v_cold_hx_id, m_hot, v_hot_id, m_intake, v_0, p_wp,
                      T_wp, equ_wp, fuel_type, pa, LHV):
    sfc = mdot_fuel / F  # Thrust specific fuel consumption

    # Kinetic power ( cold stream + hot stream - intake stream)
    P_kin = 0.5 * (m_cold * v_cold_id ** 2 + m_cold_hx * v_cold_hx_id ** 2 +
                   m_hot * v_hot_id ** 2 - m_intake * v_0 ** 2)

    # Thrust power
    P_thrust = F * v_0

    # Core power

    WP_core = thermo.work_potential(T_wp, p_wp, equ_wp, pa, fuel_type)  # pa or p0 as ambient? Should be pa I think
    P_core = m_hot * (WP_core - 0.5 * v_0 ** 2)  # use flow before or after adding fuel? cooling?

    # Fuel power
    P_fuel = mdot_fuel * LHV

    # Propulsive efficiency
    eta_p = P_thrust / P_kin

    # Core efficiency
    eta_core = P_core / P_fuel

    # Transmission efficiency
    eta_transmission = P_kin / P_core

    # Thermal efficiency
    eta_th = eta_core * eta_transmission
    eta_o = eta_p * eta_th

    # Specific thrust
    Fs = F / m_intake

    return sfc, eta_core, eta_transmission, eta_th, eta_p, eta_o, Fs
