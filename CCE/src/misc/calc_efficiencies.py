import thermo


def calc_efficiencies_cce(
    F,
    mdot_fuel,
    m_cold,
    v_cold_id,
    m_cold_hx,
    v_cold_hx_id,
    m_hot,
    v_hot_id,
    m_intake,
    v_0,
    p_wp,
    T_wp,
    equ_wp,
    fuel_type,
    pa,
    LHV,
    m_core_nofuel,
    p_beforehx,
    T_beforehx,
    p_afterhx,
    T_afterhx,
):
    sfc = mdot_fuel / F  # Thrust specific fuel consumption

    # Kinetic power ( cold stream + hot stream - intake stream)
    P_kin = 0.5 * (
        m_cold * v_cold_id**2
        + m_cold_hx * v_cold_hx_id**2
        + m_hot * v_hot_id**2
        - m_intake * v_0**2
    )

    # Thrust power
    P_thrust = F * v_0

    # Core power

    #work potential of the core after powering itself
    WP_core = thermo.work_potential(
        T_wp, p_wp, equ_wp, pa, fuel_type
    )

    # 0.5 v_0^2 of the total properties before the inlet is the same as the work potential
    P_core = m_hot * WP_core - 0.5 * m_core_nofuel *  v_0**2


    # work potential before bypass hx
    WP_beforehx = thermo.work_potential(T_beforehx, p_beforehx, 0.0, pa, fuel_type)

    # work potential after bypass hx
    WP_afterhx = thermo.work_potential(T_afterhx, p_afterhx, 0.0, pa, fuel_type)

    # work potential increase due to heating the bypass
    P_hx = m_cold_hx * (WP_afterhx - WP_beforehx)

    P_core2 = P_core + P_hx

    #print(f"Pcore: {P_core}")
    #print(f"Pcore2: {P_core2}")
    #print(f"Phx: {P_hx}")


    P_core_spec = P_core / m_core_nofuel

    # Fuel power
    P_fuel = mdot_fuel * LHV

    # Propulsive efficiency
    eta_p = P_thrust / P_kin

    # Core efficiency
    eta_core = P_core / P_fuel
    eta_core2 = P_core2 / P_fuel

    #print(f"eta core: {eta_core*100}")
    #print(f"eta core2: {eta_core2*100}")

    # Transmission efficiency
    eta_transfer = P_kin / P_core

    # Thermal efficiency
    eta_th = eta_core * eta_transfer
    eta_o = eta_p * eta_th

    # Specific thrust
    Fs = F / m_intake

    output_dict = {
        "sfc": sfc,
        "core eff": eta_core,
        "transfer eff": eta_transfer,
        "thermal eff": eta_th,
        "propulsive eff": eta_p,
        "overall eff": eta_o,
        "specific thrust": Fs,
        "core specific power": P_core_spec
    }
    return output_dict


def calc_efficiencies_recuperated_h2_geared(
    F,
    mdot_fuel,
    m_cold,
    v_cold_id,
    m_hot,
    v_hot_id,
    m_rec,
    v_rec_id,
    m_intake,
    v_0,
    p_wp,
    T_wp,
    equ_wp,
    fuel_type,
    pa,
    LHV,
):
    sfc = mdot_fuel / F  # Thrust specific fuel consumption

    # Kinetic power ( cold stream + hot stream - intake stream)
    P_kin = 0.5 * (
        m_cold * v_cold_id**2
        + m_rec * v_rec_id**2
        + m_hot * v_hot_id**2
        - m_intake * v_0**2
    )

    # Thrust power
    P_thrust = F * v_0

    # Work potential of core after it has powered "itself", meaning inner fan, lpc and hpc are powered.
    WP_core = thermo.work_potential(
        T_wp, p_wp, equ_wp, pa, fuel_type
    )

    # Core power
    P_core = (m_hot + m_rec) * (
        WP_core - 0.5 * v_0**2
    )

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


def calc_efficiencies_jetA_geared(
    F,
    mdot_fuel,
    m_cold,
    v_cold_id,
    m_hot,
    v_hot_id,
    m_intake,
    v_0,
    p_wp,
    T_wp,
    equ_wp,
    fuel_type,
    pa,
    LHV,
):
    sfc = mdot_fuel / F  # Thrust specific fuel consumption

    # Kinetic power ( cold stream + hot stream - intake stream)
    P_kin = 0.5 * (
        m_cold * v_cold_id**2
        + m_hot * v_hot_id**2
        - m_intake * v_0**2
    )

    # Thrust power
    P_thrust = F * v_0

    # Work potential of core after it has powered "itself", meaning inner fan, lpc and hpc are powered.
    WP_core = thermo.work_potential(
        T_wp, p_wp, equ_wp, pa, fuel_type
    )

    # Core power
    P_core = m_hot * (
        WP_core - 0.5 * v_0**2
    )

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
