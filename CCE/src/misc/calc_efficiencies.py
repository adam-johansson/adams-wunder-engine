import thermo


def calc_efficiencies_cce(
    F,
    mdot_fuel,
    mdot_fuel_pe,
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
    v_intercooler_id,
    T_intercooler,
    p_intercooler,
    m_intercooler,
    T26,
    p26, 
    m26,
    T35,
    p35, 
    m35,
    equ35,
    displacement,
    piston_indicated_power,
    v_mean,
    stroke,
    m_cool,
    heatloss_pe,
    total_friction_pe,
):
    sfc = mdot_fuel / F  # Thrust specific fuel consumption

    # Kinetic power ( cold stream + hot stream - intake stream)
    P_kin = 0.5 * (
        m_cold * v_cold_id**2
        + m_cold_hx * v_cold_hx_id**2
        + m_hot * v_hot_id**2
        + m_intercooler * v_intercooler_id ** 2
        - m_intake * v_0**2
    )

    # Thrust power
    P_thrust = F * v_0

    # Gas generator power
    WP_before_hpc = thermo.work_potential(T26, p26, 0.0, pa, fuel_type)

    WP_after_pe = thermo.work_potential(T35, p35, equ35, pa, fuel_type)


    #h26, _, _, _, _, _, _, _ = thermo.mixture(T26, p26, 0.0, fuel_type)
    #h35, _, _, _, _, _, _, _ = thermo.mixture(T35, p35, equ35, fuel_type)

    P_gg = WP_after_pe * m35 - WP_before_hpc * m26
    #P_gg = h35 * m35 - h26 * m26

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

    # temperature and pressure before intercooler and oil/piston engine cooler are the same

    # work potential before bypass hx
    WP_before_ic = thermo.work_potential(T_beforehx, p_beforehx, 0.0, pa, fuel_type)

    # work potential after bypass hx
    WP_after_ic = thermo.work_potential(T_intercooler, p_intercooler, 0.0, pa, fuel_type)

    # work potential increase due to heating the bypass
    P_ic = m_intercooler * (WP_after_ic - WP_before_ic)

    P_core2 = P_core + P_hx + P_ic


    P_core_spec = P_core / m_core_nofuel

    # Fuel power (piston engine + burner)
    P_fuel = mdot_fuel * LHV

    P_fuel_pe = mdot_fuel_pe * LHV

    # Propulsive efficiency
    eta_p = P_thrust / P_kin

    # Core efficiency
    eta_core = P_core / P_fuel
    eta_core2 = P_core2 / P_fuel

    # Gas generator efficency
    eta_gg = P_gg / P_fuel_pe

    # Mass specific gas generator power J/kg
    P_gg_spec_mass = P_gg / m26

    # Gas generator power per liter of piston engine displacement W / m3
    P_gg_spec_displacement = P_gg / displacement

    # Mean effective pressure
    # if twostroke change 0.5 to 1
    rps = v_mean / (2 * stroke)
    imep = piston_indicated_power / (0.5 * rps * displacement)

    #print(f"rpm: {rps * 60}, IMEP: {imep * 1e-5} bar")

    # percentage of fuel energy of piston engine goes to heat through the walls
    eta_heatloss = heatloss_pe / P_fuel_pe

    # perceantage of fuel energy of piston engine going to friction, oil pump and fuel pump
    eta_friction = total_friction_pe / P_fuel_pe

    cooling_ratio = m_cool / m26


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
        "gas generator eff": eta_gg,
        "overall eff": eta_o,
        "specific thrust": Fs,
        "core specific power": P_core_spec,
        "gas generator mass specific power": P_gg_spec_mass,
        "gas generator spec displacement": P_gg_spec_displacement,
        "GG power": P_gg,
        "cooling ratio": cooling_ratio,
        "heatloss percentage": eta_heatloss,
        "friction percentage": eta_friction,
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
