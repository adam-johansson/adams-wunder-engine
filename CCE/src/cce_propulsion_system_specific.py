import numpy as np
import importlib
from timeit import default_timer as timer

from CCE.src import components, compressible, misc
from CCE.src.gas_props.air_properties import isa

from thermo import fuel_props, work_potential


def run_cce(input, input_piston, flags, meta_model):

    Fn = input['Fn']
    dTisa = input['dTisa']
    bpr = input['bpr']
    T4_req = input['T4']
    fpr_outer = input['fpr_outer']
    Fs_req = input['Fs_req']
    dp_intake = input['dp_intake']
    dp_bypass = input['dp_bypass']
    Mach = input['M']
    eta_fan = input['eta_fan']
    eta_p_lpc = input['eta_p_lpc']
    eta_p_hpc = input['eta_p_hpc']
    eta_b = input['eta_b']
    dPcomb = input['dPcomb']
    eta_s = input['eta_s']
    eta_g = input['eta_g']
    eta_s_lpt = input['eta_lpt']
    cfg_core = input['cfg_core']
    cfg_bypass = input['cfg_bypass']
    cd_nozzle = input['cd_nozzle']
    alt = input['alt']
    fuel_type = input['fuel']
    pi_pe = input['pi_pe']
    surrogate = input['surrogate']
    cr = input['cr']
    OPR = input['OPR']
    PR = input['PR']
    bore = input['bore']
    second_burner = input['second_burner']
    t_fuel = input['t_fuel']
    t_tank = input['t_tank']
    power_offtake = input['power_offtake']
    far_piston = input['far piston']
    effectiveness_IC = input['effectiveness IC']
    dp_inter_compressor = input['dp_inter_compressor']
    v_mean = input['v_mean']

    error = False
    minor_error_mass = False
    minor_error_pressure = False

    # number of output variables from function
    outputs = 13

    if input["specific"]:
        # unit mass flow
        m0 = 1.0
    else:
        m0 = Fn / Fs_req

    # calculate pressure ratios based on OPR and pressure ratio split
    fpr_inner = 0.913 * fpr_outer
    OPR_c = OPR / fpr_inner  # OPR of the compressor, excluding fan
    OPR_c = OPR_c / (1-dp_inter_compressor) #account for the pressure loss after LPC
    pi_ipc = OPR_c**PR
    pi_hpc = OPR_c / pi_ipc

    # fuel props
    far_s, LHV = fuel_props(fuel_type)

    # ISA table
    pa, Ta, a = isa(
        alt, dTisa, False
    )  # static pressure, static temperature and speed of sound

    # calculate optimum fpr from Gouya paper (Unsure if this works for me)
    # fpr_outer = misc.fpr_opt(Mach, bpr, Fs_req, Ta, eta_s_lpt, eta_outer_fan, cfg_bypass, cd_nozzle, dp_bypass)

    # Total properties
    p0, T0 = compressible.stagnation(pa, Ta, Mach)

    # Inlet
    p2, T2 = components.inlet(p0, T0, dp_intake)
    m2 = m0

    # Split core and bypass mass flow
    m21 = m2 / (1 + bpr)  # core flow
    m13 = m2 * bpr / (1 + bpr)  # bypass stream

    # Convert polytropic efficiency to isentropic for the fan

    # Inner fan
    p21, T21, P_inner_fan = components.compressor_isentropic(T2, p2, m21, eta_fan, fpr_inner)

    # Outer fan
    p13, T13, P_outer_fan = components.compressor_isentropic(T2, p2, m13, eta_fan, fpr_outer)

    # Compressor intake loss
    p22 = p21


    # LPC
    p25, T25, P_lpc = components.compressor(T21, p22, m21, eta_p_lpc, pi_ipc)
    m25 = m21

    # Inter compressor loss DO I NEED THIS??
    p255 = p25 * (1-dp_inter_compressor)

    # Intercooler
    if input["intercooler"]:

        p26, T26, p_intercooler, T_intercooler, m_intercooler =\
            components.intercool(p255, T25, m25, p13, T13, effectiveness_IC)

        #print(f"Delta T intercooler: {T25 - T26}")
        dT_intercooler = T25 - T26

    else:
        p26 = p255
        T26 = T25
        p_intercooler = p13
        T_intercooler = T13
        m_intercooler = 0.0
        dT_intercooler = 0.0

    # HPC
    p3, T3, P_hpc = components.compressor(T26, p26, m25, eta_p_hpc, pi_hpc)
    m3 = m25

    p_loss_piston_in = 0.03
    p_loss_piston_out = 0.0


    # Piston intake duct pressure loss
    p31 = (1-p_loss_piston_in) * p3
    T31 = T3
    m31 = m3


    # Piston engine
    input_piston["far piston"] = far_piston
    input_piston["p_in"] = p31
    input_piston["T_in"] = T31
    input_piston["p_ratio"] = pi_pe
    input_piston["cr"] = cr
    input_piston["v_mean"] = v_mean
    input_piston["T_fuel"] = t_fuel
    input_piston["p_loss_in"] = p_loss_piston_in
    input_piston["p_loss_out"] = p_loss_piston_out

    power_req = P_hpc / eta_g + power_offtake
    # Piston engine powers the HPC + gearbox for hpc ( no shaft efficiency now) (circumv flow is within
    # the piston model)


    if input["specific"]:
        piston_output = misc.match_power_specific(
            input_piston,
            meta_model,
            power_req,
            core_flow=m31,
        )

    else:
        piston_output = misc.match_power_bore(
            input_piston,
            meta_model,
            power_req,
            core_flow=m31,
        )


    # if engine was not able to match power requirements, negative air flow or input outside surrogate limits, return error
    if piston_output["error"]:
        error = True
        #print("problem with piston engine matching")
        output_dict = {
            "sfc": 999,
            "error": error,
            "error_type": "PISTON"

        }
        return output_dict


    p34 = piston_output["p34"]
    p35 = piston_output["p35"]
    m32 = piston_output["m32"]
    m34 = piston_output["m34"]
    #m35 = piston_output["m35"]
    T34 = piston_output["T34"]
    T_cool = piston_output["T_circumv"]
    #T35 = piston_output["T35"]
    far34 = piston_output["far34"]
    #far35 = piston_output["far35"]
    m_nox = piston_output["m NO"]
    friction_loss_pe = piston_output["friction loss"]
    P_circumv = piston_output["P comp circumvent air"]
    piston_aux_loss = piston_output["aux loss"]
    piston_heat_loss = piston_output["heat loss"]
    P_fuel_pump = piston_output["fuel pump"]
    p_max = piston_output["p max"]
    T_max = piston_output["T max"]
    piston_power_net = piston_output["power_net"] #power of the piston shaft after friction, aux, fuel pump
    # and compressing circumventing air
    piston_indicated_p = piston_output["power_indicated"]
    #displacement = piston_output["tot engine displacement"]


    # pure air massflow to the engine (m32) is the basis for the fuel air ratio
    fuel_flow_piston = m32 * far34

    # fraction of air led around engine (based on m31)
    bpr_piston = (m31 - m32) / m31

    # power requirement on the LPT
    power_required_LPT = (P_lpc + (P_inner_fan + P_outer_fan) / eta_g) / eta_s

    input_burner_turbine = {
        "m31": m31,
        "m32": m32,
        "m34": m34,
        "T_cooling": T_cool,
        "T34": T34,
        "T4_req": T4_req,
        "far34": far34,
        "fuel_type": fuel_type,
        "T_fuel": t_fuel,
        "dP_comb": dPcomb,
        "eta_b": eta_b,
        "p34": p34,
        "power_req": power_required_LPT,
        "eta_s_lpt": eta_s_lpt,
        "second burner": second_burner,
    }

    output_burner_turbine = components.burner_turbine(input_burner_turbine)

    # if error in burner_turbine
    if output_burner_turbine["error"]:
        error = True
        #print("problem with piston engine matching")
        output_dict = {
            "sfc": 999,
            "error": error,
            "error_type": "PISTON"

        }
        return output_dict

    # number 6 is just after 1% pressure loss after turbine duct
    equ35 = output_burner_turbine["equ35"]
    equ4 = output_burner_turbine["equ4"]
    equ46 = output_burner_turbine["equ46"]
    equ5 = output_burner_turbine["equ5"]
    m35 = output_burner_turbine["m35"]
    m4 = output_burner_turbine["m4"]
    m46 = output_burner_turbine["m46"]
    m6 = output_burner_turbine["m5"]
    p4 = output_burner_turbine["p4"]
    p5 = output_burner_turbine["p5"]
    p6 = output_burner_turbine["p6"]
    T35 = output_burner_turbine["T35"]
    T4 = output_burner_turbine["T4"]
    # T46 is after ngv cooling
    T46 = output_burner_turbine["T46"]
    # T47 is after rotor cooling
    T47 = output_burner_turbine["T47"]
    T6 = output_burner_turbine["T5"]
    fuel_flow_burner = output_burner_turbine["fuel_flow_burner"]
    m_cool = output_burner_turbine["m_cool"]
    q_ngv = output_burner_turbine["q_ngv"]

    # Hot nozzle
    equ6 = equ5
    m8 = m6
    F8, v8_id, v8, error = components.nozzle(
        p6, T6, pa, equ6, m6, cfg_core, cd_nozzle, fuel_type
    )

    if error:
        #print('Hot nozzle fails. Prob too low pressure.')
        output_dict={
            "sfc": 999,
            "error": error,
            "error_type": "Hot nozzle",
            "error_info": pa-p6

        }
        return output_dict

    # Heating the fuel
    if fuel_type == "H2":
        heating_fuel, oil_temp_1 = components.fuel_oil_hx(
            300e5, t_tank, 0.08, t_fuel, fuel_flow_piston + fuel_flow_burner
        )
    else:
        heating_fuel = 0.0
        oil_temp_1 = 400


    # Bypass stream
    # Split stream into cooling air and not cooling air
    # total amount of heat produced by piston engine
    cooling_req = (
        piston_heat_loss + friction_loss_pe
    )

    # total amount of heat needed to be rejected (some heat used to heat fuel)
    heating_bypass = (
        cooling_req - heating_fuel
    )


    # Oil to air heat exchanger in the bypass
    p15, T15, m15, m_oil = components.hx_NASA(
        p13, T13, heating_bypass, oil_temp_1
    )

    #print(f"Mass flow of engine oil: {m_oil} kg/s")

    m14 = m13 - m15 - m_intercooler  # mass flow not going through heat exchanger
    p14 = (1 - dp_bypass) * p13
    T14 = T13  # adiabatic bypass

    # Cold nozzle ( no heat exchanger)
    m18 = m14
    F18, v18_id, v18, error = components.nozzle(
        p14, T14, pa, equ=0, m=m18, cfg=cfg_bypass, cd=cd_nozzle, fuel_type=fuel_type
    )
    if error:
        #print('Prob too low pressure and temperature in cold nozzle')
        cost = 999 + (pa - p14)
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros

    # Cooling flow nozzle
    m17 = m15
    F17, v17_id, v17, error = components.nozzle(
        p15, T15, pa, equ=0, m=m17, cfg=cfg_bypass, cd=cd_nozzle, fuel_type=fuel_type
    )

    if error:
        #print('Prob too low pressure and temperature in cooling nozzle')
        cost = 999 + (pa - p15)
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros

    # Intercooler nozzle
    F_ic_nozzle, v_ic_nozzle_id, v_ic_nozzle, error = components.nozzle(
        p_intercooler, T_intercooler, pa, equ=0, m=m_intercooler, cfg=cfg_bypass, cd=cd_nozzle, fuel_type=fuel_type
    )

    if error:
        #print('Prob too low pressure and temperature in cooling nozzle')
        cost = 999 + (pa - p15)
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros

    # Thrust
    v_0 = a * Mach  # air speed
    F = F8 + F18 + F17 + F_ic_nozzle - v_0 * m2  # net thrust

    # Total fuel flow
    mdot_fuel = fuel_flow_piston + fuel_flow_burner

    # NOx emission index from piston engine
    EI_nox_PE = 1e3 * m_nox / (mdot_fuel)

    if second_burner:
        # NOx emission index from burner
        EI_nox_burner = 0.007549 * T4 * (p35*1e-3 /3027)**0.37 * np.exp((1.8*T35 - 1471)/345) * (fuel_flow_burner/mdot_fuel)
    else:
        EI_nox_burner = 0.0

    # total
    EI_nox = EI_nox_PE + EI_nox_burner

    # Ideal jet velocity ratio NOT VALID ANYMORE
    vel_ratio = v17_id / v8_id
    #print(vel_ratio)

    power_hpc = (
        P_hpc / eta_g
    )  # power required by the hpc divided by mechanical efficiency (from piston engine/HPT)

    # Calculating the work potential left after powering lpc and inner fan
    p_wp, _, _, T_wp, _, m_wp, _, equ_wp, error = components.turbine(
        T4,
        p4,
        m4,
        equ4,
        (P_lpc + P_inner_fan / eta_g) / eta_s,
        eta_s_lpt,
        fuel_type,
        cooling=True,
        t_cool=T_cool,
        m1_cool=m_cool,
        q_ngv=q_ngv,
    )

    # Efficiencies
    eff_dict = misc.calc_efficiencies_cce(
        F,
        mdot_fuel,
        m14,
        v18_id,
        m15,
        v17_id,
        m6,
        v8_id,
        m0,
        v_0,
        p_wp,
        T_wp,
        equ_wp,
        fuel_type,
        pa,
        LHV,
        m21,
        p13,
        T13,
        p15,
        T15,
        v_ic_nozzle_id,
        T_intercooler,
        p_intercooler,
        m_intercooler,
    )


    sfc = eff_dict["sfc"]
    eta_core = eff_dict["core eff"]
    eta_transfer = eff_dict["transfer eff"]
    eta_th = eff_dict["thermal eff"]
    eta_p = eff_dict["propulsive eff"]
    eta_o = eff_dict["overall eff"]
    Fs = eff_dict["specific thrust"]
    core_spec_power = eff_dict["core specific power"]

    # calculate piston engine displacement based on real mass flow
    if input["specific"]:
        m_real = Fn / Fs
        m32_real = m32 * m_real
        m32_per_cylinder = m32_real / 24

        slope = piston_output["slope"]
        intercept = piston_output["intercept"]

        bore = np.sqrt((m32_per_cylinder - intercept) / slope)

        bsr = input_piston["bsr"]

        # Calculate displacement
        stroke = bore / bsr  # using bore-to-stroke ratio
        displacement = np.pi / 4 * bore ** 2 * stroke  # m³
        displacement_tot = displacement * 24
    else:
        bore = piston_output["bore"]
        displacement = piston_output["displacement"]
        displacement_tot = displacement * 24
        m32_real = m32



    if "print_output" in flags:

        # Creating array for output
        p_array = (
            np.array(
                [
                    p2,
                    p13,
                    p14,
                    p15,
                    p21,
                    p25,
                    p3,
                    p31,
                    p31,
                    p34,
                    p35,
                    p4,
                    p4,
                    p5,
                    p5,
                    p6,
                ]
            )
            * 1e-3
        )
        T_array = (
            T2,
            T13,
            T14,
            T15,
            T21,
            T25,
            T3,
            T31,
            T31,
            T34,
            T35,
            T4,
            T4,
            T46,
            T6,
            T6,
        )
        m_array = (
            m2,
            m13,
            m14,
            m15,
            m21,
            m25,
            m3,
            m31,
            m32,
            m34,
            m35,
            m4,
            m4,
            m46,
            m6,
            m6,
        )

        far_array = (
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    far34,
                    equ35 * far_s,
                    equ4 * far_s,
                    equ46 * far_s,
                    equ46 * far_s,
                    equ5 * far_s,
                    equ5 * far_s,
                    equ6 * far_s,
                ]
            )
        )

        s_array = misc.entropy_array(p_array, T_array, far_array, fuel_type)

        misc.print_output(
            m32,
            m32_real,
            sfc,
            F,
            m0,
            m21,
            piston_power_net,
            power_hpc,
            power_offtake,
            power_required_LPT,
            p3,
            T3,
            p35,
            T34,
            T4,
            p_max,
            T_max,
            far34 / far_s,
            far_s,
            piston_indicated_p,
            fuel_flow_piston,
            fuel_flow_burner,
            displacement_tot,
            bore,
            friction_loss_pe,
            piston_aux_loss,
            piston_heat_loss,
            bpr_piston,
            m_nox,
            fpr_outer,
            bpr,
        )

        misc.print_efficiencies(eta_o, eta_p, eta_th, eta_transfer, eta_core, Fs)

        misc.plot_stations_cce(p_array, T_array)

        misc.csv_output_cce(p_array, T_array, m_array, far_array, s_array)
        # power lost in the compressor gearbox
        # assuming offtake is taken before gearbox


        # I dont divide the power needed to compress circumventing air and cooling air right now
        P_cooling = 0.0

        piston_gearbox = (
            piston_indicated_p
            - P_fuel_pump
            - P_circumv
            - friction_loss_pe
            - piston_aux_loss
            - P_cooling
            - power_offtake
        ) * (1 - eta_g)
        misc.power_balance(
            piston_indicated_p,
            friction_loss_pe,
            piston_aux_loss,
            P_fuel_pump,
            P_circumv,
            P_cooling,
            piston_gearbox,
            power_offtake,
            P_hpc,
            power_required_LPT,
            power_required_LPT * (1 - eta_s),
            P_lpc,
            (P_inner_fan + P_outer_fan) * (1 - eta_g) / eta_g,
            P_inner_fan + P_outer_fan,
            piston_heat_loss,
            heating_fuel,
            heating_bypass,
        )

        misc.energy_flow_fuel(
            LHV,
            fuel_flow_piston,
            fuel_flow_burner,
            t_tank,
            fuel_type,
            t_fuel,
            piston_indicated_p,
            piston_heat_loss,
            T31,
            T34,
            p31,
            p34,
            m32,
            m34,
            far34 / far_s,
            T35,
            equ35,
            m35,
            T4,
            equ4,
            m4,
            pa,
        )

        # print(f'ideal jet velocity ratio: {vel_ratio}')

        # cost function if minor errors NOT USED FOR NOW
        #if minor_error_mass:
        #    punish_factor = 1e-4
        #    sfc = sfc - punish_factor * m_circumv

        if minor_error_pressure:
            punish_factor = 1e-4
            print(p_max)
            sfc = sfc + punish_factor * (p_max - 300)
    # print(f'FAR: {far_piston}, T_before: {T32}, T_after: {T34}, bore: {bore_match} ')

    output_dict = {
        "sfc": sfc,
        "vel_ratio": vel_ratio,
        "thrust": F,
        "specific thrust": Fs,
        "core specific power": core_spec_power,
        "mass flow": m0,
        "p_max": p_max,
        "T_max": T_max,
        "T31": T31,
        "T34": T34,
        "T35": T35,
        "T4": T4,
        "far_piston": far34,
        "EI_nox": EI_nox,
        "EI_nox_PE": EI_nox_PE,
        "EI_nox_burner": EI_nox_burner,
        "core efficiency": eta_core,
        "transmission efficiency": eta_transfer,
        "thermal efficiency": eta_th,
        "propulsive efficiency": eta_p,
        "overall efficiency": eta_o,
        "cold bypass thrust": F18,
        "hot bypass thrust": F17,
        "core thrust": F8,
        "piston fuelflow": fuel_flow_piston,
        "burner fuelflow": fuel_flow_burner,
        "dT intercooler": dT_intercooler,
        "engine displacement": displacement,
        "error": error
    }

    if p_max > 250*1e5:
        #print(f"Warning: pmax {p_max*1e-5} bar larger than 250 bar")
        output_dict["error_type"] = "p_max"
    if T4 < T35:
        #print(f"Warning: T4 {T4} smaller than T35 {T35}")
        output_dict["error_type"] = "T4"
    if T34 > 1200:
        #print(f"Warning: T_out {T34} of piston larger than 1200K")
        output_dict["error_type"] = "T34"

    #print(p0*1e-5, p_max*1e-5, p_max/p0)
    #print(T_max)

    return output_dict
