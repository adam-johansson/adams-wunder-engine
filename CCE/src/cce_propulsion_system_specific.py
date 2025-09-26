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
    q_ngv = input['q_ngv']
    bpr_c = input['bpr_c']
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

    # Remove cooling flow
    m_cool = m3 * bpr_c
    m31 = m3 - m_cool  # core flow after cooling air is removed
    p31 = p3  # pressure of main flow
    T31 = T3  # temperature of main flow after removal of cooling flow

    p_loss_piston_in = 0.03
    p_loss_piston_out = 0.03


    if second_burner:
        p4 = p31 * (1-p_loss_piston_in) * pi_pe * (1-p_loss_piston_out) * (1 - dPcomb)
    else:
        p4 = p31 * (1-p_loss_piston_in) * pi_pe * (1-p_loss_piston_out)


    # Compress cooling flow
    if p4 > p3:
        pi_cool = p4 / p3
        p_cool, T_cool, P_cooling = components.compressor(
            T3, p3, m_cool * q_ngv, 0.85, pi_cool
        )
    else:
        P_cooling = 0.0
        T_cool = T3
        p_cool = p3


    # Piston intake duct pressure loss
    p32 = (1-p_loss_piston_in) * p31
    T32 = T31


    # Piston engine
    input_piston["far piston"] = far_piston
    input_piston["p_in"] = p32
    input_piston["T_in"] = T32
    input_piston["p_ratio"] = pi_pe
    input_piston["cr"] = cr
    input_piston["T_fuel"] = t_fuel
    input_piston["p_loss_in"] = p_loss_piston_in
    input_piston["p_loss_out"] = p_loss_piston_out

    power_req = P_hpc / eta_g + P_cooling + power_offtake
    # Piston engine powers the HPC + Cooling+ gearbox for hpc ( no shaft efficiency now) (circumv flow is within
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
        print("problem with piston engine matching")
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
    m35 = piston_output["m35"]
    T34 = piston_output["T34"]
    T35 = piston_output["T35"]
    far34 = piston_output["far34"]
    far35 = piston_output["far35"]
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


    fuel_flow_piston = m34 * far34

    # fraction of air led around engine (based on m31, after cooling flow is removed)
    bpr_piston = (m31 - m32) / m31

    equ35 = far35 / far_s


    if second_burner:
        if T4_req < T35:
            #print("T4 lower than T35")
            error = True
            output_dict = {
                "sfc": 999,
                "error": error,
                "error_type": "T4"

            }
            return output_dict

        # Second burner
        p4, T4, far_4 = components.burner(
            p35, T35, equ35, T4_req, dPcomb, eta_b, fuel_type, t_fuel=t_fuel
        )

        # fuel air ratio of added fuel
        far_burner = far_4 - far35

        # m31 is pure air. After cooling flow removed but before piston.
        m4 = (
            m31 + fuel_flow_piston + far_burner * m31
        )  # flow after burner. air + fuel piston + fuel burner

        # burner fuel flow
        fuel_flow_burner = far_burner * m31

        # fuel air ratio after burner and piston engine
        equ4 = far_4 / far_s

    else:
        # skipping second burner
        p4 = p35
        T4 = T35
        m4 = m35
        fuel_flow_burner = 0
        far_burner = 0
        far4 = far35

        equ4 = far4 / far_s




    # Low pressure turbine, powering fan and IPC
    # IPC + inner and outer fan (with gearbox efficiency) + everything shaft efficiency
    power_lpt = (P_lpc + (P_inner_fan + P_outer_fan) / eta_g) / eta_s
    p5, T46, T47, T5, m46, m5, equ46, equ5, error = components.turbine(
        T4,
        p4,
        m4,
        equ4,
        power_lpt,
        eta_s_lpt,
        fuel_type,
        cooling=True,
        t_cool=T_cool,
        m1_cool=m_cool,
        q_ngv=q_ngv,
    )

    if error:
        #print('Prob too high power demand on LPT')
        output_dict={
            "sfc": 999,
            "error": error,
            "error_type": "LPT"

        }
        return output_dict

    # Turbine exhaust duct pressure loss
    p6 = p5 * 0.99
    T6 = T5
    m6 = m5


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

    # NOx emission index
    EI_nox = m_nox / (mdot_fuel) * 1e3

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
                    p32,
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
            T32,
            T34,
            T35,
            T4,
            T4,
            T46,
            T5,
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
            m5,
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
                    far35,
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
            power_lpt,
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
            power_lpt,
            power_lpt * (1 - eta_s),
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
            T32,
            T34,
            p32,
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
        "T32": T32,
        "T34": T34,
        "T35": T35,
        "T4": T4,
        "far_piston": far34,
        "EI_nox": EI_nox,
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
        print(f"Warning: pmax {p_max*1e-5} bar larger than 250 bar")
    if T4 < T35:
        print(f"Warning: T4 {T4} smaller than T35 {T35}")
    if T34 > 1200:
        print(f"Warning: T_out {T34} of piston larger than 1200K")

    #print(p0*1e-5, p_max*1e-5, p_max/p0)
    #print(T_max)

    return output_dict
