from piston_engine.src.misc import post_processing
from scipy.optimize import brentq
from CCE.src import components
from thermo import fuel_props

import numpy as np


def match_power_specific(input, meta_model, power_req, core_flow):


    # calculate the specific power of the piston engine
    # first specific indicated power and thereafter remove auxiliary and friction losses
    # finally match how big mass flow is needed to match the power requirements

    # input to surrogate
    pin = input["p_in"]
    Tin = input["T_in"]
    cr = input["cr"]
    p_ratio = input["p_ratio"]
    v_mean = input["v_mean"]
    T_fuel = input["T_fuel"]
    p_loss_in = input["p_loss_in"]
    p_loss_out = input["p_loss_out"]
    far34 = input["far piston"]


    cylinders = 12 #assume this for friction calculations????
    bore = 0.15  # assume bore size??? we fix this and then make everything specific


    piston_input = np.atleast_2d(
        np.array([pin, Tin, p_ratio, cr, bore, v_mean, T_fuel, far34])
    )

    # use meta model to get outputs from the piston engine
    output = meta_model.inference(piston_input)[0]

    # mass flow of air into the engine
    m_in = output[7]

    # specific power and specific heat loss
    indicated_power = output[0] / m_in
    heat_loss = output[1] / m_in
    nox_ppm = output[2]
    p_tdc = output[3]
    p_max = output[4]
    T_max = output[5]
    T34 = output[6]
    p34 = pin * p_ratio

    fuel_type = input["fuel"]
    far_s, LHV = fuel_props(fuel_type)

    #specific fuel flow (since fuel_flow_tot = far34 * m_in)
    fuel_flow = far34 * m_in

    # specific power needed to pressurise the fuel
    P_fuel_pump = components.fuel_pump(p_tdc, fuel_type, fuel_flow)

    # things needed for aux and friction losses
    bsr = input["bsr"]
    stroke = bore / bsr
    lv_max = bore * 0.1
    rpm = v_mean / (2 * stroke) * 60
    cycle = input["cycle"]

    # auxiliary losses and friction losses (for 12 cylinders)
    friction_loss, aux_loss, _ = post_processing.friction_patton(
        bore, rpm, stroke, v_mean, pin, cr, cylinders, lv_max, cycle
    )

    # convert to one cylinder (this is per cylinder friction loss for a V12 cylinder engine)
    friction_loss = friction_loss / cylinders
    aux_loss = aux_loss / cylinders

    # make it specific
    friction_loss = friction_loss / m_in
    aux_loss = aux_loss / m_in



    def find_match(x):

        # fraction of flow going into the piston engine
        piston_flow_fraction = x

        mdot_in = piston_flow_fraction * core_flow

        mdot_bypass = core_flow - mdot_in

        # dont need compress negative pressure ratio
        if p_ratio * (1-p_loss_in) * (1-p_loss_out) <= 1:
            pressure_circ, T_circumv, P_circumv = 0.0, Tin, 0.0
        else:
            pressure_circ, T_circumv, P_circumv = components.compressor(
                Tin, pin / (1-p_loss_in), mdot_bypass, 0.85, p_ratio * (1 - p_loss_out) * (1-p_loss_in)
            )

        # the specific piston output power (per kg/s of air into the piston engine)
        shaft_power_specific = indicated_power - aux_loss - friction_loss - P_fuel_pump
        # total power (not specific)
        shaft_power = shaft_power_specific * mdot_in

        # the total power left to power the HPC
        power_piston = shaft_power - P_circumv

        # power out should match power required
        residual = np.array([power_piston - power_req])
        return residual


    fraction = brentq(find_match, 0, 1)
    print(fraction)

    # mass flow into the piston engine that matches HPC power
    mdot_in = fraction * core_flow

    mdot_bypass = core_flow - mdot_in

    # dont need compress negative pressure ratio
    if p_ratio * (1 - p_loss_in) * (1 - p_loss_out) <= 1:
        p_ratio_circ = 1
        pressure_circ, T_circumv, P_circumv = 0.0, Tin, 0.0
    else:
        pressure_circ, T_circumv, P_circumv = components.compressor(
            Tin, pin / (1 - p_loss_in), mdot_bypass, 0.85, p_ratio * (1 - p_loss_out) * (1 - p_loss_in)
        )

    # the specific piston output power
    shaft_power_specific = indicated_power - aux_loss - friction_loss - P_fuel_pump

    # the total power
    shaft_power = shaft_power_specific * mdot_in

    # power after compressing bypass air
    power_net = shaft_power - P_circumv

    #total indicated power
    indicated_power_tot = indicated_power * mdot_in

    # total aux
    aux_loss_tot = aux_loss * mdot_in

    # total fuel pump
    P_fuel_pump_tot = P_fuel_pump * mdot_in

    # total friction
    friction_loss_tot = friction_loss * mdot_in

    # total heat loss
    heat_loss_tot = heat_loss * mdot_in

    # mix circumventing flow
    equ34 = far34 / far_s
    fuel_flow = far34 * mdot_in
    m34 = mdot_in + fuel_flow  # outflow of piston engine (air + fuel)
    m35 = m34 + mdot_bypass # flow after mixing
    T35, equ35 = components.mix(
        m34,
        T34,
        equ34,
        mdot_bypass,
        T_circumv,
        equ2=0,
        fuel_type=fuel_type,
    )

    far35 = equ35 * far_s

    p35 = p34 * (1- p_loss_out)

    m_NOX = nox_ppm * m34 * 1e-6

    output_dict={
        "power_net": power_net,
        "power_indicated": indicated_power_tot,
        "heat loss": heat_loss_tot,
        "aux loss": aux_loss_tot,
        "friction loss": friction_loss_tot,
        "fuel pump": P_fuel_pump_tot,
        "P comp circumvent air": P_circumv,
        "piston flow fraction": fraction,
        "T34": T34,
        "T35": T35,
        "p34": p34,
        "p35": p35,
        "m32": mdot_in,
        "m34": m34,
        "m35": m35,
        "far34": far34,
        "far35": far35,
        "p max": p_max,
        "T max": T_max,
        "m NO": m_NOX,

    }

    return output_dict