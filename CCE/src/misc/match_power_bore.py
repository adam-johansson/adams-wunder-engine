from piston_engine.src.misc import post_processing
from scipy.optimize import brentq
from CCE.src import components
from thermo import fuel_props
from neural_network.src import input_outside_limits

import numpy as np


def match_power_bore(input, meta_model, power_req, core_flow):


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

    cycle = input["cycle"]
    bsr = input["bsr"]
    fuel_type = input["fuel"]

    cylinders = 12 #assume this for friction calculations????
    nr_engines = 2

    # check input:
    piston_input = np.atleast_2d(
        np.array([pin, Tin, p_ratio, cr, 0.15, v_mean, T_fuel, far34])
    )

    if input_outside_limits(piston_input):
        output = {
            "error": True
        }
        return output

    # Store the last valid outputs
    last_outputs = {}

    def find_match(x):
        nonlocal last_outputs  # Allow modification of outer scope variable

        bore = x


        piston_input = np.atleast_2d(
            np.array([pin, Tin, p_ratio, cr, bore, v_mean, T_fuel, far34])
        )

        # use meta model to get outputs from the piston engine
        output = meta_model.inference(piston_input)[0]

        # mass flow of air into the engine
        m_in = output[7]

        # specific power and specific heat loss
        indicated_power = output[0]
        heat_loss = output[1]
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
        stroke = bore / bsr
        lv_max = bore * 0.1
        rpm = v_mean / (2 * stroke) * 60


        # auxiliary losses and friction losses (for 12 cylinders)
        friction_loss, aux_loss, _ = post_processing.friction_patton(
            bore, rpm, stroke, v_mean, pin, cr, cylinders, lv_max, cycle
        )

        # convert to one cylinder (this is per cylinder friction loss for a V12 cylinder engine)
        friction_loss = friction_loss / cylinders
        aux_loss = aux_loss / cylinders

        mdot_bypass = core_flow / (cylinders * nr_engines) - m_in

        # dont need compress negative pressure ratio
        if p_ratio * (1-p_loss_in) * (1-p_loss_out) <= 1:
            pressure_circ, T_circumv, P_circumv = 0.0, Tin, 0.0
        else:
            pressure_circ, T_circumv, P_circumv = components.compressor(
                Tin, pin / (1-p_loss_in), mdot_bypass, 0.85, p_ratio * (1 - p_loss_out) * (1-p_loss_in)
            )

        # piston output power
        shaft_power = indicated_power - aux_loss - friction_loss - P_fuel_pump

        # the total power left to power the HPC
        power_piston = shaft_power - P_circumv

        # Store all the calculated values
        last_outputs.update({
            'bore': bore,
            'output': output,
            'm_in': m_in,
            'indicated_power': indicated_power,
            'heat_loss': heat_loss,
            'nox_ppm': nox_ppm,
            'p_tdc': p_tdc,
            'p_max': p_max,
            'T_max': T_max,
            'T34': T34,
            'fuel_flow': fuel_flow,
            'friction_loss': friction_loss,
            'aux_loss': aux_loss,
            'mdot_bypass': mdot_bypass,
            'pressure_circ': pressure_circ,
            'T_circumv': T_circumv,
            'P_circumv': P_circumv,
            "P_fuel_pump": P_fuel_pump,
            'shaft_power': shaft_power,
            'power_piston': power_piston
        })

        # power out should match power required
        residual = np.array([power_piston - power_req / (nr_engines * cylinders)])
        return residual


    bore_match = brentq(find_match, 0.1, 0.2)

    # Now use the stored values from the last iteration
    mdot_in = last_outputs['m_in'] * nr_engines * cylinders
    T34 = last_outputs['T34']
    p34 = pin * p_ratio
    p_max = last_outputs["p_max"]
    T_max = last_outputs["T_max"]
    nox_ppm = last_outputs["nox_ppm"]
    T_circumv = last_outputs["T_circumv"]
    P_circumv = last_outputs["P_circumv"]
    power_piston = last_outputs["power_piston"] * nr_engines * cylinders
    heat_loss = last_outputs["heat_loss"] * nr_engines * cylinders
    indicated_power = last_outputs["indicated_power"] * nr_engines * cylinders
    friction_loss = last_outputs["friction_loss"] * nr_engines * cylinders
    aux_loss = last_outputs["aux_loss"] * nr_engines * cylinders
    P_fuel_pump = last_outputs["P_fuel_pump"] * nr_engines * cylinders

    mdot_bypass = core_flow - mdot_in

    fraction = mdot_in / core_flow

    far_s, LHV = fuel_props(fuel_type)

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

    # Calculate displacement
    stroke = bore_match / bsr  # using bore-to-stroke ratio
    displacement = np.pi / 4 * bore_match ** 2 * stroke  # m³
    displacement_per_cyl = displacement / 24 # liters

    output_dict={
        "power_net": power_piston,
        "power_indicated": indicated_power,
        "heat loss": heat_loss,
        "aux loss": aux_loss,
        "friction loss": friction_loss,
        "fuel pump": P_fuel_pump,
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
        "bore": bore_match,
        "displacement": displacement_per_cyl,
        "error": False,

    }

    return output_dict