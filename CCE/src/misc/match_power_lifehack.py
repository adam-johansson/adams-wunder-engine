from piston_engine.src.misc import post_processing
from scipy.optimize import brentq
from CCE.src import components
from thermo import fuel_props, JETA_L, H2, mixture
from neural_network.src import input_outside_limits
from piston_engine.engine import run_piston_engine


import numpy as np


def match_power_lifehack(input, power_req, core_flow, life_hack):


    # calculate the specific power of the piston engine
    # first specific indicated power and thereafter remove auxiliary and friction losses
    # finally match how big mass flow is needed to match the power requirements

    far34 = input["far_goal"]

    if life_hack == "Simulate":
        # calculate stuff for bore between 10 and 20 cm

        flags = ["sweep"]


        bore_mid = 0.15
        bore_high = 0.20

        input["bore"] = bore_high

        (
            T_out_high,
            _,
            eta_th_high,
            _,
            _,
            _,
            _,
            _,
            indicated_power_high,
            _,
            _,
            heat_loss_high,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = run_piston_engine(input, flags)

        #print("Sim 1 done")
        #print(f"Heat loss sim 1: {heat_loss_high*24*1e-3} kW")

        input["bore"] = bore_mid

        (
            T_out_mid,
            _,
            eta_th_mid,
            m_in,
            p_max,
            T_max,
            _,
            _,
            indicated_power,
            _,
            _,
            heat_loss_mid,
            p_tdc,
            _,
            nox_ppm,
            _,
            EI_nox,
            _,
            nox_spec,
            _
        ) = run_piston_engine(input, flags)

        #print("Sim 2 done")
        #print(f"pmax: {p_max*1e-5} bar ")
        #print(f"Heat loss sim 2: {heat_loss_mid*24*1e-3} kW")
        #print(f"Power sim 2: {indicated_power*24*1e-3} kW")
        #print(f"m sim 2: {m_in*24} kg/s")
        #print(f"Fuel flow sim 2: {m_in * far34} kW")
        #print(f"thermal eff sim 2: {eta_th_mid*100} %")


        specific_power = indicated_power / m_in


        ### assume mdot = k*bore**2
        k_m = m_in / bore_mid ** 2


        # try to model heat loss and T out
        # assume T_out = c0 + c1*bore

        k1_T = (T_out_high - T_out_mid) / (bore_high - bore_mid)
        k0_T = T_out_mid - k1_T * bore_mid

        # assume heat_loss = k0 + k1*bore**2
        k1_H = (heat_loss_high - heat_loss_mid) / (bore_high ** 2 - bore_mid ** 2)
        k0_H = heat_loss_mid - k1_H * bore_mid ** 2

    elif life_hack == "Simulate_final":
        # calculate NOX for bore that was matching
        flags = ["sweep"]

        (
            _,
            _,
            _,
            _,
            p_max,
            T_max,
            _,
            _,
            _,
            _,
            _,
            heat_loss_final_sim,
            p_tdc,
            _,
            nox_ppm,
            _,
            _,
            _,
            _,
            T_max_twozone,
        ) = run_piston_engine(input, flags)

        #print("Final sim done")
        #print(f"Heta loss final sim: {heat_loss_final_sim*24*1e-3} kW")
        specific_power = input["piston_specific_power"]
        k_m = input["k_m"]
        k1_T = input["k1_T"]
        k0_T = input["k0_T"]
        k1_H = input["k1_H"]
        k0_H = input["k0_H"]


    else:
        #import specific power and coefficients
        specific_power = input["piston_specific_power"]
        k_m = input["k_m"]
        k1_T = input["k1_T"]
        k0_T = input["k0_T"]
        k1_H = input["k1_H"]
        k0_H = input["k0_H"]
        nox_ppm = 99999999
        p_max = 1e5
        T_max = 999



    pin = input["p_in"]
    Tin = input["T_in"]
    cr = input["cr"]
    p_ratio = input["p_ratio"]
    v_mean = input["v_mean"]
    T_fuel = input["T_fuel"]
    p_loss_in = input["p_loss_in"]
    p_loss_out = input["p_loss_out"]
    far34 = input["far_goal"]

    cycle = input["cycle"]
    bsr = input["bsr"]
    fuel_type = input["fuel"]

    far_stoich, LHV = fuel_props(fuel_type)

    if fuel_type == "jetA":
        p_tdc = 10e5


    cylinders = 12 #assume this for friction calculations????
    nr_engines = 2

    # Store the last valid outputs
    last_outputs = {}

    def find_match(x):
        nonlocal last_outputs  # Allow modification of outer scope variable

        bore = x

        # approximate mass flow and power
        approx_m_in = bore * bore * k_m
        approx_power = approx_m_in * specific_power

        approx_T34 = k0_T + k1_T * bore


        #fuel flow (since fuel_flow_tot = far34 * m_in)
        fuel_flow = far34 * approx_m_in


        # power needed to pressurise the fuel
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

        # this is per cylinder
        mdot_bypass = core_flow / (cylinders * nr_engines) - approx_m_in

        # dont need compress negative pressure ratio
        if p_ratio * (1-p_loss_in) * (1-p_loss_out) <= 1:
            pressure_circ, T_circumv, P_circumv = 0.0, Tin, 0.0
        else:
            pressure_circ, T_circumv, P_circumv = components.compressor(
                Tin, pin / (1-p_loss_in), mdot_bypass, 0.85, p_ratio * (1 - p_loss_out) * (1-p_loss_in)
            )

        # piston output power
        shaft_power = approx_power - aux_loss - friction_loss - P_fuel_pump

        # the total power left to power the HPC
        power_piston = shaft_power - P_circumv

        #print(approx_power, aux_loss, friction_loss, P_fuel_pump, P_circumv)

        # Store all the calculated values
        last_outputs.update({
            'bore': bore,
            'm_in': approx_m_in,
            'indicated_power': approx_power,
            'T34': approx_T34,
            'friction_loss': friction_loss,
            'aux_loss': aux_loss,
            'pressure_circ': pressure_circ,
            'T_circumv': T_circumv,
            'P_circumv': P_circumv,
            "P_fuel_pump": P_fuel_pump,
            'shaft_power': shaft_power,
            'power_piston': power_piston
        })

        # power out should match power required
        residual = np.array([power_piston - power_req / (nr_engines * cylinders)])
        #print(residual)
        return residual

    try:
        bore_match = brentq(find_match, 0.1, 0.2)
    except ValueError:
        output_dict = {
            "error": True
        }
        return output_dict


    # Now use the stored values from the last iteration
    mdot_in = last_outputs['m_in'] * nr_engines * cylinders
    T34 = last_outputs['T34']
    p34 = pin * p_ratio


    T_circumv = last_outputs["T_circumv"]
    P_circumv = last_outputs["P_circumv"] * nr_engines * cylinders
    friction_loss = last_outputs["friction_loss"] * nr_engines * cylinders
    aux_loss = last_outputs["aux_loss"] * nr_engines * cylinders
    P_fuel_pump = last_outputs["P_fuel_pump"] * nr_engines * cylinders

    power_piston = last_outputs["power_piston"] * nr_engines * cylinders
    indicated_power = last_outputs["indicated_power"] * nr_engines * cylinders



    mdot_bypass = core_flow - mdot_in

    fraction = mdot_in / core_flow

    # calcualate heat loss to conserve energy
    # ADJUST POWER SO THAT ENERGY IS CONSERVED

    if fuel_type == "jetA":
        _, h_fuel, _, _ = JETA_L(T_fuel)

    else:
        _, h_fuel, _, _ = H2(T_fuel)


    equ34 = far34 / far_stoich
    # mix circumventing flow
    fuel_flow = far34 * mdot_in

    m34 = mdot_in + fuel_flow  # outflow of piston engine (air + fuel)



    h_in, _, _, _, _, _, _, _ = mixture(Tin, pin)
    h_out, _, _, _, _, _, _, _ = mixture(T34, p34, equ34, fuel_type=fuel_type)

    # Enthalpy in, out and fuel
    H_in = h_in * mdot_in
    H_fuel = h_fuel * fuel_flow
    H_out = h_out * m34

    # Conservation of energy gives heat_loss
    heat_loss = H_in + H_fuel - H_out - indicated_power

    heat_loss_approx = (k0_H + k1_H * bore_match ** 2) * cylinders * nr_engines

    term1 = H_in + H_fuel
    term2 = H_out + indicated_power + heat_loss_approx

    #print(f"Energy conservation: {(term2 - term1)*1e-3} kW")


    #print(f"Fuel energy: {LHV * mdot_in * far34 * 1e-3} kW")
    #print(f"Indicated power: {indicated_power * 1e-3} kW")
    #print(f"Heat loss: {heat_loss * 1e-3} kW")
    #print(f"Heat loss approx: {heat_loss_approx * 1e-3} kW")
    #print(f"Heat loss percentage diff (approx vs energy conserv): {100 * (heat_loss_approx - heat_loss) / heat_loss} %")

    if life_hack == "Simulate_final":
        heat_loss_diff = (heat_loss - heat_loss_final_sim*24) / (24 *heat_loss_final_sim)
        if heat_loss_diff > 1e-2:
            print(f"Heat loss error larger than 1 percent: {heat_loss_diff}")

    else:
        # we only extract max temp from last simulation
        T_max_twozone = 0


    #    print(f"Heat loss percentage diff (energy conserv vs sim): {100 * (heat_loss - heat_loss_final_sim*24) / (24 *heat_loss_final_sim)} %")
    #print(f"Increase in enthalpy over engine: {(H_in - H_out) * 1e-3} kW")
    #print(f"Specific power: {(specific_power) * 1e-3} kW/kg/s")
    #print(f"mdot in: {mdot_in}")



    m35 = m34 + mdot_bypass # flow after mixing
    #print(m34, T34, equ34, mdot_bypass, T_circumv, fuel_type)
    T35, equ35 = components.mix(
        m34,
        T34,
        equ34,
        mdot_bypass,
        T_circumv,
        equ2=0,
        fuel_type=fuel_type,
    )

    far35 = equ35 * far_stoich

    p35 = p34 * (1- p_loss_out)

    m_NOX = nox_ppm * m34 * 1e-6

    # Calculate displacement
    stroke = bore_match / bsr  # using bore-to-stroke ratio
    displacement = np.pi / 4 * bore_match ** 2 * stroke  # m³
    displacement_per_cyl = displacement / 24

    output_dict={
        "power_net": power_piston,
        "power_indicated": indicated_power,
        "heat loss": heat_loss,
        "aux loss": aux_loss,
        "friction loss": friction_loss,
        "fuel pump": P_fuel_pump,
        "P comp circumvent air": P_circumv,
        "T34": T34,
        "T35": T35,
        "T_circumv": T_circumv,
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
        "k_m": k_m,
        "k0_T": k0_T,
        "k1_T": k1_T,
        "k0_H": k0_H,
        "k1_H": k1_H,
        "specific_power": specific_power,
        "T_max_twozone": T_max_twozone,
        "error": False,

    }

    return output_dict