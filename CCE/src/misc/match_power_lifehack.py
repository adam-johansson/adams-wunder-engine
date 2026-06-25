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

    far33 = input["far_goal"]

    if life_hack == "Simulate":
        # calculate stuff for bore between 10 and 20 cm

        flags = ["sweep"]


        bore_mid = 0.15
        bore_high = 0.20

        input["bore"] = bore_high

  

        piston_output = run_piston_engine(input, flags)

        T_out_high = piston_output["T_out"]
        heat_loss_high = piston_output["heat_loss"]

        #print("Sim 1 done")
        #print(f"Heat loss sim 1: {heat_loss_high*24*1e-3} kW")

        input["bore"] = bore_mid

        
        piston_output = run_piston_engine(input, flags)

        try:
            T_out_mid = piston_output["T_out"]
        except IndexError:
            # piston simulation not converging
            output_dict = {
               "error": True
            }
            return output_dict


        heat_loss_mid = piston_output["heat_loss"]
        m_in = piston_output["intake massflow"]
        p_max = piston_output["peak pressure"]
        T_max = piston_output["peak temperature"]
        indicated_power = piston_output["indicated power"]
        p_tdc = piston_output["p_tdc"]
        nox_ppm = piston_output["no_ppm"]

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
        piston_output = run_piston_engine(input, flags)
       
       
        try:
            heat_loss_final_sim = piston_output["heat_loss"]
        except IndexError:
            # piston simulation not converging
            output_dict = {
                "error": True
            }
            return output_dict

        p_max = piston_output["peak pressure"]
        T_max = piston_output["peak temperature"]
        p_tdc = piston_output["p_tdc"]
        nox_ppm = piston_output["no_ppm"]   
        T_max_twozone = piston_output["peak temperature hot zone"]


        #print("Final sim done")
        #print(f"Heta loss final sim: {heat_loss_final_sim*24*1e-3} kW")
        specific_power = input["piston_specific_power"]
        k_m = input["k_m"]
        k1_T = input["k1_T"]
        k0_T = input["k0_T"]
        k1_H = input["k1_H"]
        k0_H = input["k0_H"]

        # This is only to check
        indicated_power_check = piston_output["indicated power"]
        T_out_mid_check = piston_output["T_out"]
        heat_loss_mid_check = piston_output["heat_loss"]
        m_in_check = piston_output["intake massflow"]


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
    far33 = input["far_goal"]
    equ_in = input["equ_in"]

    EGR_rate = input["EGR_rate"]

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

        approx_T33 = k0_T + k1_T * bore


        #fuel flow from calculations
        #fuel_flow = far34 * approx_m_in
        fuel_flow = approx_m_in * (far33 - equ_in * far_stoich) / (1 + equ_in * far_stoich)

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

        # this is the massflow of core air into the piston engine (total piston intake mass flow can also consist of EGR mass flow)
        core_air_in = approx_m_in * (1 - EGR_rate)

        # this is per cylinder
        # air bypassing the piston engine
        # core air - core air mass flow into the piston engine
        mdot_bypass = core_flow / (cylinders * nr_engines) - core_air_in

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
            'core_air_in': core_air_in,
            'indicated_power': approx_power,
            'T33': approx_T33,
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
        bore_match = brentq(find_match, 0.1, 0.25)
    except ValueError:
        #print(f"No bore found to match power")
        output_dict = {
            "error": True
        }
        return output_dict


    # Now use the stored values from the last iteration
    mdot_in = last_outputs['m_in'] * nr_engines * cylinders
    core_air_in = last_outputs['core_air_in'] * nr_engines * cylinders
    T33 = last_outputs['T33']
    p33 = pin * p_ratio


    T_circumv = last_outputs["T_circumv"]
    P_circumv = last_outputs["P_circumv"] * nr_engines * cylinders
    friction_loss = last_outputs["friction_loss"] * nr_engines * cylinders
    aux_loss = last_outputs["aux_loss"] * nr_engines * cylinders
    P_fuel_pump = last_outputs["P_fuel_pump"] * nr_engines * cylinders

    power_piston = last_outputs["power_piston"] * nr_engines * cylinders
    indicated_power = last_outputs["indicated_power"] * nr_engines * cylinders


    mdot_bypass = core_flow - core_air_in

    fraction = mdot_in / core_flow

    # calcualate heat loss to conserve energy
    # ADJUST POWER SO THAT ENERGY IS CONSERVED

    if fuel_type == "jetA":
        _, h_fuel, _, _ = JETA_L(T_fuel)

    else:
        _, h_fuel, _, _ = H2(T_fuel)

    
    # equ out of the piston engine
    equ33 = far33 / far_stoich
  
    fuel_flow = mdot_in * (far33 - equ_in * far_stoich) / (1 + equ_in * far_stoich)

    m33 = mdot_in + fuel_flow  # outflow of piston engine (intake flow + fuel) (BEFORE EGR IS EXTRACTED)


    h_in, _, _, _, _, _, _, _ = mixture(Tin, pin, equ_in, fuel_type=fuel_type)
    h_out, _, _, _, _, _, _, _ = mixture(T33, p33, equ33, fuel_type=fuel_type)

    # Enthalpy in, out and fuel
    H_in = h_in * mdot_in
    H_fuel = h_fuel * fuel_flow
    H_out = h_out * m33

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
        #print(f"heat loss diff: {heat_loss_diff}")

        # just to check

        power_diff = indicated_power / (nr_engines * cylinders) - indicated_power_check
        T_out_diff = T33 - T_out_mid_check
        m_in_diff = mdot_in / (nr_engines * cylinders) - m_in_check

        #print(f"power diff: {power_diff*1e-3} kW fraction: {power_diff / indicated_power_check}")
        #print(f"Tout diff: {T_out_diff} K fraction: {T_out_diff / T_out_mid_check}")
        #print(f"mass diff: {m_in_diff} kg/s fraction: {m_in_diff / m_in_check}")

        if heat_loss_diff > 1e-2:
            print(f"Heat loss error larger than 1 percent: {heat_loss_diff}")

    else:
        # we only extract max temp from last simulation
        T_max_twozone = 0


    #    print(f"Heat loss percentage diff (energy conserv vs sim): {100 * (heat_loss - heat_loss_final_sim*24) / (24 *heat_loss_final_sim)} %")
    #print(f"Increase in enthalpy over engine: {(H_in - H_out) * 1e-3} kW")
    #print(f"Specific power: {(specific_power) * 1e-3} kW/kg/s")
    #print(f"mdot in: {mdot_in}")

    # output in kg
    m_NOX = nox_ppm * m33 * 1e-6

    # Calculate displacement for one cylinder
    stroke = bore_match / bsr  # using bore-to-stroke ratio
    displacement = np.pi / 4 * (bore_match ** 2 ) * stroke  # m³


    output_dict={
        "power_net": power_piston,
        "power_indicated": indicated_power,
        "heat loss": heat_loss,
        "aux loss": aux_loss,
        "friction loss": friction_loss,
        "fuel pump": P_fuel_pump,
        "P comp circumvent air": P_circumv,
        "T33": T33,
        #"T35": T35,
        "T_circumv": T_circumv,
        "p33": p33,
        #"p35": p35,
        "m31": core_air_in,
        "m32": mdot_in,
        "m33": m33,
        #"m35": m35,
        "far33": far33,
        #"far35": far35,
        "fuel_flow": fuel_flow,
        "p max": p_max,
        "T max": T_max,
        "m NO": m_NOX,
        "bore": bore_match,
        "displacement": displacement,
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