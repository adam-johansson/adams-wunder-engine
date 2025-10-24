import numpy as np
from piston_engine.src.misc.seiliger import seiliger
from piston_engine.src.misc.temp_lim import t_in_lim
from thermo import fuel_props, mixture, H2, JETA_L

def nn_output_energy_conserved(meta_model, input_array, fuel_type):

    input_array = input_array[0]

    # input
    p_in = input_array[0]
    T_in = input_array[1]
    p_ratio = input_array[2]
    cr = input_array[3]
    bore = input_array[4]
    v_mean = input_array[5]
    T_fuel = input_array[6]
    far_goal = input_array[7]



    far_stoich, LHV = fuel_props(fuel_type)

    # GET THE RAW OUTPUT FROM THE SURROGATE MODEL
    output_raw = meta_model.inference(input_array)[0]

    indicated_power = output_raw[0]
    nox_ppm = output_raw[1]
    p_tdc = output_raw[2]
    p_max = output_raw[3]
    T_out = output_raw[4]
    T_max = output_raw[5]
    m_in = output_raw[6]



    # ADJUST POWER SO THAT ENERGY IS CONSERVED
    if fuel_type == "jetA":
        _, h_fuel, _, _ = JETA_L(T_fuel)
    else:
        _, h_fuel, _, _ = H2(T_fuel)

    h_in, _, _, _, _, _, _, _ = mixture(T_in, p_in, equivalence_ratio=0.0, fuel_type=fuel_type)
    h_out, _, _, _, _, _, s, _ = mixture(T_out, p_in * p_ratio, far_goal / far_stoich, fuel_type=fuel_type)

    # Enthalpy in, out and fuel
    H_in = h_in * m_in
    H_fuel = h_fuel * m_in * far_goal
    H_out = h_out * m_in * (1 + far_goal)

    # Conservation of energy gives heat_loss
    heat_loss = H_in + H_fuel - H_out - indicated_power

    output_array = np.array([indicated_power, nox_ppm, p_tdc, p_max, T_max, T_out, m_in, heat_loss])

    return output_array