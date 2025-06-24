import numpy as np
from piston_engine.src.misc.seiliger import seiliger
from piston_engine.src.misc.temp_lim import t_in_lim
from thermo import fuel_props, mixture, H2, JETA_L

def nn_output_energy_conserved(meta_model, input_array, fuel_type):

    # input
    p_in = input_array[0]
    T_in = input_array[1]
    cr = input_array[2]
    bore = input_array[3]
    far_goal = input_array[4]
    p_ratio = input_array[5]
    v_mean = input_array[6]
    T_fuel = input_array[7]



    # check for valid input
    far_stoich, LHV = fuel_props(fuel_type)


    # 400 bar limit
    pmax_limit = 400 * 1e5

    base_check = (1e5 < p_in < 35e5 and 250 < T_in < 1000 and 4 < cr < 16 and
                  0.10 < bore < 0.20 and 0.9 < p_ratio < 1.5 and 8 < v_mean < 15 and
                  far_stoich / 3.0 < far_goal < far_stoich / 1.0)

    temp_check = (220 < T_fuel < 550) if fuel_type == "jetA" else (300 < T_fuel < 500) if fuel_type == "H2" else False

    if base_check and temp_check:
        invalid_input = False
    else:
        invalid_input = True
        print(f"Input to surrogate is out of bounds: {input_array}")


    # rough estimiation of the peak pressure
    pmax_seiliger = seiliger(p_in, T_in, cr, far_goal, bore, fuel_type)

    # estimation of the highest possible temperature for given pressure (assuming heating only from compression)
    t_limit = t_in_lim(p_in)

    # if predicted pressure under 400 bar and temperature under limit
    if pmax_seiliger > pmax_limit or T_in > t_limit:
        invalid_input = True
        print(f"Input to surrogate is too high pressure or temperature: {input_array}")

    if invalid_input:
        return np.zeros(9)

    # GET THE RAW OUTPUT FROM THE SURROGATE MODEL
    output_raw = meta_model.inference(input_array)

    T_out = output_raw[0][0]
    #thermal_efficiency = output_raw[0][1]
    air_flow = output_raw[0][2]
    p_max = output_raw[0][3]
    T_max = output_raw[0][4]
    #indicated_power = output_raw[0][5]
    heat_loss = output_raw[0][6]
    p_tdc = output_raw[0][7]
    nox_ppm = output_raw[0][8]


    # ADJUST POWER SO THAT ENERGY IS CONSERVED

    if fuel_type == "jetA":
        _, h_fuel, _, _ = JETA_L(T_fuel)
    else:
        _, h_fuel, _, _ = H2(T_fuel)

    h_in, _, _, _, _, _, _, _ = mixture(T_in, p_in, equivalence_ratio=0.0, fuel_type=fuel_type)
    h_out, _, _, _, _, _, s, _ = mixture(T_out, p_in * p_ratio, far_goal / far_stoich, fuel_type=fuel_type)

    # Enthalpy in, out and fuel
    H_in = h_in * air_flow
    H_fuel = h_fuel * air_flow * far_goal
    H_out = h_out * air_flow * (1 + far_goal)

    # Conservation of energy gives power
    indicated_power = H_in + H_fuel - H_out - heat_loss

    # thermal efficiency calculated from the power
    fuel_energy_in = far_goal * air_flow * LHV
    thermal_efficiency = indicated_power / fuel_energy_in
    #diff_efficiency = np.abs((thermal_efficiency - eff)) / eff

    output_array = np.array([T_out, thermal_efficiency, air_flow, p_max, T_max, indicated_power, heat_loss, p_tdc, nox_ppm])

    return output_array