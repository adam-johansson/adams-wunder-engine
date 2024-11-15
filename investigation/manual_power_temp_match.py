import numpy as np
from CCE.src import thermo_outdated, components
from piston_engine.src.misc import post_processing

import pickle
import importlib


# load the surrogate model
filename = '../piston_engine/surrogate_data/piston_surrogate_h2.pkl'
with open(filename, "rb") as f:
    meta_model = pickle.load(f)

input_file_pist = "4stroke_hydrogen"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist

d_p = importlib.import_module(path_pist)


cr = 5
bore_dummy = 0.1
pin = 7e5
Tin = 600
pi_pe = 1.3
core_flow = 5


data= [pin, Tin, pi_pe, d_p.cycle, d_p.thermo, d_p.cooling, d_p.opposed,
               cr, bore_dummy, d_p.bsr, d_p.v_mean, d_p.lms, d_p.Twalls, d_p.ch, d_p.valve_timings,
               d_p.n_valve, d_p.lv_max, d_p.cd, d_p.eta_c, d_p.mf_tot, d_p.wa, d_p.wm, d_p.m_wiebe,
               d_p.phi_sc, d_p.phi_cd, d_p.T_fuel, d_p.p_fuel, d_p.it, d_p.wiebe_type, d_p.valve_type,
               d_p.throttle, d_p.cylinders, d_p.fuel]

# input to surrogate
pin = data[0]
Tin = data[1]
cr = data[7]
p_ratio = data[2]

# fuel type
fuel_type = data[32]
far_s, LHV = thermo_outdated.fuel_props(fuel_type)

def find_match(x):
    # change fuel air ratio and bore to match power and turbine inlet temperature

    bore = x[0]  # bore is varied to
    far34 = x[1]  # far is varied to match target output temperature

    if far34 < 0.002923 or far34 > 0.02923 or bore > 0.2:
        residual = np.array([1e6, 1e6])
        return residual

    # get the output of the surrogate
    piston_input = np.atleast_2d(np.array([pin, Tin, cr, bore, far34, p_ratio]))
    air_flow = meta_model[2].predict_values(piston_input)[0][0]
    induced_power = meta_model[5].predict_values(piston_input)[0][0] * 1e3
    p_tdc = meta_model[7].predict_values(piston_input)[0][0] * 1e5
    T34 = meta_model[0].predict_values(piston_input)[0][0]

    # pressurise circumventing flow
    m_circumvent = core_flow - air_flow
    if m_circumvent < 0:
        return np.array([1e6, 1e6])
    pressure_circ, T_circumv, P_circumv = \
        components.compressor(data[1], data[0] / 0.99, m_circumvent, 0.85, p_ratio * 0.99 * 0.99)

    # mix circumventing flow
    equ34 = far34 / far_s
    m34 = air_flow * (1 + far34)  # outflow of piston engine (air + fuel)
    # m35 = m34 + m_circumvent  # flow after mixing
    T35, equ35 = components.mix(m34, T34, equ34, m_circumvent, T_circumv, equ2=0, fuel_type=fuel_type)
    # far35 = equ35 * far_s

    # power needed to pressurise the fuel
    fuel_flow = air_flow * far34  # far_given is the same as far in the engine (at least it is supposed to be)
    P_fuel_pump = components.fuel_pump(p_tdc, fuel_type, fuel_flow)

    # things needed for aux and friction losses
    bsr = data[9]
    stroke = bore / bsr
    lv_max = bore * 0.1
    cylinders = data[31]
    v_mean = data[10]
    rpm = v_mean / (2 * stroke) * 60
    rps = rpm / 60
    Vd_tot = stroke * bore ** 2 / 4 * np.pi * cylinders
    cycle = data[3]
    if cycle == '4T':
        n_r = 2
    else:
        n_r = 1

    # auxiliary losses and friction losses. these do not depend on the trapped fuel air ratio
    fmep, fmep_aux, fmep_pe_loss = post_processing.friction_patton(bore, rpm, stroke, v_mean, pin, cr, cylinders,
                                                                   lv_max)
    friction_loss = fmep_pe_loss * Vd_tot * rps / n_r  # friction losses for total engine all cylinders
    aux_loss = fmep_aux * Vd_tot * rps / n_r  # auxiliary losses

    shaft_power = induced_power - aux_loss - friction_loss - P_circumv - P_fuel_pump

    residual = np.array([(shaft_power), T35])
    # print(residual, bore, far34)

    return residual


bores = np.linspace(0.05, 0.17, 10)
fars = np.linspace(0.015, 0.029, 10)
output = np.empty((100, 4))

i = 0
for bore in bores:
    for far in fars:
        x = np.array([bore, far])
        power, T35 = find_match(x)
        output[i, :] = np.array([power*1e-3, T35, far, bore])
        i += 1



print(output[:, 0])

#mask = np.ma.masked_where((1900 < output[:, 0]) & (output[:, 0] < 2100), output[:, 0])

#print(mask)