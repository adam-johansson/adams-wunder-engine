import numpy as np

from CCE.src import misc

import importlib
import pickle

from timeit import default_timer as timer

# Importing input parameters

# load the surrogate model
filename = '../piston_engine/surrogate_data/piston_surrogate_h2.pkl'
with open(filename, "rb") as f:
    meta_model = pickle.load(f)

input_file = "MCR_H2"
input_dir = "input"
path = input_dir + "." + input_file

input_file_pist = "4stroke_hydrogen"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist

d = importlib.import_module(path)
d_p = importlib.import_module(path_pist)

cr = 5
bore = 0.1
pin = 7e5
Tin = 600
pi_pe = 1.3

data_piston = [pin, Tin, pi_pe, d_p.cycle, d_p.thermo, d_p.cooling, d_p.opposed,
               cr, bore, d_p.bsr, d_p.v_mean, d_p.lms, d_p.Twalls, d_p.ch, d_p.valve_timings,
               d_p.n_valve, d_p.lv_max, d_p.cd, d_p.eta_c, d_p.mf_tot, d_p.wa, d_p.wm, d_p.m_wiebe,
               d_p.phi_sc, d_p.phi_cd, d_p.T_fuel, d_p.p_fuel, d_p.it, d_p.wiebe_type, d_p.valve_type,
               d_p.throttle, d_p.cylinders, d_p.fuel]


flow = 5
Ts = np.linspace(830, 870, 100)
power_req_single = 1000e3
#powers = np.linspace(1000e3, 1700e3, 100)

for T35_req in Ts:


    T34, T35, p34, p35, m34_single, m35_single, far34, far35, power_piston, air_flow, p_max, T_max, far_piston, \
        induced_power, friction_loss, aux_loss, heat_loss, P_fuel_pump, bore_match, error =\
        misc.match_tet_power(data_piston, meta_model, power_req_single, core_flow=flow, T35_req=T35_req)
    print(T35, power_piston, far34, bore_match)
"""

for power_req_single in powers:
    T34, power_piston, eta_th, air_flow, p_max, T_max, far_piston, equ_trapped, throttle_match, \
                induced_power, friction_loss, aux_loss, heat_loss, P_fuel_pump, bore_match, error \
                = misc.match_piston_surrogate(data_piston, meta_model, power_req_single, core_flow=flow)
"""