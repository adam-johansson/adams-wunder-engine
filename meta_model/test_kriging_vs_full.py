from piston_engine.engine import run_piston_engine
import pickle
import numpy as np
import importlib

from timeit import default_timer as timer




input_file_pist = "4stroke_hydrogen"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist


d_p = importlib.import_module(path_pist)



# load the surrogate model
filename = '../piston_engine/surrogate_data/piston_surrogate_h2_16th.pkl'
with open(filename, "rb") as f:
    meta_model = pickle.load(f)


far34 = 0.0164544324463330
cr = 6.41  # piston engine geometric compression ratio
p_ratio = 0.911  # Piston engine pressure ratio
bore = 0.12015  # piston bore
p_in = 700368.0925487876
T_in = 625.4502715206818


data_piston = [p_in, T_in, p_ratio, d_p.cycle, d_p.thermo, d_p.cooling, d_p.opposed,
               cr, bore, d_p.bsr, d_p.v_mean, d_p.lms, d_p.Twalls, d_p.ch, d_p.valve_timings,
               d_p.n_valve, d_p.lv_max, d_p.cd, d_p.eta_c, d_p.mf_tot, d_p.wa, d_p.wm, d_p.m_wiebe,
               d_p.phi_sc, d_p.phi_cd, d_p.T_fuel, d_p.p_fuel, d_p.it, d_p.wiebe_type, d_p.valve_type,
               far34, d_p.cylinders, d_p.fuel, d_p.c1, d_p.c4, d_p.c5]


# get the output of the surrogate
start = timer()
piston_input = np.atleast_2d(np.array([p_in, T_in, cr, bore, far34, p_ratio, d_p.v_mean]))
air_flow_surrogate = meta_model[2].predict_values(piston_input)[0][0]
induced_power_surrogate = meta_model[5].predict_values(piston_input)[0][0] * 1e3
p_tdc_surrogate = meta_model[7].predict_values(piston_input)[0][0] * 1e5
T34_surrogate = meta_model[0].predict_values(piston_input)[0][0]
end = timer()
print(end - start)

flags = ['sweep']
data_piston[30] = far34  # fuel air ratio
T34, power_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, \
    aux_loss, heat_loss, p_tdc = run_piston_engine(data_piston, flags)


print(air_flow_surrogate, air_flow)
print(induced_power_surrogate, induced_power)
print(p_tdc_surrogate, p_tdc)
print(T34_surrogate, T34)