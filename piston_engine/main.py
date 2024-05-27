from timeit import default_timer as timer
import matplotlib.pyplot as plt
import importlib

import numpy as np

from engine import run_piston_engine  # import the piston engine function
from CCE.src.thermo.fuel_func import fuel_props

# import all the input variables

# input_file = "4T_HP"
#input_file = "4stroke"
#input_file = "4stroke_kaiser"
#input_file = "4stroke_hydrogen"
input_file = "H2_validation_italian.4stroke_hydrogen_validation_italian_08_v2"
#input_file = "nasa_validation_singlewiebe"
input_dir = "input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

# flags: plot_all, plot_essentials, plot_convergence, validation, output_all, output_power
# sweep, plot_details

#flags = ['validation', 'output_all', 'single', 'plot_convergence', 'plot_essentials', 'save']  # validation case
#flags = ['plot_essentials', 'output', 'output_all', 'plot_convergence', 'single', 'save']  # normal case
flags = ['single', 'save', 'output_all', 'load']  # normal case no plots
#flags = ['sweep']  # parametric study
#flags = ['optimise']  # optimisation
#flags = ['load']

data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
        d.v_mean, d.lms, d.Twalls, d.ch,
        d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, d.throttle,
        d.cylinders, d.fuel, d.c1, d.c4, d.c5]

if 'single' in flags:
    start = timer()
    T4, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss,\
        heat_loss, p_tdc = run_piston_engine(data, flags)
    end = timer()
    print(far / 0.02918)
    #0.01124952812
    #print(f'Induced power: {induced_power * 1e-3} [kW]')
    #print(f'PE losses: {friction_loss * 1e-6} [MW]')
    #print(f'Aux losses: {aux_loss * 1e-6} [MW]')
    #print(f'Heat losses: {heat_loss * 1e-6} [MW]')
    #print(f'Pressure at TDC: {p_tdc*1e-5}')

    afr_stoch, lhv = fuel_props(d.fuel)
    #print(f"FAR TRAPPED: {equ_trapped * afr_stoch} ")  # for H2
    #print(f'Far: {far}')
    #print(f'airflow: {air_flow}')
    #print(f'mass flow out: {air_flow * (1 + far)}')
    #print(p_max)
    #print(T4)
    #print(eta_th)
    #print(T_max)


elif 'sweep' in flags:
    from piston_engine.src.misc.sweep import sweep_pressure_ratio, sweep_valve_timings, sweep_cr, sweep_sc, \
        sweep_lift, sweep_throttle, sweep_far_surrogate
    sweep_valve_timings(d=d)
    #sweep_cr(d=d)
    #sweep_sc(d=d)
    #sweep_lift(d=d)
    #sweep_throttle(d=d)
    #sweep_far_surrogate(d=d)

elif 'optimise' in flags:
    from piston_engine.src.misc.optimisation import optimisation_combust_start, optimisation_valve_timings, \
        optimisation_combustion
    #optimisation_combust_start(d=d)
    optimisation_valve_timings(d=d)
    #optimisation_combustion(d=d)

elif 'load' in flags:
    from src.misc.plot_output import plot_validation
    from src.misc.post_processing import validation_error
    from numpy import genfromtxt

    P = genfromtxt('simulation_data/P.csv', delimiter=',')
    T = genfromtxt('simulation_data/T.csv', delimiter=',')
    m = genfromtxt('simulation_data/m.csv', delimiter=',')
    equ = genfromtxt('simulation_data/equ.csv', delimiter=',')
    phi = genfromtxt('simulation_data/phi.csv', delimiter=',')
    plot_validation(phi, P, T, m, equ)
    validation_error(phi, P, T, m, equ)

else:
    print('What to you want to do?')
