import numpy as np
import importlib

from timeit import default_timer as timer

from engine import run_piston_engine  # import the piston engine function

from scipy.stats import qmc

import pandas as pd


# Setting up the piston engine

input_file = "4stroke_hydrogen"
input_dir = "input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

# flags: plot, output, validation, sweep
flags = ["sweep"]
# just to be able to get 1 output instead of 4

ndim = 8  # number of input variables
n_out = 8  # Number of outputs from the piston model


start_sampling = timer()

# limits for the sampling
p_lim = [2e5, 70e5]  # limits for input pressure (3e5, 10e5)
T_lim = [300, 1200]  # limits for input temperature (400, 1000)
cr_lim = [4, 18]  # limits for geometric compression ratio (6, 12)
d_lim = [0.05, 0.30]  # limits for bore (piston diameter) (0.08, 0.17)

# THIS THROTTLE LIM IS FOR HYDROGEN
throttle_lim = [0.02923 / 6, 0.02923 / 1.0]  # (0.02923 / 5, 0.02923 / 1.5)
fuel_t_lim = [100, 600]

# THIS IS FOR JETA
#throttle_lim = [0.06821 / 6, 0.06821 / 1.0]

p_ratio_lim = [0.7, 2.0]   # (1.0, 1.5)
v_mean_lim = [5, 20]

# could add wall temperatures

xlimits = np.array([p_lim, T_lim, cr_lim, d_lim, throttle_lim, p_ratio_lim, v_mean_lim, fuel_t_lim])

# Construction of the DOE, the training points  #approx 700 seconds for 60 training 60 validation
npoints = 3000  # points per variable
ndoe = ndim * npoints

# create sampling on unit hypercube
# optimization greatly improves discrepancy
# strength=2 requires p**2 sample points with p being a prime number
# however strength 1 + optimization seems to give best results
sampler = qmc.LatinHypercube(d=ndim, seed=42, optimization="random-cd", strength=1)
sample = sampler.random(n=ndoe)


# lower bounds
l_bounds = xlimits[:, 0]
# upper bounds
u_bounds = xlimits[:, 1]
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
print(f'Discrepancy: {qmc.discrepancy(sample)} (Low number good. How well the parameter space is covered)')

# initiate output array
y = np.zeros([ndoe, n_out])


print(f'Total number of sampling points: {ndoe}')
start_simulating = timer()
i = 0
for p, T, cr, bore, throttle, p_ratio, v_mean, fuel_t in sample_scaled:
    i += 1

    lv_max = 0.1 * bore
    data = [p, T, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
            v_mean, d.lms, d.Twalls, d.ch,
            d.valve_timings, d.n_valve, lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
            d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, fuel_t, d.p_fuel, d.it, d.wiebe_type, d.valve_type, throttle,
            d.cylinders, d.fuel, d.c1, d.c4, d.c5]

    # run the simulation
    T_out, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss, \
        heat_loss, p_tdc = run_piston_engine(data, flags)
    # save the output that is relevant
    # eta_th is redundant I suppose
    y[i - 1, :] = T_out, eta_th, air_flow, p_max, T_max, induced_power * 1e-3, heat_loss * 1e-3, p_tdc * 1e-5

    if not (i % (ndoe // 1000)):
        mellantid = timer()
        elapsed_time = mellantid - start_simulating
        avg_iteration_time = elapsed_time / i
        total_time = avg_iteration_time * ndoe
        print(f'Simulation {i} out of {ndoe}.'
              f'Elapsed time: {elapsed_time} [s]'
              f'Avg iteration time: {avg_iteration_time} [s]'
              f'Estimated total sampling time: {total_time} [s]'
              f'Progress in percent: {elapsed_time / total_time * 100}')


# Saving the arrays in sampled_data folder
# Labels
headers_input = np.array(['p_in', 'T_in', 'cr', 'bore', 'far', 'PI', 'v_mean', 'T_fuel'])
headers_output = np.array(['T_out', 'eff', 'air_flow', 'p_max', 'T_max', 'power', 'heat_loss', 'p_tdc'])

# Adding the labels to the data
x_data = pd.DataFrame(sample_scaled, columns=headers_input)
y_data = pd.DataFrame(y, columns=headers_output)

# Writing data to file
x_data.to_csv('sampled_data/x.csv')
y_data.to_csv('sampled_data/y.csv')
end_sampling = timer()
print(f'Total time for sampling data: {end_sampling - start_sampling} [s]')


