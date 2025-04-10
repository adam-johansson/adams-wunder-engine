import numpy as np
import importlib

from timeit import default_timer as timer

from piston_engine.engine import run_piston_engine  # import the piston engine function

from scipy.stats import qmc

import pandas as pd

from piston_engine.src.misc.seiliger import seiliger
from piston_engine.src.misc.temp_lim import t_in_lim

from numba import jit

from thermo import fuel_props

# Setting up the piston engine
input_file = "4stroke_hydrogen_sampling"

if input_file == "4stroke_hydrogen_sampling":
    fuel = "H2"
else:
    fuel = "jetA"

input_dir = "piston_engine.input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

# flags: plot, output, validation, sweep
flags = ["sweep"]
# just to be able to get 1 output instead of 4

ndim = 8  # number of input variables
n_out = 9  # Number of outputs from the piston model


start_sampling = timer()

# limits for the sampling
p_lim = [2e5, 35e5]  # limits for input pressure
T_lim = [250, 1000]  # limits for input temperature
cr_lim = [6, 12]  # limits for geometric compression ratio
d_lim = [0.10, 0.20]  # limits for bore (piston diameter)
p_ratio_lim = [0.9, 1.5]
v_mean_lim = [8, 15]


far_s, _ = fuel_props(fuel)
if fuel == "H2":
    # THIS THROTTLE LIM IS FOR HYDROGEN
    far_lim = [far_s / 3.0, far_s / 1.2]  # (0.02923 / 5, 0.02923 / 1.5)
    fuel_t_lim = [300, 500]

elif fuel == "jetA":
    # THIS IS FOR JETA
    far_lim = [far_s / 3.0, far_s / 1.2]
    fuel_t_lim = [220, 550]


# could add wall temperatures

xlimits = np.array([p_lim, T_lim, cr_lim, d_lim, far_lim, p_ratio_lim, v_mean_lim, fuel_t_lim])

# Construction of the DOE, the training points  #approx 700 seconds for 60 training 60 validation
npoints = 10  # points per variable
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
remove = 0

for p, T, cr, bore, far_goal, p_ratio, v_mean, fuel_t in sample_scaled:
    i += 1


    # TODO: Add variable combustion duration and shape ...

    # rough estimiation of the peak pressure
    pmax_seiliger = seiliger(p, T, cr, far_goal, bore, fuel)

    # estimation of the highest possible temperature for given pressure
    t_limit = t_in_lim(p)

    #t_limit = 1000000
    # if predicted pressure under 400 bar
    if pmax_seiliger < 400*1e5 and T < t_limit:

        lv_max = 0.1 * bore
        data = [p, T, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
                v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, fuel_t, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_goal,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5, d.premixed]

        # run the simulation
        print(p*1e-5, T, cr, bore, far_goal, p_ratio, v_mean, fuel_t)
        T_out, work_piston, eta_th, air_flow, p_max, T_max, _, equ_trapped, induced_power, _, _, \
            heat_loss, p_tdc, _, nox, _, EI_nox, volume_eff, nox_spec = run_piston_engine(data, flags)
        # save the output that is relevant
        if equ_trapped > 1.0:
            y[i - 1, :] = np.zeros(n_out)
            remove = remove + 1
            print(f"Removed data point because too high far.")
        else:
            # eta_th is redundant I suppose
            y[i - 1, :] = T_out, eta_th, air_flow, p_max, T_max, induced_power, heat_loss, p_tdc, nox
            #print(y[i - 1, :])


    else:
        # if predicted peak pressure is over 400 bar or temperature is too high
        y[i - 1, :] = np.zeros(n_out)
        remove = remove + 1
        #print(f"Number of data points removed: {remove} out of {i} in total")

    if not (i % (ndoe // 80)):
        mellantid = timer()
        elapsed_time = mellantid - start_simulating
        avg_iteration_time = elapsed_time / i
        total_time = avg_iteration_time * ndoe
        print(f'Simulation {i} out of {ndoe}.'
              f'Datapoints removed: {remove}. '
              f'Elapsed time: {elapsed_time} [s]'
              f'Avg iteration time: {avg_iteration_time} [s]'
              f'Estimated total sampling time: {total_time} [s]'
              f'Progress in percent: {elapsed_time / total_time * 100}')


# Saving the arrays in sampled_data folder
# Labels
headers_input = np.array(['p_in', 'T_in', 'cr', 'bore', 'far', 'PI', 'v_mean', 'T_fuel'])
headers_output = np.array(['T_out', 'eff', 'air_flow', 'p_max', 'T_max', 'power', 'heat_loss', 'p_tdc', 'nox'])

# Adding the labels to the data
x_data = pd.DataFrame(sample_scaled, columns=headers_input)
y_data = pd.DataFrame(y, columns=headers_output)

# Writing data to file
x_data.to_csv('sampled_data/' + fuel + '/x.csv')
y_data.to_csv('sampled_data/' + fuel + '/y.csv')
end_sampling = timer()
print(f'Total time for sampling data: {end_sampling - start_sampling} [s]')

