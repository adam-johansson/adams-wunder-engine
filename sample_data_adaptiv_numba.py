import numpy as np
import importlib

from timeit import default_timer as timer

from piston_engine.engine import run_piston_engine  # import the piston engine function

from scipy.stats import qmc

import pandas as pd

from piston_engine.src.misc.seiliger import seiliger
from piston_engine.src.misc.temp_lim import t_in_lim

from numba import njit

from thermo import fuel_props

# Setting up the piston engine
input_file = "4stroke_sampling"

if input_file == "4stroke_hydrogen_sampling":
    fuel = "H2"
else:
    fuel = "jetA"

input_dir = "piston_engine.input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

piston_input = {
        'cycle': d.cycle,
        'cooling': d.cooling,
        'opposed': d.opposed,
        'bsr': d.bsr,
        'lms': d.lms,
        'Twalls': d.Twalls,
        'ch': d.ch,
        'valve_timings': d.valve_timings,
        'n_valve': d.n_valve,
        'cd': d.cd,
        'eta_c': d.eta_c,
        'mf_tot': d.mf_tot,
        'wa': d.wa,
        'wm': d.wm,
        'm_wiebe': d.m_wiebe,
        'phi_sc': d.phi_sc,
        'phi_cd': d.phi_cd,
        'T_fuel': d.T_fuel,
        'p_fuel': d.p_fuel,
        'it': d.it,
        'wiebe_type': d.wiebe_type,
        'valve_type': d.valve_type,
        'cylinders': d.cylinders,
        'fuel': d.fuel,
        'c1': d.c1,
        'c4': d.c4,
        'c5': d.c5,
        'premixed': d.premixed,
        'mode': d.mode,
    }

# flags: plot, output, validation, sweep
flags = ["sweep"]
# just to be able to get 1 output instead of 4

# 7 without fuel temp
ndim = 7  # number of input variables
n_out = 12  # Number of outputs from the piston model


start_sampling = timer()

# limits for the sampling
p_lim = [1e5, 35e5]  # limits for input pressure (maybe lower p_lim slightly to never be on the limit)
T_lim = [250, 1000]  # limits for input temperature
cr_lim = [4, 16]  # limits for geometric compression ratio
d_lim = [0.10, 0.20]  # limits for bore (piston diameter)
p_ratio_lim = [0.9, 1.5]
v_mean_lim = [8, 15]


far_s, _ = fuel_props(fuel)
if fuel == "H2":
    # THIS THROTTLE LIM IS FOR HYDROGEN
    far_lim = [far_s / 3.0, far_s / 1.0]  # (0.02923 / 5, 0.02923 / 1.5)
    fuel_t_lim = [300, 500]

elif fuel == "jetA":
    # THIS IS FOR JETA
    far_lim = [far_s / 3.3, far_s / 1.1] #roughly 2% to 6%
    #fuel_t_lim = [220, 550]


# could add wall temperatures

xlimits = np.array(
    [p_lim, T_lim, cr_lim, d_lim, far_lim, p_ratio_lim, v_mean_lim]
)


# Construction of the DOE, the training points  #approx 700 seconds for 60 training 60 validation
npoints = 15000   # points per variable #6000 takes 20h (gives 18000 nonzero points) (for 8 dimensions)
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
print(
    f"Discrepancy: {qmc.discrepancy(sample)} (Low number good. How well the parameter space is covered)"
)

# initiate output array
y = np.zeros([ndoe, n_out])


print(f"Total number of sampling points: {ndoe}")
start_simulating = timer()
i = 0
remove = 0


for p, T, cr, bore, far_goal, p_ratio, v_mean in sample_scaled:
    i += 1

    # rough estimiation of the peak pressure
    pmax_seiliger = seiliger(p, T, cr, far_goal, bore, fuel)

    # estimation of the highest possible temperature for given pressure
    t_limit = t_in_lim(p)

    # t_limit = 1000000
    # if predicted pressure under 400 bar
    if pmax_seiliger < 400 * 1e5 and T < t_limit:

        # inside loop, only update what changes:
        piston_input.update({
            'p_in': p,
            'T_in': T,
            'p_ratio': p_ratio,
            'cr': cr,
            'bore': bore,
            'v_mean': v_mean,
            'lv_max': 0.1 * bore,
            'far_goal': far_goal,
        })
      

        # run the simulation
        #print(p * 1e-5, T, cr, bore, far_goal, p_ratio, v_mean, fuel_t)
        piston_output = run_piston_engine(piston_input, flags)

        try:
            T_out = piston_output["T_out"]   
        except (IndexError, KeyError):
            # piston simulation not converging
            y[i - 1, :] = np.zeros(n_out)
            remove = remove + 1
        else:

            y[i - 1, :] = (
                T_out,
                piston_output["eta_th"],
                piston_output["air_flow"],
                piston_output["peak pressure"],
                piston_output["peak temperature"],
                piston_output["indicated power"],
                piston_output["heat_loss"],
                piston_output["p_tdc"],
                piston_output["no_ppm"],
                piston_output["flame temperature"],
                piston_output["T start of combustion"],
                piston_output["p start of combustion"],
            )

    else:
        # if predicted peak pressure is over 400 bar or temperature is too high
        y[i - 1, :] = np.zeros(n_out)
        remove = remove + 1
        # print(f"Number of data points removed: {remove} out of {i} in total")

    if not (i % (ndoe // 10  )):
        mellantid = timer()
        elapsed_time = mellantid - start_simulating
        avg_iteration_time = elapsed_time / i
        total_time = avg_iteration_time * ndoe
        print(
            f"Simulation {i} out of {ndoe}."
            f"Datapoints removed: {remove}. "
            f"Elapsed time: {elapsed_time} [s]"
            f"Avg iteration time: {avg_iteration_time} [s]"
            f"Estimated total sampling time: {total_time} [s]"
            f"Progress in percent: {elapsed_time / total_time * 100}"
        )


# Saving the arrays in sampled_data folder
# Labels

headers_input = np.array(
    ["p_in", "T_in", "cr", "bore", "far_goal", "PI", "v_mean"]
)

headers_output = np.array(
    ["T_out", "eff", "mdot_in", "p_max", "T_max", "power", "heat_loss", "p_tdc", "nox_ppm", "flame_temp", "T_sc", "p_sc"]
)

# Adding the labels to the data
x_data = pd.DataFrame(sample_scaled, columns=headers_input)
y_data = pd.DataFrame(y, columns=headers_output)
combined_data = pd.concat([x_data, y_data], axis=1)

# Writing data to file
combined_data.to_csv("piston_engine/sampled_data/" + fuel + "/data.csv")
end_sampling = timer()
print(f"Total time for sampling data: {end_sampling - start_sampling} [s]")
