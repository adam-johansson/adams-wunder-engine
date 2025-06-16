from thermo import fuel_props

import numpy as np
import pandas as pd
import importlib

from piston_engine.engine import run_piston_engine


### LOAD THE MODELS

input_file_pist = "4stroke_sampling"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist
d = importlib.import_module(path_pist)
piston_input = {
    'p_in': d.p_in,
    'T_in': d.T_in,
    'p_ratio': d.p_ratio,
    'cycle': d.cycle,
    'cooling': d.cooling,
    'opposed': d.opposed,
    'cr': d.cr,
    'bore': d.d,
    'bsr': d.bsr,
    'v_mean': d.v_mean,
    'lms': d.lms,
    'Twalls': d.Twalls,
    'ch': d.ch,
    'valve_timings': d.valve_timings,
    'n_valve': d.n_valve,
    'lv_max': d.lv_max,
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
    'far_goal': d.far_goal,
    'cylinders': d.cylinders,
    'fuel': d.fuel,
    'c1': d.c1,
    'c4': d.c4,
    'c5': d.c5,
    'premixed': d.premixed,
}

### CODE STARTS HERE ###

# DEFINE INPUT #
cr = 10
bore = 0.15
PI = 1.0
v_mean = 10.0
T_fuel = 300


far_stoich, _ = fuel_props("jetA")

lambdas = [1.1, 1.5, 2.0, 2.5, 3.0]

num = 5
p_in_s = np.linspace(1e5, 10e5, num)
T_in_s = np.linspace(300, 700, num)

# Create 2D meshgrid
X, Y = np.meshgrid(p_in_s, T_in_s)

# Store results for plotting later
results = []

# Compute all data first
for lamba in lambdas:
    print(f"Computing for lambda = {lamba}...")
    equ = (1 / lamba)
    far = equ * far_stoich
    piston_input["far_goal"] = far  # fuel air ratio

    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    Z3 = np.zeros_like(X)
    Z4 = np.zeros_like(X)
    Z5 = np.zeros_like(X)
    Z6 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p_in = X[i, j]
            T_in = Y[i, j]
            flags = ["sweep"]
            piston_input["p_in"] = p_in  # fuel air ratio
            piston_input["T_in"] = T_in  # fuel air ratio
            (
                T_out,
                _,
                _,
                air_flow,
                p_max,
                T_max,
                _,
                _,
                indicated_power,
                _,
                _,
                heat_loss,
                p_tdc,
                _,
                nox_ppm,
                _,
                EI_nox,
                _,
                nox_spec,
            ) = run_piston_engine(piston_input, flags)

            Z1[i, j] = EI_nox
            Z2[i, j] = nox_ppm
            Z3[i, j] = air_flow
            Z4[i, j] = p_max
            Z5[i, j] = T_max
            Z6[i, j] = T_out

    # Store the results
    results.append({
        'lambda': lamba,
        'Z1': Z1.copy(),  # Important: copy the array
        'Z2': Z2.copy(),  # Important: copy the array
        'Z3': Z3.copy(),  # Important: copy the array
        'Z4': Z4.copy(),  # Important: copy the array
        'Z5': Z5.copy(),  # Important: copy the array
        'Z6': Z6.copy(),  # Important: copy the array
        'X': X,
        'Y': Y
    })

print("Computation complete. Saving data...")

# Save as CSV - flatten the 2D data into rows

# Create a comprehensive dataframe with all results
all_data = []
for result in results:
    lamba_val = result['lambda']
    X_flat = result['X'].flatten()
    Y_flat = result['Y'].flatten()
    Z1_flat = result['Z1'].flatten()
    Z2_flat = result['Z2'].flatten()
    Z3_flat = result['Z3'].flatten()
    Z4_flat = result['Z4'].flatten()
    Z5_flat = result['Z5'].flatten()
    Z6_flat = result['Z6'].flatten()

    # Create dataframe for this lambda
    df_lambda = pd.DataFrame({
        'lambda': lamba_val,
        'p_in_Pa': X_flat,
        'p_in_bar': X_flat * 1e-5,
        'T_in_K': Y_flat,
        'NOx_g_per_kg': Z1_flat,
        'NOx_ppm': Z2_flat,
        'air_flow': Z3_flat,
        'p_max': Z4_flat,
        'T_max': Z5_flat,
        'T_out': Z6_flat,
    })
    all_data.append(df_lambda)

# Combine all data
df_all = pd.concat(all_data, ignore_index=True)

# Save the complete dataset
df_all.to_csv('nox_contour_data/jetA_simulation.csv', index=False)
print(f"Complete data saved to 'nox_contour_data_jetA_simulation.csv' ({len(df_all)} rows)")
