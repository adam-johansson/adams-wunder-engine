from neural_network.src import load_ANN
from thermo import fuel_props

import numpy as np
import pandas as pd
import importlib

from piston_engine.engine import run_piston_engine


### LOAD THE MODELS

folder = "jetA"
# Load the trained model
hidden_dim = 128
layers = 2
model = load_ANN(f"../neural_network/models/{folder}_{hidden_dim}_{layers}.pth")
model = model.double()
print(model)

### CODE STARTS HERE ###

# DEFINE INPUT #
cr = 10
bore = 0.15
PI = 1.0
v_mean = 10.0
T_fuel = 300


far_stoich, _ = fuel_props("jetA")

lambdas = [1.1, 1.5, 2.0, 2.5, 3.0]

num = 100
p_in_s = np.linspace(1e5, 10e5, num)
T_in_s = np.linspace(300, 700, num)

# Create 2D meshgrid
X, Y = np.meshgrid(p_in_s, T_in_s)

def nn_func_2d(x, y):
    temp = model.inference(np.array([x, y, cr, bore, far, PI, v_mean, T_fuel]))

    nox_ppm = temp[0][8]

    return nox_ppm


# PLOTTING SETTINGS
levels = 20

# Store results for plotting later
results = []

# Compute all data first
for lamba in lambdas:
    print(f"Computing for lambda = {lamba}...")
    equ = (1 / lamba)
    far = equ * far_stoich

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p_in = X[i, j]
            T_in = Y[i, j]
            temp = model.inference(np.array([p_in, T_in, cr, bore, far, PI, v_mean, T_fuel]))

            p_max = temp[0][3]
            T_max = temp[0][4]

            air_flow = temp[0][2]
            nox_ppm = temp[0][8]

            # mass flow out of the engine
            mdot_out = air_flow * (1 + far)

            # calculate mass of NOx (and convert to fraction from ppm)
            mdot_nox = nox_ppm * mdot_out * 1e-6

            # mass flow of fuel
            mdot_fuel = far * air_flow

            # emission index (g of NOX per kg of fuel)
            EI_nox = (mdot_nox / mdot_fuel) * 1e3

            #EI_nox = nox_ppm * (1 + far) * 1e-3 / far

            Z[i, j] = EI_nox

    # Store the results
    results.append({
        'lambda': lamba,
        'Z': Z.copy(),  # Important: copy the array
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
    Z_flat = result['Z'].flatten()

    # Create dataframe for this lambda
    df_lambda = pd.DataFrame({
        'lambda': lamba_val,
        'p_in_Pa': X_flat,
        'p_in_bar': X_flat * 1e-5,
        'T_in_K': Y_flat,
        'NOx_g_per_kg': Z_flat
    })
    all_data.append(df_lambda)

# Combine all data
df_all = pd.concat(all_data, ignore_index=True)

# Save the complete dataset
df_all.to_csv('nox_contour_data.csv', index=False)
print(f"Complete data saved to 'nox_contour_data.csv' ({len(df_all)} rows)")
