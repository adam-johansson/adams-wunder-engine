import pandas as pd
import numpy as np
import thermo
import matplotlib.pyplot as plt
from neural_network.src import load_ANN, surrogate_wrapper

# KOLLA DATAPUNKTERNA SOM MODELLEN ÄR TRÄNAD PÅ
fuel = "jetA"

data = pd.read_csv("../../neural_network/input_data/" + fuel + "/data.csv", index_col=0)

data = data[data.eff != 0]

# input
p_in = np.array((data.p_in))
T_in = np.array((data.T_in))
far = np.array((data.far_goal))
p_ratio = np.array((data.PI))
T_fuel = np.array((data.T_fuel))
cr = np.array((data.cr))
bore = np.array((data.bore))
v_mean = np.array((data.v_mean))

# output
T_out = np.array((data.T_out))
eff = np.array((data.eff))
air_flow = np.array((data.air_flow))
power = np.array((data.power))
heat_loss = np.array((data.heat_loss))

# Initialise arrays for energies
h_in = np.zeros_like(p_in)
h_fuel = np.zeros_like(p_in)
h_out = np.zeros_like(p_in)

far_stoich, LHV = thermo.fuel_props(fuel)

equ_in = 0.0

# Specific energy in, out and fuel enthalpy
for i in range(h_in.shape[0]):
    h_in_temp, _, _, _, _, _, s, _ = thermo.mixture(T_in[i], p_in[i], equ_in, fuel_type=fuel)
    h_out_temp, _, _, _, _, _, s, _ = thermo.mixture(T_out[i], p_in[i] * p_ratio[i], far[i] / far_stoich, fuel_type=fuel)
    _, h_fuel_temp, s, _ = thermo.JETA_L(T_fuel[i])
    h_in[i] = h_in_temp
    h_out[i] = h_out_temp
    h_fuel[i] = h_fuel_temp

# Enthalpy in, out and fuel
H_in = h_in * air_flow
H_fuel = h_fuel * air_flow * far
H_out = h_out * air_flow * (1 + far)


# Conservation of energy
diff = H_in + H_fuel - H_out - power - heat_loss
fuel_energy_in = far * air_flow * LHV
diff_percentage = diff / fuel_energy_in
thermal_efficiency = power / fuel_energy_in
diff_efficiency = np.abs((thermal_efficiency - eff)) / eff

# PLOT THE RESULTS
fig, ax1 = plt.subplots()
#ax1.plot(far_s, outputs_nn[:, 0], label="nn")
ax1.plot(diff, "r+", label="simulation")
ax1.set_title("Energy conservation [J]")
#plt.legend()

fig, ax2 = plt.subplots()
#ax1.plot(far_s, outputs_nn[:, 0], label="nn")
ax2.plot(diff_percentage * 100, "r+", label="simulation")
ax2.set_title("Energy conservation [\% of fuel energy]")
#plt.legend()

fig, ax3 = plt.subplots()
#ax1.plot(far_s, outputs_nn[:, 0], label="nn")
ax3.plot(diff_efficiency * 100, "g.", label="simulation")
ax3.set_title("Diff in thermal efficiency [\%]")
#plt.legend()

plt.show()


h_in_nn = h_in
h_fuel_nn = h_fuel
h_out_nn = np.zeros_like(p_in)
power_nn = np.zeros_like(p_in)
heat_loss_nn = np.zeros_like(p_in)
air_flow_nn = np.zeros_like(p_in)
efficiency_nn = np.zeros_like(p_in)

### LOAD THE MODELS
# Load the trained model
hidden_dim = 128
layers = 2
model = load_ANN(f"../../neural_network/models/{fuel}_{hidden_dim}_{layers}_pinn.pth")
model = model.double()
print(model)


# KOLLA NN (GÖR SAMMA SOM FÖR DATAPUNKTERNA)
for i in range(p_in.shape[0]):

    output_temp = model.inference(np.array([p_in[i], T_in[i], cr[i], bore[i], far[i],
                                            p_ratio[i], v_mean[i], T_fuel[i]]))

    T_out_temp = output_temp[0][0]
    efficiency_temp = output_temp[0][1]
    air_flow_temp = output_temp[0][2]
    power_temp = output_temp[0][5]
    heat_loss_temp = output_temp[0][6]



    h_out_temp, _, _, _, _, _, s, _ = thermo.mixture(T_out_temp, p_in[i] * p_ratio[i], far[i] / far_stoich, fuel_type=fuel)
    h_out_nn[i] = h_out_temp
    power_nn[i] = power_temp
    heat_loss_nn[i] = heat_loss_temp
    air_flow_nn[i] = air_flow_temp
    efficiency_nn[i] = efficiency_temp



# Enthalpy in, out and fuel
H_in_nn = h_in_nn * air_flow_nn
H_fuel_nn = h_fuel_nn * air_flow_nn * far
H_out_nn = h_out_nn * air_flow_nn * (1 + far)


# Conservation of energy
diff_nn = H_in_nn + H_fuel_nn - H_out_nn - power_nn - heat_loss_nn
fuel_energy_in_nn = far * air_flow_nn * LHV
diff_percentage_nn = diff_nn / fuel_energy_in_nn
thermal_efficiency_nn = power_nn / fuel_energy_in_nn
diff_efficiency_nn = np.abs((thermal_efficiency_nn - eff)) / eff

# PLOT THE RESULTS
fig, ax1 = plt.subplots()
ax1.plot(diff_nn, "r+", label="nn")
ax1.set_title("Energy conservation [J]")
#plt.legend()

fig, ax2 = plt.subplots()
ax2.plot(diff_percentage_nn * 100, "r+", label="nn")
ax2.set_title("Energy conservation [\% of fuel energy]")
#plt.legend()

fig, ax3 = plt.subplots()
#ax1.plot(far_s, outputs_nn[:, 0], label="nn")
ax3.plot(diff_efficiency_nn * 100, "g.", label="nn")
ax3.set_title("Diff in thermal efficiency [\%]")
#plt.legend()

plt.show()

# KOLLA / SKAPA NN WRAPPER (dvs bevara energin) Jämför sedan men ren NN
h_in_nn_wrapper = h_in
h_fuel_nn_wrapper = h_fuel
h_out_nn_wrapper = np.zeros_like(p_in)
power_nn_wrapper = np.zeros_like(p_in)
heat_loss_nn_wrapper = np.zeros_like(p_in)
air_flow_nn_wrapper = np.zeros_like(p_in)
efficiency_nn_wrapper = np.zeros_like(p_in)


for i in range(p_in.shape[0]):

    input_array = np.array([p_in[i], T_in[i], cr[i], bore[i], far[i],
                                            p_ratio[i], v_mean[i], T_fuel[i]])

    output_temp = surrogate_wrapper.nn_output_energy_conserved(model, input_array, fuel_type=fuel)

    T_out_temp = output_temp[0]
    efficiency_temp = output_temp[1]
    air_flow_temp = output_temp[2]
    power_temp = output_temp[5]
    heat_loss_temp = output_temp[6]

    h_out_temp, _, _, _, _, _, s, _ = thermo.mixture(T_out_temp, p_in[i] * p_ratio[i], far[i] / far_stoich, fuel_type=fuel)
    h_out_nn_wrapper[i] = h_out_temp
    power_nn_wrapper[i] = power_temp
    heat_loss_nn_wrapper[i] = heat_loss_temp
    air_flow_nn_wrapper[i] = air_flow_temp
    efficiency_nn_wrapper[i] = efficiency_temp



# Enthalpy in, out and fuel
H_in_nn_wrapper = h_in_nn_wrapper * air_flow_nn_wrapper
H_fuel_nn_wrapper = h_fuel_nn_wrapper * air_flow_nn_wrapper * far
H_out_nn_wrapper = h_out_nn_wrapper * air_flow_nn_wrapper * (1 + far)


# Conservation of energy
diff_nn_wrapper = H_in_nn_wrapper + H_fuel_nn_wrapper - H_out_nn_wrapper - power_nn_wrapper - heat_loss_nn_wrapper
fuel_energy_in_nn_wrapper = far * air_flow_nn_wrapper * LHV
diff_percentage_nn_wrapper = diff_nn_wrapper / fuel_energy_in_nn_wrapper
thermal_efficiency_nn_wrapper = power_nn_wrapper / fuel_energy_in_nn
diff_efficiency_nn_wrapper = np.abs((thermal_efficiency_nn_wrapper - eff)) / eff

# PLOT THE RESULTS
fig, ax1 = plt.subplots()
ax1.plot(diff_nn_wrapper, "r+", label="nn conserved")
ax1.set_title("Energy conservation [J]")
#plt.legend()

fig, ax2 = plt.subplots()
ax2.plot(diff_percentage_nn_wrapper * 100, "r+", label="nn conserved")
ax2.set_title("Energy conservation [\% of fuel energy]")
#plt.legend()

fig, ax3 = plt.subplots()
#ax1.plot(far_s, outputs_nn[:, 0], label="nn")
ax3.plot(diff_efficiency_nn_wrapper * 100, "g.", label="nn conserved")
ax3.set_title("Diff in thermal efficiency [\%]")
#plt.legend()

plt.show()
