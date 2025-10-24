import matplotlib.pyplot as plt

from neural_network.src import load_ANN
import numpy as np

from thermo import fuel_props



folder = "jetA"
# Load the trained model
hidden_dim = 128
layers = 2
model = load_ANN(f"../models/{folder}_{hidden_dim}_{layers}.pth")
model = model.double()
print(model)


#thermal efficiency
# power
# nox
# Tout
# mass flow

p_ratio = 1.2
cr = 10
bore = 0.15
v_mean = 15
T_fuel = 350
far = 0.045
p_in = 10e5
T_in = 650


num = 100
params = np.linspace(0.03,0.055,num)
param_name = "far"


powers = np.zeros(num)
noxs = np.zeros(num)
Touts = np.zeros(num)
mdots = np.zeros(num)
effs = np.zeros(num)

fuel_type = folder

far_stoich, LHV = fuel_props(fuel_type)



for i in range(num):

    far = params[i]

    piston_input = np.atleast_2d(
        np.array([p_in, T_in, p_ratio, cr, bore, v_mean, T_fuel, far])
    )

    output_raw = model.inference(piston_input)[0]

    indicated_power = output_raw[0]
    nox_ppm = output_raw[1]
    p_tdc = output_raw[2]
    p_max = output_raw[3]
    T_out = output_raw[4]
    T_max = output_raw[5]
    m_in = output_raw[6]


    fuel_flow = m_in * far
    out_flow = m_in + fuel_flow


    # calculate EI NOX (convert from ppm to fraction and from kg to g)
    nox_gram = nox_ppm * out_flow * 1e-3
    EI_nox = nox_gram / fuel_flow

    powers[i] = indicated_power
    noxs[i] = EI_nox
    effs[i] = indicated_power / (fuel_flow * LHV)
    mdots[i] = m_in





_, ax1 = plt.subplots()

ax1.plot(params, powers*1e-3)
ax1.set_xlabel(f"{param_name}")
ax1.set_ylabel(f"power [kW]")

_, ax2 = plt.subplots()

ax2.plot(params, effs*1e2)
ax2.set_xlabel(f"{param_name}")
ax2.set_ylabel(f"efficiency [percent]")

_, ax3 = plt.subplots()

ax3.plot(params, mdots)
ax3.set_xlabel(f"{param_name}")
ax3.set_ylabel(f"mass flow [kg/s]")

plt.show()
