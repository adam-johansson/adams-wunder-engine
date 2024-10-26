## compare simulation vs meta model

import numpy as np

import matplotlib.pyplot as plt

from neural_network.src import load_ANN

from piston_engine.engine import run_piston_engine  # import the piston engine function
import importlib

# import all the input files for real piston simulation
input_file_pist = "4stroke_hydrogen"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist

d = importlib.import_module(path_pist)

flags = ['sweep']  # normal case no plots


# Load the trained model
hidden_dim = 128
layers = 1
model = load_ANN(f'./../models/straight_{hidden_dim}_{layers}.pth')
print(model)



# sweep of far and inlet temp
p_in = 10.5e5
T_in = 570
cr = 7
bore = 0.17
far = 0.02923 / 1.3
PI = 1.0
v_mean = 10.5
T_fuel = 430


fars_nn = np.linspace(0.02923 / 1.5, 0.02923 / 1.1, 1000)

powers_nn = []
for far in fars_nn:
    temp = model.inference(np.array([p_in, T_in, cr, bore, far, PI, v_mean, T_fuel]))
    powers_nn.append(temp[0][0])


fars = np.linspace(0.02923 / 1.5, 0.02923 / 1.1, 10)

powers = []

for far in fars:
    # input data to real simulation

    data = [p_in, T_in, PI, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
            v_mean, d.lms, d.Twalls, d.ch,
            d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
            d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far,
            d.cylinders, d.fuel, d.c1, d.c4, d.c5]

    T4, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, indicated_power, friction_loss, aux_loss, \
        heat_loss, p_tdc = run_piston_engine(data, flags)

    powers.append(indicated_power*1e-3)


# plot the real values for power and the predicted one
fig, ax6 = plt.subplots()
ax6.plot(fars_nn, powers_nn, label="Meta model")
ax6.plot(fars, powers, label="Simulation", marker='o')
ax6.set_xlabel(r'far [-]')
ax6.set_ylabel(r'P [kW]')
ax6.set_title(fr'Meta model vs simulations')
#ax2.set_ylim(-10, 10)
#ax1.set_xlim(100, epochs)
ax6.legend()


plt.show()