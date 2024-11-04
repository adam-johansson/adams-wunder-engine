## compare simulation vs meta model

import numpy as np

import matplotlib.pyplot as plt

from neural_network.src import load_ANN

from piston_engine.engine import run_piston_engine  # import the piston engine function
import importlib

# import all the input files for real piston simulation
input_file_pist = "4stroke_hydrogen_bad_point"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist

d = importlib.import_module(path_pist)

flags = ['sweep']  # normal case no plots


# Load the trained model
simple = False
hidden_dim = 128
layers = 1
model = load_ANN(f'./../models/straight_{hidden_dim}_{layers}.pth')
print(model)



# sweep of far and inlet temp
#p_in = 10.5e5
#T_in = 570
#cr = 7
#bore = 0.17
#far = 0.02923 / 1.3
#PI = 1.0
#v_mean = 10.5
#T_fuel = 430

p_in = d.p_in
T_in = d.T_in
cr = d.cr
bore = d.d
far = d.throttle
PI = d.p_ratio
v_mean = d.v_mean
T_fuel = d.T_fuel


points_nn = 1000

#fars_nn = np.linspace(0.02923 / 3.0, 0.02923 / 1.1, 1000)
Tfuels_nn = np.linspace(300, 500, points_nn)

points = 20

#fars = np.linspace(0.02923 / 3.0, 0.02923 / 1.1, 10)
Tfuels = np.linspace(300, 500, points)

variable_nn = Tfuels_nn
variable = Tfuels


if simple:
    outputs_nn = []
else:
    outputs_nn = np.zeros((points_nn, 8))

i = 0
for T_fuel in variable_nn:
    temp = model.inference(np.array([p_in, T_in, cr, bore, far, PI, v_mean, T_fuel]))
    if simple:
        outputs_nn.append(temp[0][0])
    else:
        outputs_nn[i, :] = temp[0]
        i = i + 1


if simple:
    outputs = []
else:
    outputs = np.zeros((points, 8))


i = 0
for T_fuel in variable:
    # input data to real simulation

    data = [p_in, T_in, PI, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
            v_mean, d.lms, d.Twalls, d.ch,
            d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
            d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far,
            d.cylinders, d.fuel, d.c1, d.c4, d.c5]

    T4, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, indicated_power, friction_loss, aux_loss, \
        heat_loss, p_tdc = run_piston_engine(data, flags)

    if simple:
        outputs.append(indicated_power*1e-3)
    else:
        outputs[i, :] = np.array([T4, eta_th, air_flow, p_max, T_max, indicated_power*1e-3, heat_loss*1e-3, p_tdc*1e-5])
        i = i + 1


if simple:
    # plot the real values for power and the predicted one
    fig, ax6 = plt.subplots()
    ax6.plot(Tfuels_nn, powers_nn, label="Meta model")
    ax6.plot(Tfuels, powers, label="Simulation", marker='o')
    ax6.set_xlabel(r'far [-]')
    ax6.set_ylabel(r'P [kW]')
    ax6.set_title(fr'Meta model vs simulations')
    #ax2.set_ylim(-10, 10)
    #ax1.set_xlim(100, epochs)
    ax6.legend()

else:
    for i in range(8):
        plt.figure()
        plt.plot(variable_nn, outputs_nn[:, i], label="Meta model")
        plt.plot(variable, outputs[:, i], label="Simulation", marker='o')
        plt.legend()
        plt.xlabel(f'Variable ')
        plt.ylabel(f'Output: {i}')

        # Show/save figure as desired.
        #plt.show()




# Can show all figures at once by calling plt.show() here, outside the loop.
plt.show()
