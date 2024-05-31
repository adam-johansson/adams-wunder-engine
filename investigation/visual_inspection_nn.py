import matplotlib.pyplot as plt
import numpy as np

from CCE.src import surrogate
import importlib

# import all the input files for real piston simulation
input_file = "4stroke_hydrogen"

path = input_file

d = importlib.import_module(path)

flags = ['sweep']  # normal case no plots


# neural network
model = surrogate.load_nn()


pin = 5e5     # också denna
Tin = 700     # denna också
cr = 10
bore = 0.11
p_ratio = 1.2
v_mean = 18
T_fuel = 450
far = 0.02923/1.5  # denna varieras under optimeringen


nr = 10000

far_s = np.linspace(0.02923/3.0, 0.02923/1.0, nr)
pins = np.linspace(3e5, 9e5, nr)
Tins = np.linspace(400, 1000, nr)
outputs_nn = np.zeros((nr, 7))
outputs_real = np.zeros((nr, 7))

xs = Tins

# problems: p_tdc and air flow vs far. heat loss vs Tin,

i = 0

for Tin in xs:
    input_pist = np.atleast_2d(np.array([pin, Tin, cr, bore, far, p_ratio, v_mean, T_fuel]))
    output_nn = surrogate.nn_output(input_pist, model)
    outputs_nn[i, :] = output_nn

    #print(i)
    i += 1

fig, ax1 = plt.subplots()
ax1.plot(xs, outputs_nn[:, 0], label="nn")
ax1.set_title('Tout')
plt.legend()

fig, ax2 = plt.subplots()
ax2.plot(xs, outputs_nn[:, 1], label="nn")
ax2.set_title('air flow')
plt.legend()

fig, ax3 = plt.subplots()
ax3.plot(xs, outputs_nn[:, 2], label="nn")
ax3.set_title('p_max')
plt.legend()

fig, ax4 = plt.subplots()
ax4.plot(xs, outputs_nn[:, 3], label="nn")
ax4.set_title('Tmax')
plt.legend()

fig, ax5 = plt.subplots()
ax5.plot(xs, outputs_nn[:, 4], label="nn")
ax5.set_title('power')
plt.legend()

fig, ax6 = plt.subplots()
ax6.plot(xs, outputs_nn[:, 5], label="nn")
ax6.set_title('heat loss')
plt.legend()

fig, ax7 = plt.subplots()
ax7.plot(xs, outputs_nn[:, 6], label="nn")
ax7.set_title('p_tdc')
plt.legend()

plt.show()