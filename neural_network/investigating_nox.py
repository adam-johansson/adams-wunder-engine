import numpy as np
import matplotlib.pyplot as plt
import importlib


from neural_network.src import load_ANN
from thermo import fuel_props

from piston_engine.engine import run_piston_engine



folder = "jetA"
# Load the trained model
hidden_dim = 128
layers = 2
model = load_ANN(f"./models/{folder}_{hidden_dim}_{layers}.pth")
model = model.double()
print(model)

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







# sweep of far and inlet temp
p_in = 2e5
T_in = 400
cr = 10
bore = 0.15
PI = 1.0
v_mean = 10.0
T_fuel = 300


far_stoich, _ = fuel_props("jetA")
fars = np.linspace(far_stoich / 3.0, far_stoich / 1.1, 1000)


nox_nn = []
p_max_nn = []
T_max_nn = []

for far in fars:
    temp = model.inference(np.array([p_in, T_in, cr, bore, far, PI, v_mean, T_fuel]))
    p_max_nn.append(temp[0][3])
    T_max_nn.append(temp[0][4])
    nox_nn.append(temp[0][8])


p_max_nn = np.array(p_max_nn)
T_max_nn = np.array(T_max_nn)
nox_nn = np.array(nox_nn)

piston_input["p_in"] = p_in
piston_input["T_in"] = T_in
piston_input["p_ratio"] = PI
piston_input["cr"] = cr
piston_input["bore"] = bore
lv_max = 0.1 * bore
piston_input["lv_max"] = lv_max
piston_input["T_fuel"] = T_fuel

fars_sparse = np.linspace(far_stoich / 3.0, far_stoich / 1.1, 10)


nox_simulation = []
p_max_simulation = []
T_max_simulation = []

for far in fars_sparse:
    flags = ["sweep"]
    piston_input["far_goal"] = far  # fuel air ratio
    (
        T_out,
        _,
        _,
        air_flow,
        p_max_temp,
        T_max_temp,
        _,
        _,
        indicated_power,
        _,
        _,
        heat_loss,
        p_tdc,
        _,
        nox_temp,
        _,
        EI_nox,
        _,
        nox_spec,
    ) = run_piston_engine(piston_input, flags)

    nox_simulation.append(nox_temp)
    p_max_simulation.append(p_max_temp)
    T_max_simulation.append(T_max_temp)

nox_simulation = np.array(nox_simulation)
p_max_simulation = np.array(p_max_simulation)
T_max_simulation = np.array(T_max_simulation)


fig, ax1 = plt.subplots()
ax1.plot(far_stoich/fars, nox_nn, label='Neural Network', linewidth=2)
ax1.plot(far_stoich/fars_sparse, nox_simulation, label='Simulation',
         marker='o', markersize=6, linewidth=1.5, markerfacecolor='white',
         markeredgewidth=1.5)
ax1.set_xlabel(r"Lambda [-]")
ax1.set_ylabel(r"NOX [ppm]")
ax1.legend()
ax1.grid(True, alpha=0.3)

fig, ax2 = plt.subplots()
ax2.plot(far_stoich/fars, p_max_nn * 1e-5, label='Neural Network', linewidth=2)
ax2.plot(far_stoich/fars_sparse, p_max_simulation * 1e-5, label='Simulation',
         marker='o', markersize=6, linewidth=1.5, markerfacecolor='white',
         markeredgewidth=1.5)
ax2.set_xlabel(r"Lambda [-]")
ax2.set_ylabel(r"p max [bar]")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig, ax3 = plt.subplots()
ax3.plot(far_stoich/fars, T_max_nn, label='Neural Network', linewidth=2)
ax3.plot(far_stoich/fars_sparse, T_max_simulation, label='Simulation',
         marker='o', markersize=6, linewidth=1.5, markerfacecolor='white',
         markeredgewidth=1.5)
ax3.set_xlabel(r"Lambda [-]")
ax3.set_ylabel(r"T max [K]")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.show()
