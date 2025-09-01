import matplotlib.pyplot as plt

from neural_network.src import load_ANN
import numpy as np

from thermo import fuel_props

from piston_engine.engine import run_piston_engine

import importlib


folder = "jetA"
# Load the trained model
hidden_dim = 128
layers = 2
model = load_ANN(f"../models/{folder}_{hidden_dim}_{layers}_pinn.pth")
model = model.double()
print(model)


p_ratio = 1.0
cr = 10
bore = 0.15
v_mean = 10
T_fuel = 300
far = 0.035
Tin = 300
pin = 1.1e5

far_stoich, LHV = fuel_props(folder)

input_file = "4stroke_sampling"

input_dir = "piston_engine.input"
path = input_dir + "." + input_file

d = importlib.import_module(path)


piston_input = {
    'p_in': pin,
    'T_in': Tin,
    'p_ratio': p_ratio,
    'cycle': d.cycle,
    'cooling': d.cooling,
    'opposed': d.opposed,
    'cr': cr,
    'bore': bore,
    'bsr': d.bsr,
    'v_mean': v_mean,
    'lms': d.lms,
    'Twalls': d.Twalls,
    'ch': d.ch,
    'valve_timings': d.valve_timings,
    'n_valve': d.n_valve,
    'lv_max': bore*0.1,
    'cd': d.cd,
    'eta_c': d.eta_c,
    'mf_tot': d.mf_tot,
    'wa': d.wa,
    'wm': d.wm,
    'm_wiebe': d.m_wiebe,
    'phi_sc': d.phi_sc,
    'phi_cd': d.phi_cd,
    'T_fuel': T_fuel,
    'p_fuel': d.p_fuel,
    'it': d.it,
    'wiebe_type': d.wiebe_type,
    'valve_type': d.valve_type,
    'far_goal': far,
    'cylinders': d.cylinders,
    'fuel': d.fuel,
    'c1': d.c1,
    'c4': d.c4,
    'c5': d.c5,
    'premixed': d.premixed,
}

flags = ["sweep"]


num = 1000
fars = np.linspace(far_stoich/3.0,far_stoich/1.1,num)

powers = np.zeros([num])
noxs = np.zeros([num])
p_maxs = np.zeros([num])
T_maxs = np.zeros([num])
effs = np.zeros([num])
airflows = np.zeros([num])
heatlosses = np.zeros([num])


num_sim = 10
fars_sim = np.linspace(far_stoich/3.0,far_stoich/1.1,num_sim)

powers_sim = np.zeros([num_sim])
noxs_sim = np.zeros([num_sim])
p_maxs_sim = np.zeros([num_sim])
T_maxs_sim = np.zeros([num_sim])
effs_sim = np.zeros([num_sim])
airflows_sim = np.zeros([num_sim])
heatlosses_sim = np.zeros([num_sim])


for i in range(num_sim):

    piston_input["far_goal"] = fars_sim[i]

    (
        T4,
        work_piston,
        eta_th,
        air_flow,
        p_max,
        T_max,
        far,
        equ_trapped,
        indicated_power,
        friction_loss,
        aux_loss,
        heat_loss,
        p_tdc,
        outflow,
        no,
        imep,
        EI_nox,
        volume_eff,
        nox_spec,
    ) = run_piston_engine(piston_input, flags)


    powers_sim[i] = indicated_power
    noxs_sim[i] = EI_nox
    T_maxs_sim[i] = T_max
    p_maxs_sim[i] = p_max
    effs_sim[i] = eta_th
    airflows_sim[i] = air_flow
    heatlosses_sim[i] = heat_loss



for i in range(num):

    far_goal = fars[i]

    piston_input = np.atleast_2d(
        np.array([pin, Tin, p_ratio, cr, bore, v_mean, T_fuel, far_goal])
    )

    # use meta model to get outputs from the piston engine
    output = model.inference(piston_input)[0]


    indicated_power = output[0]
    heat_loss = output[1]
    nox_ppm = output[2]
    p_tdc = output[3]
    p_max = output[4]
    T_max = output[5]
    T34 = output[6]
    air_flow = output[7]

    fuel_flow = air_flow * far_goal

    out_flow = air_flow + fuel_flow

    # calculate EI NOX (convert from ppm to fraction and from kg to g)
    nox_gram = nox_ppm * out_flow * 1e-3
    EI_nox = nox_gram / fuel_flow

    # thermal efficiency
    eff = indicated_power / (fuel_flow * LHV)

    powers[i] = indicated_power
    noxs[i] = EI_nox
    T_maxs[i] = T_max
    p_maxs[i] = p_max
    effs[i] = eff
    airflows[i] = air_flow
    heatlosses[i] = heat_loss


fig, ax1 = plt.subplots()
ax1.plot(fars, noxs)
ax1.plot(fars_sim, noxs_sim)
ax1.set_xlabel("far [-]")
ax1.set_ylabel("EI_nox [g/kg]")

fig, ax2 = plt.subplots()
ax2.plot(fars, powers*1e-3)
ax2.plot(fars_sim, powers_sim*1e-3)
ax2.set_xlabel("far [-]")
ax2.set_ylabel("P [kW]")

fig, ax3 = plt.subplots()
ax3.plot(fars, p_maxs*1e-5)
ax3.plot(fars_sim, p_maxs_sim*1e-5)
ax3.set_xlabel("far [-]")
ax3.set_ylabel("pmax [bar]")

fig, ax4 = plt.subplots()
ax4.plot(fars, T_maxs)
ax4.plot(fars_sim, T_maxs_sim)
ax4.set_xlabel("far [-]")
ax4.set_ylabel("Tmax [K]")

fig, ax5 = plt.subplots()
ax5.plot(fars, effs)
ax5.plot(fars_sim, effs_sim)
ax5.set_xlabel("far [-]")
ax5.set_ylabel("thermal efficiency [-]")

fig, ax6 = plt.subplots()
ax6.plot(fars, airflows)
ax6.plot(fars_sim, airflows_sim)
ax6.set_xlabel("far [-]")
ax6.set_ylabel("air flow [kg/s]")


fig, ax7 = plt.subplots()
ax7.plot(fars, heatlosses*1e-3)
ax7.plot(fars_sim, heatlosses_sim*1e-3)
ax7.set_xlabel("far [-]")
ax7.set_ylabel("heat loss [kW]")

plt.show()


