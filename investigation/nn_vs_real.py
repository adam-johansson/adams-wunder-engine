import matplotlib.pyplot as plt
import numpy as np

from CCE.src import surrogate
from piston_engine.engine import run_piston_engine  # import the piston engine function
import importlib

# import all the input files for real piston simulation
input_file = "4stroke_hydrogen"

path = input_file

d = importlib.import_module(path)

flags = ["sweep"]  # normal case no plots


# neural network
model = surrogate.load_nn()


pin = 5e5
Tin = 700
cr = 10
bore = 0.11
p_ratio = 1.2
v_mean = 18
T_fuel = 450


nr = 10

far_s = np.linspace(0.02923 / 3.0, 0.02923 / 1.0, nr)
outputs_nn = np.zeros((nr, 7))
outputs_real = np.zeros((nr, 7))

i = 0

for far in far_s:
    input_pist = np.atleast_2d(
        np.array([pin, Tin, cr, bore, far, p_ratio, v_mean, T_fuel])
    )
    output_nn = surrogate.nn_output(input_pist, model)
    outputs_nn[i, :] = output_nn

    data = [
        pin,
        Tin,
        p_ratio,
        d.cycle,
        d.thermo,
        d.cooling,
        d.opposed,
        cr,
        bore,
        d.bsr,
        v_mean,
        d.lms,
        d.Twalls,
        d.ch,
        d.valve_timings,
        d.n_valve,
        0.1 * bore,
        d.cd,
        d.eta_c,
        d.mf_tot,
        d.wa,
        d.wm,
        d.m_wiebe,
        d.phi_sc,
        d.phi_cd,
        d.T_fuel,
        d.p_fuel,
        d.it,
        d.wiebe_type,
        d.valve_type,
        far,
        d.cylinders,
        d.fuel,
        d.c1,
        d.c4,
        d.c5,
    ]

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
    ) = run_piston_engine(data, flags)

    output_real = (
        T4,
        air_flow,
        p_max,
        T_max,
        indicated_power * 1e-3,
        heat_loss * 1e-3,
        p_tdc * 1e-5,
    )
    outputs_real[i, :] = output_real
    print(i)
    i += 1

fig, ax1 = plt.subplots()
ax1.plot(far_s, outputs_nn[:, 0], label="nn")
ax1.plot(far_s, outputs_real[:, 0], label="simulation")
ax1.set_title("Tout")
plt.legend()

fig, ax2 = plt.subplots()
ax2.plot(far_s, outputs_nn[:, 1], label="nn")
ax2.plot(far_s, outputs_real[:, 1], label="simulation")
ax2.set_title("air flow")
plt.legend()

fig, ax3 = plt.subplots()
ax3.plot(far_s, outputs_nn[:, 2], label="nn")
ax3.plot(far_s, outputs_real[:, 2], label="simulation")
ax3.set_title("p_max")
plt.legend()

fig, ax4 = plt.subplots()
ax4.plot(far_s, outputs_nn[:, 3], label="nn")
ax4.plot(far_s, outputs_real[:, 3], label="simulation")
ax4.set_title("Tmax")
plt.legend()

fig, ax5 = plt.subplots()
ax5.plot(far_s, outputs_nn[:, 4], label="nn")
ax5.plot(far_s, outputs_real[:, 4], label="simulation")
ax5.set_title("power")
plt.legend()

fig, ax6 = plt.subplots()
ax6.plot(far_s, outputs_nn[:, 5], label="nn")
ax6.plot(far_s, outputs_real[:, 5], label="simulation")
ax6.set_title("heat loss")
plt.legend()

fig, ax7 = plt.subplots()
ax7.plot(far_s, outputs_nn[:, 6], label="nn")
ax7.plot(far_s, outputs_real[:, 6], label="simulation")
ax7.set_title("p_tdc")
plt.legend()


nr = 10

far = 0.02923 / 1.5
bores = np.linspace(0.05, 0.2, nr)
outputs_nn = np.zeros((nr, 7))
outputs_real = np.zeros((nr, 7))

i = 0

for bore in bores:
    input_pist = np.atleast_2d(
        np.array([pin, Tin, cr, bore, far, p_ratio, v_mean, T_fuel])
    )
    output_nn = surrogate.nn_output(input_pist, model)
    outputs_nn[i, :] = output_nn

    data = [
        pin,
        Tin,
        p_ratio,
        d.cycle,
        d.thermo,
        d.cooling,
        d.opposed,
        cr,
        bore,
        d.bsr,
        v_mean,
        d.lms,
        d.Twalls,
        d.ch,
        d.valve_timings,
        d.n_valve,
        0.1 * bore,
        d.cd,
        d.eta_c,
        d.mf_tot,
        d.wa,
        d.wm,
        d.m_wiebe,
        d.phi_sc,
        d.phi_cd,
        d.T_fuel,
        d.p_fuel,
        d.it,
        d.wiebe_type,
        d.valve_type,
        far,
        d.cylinders,
        d.fuel,
        d.c1,
        d.c4,
        d.c5,
    ]

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
    ) = run_piston_engine(data, flags)

    output_real = (
        T4,
        air_flow,
        p_max,
        T_max,
        indicated_power * 1e-3,
        heat_loss * 1e-3,
        p_tdc * 1e-5,
    )
    outputs_real[i, :] = output_real
    print(i)
    i += 1

fig, ax8 = plt.subplots()
ax8.plot(bores, outputs_nn[:, 0], label="nn")
ax8.plot(bores, outputs_real[:, 0], label="simulation")
ax8.set_title("Tout, bore")
plt.legend()

fig, ax9 = plt.subplots()
ax9.plot(bores, outputs_nn[:, 1], label="nn")
ax9.plot(bores, outputs_real[:, 1], label="simulation")
ax9.set_title("air flow, bore")
plt.legend()

fig, ax10 = plt.subplots()
ax10.plot(bores, outputs_nn[:, 2], label="nn")
ax10.plot(bores, outputs_real[:, 2], label="simulation")
ax10.set_title("p_max, bore")
plt.legend()

fig, ax11 = plt.subplots()
ax11.plot(bores, outputs_nn[:, 3], label="nn")
ax11.plot(bores, outputs_real[:, 3], label="simulation")
ax11.set_title("Tmax, bore")
plt.legend()

fig, ax12 = plt.subplots()
ax12.plot(bores, outputs_nn[:, 4], label="nn")
ax12.plot(bores, outputs_real[:, 4], label="simulation")
ax12.set_title("power, bore")
plt.legend()

fig, ax13 = plt.subplots()
ax13.plot(bores, outputs_nn[:, 5], label="nn")
ax13.plot(bores, outputs_real[:, 5], label="simulation")
ax13.set_title("heat loss, bore")
plt.legend()

fig, ax14 = plt.subplots()
ax14.plot(bores, outputs_nn[:, 6], label="nn")
ax14.plot(bores, outputs_real[:, 6], label="simulation")
ax14.set_title("p_tdc, bore")
plt.legend()

plt.show()

# Main problems air flow and p_tdc [6], but probably most air flow [1]
