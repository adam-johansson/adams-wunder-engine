import matplotlib.pyplot as plt
import numpy as np
import importlib

from piston_engine.engine import run_piston_engine  # import the piston engine function

from piston_engine.src.misc.plot_output import plot_validation_nasa
from piston_engine.src.misc.post_processing import validation_error
from numpy import genfromtxt

# Setting up the piston engine

input_dir = "piston_engine.input.H2_validation_italian"
# flags: plot, output, validation, sweep

flags = ["save"]
# just to be able to get 1 output instead of 4

p = []
V = []
Q = []
phi = []


for i in range(3):
    print(i)

    if i == 0:
        input_file = "4stroke_hydrogen_validation_italian_04_v2"
    elif i == 1:
        input_file = "4stroke_hydrogen_validation_italian_06_v2"
    else:
        input_file = "4stroke_hydrogen_validation_italian_08_v2"


    path = input_dir + "." + input_file

    d = importlib.import_module(path)

    v_mean = d.v_mean
    s = d.d / d.bsr

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
        'far_goal': d.throttle,
        'cylinders': d.cylinders,
        'fuel': d.fuel,
        'c1': d.c1,
        'c4': d.c4,
        'c5': d.c5,
        'premixed': d.premixed,
    }

    # run the simulation
    #print(p * 1e-5, T, cr, bore, far_goal, p_ratio, v_mean, fuel_t)
    (
        T_out,
        work_piston,
        eta_th,
        air_flow,
        p_max,
        T_max,
        far_avg,
        equ_trapped,
        induced_power,
        _,
        _,
        heat_loss,
        p_tdc,
        _,
        nox,
        _,
        EI_nox,
        volume_eff,
        nox_spec,
    ) = run_piston_engine(piston_input, flags)

    print(induced_power*1e-3)

    P_temp = genfromtxt("piston_engine/simulation_data/P.csv", delimiter=",")
    V_temp = genfromtxt("piston_engine/simulation_data/V.csv", delimiter=",")
    Q_temp = genfromtxt("piston_engine/simulation_data/Q_in.csv", delimiter=",")
    phi_temp = genfromtxt("piston_engine/simulation_data/phi.csv", delimiter=",")

    p.append(P_temp)
    V.append(V_temp)
    Q.append(Q_temp)
    phi.append(phi_temp)


p = np.array(p)
Q = np.array(Q)
V = np.array(V)
phi = np.array(phi)

p = np.atleast_2d(p)
Q = np.atleast_2d(Q)
V = np.atleast_2d(V)
phi = np.atleast_2d(phi)

# calculate ARHH
# testa variablet gamma
gamma = 1.35
dtdphi = s / (np.pi * v_mean)
#dtdphi = np.pi / 180

dpdphi0 = np.gradient(p[0, :], phi[0, :], axis=0)
dVdphi0 = np.gradient(V[0, :], phi[0, :], axis=0)

dpdphi1 = np.gradient(p[1,:], phi[1,:], axis=0)
dVdphi1 = np.gradient(V[1,:], phi[1,:], axis=0)

dpdphi2 = np.gradient(p[2,:], phi[2,:], axis=0)
dVdphi2 = np.gradient(V[2,:], phi[2,:], axis=0)

dpdphi = np.concatenate((np.atleast_2d(dpdphi0), np.atleast_2d(dpdphi1), np.atleast_2d(dpdphi2)), axis=0)
dVdphi = np.concatenate((np.atleast_2d(dVdphi0), np.atleast_2d(dVdphi1), np.atleast_2d(dVdphi2)), axis=0)

dqdphi_apparent = (V * dpdphi + gamma * p * dVdphi) / (gamma - 1)


#dqdt_apparent = dqdphi_apparent / dtdphi
ddegdphi = np.pi / 180
dqddeg_apparent = dqdphi_apparent / ddegdphi

# wrong name but had already implemented the name Q later
#Q = dqdt_apparent
Q = dqddeg_apparent * 1e-3

# get units right. Bar and kW and degrees
p = p * 1e-5 #bar
#Q = Q * 1e-3 #kW
phi = phi * 180 / np.pi #deg


p_04_sim = np.concatenate((np.atleast_2d(p[0, :]), np.atleast_2d(phi[0, :])), axis=0)
p_06_sim = np.concatenate((np.atleast_2d(p[1, :]), np.atleast_2d(phi[1, :])), axis=0)
p_08_sim = np.concatenate((np.atleast_2d(p[2, :]), np.atleast_2d(phi[2, :])), axis=0)

Q_04_sim = np.concatenate((np.atleast_2d(Q[0, :]), np.atleast_2d(phi[0, :])), axis=0)
Q_06_sim = np.concatenate((np.atleast_2d(Q[1, :]), np.atleast_2d(phi[1, :])), axis=0)
Q_08_sim = np.concatenate((np.atleast_2d(Q[2, :]), np.atleast_2d(phi[2, :])), axis=0)


p_04_sim = np.transpose(p_04_sim)
p_06_sim = np.transpose(p_06_sim)
p_08_sim = np.transpose(p_08_sim)

Q_04_sim = np.transpose(Q_04_sim)
Q_06_sim = np.transpose(Q_06_sim)
Q_08_sim = np.transpose(Q_08_sim)


# load validation data

p_04_val = genfromtxt("piston_engine/validation_output_data/H2/p04_val.dat")
p_06_val = genfromtxt("piston_engine/validation_output_data/H2/p06_val.dat")
p_08_val = genfromtxt("piston_engine/validation_output_data/H2/p08_val.dat")

Q_04_val = genfromtxt("piston_engine/validation_output_data/H2/Q04_val.dat")
Q_06_val = genfromtxt("piston_engine/validation_output_data/H2/Q06_val.dat")
Q_08_val = genfromtxt("piston_engine/validation_output_data/H2/Q08_val.dat")

p_04_val[:, 0] = p_04_val[:, 0] - 360
p_06_val[:, 0] = p_06_val[:, 0] - 360
p_08_val[:, 0] = p_08_val[:, 0] - 360

Q_04_val[:, 0] = Q_04_val[:, 0] - 360
Q_06_val[:, 0] = Q_06_val[:, 0] - 360
Q_08_val[:, 0] = Q_08_val[:, 0] - 360


# cut the simulation data
phi_min = np.min([p_04_val[0, 0], p_06_val[0, 0], p_08_val[0, 0]])
phi_max = np.min([p_04_val[-1, 0], p_06_val[-1, 0], p_08_val[-1, 0]])

p_04_sim = p_04_sim[(p_04_sim[:, 1] > phi_min) & (p_04_sim[:, 1] < phi_max)]
p_06_sim = p_06_sim[(p_06_sim[:, 1] > phi_min) & (p_06_sim[:, 1] < phi_max)]
p_08_sim = p_08_sim[(p_08_sim[:, 1] > phi_min) & (p_08_sim[:, 1] < phi_max)]

phi_min = np.min([Q_04_val[0, 0], Q_06_val[0, 0], Q_08_val[0, 0]])
phi_max = np.min([Q_04_val[-1, 0], Q_06_val[-1, 0], Q_08_val[-1, 0]])

Q_04_sim = Q_04_sim[(Q_04_sim[:, 1] > phi_min) & (Q_04_sim[:, 1] < phi_max)]
Q_06_sim = Q_06_sim[(Q_06_sim[:, 1] > phi_min) & (Q_06_sim[:, 1] < phi_max)]
Q_08_sim = Q_08_sim[(Q_08_sim[:, 1] > phi_min) & (Q_08_sim[:, 1] < phi_max)]

n = 100
p_04_sim = p_04_sim[::n]
p_06_sim = p_06_sim[::n]
p_08_sim = p_08_sim[::n]

Q_04_sim = Q_04_sim[::n]
Q_06_sim = Q_06_sim[::n]
Q_08_sim = Q_08_sim[::n]




fig, ax1 = plt.subplots()

ax1.plot(
    p_04_sim[:,1], p_04_sim[:,0], label="Piston model"
)

ax1.plot(
    p_06_sim[:, 1], p_06_sim[:, 0], label="Piston model"
)

ax1.plot(
    p_08_sim[:, 1], p_08_sim[:, 0], label="Piston model"
)

ax1.plot(
    p_04_val[:,0], p_04_val[:, 1], label="Exp", linestyle='dashed'
)

ax1.plot(
    p_06_val[:, 0], p_06_val[:, 1], label="Exp", linestyle='dashed'
)

ax1.plot(
    p_08_val[:, 0], p_08_val[:, 1], label="Exp", linestyle='dashed'
)

fig, ax2 = plt.subplots()
ax2.plot(
    Q_04_sim[:,1], Q_04_sim[:,0], label="Piston model"
)

ax2.plot(
    Q_06_sim[:, 1], Q_06_sim[:, 0], label="Piston model"
)

ax2.plot(
    Q_08_sim[:, 1], Q_08_sim[:, 0], label="Piston model"
)

ax2.plot(
    Q_04_val[:,0], Q_04_val[:, 1], label="Exp", linestyle='dashed'
)

ax2.plot(
    Q_06_val[:, 0], Q_06_val[:, 1], label="Exp", linestyle='dashed'
)

ax2.plot(
    Q_08_val[:, 0], Q_08_val[:, 1], label="Exp", linestyle='dashed'
)

plt.show()

np.savetxt("piston_engine/validation_output_data/H2_performance/p_04_sim.dat", p_04_sim, fmt="%.5f")
np.savetxt("piston_engine/validation_output_data/H2_performance/p_04_val.dat", p_04_val, fmt="%.5f")

np.savetxt("piston_engine/validation_output_data/H2_performance/p_06_sim.dat", p_06_sim, fmt="%.5f")
np.savetxt("piston_engine/validation_output_data/H2_performance/p_06_val.dat", p_06_val, fmt="%.5f")

np.savetxt("piston_engine/validation_output_data/H2_performance/p_08_sim.dat", p_08_sim, fmt="%.5f")
np.savetxt("piston_engine/validation_output_data/H2_performance/p_08_val.dat", p_08_val, fmt="%.5f")


np.savetxt("piston_engine/validation_output_data/H2_performance/Q_04_sim.dat", Q_04_sim, fmt="%.5f")
np.savetxt("piston_engine/validation_output_data/H2_performance/Q_04_val.dat", Q_04_val, fmt="%.5f")

np.savetxt("piston_engine/validation_output_data/H2_performance/Q_06_sim.dat", Q_06_sim, fmt="%.5f")
np.savetxt("piston_engine/validation_output_data/H2_performance/Q_06_val.dat", Q_06_val, fmt="%.5f")

np.savetxt("piston_engine/validation_output_data/H2_performance/Q_08_sim.dat", Q_08_sim, fmt="%.5f")
np.savetxt("piston_engine/validation_output_data/H2_performance/Q_08_val.dat", Q_08_val, fmt="%.5f")








