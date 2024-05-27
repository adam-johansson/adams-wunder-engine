from CCE.src import thermo, components
from piston_engine.src.misc import post_processing
import matplotlib.pyplot as plt
import pickle
import numpy as np

# load the surrogate model
filename = '../piston_engine/surrogate_data/piston_surrogate_h2_noisy.pkl'
with open(filename, "rb") as f:
    meta_model_noisy = pickle.load(f)

# load the surrogate model
filename = '../piston_engine/surrogate_data/piston_surrogate_h2.pkl'
with open(filename, "rb") as f:
    meta_model_unnoisy = pickle.load(f)


error = False

# input to surrogate
pin = 7e5
Tin = 600
cr = 5
p_ratio = 1.3

# fuel type
fuel_type = 'H2'
far_s, LHV = thermo.fuel_props(fuel_type)

core_flow = 5


def find_match(x, meta_model):
    # change fuel air ratio and bore to match power and turbine inlet temperature

    bore = x[0]  # bore is varied to
    far34 = x[1]  # far is varied to match target output temperature


    # get the output of the surrogate
    piston_input = np.atleast_2d(np.array([pin, Tin, cr, bore, far34, p_ratio]))
    air_flow = meta_model[2].predict_values(piston_input)[0][0]
    induced_power = meta_model[5].predict_values(piston_input)[0][0]*1e3
    p_tdc = meta_model[7].predict_values(piston_input)[0][0]*1e5
    T34 = meta_model[0].predict_values(piston_input)[0][0]

    # pressurise circumventing flow
    m_circumvent = core_flow - air_flow
    if m_circumvent < 0:
        return np.array([0, 0, 0, 0])
    pressure_circ, T_circumv, P_circumv = \
        components.compressor(Tin, pin / 0.99, m_circumvent, 0.85, p_ratio * 0.99 * 0.99)

    # mix circumventing flow
    equ34 = far34 / far_s
    m34 = air_flow * (1 + far34)  # outflow of piston engine (air + fuel)
    #m35 = m34 + m_circumvent  # flow after mixing
    T35, equ35 = components.mix(m34, T34, equ34, m_circumvent, T_circumv, equ2=0, fuel_type=fuel_type)
    #far35 = equ35 * far_s

    # power needed to pressurise the fuel
    fuel_flow = air_flow * far34  # far_given is the same as far in the engine (at least it is supposed to be)
    P_fuel_pump = components.fuel_pump(p_tdc, fuel_type, fuel_flow)

    # things needed for aux and friction losses
    bsr = 1.0
    stroke = bore / bsr
    lv_max = bore * 0.1
    cylinders = 12
    v_mean = 18
    rpm = v_mean / (2 * stroke) * 60
    rps = rpm / 60
    Vd_tot = stroke * bore ** 2 / 4 * np.pi * cylinders
    cycle = '4T'
    if cycle == '4T':
        n_r = 2
    else:
        n_r = 1

    # auxiliary losses and friction losses. these do not depend on the trapped fuel air ratio
    fmep, fmep_aux, fmep_pe_loss = post_processing.friction_patton(bore, rpm, stroke, v_mean, pin, cr, cylinders,
                                                                   lv_max)
    friction_loss = fmep_pe_loss * Vd_tot * rps / n_r  # friction losses for total engine all cylinders
    aux_loss = fmep_aux * Vd_tot * rps / n_r  # auxiliary losses

    shaft_power = induced_power - aux_loss - friction_loss - P_circumv - P_fuel_pump

    residual = np.array([shaft_power, T35, air_flow, T34])
    #print(residual, bore, far34)

    return residual


xpoints = 100
ypoints = 100

bores = np.linspace(0.05, 0.20, xpoints)
fars = np.linspace(0.01, 0.02923, ypoints)

powers = np.zeros((xpoints, ypoints))
TETs = np.zeros((xpoints, ypoints))
airflows = np.zeros((xpoints, ypoints))
T34s = np.zeros((xpoints, ypoints))

i = 0

for bore in bores:
    j = 0
    for far in fars:
        print(i, j)
        x = np.array([bore, far])
        output = find_match(x, meta_model_unnoisy)
        powers[i, j] = output[0]
        TETs[i, j] = output[1]
        airflows[i, j] = output[2]
        T34s[i, j] = output[3]
        j += 1
    i += 1



"""





far = 0.028
power2 = []
TET2 = []
airflow2 = []
T342 = []

bores2 = np.linspace(0.05, 0.20, 100)
for bore in bores2:
    x = np.array([bore, far])
    output = find_match(x, meta_model_unnoisy)
    power2.append(output[0])
    TET2.append(output[1])
    airflow2.append(output[2])
    T342.append((output[3]))
"""

bore2 = 0.1
fars2 = np.linspace(0.015, 0.029, 1000)

power3 = []
TET3 = []
airflow3 = []
T343 = []
i = 0
for far in fars2:
    print(i)
    x = np.array([bore2, far])
    output = find_match(x, meta_model_unnoisy)
    power3.append(output[0])
    TET3.append(output[1])
    airflow3.append(output[2])
    T343.append((output[3]))
    i += 1

"""
far = 0.028
power2_noisy = []
TET2_noisy = []
airflow2_noisy = []
T342_noisy = []

for bore in bores2:
    x = np.array([bore, far])
    output = find_match(x, meta_model_noisy)
    power2_noisy.append(output[0])
    TET2_noisy.append(output[1])
    airflow2_noisy.append(output[2])
    T342_noisy.append((output[3]))

"""
bore2 = 0.1

power3_noisy = []
TET3_noisy = []
airflow3_noisy = []
T343_noisy = []
for far in fars2:
    x = np.array([bore2, far])
    output = find_match(x, meta_model_noisy)
    power3_noisy.append(output[0])
    TET3_noisy.append(output[1])
    airflow3_noisy.append(output[2])
    T343_noisy.append((output[3]))


# Import libraries
from mpl_toolkits import mplot3d


# Creating dataset
x, y = np.meshgrid(fars, bores)

# Creating figure
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')

# Creating plot
ax.plot_surface(x, y, powers)

# Creating figure
fig2 = plt.figure(figsize=(14, 9))
ax2 = plt.axes(projection='3d')

# Creating plot
ax2.plot_surface(x, y, TETs)

# Creating figure
fig3 = plt.figure(figsize=(14, 9))
ax3 = plt.axes(projection='3d')

# Creating plot
ax3.plot_surface(x, y, airflows)

# Creating figure
fig1337 = plt.figure(figsize=(14, 9))
ax1337 = plt.axes(projection='3d')

# Creating plot
ax1337.plot_surface(x, y, T34s)

"""
# Creating figure
fig4 = plt.figure(figsize=(14, 9))
ax4 = plt.plot(bores2, airflow2)
ax4 = plt.plot(bores2, airflow2_noisy)

fig5 = plt.figure(figsize=(14, 9))
ax5 = plt.plot(bores2, TET2)
ax5 = plt.plot(bores2, TET2_noisy)


fig6 = plt.figure(figsize=(14, 9))
ax6 = plt.plot(bores2, power2)
ax6 = plt.plot(bores2, power2_noisy)

fig7 = plt.figure(figsize=(14, 9))
ax7 = plt.plot(bores2, T342)
ax7 = plt.plot(bores2, T342_noisy)
"""
# Creating figure
fig8 = plt.figure(figsize=(14, 9))
ax8 = plt.plot(fars2, airflow3, label='less_trained')
ax8 = plt.plot(fars2, airflow3_noisy)
plt.legend()

fig9 = plt.figure(figsize=(14, 9))
ax9 = plt.plot(fars2, TET3, label='less_trained')
ax9 = plt.plot(fars2, TET3_noisy)
plt.legend()

fig10 = plt.figure(figsize=(14, 9))
ax10 = plt.plot(fars2, power3, label='less_trained')
ax10 = plt.plot(fars2, power3_noisy)
plt.legend()
"""
fig11 = plt.figure(figsize=(14, 9))
ax11 = plt.plot(fars2, T343)
ax11 = plt.plot(fars2, T343_noisy)
"""

# show plot
plt.show()

