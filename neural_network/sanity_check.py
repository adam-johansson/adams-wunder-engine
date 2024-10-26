import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import pandas as pd
import numpy as np
from timeit import default_timer as timer

import matplotlib.pyplot as plt

from src import load_inference

from piston_engine.engine import run_piston_engine  # import the piston engine function
import importlib

# import all the input files for real piston simulation
input_file_pist = "4stroke_hydrogen"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist

d = importlib.import_module(path_pist)

flags = ['sweep']  # normal case no plots



# Create new model and load states
model = load_inference("./models/narrowing_512_2.pth")
print(model)

# putting the model in output mode
model.eval()

p_in = 5e5
t_in = 500
cr = 15
bore = 0.15
far = 0.02923 / 3
p_ratio = 1.2
v_mean = 15
fuel_t = 400





X = pd.read_csv('./input_data/H2/x_cleaned.csv', index_col=0)
y = pd.read_csv('./input_data/H2/y_cleaned.csv', index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42)


# test and train predictions

y_pred_test = model.inference(X_test)



# go from 12 cylinders to 1, for airflow, power and heat loss
#divider = np.array([1, 1, 1/12, 1, 1, 1/12, 1/12, 1])
#y_pred_test = y_pred_test * divider
#y_test = y_test * divider


# root square error
RSE = np.sqrt(np.square(np.subtract(y_test, y_pred_test) ) )

# relative error
rel_error = np.divide(RSE, y_test)

# mean relative error
MRE = np.mean(rel_error, axis=0)


print(f'MRE T_2: {MRE[0] * 100:.2f} % \n'  
      f'MRE eff: {MRE[1] * 100:.2f} % \n' 
      f'MRE airflow: {MRE[2]*100:.2f} % \n' 
      f'MRE pmax: {MRE[3]*100:.2f} % \n'
      f'MRE T_max: {MRE[4]*100:.2f} % \n'
      f'MRE P: {MRE[5]*100:.2f} % \n'
      f'MRE Q : {MRE[6]*10:.2f} % \n'
      f'MRE p_tdc: {MRE[7]*100:.2f} %'
      )


points = 1000
points_real = 4
load = True

outputs = np.zeros([points, 8, 8])
outputs_real = np.zeros([points_real, 8, 8])

bores = np.linspace(0.05, 0.2, points)
far_s = np.linspace(0.02923 / 6.0, 0.02923 / 1.0, points)
v_means = np.linspace(5, 20, points)
p_ratios = np.linspace(0.7, 2.0, points)
fuel_ts = np.linspace(200, 600, points)
p_ins = np.linspace(2e5, 70e5, points)
T_ins = np.linspace(300, 1200, points)
compression_ratios = np.linspace(4, 18, points)

independents = [bores, far_s, v_means, p_ratios, fuel_ts, p_ins, T_ins, compression_ratios]
independents_labels = ["bores", "far_s", "v_means", "p_ratios", "fuel_ts", "p_ins", "T_ins", "compression_ratios"]

j = 0
start = timer()

for param, param_label in zip(independents, independents_labels):
    i = 0

    for independent in param:

        if param_label == "bores":
            x = np.array([p_in, t_in, cr, independent, far, p_ratio, v_mean, fuel_t])
        elif param_label == "far_s":
            x = np.array([p_in, t_in, cr, bore, independent, p_ratio, v_mean, fuel_t])
        elif param_label == "v_means":
            x = np.array([p_in, t_in, cr, bore, far, p_ratio, independent, fuel_t])
        elif param_label == "p_ratios":
            x = np.array([p_in, t_in, cr, bore, far, independent, v_mean, fuel_t])
        elif param_label == "fuel_ts":
            x = np.array([p_in, t_in, cr, bore, far, p_ratio, v_mean, independent])
        elif param_label == "p_ins":
            x = np.array([independent, t_in, cr, bore, far, p_ratio, v_mean, fuel_t])
        elif param_label == "T_ins":
            x = np.array([p_in, independent, cr, bore, far, p_ratio, v_mean, fuel_t])
        else:
            x = np.array([p_in, t_in, independent, bore, far, p_ratio, v_mean, fuel_t])


        with torch.no_grad():
            y = model.inference(x)
        outputs[i, :, j] = y

        i = i + 1
    j = j + 1

end = timer()
print(f'Sampling time nn: {end - start} s')

if not load:
    j = 0
    start = timer()
    for param, param_label in zip(independents, independents_labels):
        param_real = np.linspace(param[0], param[-1], points_real)
        i = 0
        for independent in param_real:
            if param_label == "bores":
                # input data to real simulation
                data = [p_in, t_in, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, independent, d.bsr,
                        v_mean, d.lms, d.Twalls, d.ch,
                        d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
                        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far,
                        d.cylinders, d.fuel, d.c1, d.c4, d.c5]
            elif param_label == "far_s":
                # input data to real simulation
                data = [p_in, t_in, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
                        v_mean, d.lms, d.Twalls, d.ch,
                        d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
                        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, independent,
                        d.cylinders, d.fuel, d.c1, d.c4, d.c5]
            elif param_label == "v_means":
                # input data to real simulation
                data = [p_in, t_in, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
                        independent, d.lms, d.Twalls, d.ch,
                        d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
                        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far,
                        d.cylinders, d.fuel, d.c1, d.c4, d.c5]
            elif param_label == "p_ratios":
                # input data to real simulation
                data = [p_in, t_in, independent, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
                        v_mean, d.lms, d.Twalls, d.ch,
                        d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
                        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far,
                        d.cylinders, d.fuel, d.c1, d.c4, d.c5]
            elif param_label == "fuel_ts":
                # input data to real simulation
                data = [p_in, t_in, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
                        v_mean, d.lms, d.Twalls, d.ch,
                        d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
                        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, independent, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far,
                        d.cylinders, d.fuel, d.c1, d.c4, d.c5]
            elif param_label == "p_ins":
                # input data to real simulation
                data = [independent, t_in, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
                        v_mean, d.lms, d.Twalls, d.ch,
                        d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
                        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far,
                        d.cylinders, d.fuel, d.c1, d.c4, d.c5]
            elif param_label == "T_ins":
                # input data to real simulation
                data = [p_in, independent, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
                        v_mean, d.lms, d.Twalls, d.ch,
                        d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
                        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far,
                        d.cylinders, d.fuel, d.c1, d.c4, d.c5]
            else:
                # input data to real simulation
                data = [p_in, t_in, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, independent, bore, d.bsr,
                        v_mean, d.lms, d.Twalls, d.ch,
                        d.valve_timings, d.n_valve, 0.1 * bore, d.cd, d.eta_c, d.mf_tot, d.wa,
                        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far,
                        d.cylinders, d.fuel, d.c1, d.c4, d.c5]


            T4, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, indicated_power, friction_loss, aux_loss,\
                heat_loss, p_tdc = run_piston_engine(data, flags)

            tempo_output = [T4, eta_th, air_flow, p_max, T_max, indicated_power * 1e-3, heat_loss * 1e-3, p_tdc * 1e-5]

            outputs_real[i, :, j] = tempo_output

            i = i + 1
        j = j + 1

    end = timer()
    print(f'Sampling time real: {end - start} s')

    # Writing data to file
    np.save('sanity_check_data/output_real', outputs_real)

if load:
    outputs_real = np.load('sanity_check_data/output_real.npy')



output_labels = ["T_2 [K]", "Thermal efficiency [%]", "Airflow [kg/s]", "p_max [bar]", "T_max [K]", "Power [kW]",
                 "Heat_loss [kW]", "p_tdc [bar]"]

# thermal efficiency converted to percent
outputs[:, 1, :] = outputs[:, 1, :] * 100
outputs_real[:, 1, :] = outputs_real[:, 1, :] * 100


variables = {"axs%s" % i: i for i in range(0, 7)}

fig = plt.figure()

j = 0
for param, param_label in zip(independents, independents_labels):
    param_real = np.linspace(param[0], param[-1], points_real)

    figs, axs = plt.subplots(3, 3)
    #plt.figure()

    i = 0
    for ax in axs.flatten():
        if i == 8:
            i = 7
        ax.plot(param, outputs[:, i, j], label="Neural network")
        ax.plot(param_real, outputs_real[:, i, j], label="Simulation")
        ax.set_xlabel(f'{param_label}')
        ax.set_ylabel(f'{output_labels[i]}')
        ax.legend()
        i = i + 1


    j = j + 1





plt.show()








