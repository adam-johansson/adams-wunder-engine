import numpy as np
import matplotlib.pyplot as plt
import importlib
import pickle
from timeit import default_timer as timer
import pandas as pd

from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
from smt.utils.misc import compute_rms_error

from engine import run_piston_engine  # import the piston engine function

# Setting up the piston engine

input_file = "4stroke"
input_dir = "input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

# flags: plot, output, validation, sweep
flags = ["sweep"]
# just to be able to get 1 output instead of 4

ndim = 7  # number of input variables
n_out = 8  # Number of outputs from the piston model
plot_status = False
sample_status = True  # True if you want to sample new data
train_status = True
save_status = True
load_status = False

if not sample_status:
    from numpy import genfromtxt

    xt = pd.read_csv('../piston_engine/surrogate_data/xt.csv', header=None)
    xval = pd.read_csv('../piston_engine/surrogate_data/xval.csv', header=None)
    yt = pd.read_csv('../piston_engine/surrogate_data/yt.csv', header=None)
    yval = pd.read_csv('../piston_engine/surrogate_data/yval.csv', header=None)

    #xt = pd.read_csv('../piston_engine/surrogate_data/xt_cleaned.csv', index_col=0)
    #xval = pd.read_csv('../piston_engine/surrogate_data/xval_cleaned.csv', index_col=0)
    #yt = pd.read_csv('../piston_engine/surrogate_data/yt_cleaned.csv', index_col=0)
    #yval = pd.read_csv('../piston_engine/surrogate_data/yval_cleaned.csv', index_col=0)

    xt = pd.DataFrame.to_numpy(xt)
    xval = pd.DataFrame.to_numpy(xval)
    yt = pd.DataFrame.to_numpy(yt)
    yval = pd.DataFrame.to_numpy(yval)

    #xt = genfromtxt('surrogate_data/backup/h2/xt.csv', delimiter=',')
    #xval = genfromtxt('surrogate_data/backup/h2/xval.csv', delimiter=',')
    #yt = genfromtxt('surrogate_data/backup/h2/yt.csv', delimiter=',')
    #yval = genfromtxt('surrogate_data/backup/h2/yval.csv', delimiter=',')

    # take less training points
    xt = xt[::16, :]
    yt = yt[::16, :]
    xval = xval[::16, :]
    yval = yval[::16, :]
else:
    start_sampling = timer()
    # Here starts the SMT
    p_lim = [1e5, 10e5]  # limits for input pressure (3e5, 10e5)
    T_lim = [300, 1200]  # limits for input temperature (400, 1000)
    cr_lim = [4, 18]  # limits for geometric compression ratio (6, 12)
    d_lim = [0.05, 0.30]  # limits for bore (piston diameter) (0.08, 0.17)
    # THIS THROTTLE LIM IS FOR HYDROGEN
    #throttle_lim = [0.02923 / 6, 0.02923 / 1.0]  # (0.02923 / 5, 0.02923 / 1.5)

    # THIS IS FOR JETA
    throttle_lim = [0.06821 / 6, 0.06821 / 1.0]

    p_ratio_lim = [0.7, 2.0]   # (1.0, 1.5)
    v_mean_lim = [5, 20]
    # could add fuel temperature here and wall temperatures

    xlimits = np.array([p_lim, T_lim, cr_lim, d_lim, throttle_lim, p_ratio_lim, v_mean_lim])
    sampling = LHS(xlimits=xlimits, random_state=1, criterion='ese')

    # Construction of the DOE, the training points  #approx 700 seconds for 60 training 60 validation
    npoints = 400  # points per variable 400
    ndoe = ndim * npoints
    xt = sampling(ndoe)
    yt = np.zeros([ndoe, n_out])

    # Construction of the validation points
    nval = ndoe
    sampling = LHS(xlimits=xlimits, random_state=69, criterion='ese')
    xval = sampling(nval)
    yval = np.zeros([nval, n_out])

    i = 0
    for p, T, cr, bore, throttle, p_ratio, v_mean in xt:
        i += 1
        print(f'Simulation {i} out of {ndoe} in training set. p: {p*1e-5}, T: {T}, cr: {cr},'
              f' bore: {bore}, far: {throttle}')
        data = [p, T, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
                v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, throttle,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5]

        # run the simulation
        T_out, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss, \
            heat_loss, p_tdc = run_piston_engine(data, flags)
        # save the output that is relevant
        # eta_th is redundant I suppose
        yt[i - 1, :] = T_out, eta_th, air_flow, p_max, T_max, induced_power * 1e-3, heat_loss * 1e-3, p_tdc * 1e-5

    i = 0
    for p, T, cr, bore, throttle, p_ratio, v_mean in xval:
        i += 1
        print(f'Simulation {i} out of {nval} in validation set')
        data = [p, T, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
                v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, throttle,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5]

        T_out, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss, \
            heat_loss, p_tdc = run_piston_engine(data, flags)
        yval[i - 1, :] = T_out, eta_th, air_flow, p_max, T_max, induced_power * 1e-3, heat_loss * 1e-3, p_tdc * 1e-5


    # Saving the arrays in surrogate_data folder
    np.savetxt("surrogate_data/xt.csv", xt, delimiter=",")
    np.savetxt("surrogate_data/xval.csv", xval, delimiter=",")
    np.savetxt("surrogate_data/yt.csv", yt, delimiter=",")
    np.savetxt("surrogate_data/yval.csv", yval, delimiter=",")
    end_sampling = timer()
    print(f'Total time for sampling data: {end_sampling - start_sampling} [s]')
if plot_status:
    # To visualize the DOE points
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(xt[:, 0], xt[:, 1], marker='x', c='b', s=200, label='Training points')
    plt.scatter(xval[:, 0], xval[:, 1], marker='.', c='k', s=200, label='Validation points')
    plt.title('DOE')
    plt.xlabel('p')
    plt.ylabel('T')
    plt.legend()
    plt.show()

if train_status:
    # The Kriging model
    y = np.zeros(np.shape(yval))
    yerr = np.zeros(np.shape(yval))
    t_list = []
    for i_out in range(np.shape(yval)[1]):
        # The variable 'theta0' is a list of length ndim.
        t = KRG(theta0=[1e-2] * ndim, print_prediction=False)
        t.set_training_values(xt, yt[:, i_out])
        t.train()
        t_list.append(t)

        # Prediction of the validation points
        y[:, i_out] = t.predict_values(xval).flatten()
        # Estimation of variance for validation points
        s2 = t.predict_variances(xval).flatten()
        yerr[:, i_out] = 2 * 3 * np.sqrt(s2)  # in order to use +/- 3 x standard deviation: 99% confidence interval
        print(f'Kriging,  error for output {i_out}: ' + str(compute_rms_error(t, xval, yval[:, i_out])))
        print(f'Theta values for output {i_out}', t.optimal_theta)

    # TODO: Save some information to specify what index correspond to what output
    if save_status:
        # Saving the trained models
        filename = 'surrogate_data/piston_surrogate.pkl'
        with open(filename, "wb") as f:
            pickle.dump(t_list, f)


if plot_status:
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    titles = ['Outlet temperature [K]', 'Thermal efficiency [-]', 'Air flow [kg/s]',
              'Maximum pressure [bar]', 'Maximum temperature [K]', 'Induced power [kW]', 'Heat losses [kW]',
              'Pressure at TDC [bar]']
    filenames = ['outlet_temperature', 'thermal_efficiency', 'air_flow', 'p_max', 'T_max', 'induced_power',
                 'heat_loss', 'p_tdc']

    for i in range(n_out):
        fig_name = 'Validation of ' + titles[i]
        fig, ax = plt.subplots()
        ax.plot(yval[:, i], yval[:, i], '-', label='$y_{true}$')
        ax.plot(yval[:, i], y[:, i], 'r.', label='$\hat{y}$')
        ax.errorbar(np.squeeze(yval[:, i]), np.squeeze(y[:, i]), yerr=np.squeeze(yerr[:, i]), fmt='none',
                    capsize=5, ecolor='lightgray', elinewidth=1, capthick=0.5, label='confidence estimate 99%')

        ax.set_xlabel('$y_{true}$')
        ax.set_ylabel('$\hat{y}$')

        plt.legend(loc='upper left')
        ax.set_title(fig_name)
        filename = 'surrogate_data/figs/' + filenames[i] + '.png'
        plt.savefig(filename)

    plt.show()


if load_status:
    # For loading the model
    filename = 'surrogate_data/piston_surrogate.pkl'
    with open(filename, "rb") as f:
        t_list_load = pickle.load(f)

    # to use the surrogate model
    p = 9e5
    Tin = 600
    cr = 8
    bore = 0.15
    far_given = 0.02923 / 3.0
    p_ratio = 1.3

    piston_input = np.atleast_2d(np.array([p, Tin, cr, bore, far_given, p_ratio]))
    T_out_load = t_list_load[0].predict_values(piston_input)
    eta_th_load = t_list_load[1].predict_values(piston_input)
    air_flow_load = t_list_load[2].predict_values(piston_input)
    p_max_load = t_list_load[3].predict_values(piston_input)
    T_max_load = t_list_load[4].predict_values(piston_input)
    induced_power_load = t_list_load[5].predict_values(piston_input)
    heat_loss_load = t_list_load[6].predict_values(piston_input)
    p_tdc_load = t_list_load[7].predict_values(piston_input)

    # compare with real run of simulation
    data = [p, Tin, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, bore, d.bsr,
            d.v_mean, d.lms, d.Twalls, d.ch,
            d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
            d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_given,
            d.cylinders, d.fuel]
    # run the simulation
    start = timer()
    T_out, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss, \
        heat_loss, p_tdc = run_piston_engine(data, flags)

    # simulation data
    print(f'Output temp surrogate: {T_out_load}. simulation: {T_out}')
    print(f'thermal eff surrogate: {eta_th_load}. simulation: {eta_th}')
    print(f'air flow surrogate: {air_flow_load}. simulation: {air_flow}')
    print(f'p_max surrogate: {p_max_load}. simulation: {p_max}')
    print(f'T_max surrogate: {T_max_load}. simulation: {T_max}')
    print(f'Induced power surrogate: {induced_power_load}. simulation: {induced_power * 1e-3}')
    print(f'heat_loss surrogate: {heat_loss_load}. simulation: {heat_loss * 1e-3}')
    print(f'p_tdc surrogate: {p_tdc_load}. simulation: {p_tdc * 1e-5}')
