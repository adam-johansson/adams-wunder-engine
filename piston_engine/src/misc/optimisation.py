from timeit import default_timer as timer
import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize, fmin_slsqp
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction

from piston_engine.engine import run_piston_engine  # import the piston engine function


def optimisation_combust_start(d):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    def flow(phi_sc):
        print(phi_sc * 180 / np.pi)
        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, phi_sc[0], d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, d.throttle,
                d.cylinders]
        T4, work_piston, eff, air_flow, p_max, T_max, far, equ_trapped, induced_power = run_piston_engine(data, flags)
        print(induced_power)
        return -induced_power

    phi_sc0 = (345 / 180) * np.pi  # angle at combustion start
    bnds = [(330 / 180 * np.pi, 360 / 180 * np.pi)]
    res = minimize(flow, phi_sc0, method='TNC', bounds=bnds, tol=1e-1)
    # res = fmin_slsqp(flow, phi_sc0, bounds=bnds)
    # methods: 'SLSQP', 'TNC', 'BFGS', ’Nelder-Mead’
    print(res.x)

    return


def optimisation_valve_timings(d):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    def black_box_function(ivo, ivc, evo, evc):

        print(ivo * 180 / np.pi, ivc * 180 / np.pi, evo * 180 / np.pi, evc * 180 / np.pi)

        timings = [ivo, ivc, evo, evc]
        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, d.throttle,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5]
        T4, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss, \
            heat_loss, p_tdc = run_piston_engine(data, flags)

        #print(induced_power * 1e-6)
        #print(air_flow)
        return eta_th


    # Set range of C to optimize for.
    # bayes_opt requires this to be a dictionary.
    pbounds = {"ivo": (710/180 * np.pi, 725/180 * np.pi), "ivc": (910/180 * np.pi, 930/180 * np.pi),
               "evo": (505/180 * np.pi, 525/180 * np.pi), "evc": (720/180 * np.pi, 740/180 * np.pi)}
    # Create a BayesianOptimization optimizer,
    # and optimize the given black_box_function.
    optimizer = BayesianOptimization(f=black_box_function,
                                     pbounds=pbounds, verbose=2,
                                     random_state=4)
    optimizer.maximize(init_points=10, n_iter=30)
    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

    return


def optimisation_combustion(d):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    def black_box_function(wa, wm, start, duration):

        print(wa, wm, start * 180 / np.pi, duration * 180 / np.pi)

        phi_sc = start
        phi_cd = duration

        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, wa,
                wm, d.m_wiebe, phi_sc, phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type,
                d.throttle, d.cylinders]
        T4, work_piston, eff, air_flow, p_max, T_max, far, equ_trapped, induced_power,\
            friction_losses, aux_losses, heat_losses = run_piston_engine(data, flags)

        #goal = -abs(induced_power * 1e-6 * 2 - 4.59)  # we want 4.59 MW induced power
        goal = induced_power
        return goal

    # Set range of C to optimize for.
    # bayes_opt requires this to be a dictionary.
    pbounds = {"wa": (6.75, 7.1), "wm": (1.2, 1.95),
               "start": (340/180 * np.pi, 350/180 * np.pi), "duration": (25/180 * np.pi, 50/180 * np.pi)}
    # Create a BayesianOptimization optimizer,
    # and optimize the given black_box_function.
    optimizer = BayesianOptimization(f=black_box_function,
                                     pbounds=pbounds, verbose=2,
                                     random_state=4)
    optimizer.maximize(init_points=10, n_iter=20)
    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

    return
