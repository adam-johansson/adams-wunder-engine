import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine


def sweep_no_diesel_greek_validation(d, flags):

    # validate against the Rakolpoulos paper
    # the three different load cases
    fuel_air_ratios = np.array([0.028, 0.0345, 0.041])

    nitrogen_oxides = []
    IMEPs = []

    for far_goal in fuel_air_ratios:
        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_goal,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5]

        T4, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss,\
            heat_loss, p_tdc, outflow, no, imep = run_piston_engine(data, flags)


        nitrogen_oxides.append(no)
        IMEPs.append(imep * 1e-5)

    # load data from Rakopoulos

    dirname = os.path.dirname(__file__)
    print(dirname)
    filename_no = os.path.join(dirname, '../../validation_output_data/NO_diesel/IMEP.txt')
    no_val = np.loadtxt(filename_no, delimiter=",")

    plt.plot(IMEPs, nitrogen_oxides, marker='o', label="Simulation")
    plt.plot(no_val[:, 0], no_val[:, 1], marker='x', label="Validation")
    plt.legend()
    plt.show()

    return


