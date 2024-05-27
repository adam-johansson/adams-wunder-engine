from CCE.src import cce_propulsion_system

import numpy as np
import matplotlib.pyplot as plt


def sweep_old(d, parameter):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    # chose parameter to investigate and how many points
    points = 30
    if parameter == 'bpr':
        x_s = np.linspace(15, 40, points)
    elif parameter == 'fpr':
        x_s = np.linspace(1.2, 1.5, points)

    sfc_s = []
    vel_ratio_s = []

    i = 1

    for x in x_s:
        if parameter == 'bpr':
            data = [d.Fn, d.dTisa, x, d.TET, d.fpr_inner, d.fpr_outer, d.dp_intake, d.dp_bypass,
                    d.M, d.eta_inner_fan, d.eta_outer_fan,
                    d.pi_hpc, d.eta_p_hpc, d.pi_ipc, d.eta_p_ipc,
                    d.eta_b, d.dPcomb, d.eta_s, d.eta_g, d.q_ngv,
                    d.bpr_c, d.eta_s_lpt, d.cfg_core, d.cfg_bypass, d.cd_nozzle, d.alt, d.fuel, d.pi_pe, d.surrogate]
        elif parameter == 'fpr':
            data = [d.Fn, d.dTisa, d.bpr, d.TET, d.fpr_inner, x, d.dp_intake, d.dp_bypass,
                    d.M, d.eta_inner_fan, d.eta_outer_fan,
                    d.pi_hpc, d.eta_p_hpc, d.pi_ipc, d.eta_p_ipc,
                    d.eta_b, d.dPcomb, d.eta_s, d.eta_g, d.q_ngv,
                    d.bpr_c, d.eta_s_lpt, d.cfg_core, d.cfg_bypass, d.cd_nozzle, d.alt, d.fuel, d.pi_pe, d.surrogate]

        print(f'Data point {i} out of {points}.')
        sfc, vel_ratio = cce_propulsion_system.run_cce(data, flags)

        if np.isnan(sfc):
            sfc_s.append(0)
            vel_ratio_s.append(0)
        else:
            sfc_s.append(sfc*1e6)
            vel_ratio_s.append(vel_ratio[0])
        i += 1

    if parameter == 'bpr':
        fig, ax = plt.subplots()
        ax.plot(x_s, sfc_s, marker='o')
        ax.set_xlabel(f'Bypass ratio [-]')
        ax.set_ylabel(r'Thrust specific fuel consumption [mg/Ns]')
        ax.set_title(f'TSFC vs BPR')

        fig, ax = plt.subplots()
        ax.plot(x_s, vel_ratio_s, marker='o')
        ax.set_xlabel(f'Bypass ratio [-]')
        ax.set_ylabel(r'Ideal jet velocity ratio [-]')
        ax.set_title(f'vel_ratio vs BPR')
        plt.show()
    elif parameter == 'fpr':
        fig, ax = plt.subplots()
        ax.plot(x_s, vel_ratio_s, marker='o')
        ax.set_xlabel(f'Fan pressure ratio [-]')
        ax.set_ylabel(r'Ideal jet velocity ratio [-]')
        ax.set_title(f'vel_ratio vs FPR')

        fig, ax = plt.subplots()
        ax.plot(x_s, sfc_s, marker='o')
        ax.set_xlabel(f'Bypass ratio [-]')
        ax.set_ylabel(r'Thrust specific fuel consumption [mg/Ns]')
        ax.set_title(f'TSFC vs BPR')
        plt.show()

    return
