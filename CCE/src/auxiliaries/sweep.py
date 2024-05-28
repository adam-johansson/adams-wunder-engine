from CCE.src import auxiliaries

import numpy as np
import matplotlib.pyplot as plt


def sweep(d, parameter):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    # chose parameter to investigate and how many points
    points = 10
    if parameter == 'bpr':
        x_s = np.linspace(15, 25, points)
    elif parameter == 'pi_pe':
        x_s = np.linspace(1.0, 1.8, points)
    elif parameter == 'v_ratio':
        x_s = np.linspace(0.84, 0.89, points)

    sfc_s = []

    i = 1

    for x in x_s:
        if parameter == 'bpr':
            data = [d.Fn, d.dTisa, x, d.TET, d.fpr_inner, d.fpr_outer, d.dp_intake, d.dp_bypass,
                    d.M, d.eta_inner_fan, d.eta_outer_fan,
                    d.pi_hpc, d.eta_p_hpc, d.pi_ipc, d.eta_p_ipc,
                    d.eta_b, d.dPcomb, d.eta_s, d.eta_g, d.q_ngv,
                    d.bpr_c, d.eta_s_lpt, d.cfg_core, d.cfg_bypass, d.cd_nozzle, d.alt, d.fuel, d.pi_pe, d.surrogate,
                    d.m0]
        elif parameter == 'pi_pe':
            data = [d.Fn, d.dTisa, d.bpr, d.TET, d.fpr_outer, d.Fs_req, d.dp_intake, d.dp_bypass,
                    d.M, d.eta_inner_fan, d.eta_outer_fan,
                    d.pi_hpc, d.eta_p_hpc, d.pi_ipc, d.eta_p_ipc,
                    d.eta_b, d.dPcomb, d.eta_s, d.eta_g, d.q_ngv,
                    d.bpr_c, d.eta_s_lpt, d.cfg_core, d.cfg_bypass, d.cd_nozzle, d.alt, d.fuel, x, d.surrogate,
                    d.m0, d.cr, d.OPR, d.PR]
        elif parameter == 'v_ratio':
            data = [d.Fn, d.dTisa, d.bpr, d.TET, d.fpr_outer, d.Fs_req, d.dp_intake, d.dp_bypass,
                    d.M, d.eta_inner_fan, d.eta_outer_fan,
                    d.pi_hpc, d.eta_p_hpc, d.pi_ipc, d.eta_p_ipc,
                    d.eta_b, d.dPcomb, d.eta_s, d.eta_g, d.q_ngv,
                    d.bpr_c, d.eta_s_lpt, d.cfg_core, d.cfg_bypass, d.cd_nozzle, d.alt, d.fuel, d.pi_pe, d.surrogate,
                    d.m0, d.cr, d.OPR, d.PR]

        print(f'Data point {i} out of {points}.')
        sfc, v_ratio, thrust, m0, error = auxiliaries.run_cce_bpr_fpr(data, flags)

        if np.isnan(sfc):
            sfc_s.append(0)
        else:
            sfc_s.append(sfc*1e6)
        i += 1

    if parameter == 'bpr':
        fig, ax = plt.subplots()
        ax.plot(x_s, sfc_s, marker='o')
        ax.set_xlabel(f'Bypass ratio [-]')
        ax.set_ylabel(r'Thrust specific fuel consumption [mg/Ns]')
        ax.set_title(f'TSFC vs BPR')
        plt.show()

    elif parameter == 'pi_pe':
        fig, ax = plt.subplots()
        ax.plot(x_s, sfc_s, marker='o')
        ax.set_xlabel(f'Piston engine pressure ratio [-]')
        ax.set_ylabel(r'Thrust specific fuel consumption [mg/Ns]')
        ax.set_title(f'TSFC vs PE pressure ratio')
        plt.show()

    elif parameter == 'v_ratio':
        fig, ax = plt.subplots()
        ax.plot(x_s, sfc_s, marker='o')
        ax.set_xlabel(f'Piston engine pressure ratio [-]')
        ax.set_ylabel(r'Thrust specific fuel consumption [mg/Ns]')
        ax.set_title(f'TSFC vs PE pressure ratio')
        plt.show()

    return
