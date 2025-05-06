from CCE.src import auxiliaries

import numpy as np
import matplotlib.pyplot as plt


def sweep(data, data_piston, meta_model, parameter):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    # chose parameter to investigate and how many points
    points = 20
    if parameter == "bpr":
        x_s = np.linspace(15, 25, points)
    elif parameter == "pi_pe":
        x_s = np.linspace(1.0, 1.8, points)
    elif parameter == "v_ratio":
        x_s = np.linspace(0.84, 0.89, points)
    elif parameter == "PR":
        x_s = np.linspace(0.05, 0.16, points)

    sfc_s = []

    i = 1

    for x in x_s:
        if parameter == "bpr":
            data[2] = x

        elif parameter == "PR":
            #print(data[32])
            data[32] = x


        print(f"Data point {i} out of {points}.")
        #sfc, v_ratio, thrust, m0, error = auxiliaries.run_cce_bpr_fpr(data, flags)
        sfc, v_ratio, thrust, m0, error, fpr, p_max, T_max, T_in_piston, T_out_piston, TET, far_piston, T35 \
                = auxiliaries.run_cce_fpr(data, data_piston, flags, meta_model)

        print(sfc, fpr, thrust, T_out_piston, TET, far_piston, T35, p_max*1e-5)

        if np.isnan(sfc):
            sfc_s.append(0)
        else:
            sfc_s.append(sfc * 1e6)
        i += 1

    if parameter == "bpr":
        fig, ax = plt.subplots()
        ax.plot(x_s, sfc_s, marker="o")
        ax.set_xlabel(f"Bypass ratio [-]")
        ax.set_ylabel(r"Thrust specific fuel consumption [mg/Ns]")
        ax.set_title(f"TSFC vs BPR")
        plt.show()

    elif parameter == "pi_pe":
        fig, ax = plt.subplots()
        ax.plot(x_s, sfc_s, marker="o")
        ax.set_xlabel(f"Piston engine pressure ratio [-]")
        ax.set_ylabel(r"Thrust specific fuel consumption [mg/Ns]")
        ax.set_title(f"TSFC vs PE pressure ratio")
        plt.show()

    elif parameter == "v_ratio":
        fig, ax = plt.subplots()
        ax.plot(x_s, sfc_s, marker="o")
        ax.set_xlabel(f"Piston engine pressure ratio [-]")
        ax.set_ylabel(r"Thrust specific fuel consumption [mg/Ns]")
        ax.set_title(f"TSFC vs PE pressure ratio")
        plt.show()

    elif parameter == "PR":
        fig, ax = plt.subplots()
        ax.plot(x_s, sfc_s, marker="o")
        ax.set_xlabel(f"Pressure split LPC HPC [-]")
        ax.set_ylabel(r"Thrust specific fuel consumption [mg/Ns]")
        ax.set_title(f"TSFC vs pressure split")
        plt.show()

    return
