import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def print_output(
    airflow_piston,
    sfc,
    F,
    m0,
    m2,
    power_tot,
    power_hpc,
    offtake,
    power_lpt,
    p3,
    T3,
    p35,
    T35,
    T4,
    p_max,
    T_max,
    equ_trapped,
    far_s,
    induced_power,
    fuel_flow_piston,
    fuel_flow_burner,
    V_d_tot,
    friction_loss,
    aux_loss,
    heat_loss,
    bpr_piston,
):

    print(f"Airflow piston: {airflow_piston} [kg/s]")

    print(f"Total displacement for the 2 V12s: {V_d_tot * 1000} [liter]")

    print(f"Thrust specific fuel consumption (TSFC): {sfc * 1e6} [mg/Ns]")
    # print(f"Nozzle pressure ratio, cold: {pi_c} and hot: {pi_h}")
    # print(f"Nozzles choked, cold: {choked_c} and hot: {choked_h}")

    # print(f"Propulsive efficiency: {eta_p}")
    # print(f"Thermal efficiency: {eta_th}")
    # print(f"Overall efficency: {eta_o}")
    # print(f"Specific thrust: {Fs} [Ns/kg]")

    print(f"Thrust: {F * 1e-3} [kN]")

    print(f"Intake flow: {m0} [kg/s]")
    print(f"Core flow: {m2} [kg/s]")
    print(f"Power from piston engine: {power_tot * 1e-3} [kW]")
    print(f"Power required by the HPC: {power_hpc * 1e-3} [kW]")
    print(f"Power offtake on HP spool: {offtake* 1e-3} [kW]")
    print(f"Power required by the fan: {power_lpt * 1e-3} [kW]")
    print(f"Max pressure in piston engine: {p_max * 1e-5} [bar]")
    print(f"Max temperature in piston engine: {T_max} [K]")
    print(f"Trapped FAR piston: {equ_trapped * far_s} [-]")
    #print(f"Trapped air-fuel-equivalence ratio: {1/equ_trapped} [-]")
    print(f"Indicated power from piston engine: {induced_power * 1e-6} [MW]")
    print(f"Fuel flow piston engine: {fuel_flow_piston * 1000} [g/s]")
    print(f"Fuel flow burner: {fuel_flow_burner * 1000} [g/s]")
    print(f"Total fuel flow: {(fuel_flow_burner + fuel_flow_piston) * 1000} [g/s]")
    print(f"PE losses: {friction_loss * 1e-6} [MW]")
    print(f"Aux losses: {aux_loss * 1e-6} [MW]")
    print(f"Heat losses: {heat_loss * 1e-6} [MW]")
    print(f"BPR around piston: {bpr_piston} [-]")

    return


def csv_output_cce(p, t, m, far, s):
    """
    saves pressures, temperatures and mass flows for all stations as csv file.
    """
    headers = ("station", "m [kg/s]", "T [K]", "p [kPa]", "FAR []", "s [kJ /(kg * K) ]")
    stations = (
        "2",
        "13",
        "14",
        "15",
        "21",
        "25",
        "3",
        "31",
        "32",
        "34",
        "35",
        "4",
        "41",
        "43",
        "5",
        "6",
    )

    stations = np.atleast_2d(stations).T
    data = np.concatenate(
        (stations, np.atleast_2d(m).T, np.atleast_2d(t).T, np.atleast_2d(p).T, np.atleast_2d(far).T, np.atleast_2d(s).T), axis=1
    )
    data = np.vstack([headers, data])
    np.savetxt("simulation_data/stations.csv", data, delimiter=",", fmt="%s")

    return

def csv_output_rec_h2_geared(p, t, m, far, s):
    """
    saves pressures, temperatures and mass flows for all stations as csv file.
    """
    headers = ("station", "m [kg/s]", "T [K]", "p [bar]", "FAR []", "s [kJ /(kg * K) ]")

    stations = (
        "a",
        "0",
        "2",
        "22",
        "25",
        "3",
        "31",
        "4",
        "41",
        "42",
        "45",
        "5",
        "6",
        "12",
        "7",
    )
    stations = np.atleast_2d(stations).T
    data = np.concatenate(
        (stations, np.atleast_2d(m).T, np.atleast_2d(t).T, np.atleast_2d(p).T, np.atleast_2d(far).T, np.atleast_2d(s).T), axis=1
    )
    data = np.vstack([headers, data])
    np.savetxt("simulation_data/stations.csv", data, delimiter=",", fmt="%s")

    return

def csv_output_jetA_geared(p, t, m, far, s):
    """
    saves pressures, temperatures and mass flows for all stations as csv file.
    """
    headers = ("station", "m [kg/s]", "T [K]", "p [bar]", "FAR []", "s [kJ /(kg * K) ]")

    stations = (
        "a",
        "0",
        "2",
        "22",
        "25",
        "3",
        "31",
        "4",
        "41",
        "42",
        "45",
        "5",
        "6",
        "12",
    )
    stations = np.atleast_2d(stations).T
    data = np.concatenate(
        (stations, np.atleast_2d(m).T, np.atleast_2d(t).T, np.atleast_2d(p).T, np.atleast_2d(far).T, np.atleast_2d(s).T), axis=1
    )
    data = np.vstack([headers, data])
    np.savetxt("simulation_data/stations.csv", data, delimiter=",", fmt="%s")

    return


def plot_stations_cce(p_array, t_array):
    # plotting the different stations

    stations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    labels = [
        "2",
        "13",
        "14",
        "15",
        "21",
        "25",
        "3",
        "31",
        "32",
        "34",
        "35",
        "4",
        "41",
        "43",
        "5",
        "6",
    ]

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("station")
    ax1.set_ylabel("pressure [kPa]", color=color)
    ax1.plot(stations, p_array, color=color, marker="o")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(
        "Temperature [K]", color=color
    )  # we already handled the x-label with ax1
    ax2.plot(stations, t_array, color=color, marker="o")
    ax2.tick_params(axis="y", labelcolor=color)
    ax1.grid(True)

    plt.xticks(stations, labels)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.margins(0.2)
    plt.show()

    return

def plot_stations_rec_h2_geared(p_array, t_array):
    # plotting the different stations

    stations = np.linspace(0,15,15)

    labels = [
        "a",
        "0",
        "2",
        "22",
        "25",
        "3",
        "31",
        "4",
        "41",
        "42",
        "45",
        "5",
        "6",
        "12",
        "7",
    ]

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("station")
    ax1.set_ylabel("pressure [bar]", color=color)
    ax1.plot(stations, p_array, color=color, marker="o")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(
        "Temperature [K]", color=color
    )  # we already handled the x-label with ax1
    ax2.plot(stations, t_array, color=color, marker="o")
    ax2.tick_params(axis="y", labelcolor=color)
    ax1.grid(True)

    plt.xticks(stations, labels)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.margins(0.2)
    plt.show()

    return

def plot_stations_jetA_geared(p_array, t_array):
    # plotting the different stations

    stations = np.linspace(0,14,14)

    labels = [
        "a",
        "0",
        "2",
        "22",
        "25",
        "3",
        "31",
        "4",
        "41",
        "42",
        "45",
        "5",
        "6",
        "12",
    ]

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("station")
    ax1.set_ylabel("pressure [bar]", color=color)
    ax1.plot(stations, p_array, color=color, marker="o")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(
        "Temperature [K]", color=color
    )  # we already handled the x-label with ax1
    ax2.plot(stations, t_array, color=color, marker="o")
    ax2.tick_params(axis="y", labelcolor=color)
    ax1.grid(True)

    plt.xticks(stations, labels)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.margins(0.2)
    plt.show()

    return


def optimisation_csv(
    sfc,
    opr,
    split,
    pi_pe,
    cr,
    bore,
    fpr,
    bpr,
    p_max,
    T_max,
    T_in_piston,
    T_out_piston,
    TET,
    far_piston,
):
    """
    saves output for each successful optimisation evaluation.
    """

    headers = (
        "TSFC [mg/Ns]",
        "OPR [-]",
        "Pressure split (LPC / HPC) [-]",
        "Pressure raise piston [-]",
        "CR [-]",
        "Cylinder bore [mm]",
        "FPR (outer) [-]",
        "BPR [-]",
        "p_max [bar]",
        "T_max [K]",
        "T_in_cyl [K]",
        "T_out_cyl [K]",
        "TET [K]",
        "FAR piston [-]",
    )

    data = np.array(
        [
            sfc,
            opr,
            split,
            pi_pe,
            cr,
            bore,
            fpr,
            bpr,
            p_max,
            T_max,
            T_in_piston,
            T_out_piston,
            TET,
            far_piston,
        ]
    )

    data = pd.DataFrame(data, headers)
    data = data.transpose()
    file = "optimisation_data/evaluations.csv"
    if os.path.exists(file) and os.path.isfile(file):
        data_load = pd.read_csv(file, index_col=0)
        data = pd.concat([data_load, data], ignore_index=True)
        data.reset_index()
        data.to_csv(file)
    else:
        data.to_csv(file)
        data = np.vstack([headers, data])
        np.savetxt(file, data, delimiter=",", fmt="%s")

    return


def print_efficiencies(eta_o, eta_p, eta_th, eta_transmission, eta_core, fs):

    print(f"Core efficiency: {eta_core}")
    print(f"Transmission efficiency: {eta_transmission}")
    print(f"Thermal efficiency: {eta_th}")
    print(f"Propulsive efficiency: {eta_p}")
    print(f"Overall efficiency: {eta_o}")
    print(f"Specific thrust: {fs} [N/(kg/s)]")

    return
