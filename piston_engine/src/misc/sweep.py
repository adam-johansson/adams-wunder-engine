from timeit import default_timer as timer
import matplotlib.pyplot as plt

import numpy as np

from piston_engine.engine import run_piston_engine  # import the piston engine function


def sweep_pressure_ratio(d):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    # chose parameter to investigate
    points = 10
    # throttles = np.linspace(0.5, 1.0, points)
    # pressures = np.linspace(1e5, 15e5, points)
    p_ratios = np.linspace(0.7, 2.5, points)
    works = []
    effs = []
    flow = []
    i = 1

    for p_ratio in p_ratios:
        data = [d.p_in, d.T_in, p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, d.throttle,
                d.cylinders]

        # start = timer()
        T4, work_piston, eff, air_flow, p_max, T_max, far, equ_trapped = run_piston_engine(data, flags)
        # end = timer()
        # print(f"Runtime of script: {end - start} [s]")
        # print(f"Outlet pressure: {p4 * 1e-5} [bar]")
        # print(f"Outlet temperature: {T4} [K]")
        # print(f"Piston power: {work_piston * 1e-3} [kW]")
        print(f'Data point {i} out of {points}. Pressure ratio: {p_ratio}')
        i += 1
        works.append(work_piston * 1e-3)
        effs.append(eff)
        flow.append(air_flow)

    fig, ax = plt.subplots()
    ax.plot(p_ratios, flow, marker='o')
    ax.set_xlabel('Pressure ratio []')
    ax.set_ylabel(r'Mass flow of air [kg/s]')
    ax.set_title('Mass flow vs pressure ratio')

    fig, ax = plt.subplots()
    ax.plot(p_ratios, works, marker='o')
    ax.set_xlabel('Pressure ratio []')
    ax.set_ylabel(r'Shaft power [kW]')
    ax.set_title('Engine power vs pressure ratio')

    fig, ax = plt.subplots()
    ax.plot(p_ratios, effs, marker='o')
    ax.set_xlabel('Pressure ratio []')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title('Thermal efficiency vs pressure ratio')
    plt.show()

    return


def sweep_valve_timings(d):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]
    # IVO, IVC, EVO, EVC
    valve = 'IVC'

    # chose parameter to investigate
    points = 10

    if valve == 'IVO':
        timings = np.linspace(670, 800, points)
    elif valve == 'IVC':
        timings = np.linspace(910, 930, points)
    elif valve == 'EVO':
        timings = np.linspace(400, 600, points)
    elif valve == 'EVC':
        timings = np.linspace(700, 760, points)
    else:
        raise Exception(f'Unknown valve.')

    works = []
    effs = []
    flow = []

    i = 1

    for timing in timings:
        if valve == 'IVO':
            valve_timings = [timing * np.pi / 180, d.valve_timings[1], d.valve_timings[2], d.valve_timings[3]]
        elif valve == 'IVC':
            valve_timings = [d.valve_timings[0], timing * np.pi / 180, d.valve_timings[2], d.valve_timings[3]]
        elif valve == 'EVO':
            valve_timings = [d.valve_timings[0], d.valve_timings[1], timing * np.pi / 180, d.valve_timings[3]]
        elif valve == 'EVC':
            valve_timings = [d.valve_timings[0], d.valve_timings[1], d.valve_timings[2], timing * np.pi / 180]
        else:
            raise Exception(f'Unknown valve.')

        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, d.throttle,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5]


        print(f'Data point {i} out of {points}. Timing: {timing}')
        T4, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss, \
            heat_loss, p_tdc = run_piston_engine(data, flags)

        # print(f"Outlet pressure: {p4 * 1e-5} [bar]")
        # print(f"Outlet temperature: {T4} [K]")
        # print(f"Piston power: {work_piston * 1e-3} [kW]")
        i += 1
        works.append(work_piston * 1e-3)
        effs.append(eta_th)
        flow.append(air_flow)

    fig, ax = plt.subplots()
    ax.plot(timings, flow, marker='o')
    ax.set_xlabel(f'{valve} [deg]')
    ax.set_ylabel(r'Mass flow of air [kg/s]')
    ax.set_title(f'Pressure ratio {d.p_ratio}')

    fig, ax = plt.subplots()
    ax.plot(timings, works, marker='o')
    ax.set_xlabel(f'{valve} [deg]')
    ax.set_ylabel(r'Shaft power [kW]')
    ax.set_title(f'Pressure ratio {d.p_ratio}')

    fig, ax = plt.subplots()
    ax.plot(timings, effs, marker='o')
    ax.set_xlabel(f'{valve} [deg]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Pressure ratio {d.p_ratio}')
    plt.show()

    return


def sweep_cr(d):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    # chose parameter to investigate
    points = 5

    cr_s = np.linspace(10, 12, points)
    works = []
    effs = []
    flow = []

    i = 1

    for cr in cr_s:
        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, d.throttle,
                d.cylinders]

        print(f'Data point {i} out of {points}. Compression ratio: {cr}')
        T4, work_piston, eff, air_flow, p_max, T_max, far, equ_trapped, power_induced = run_piston_engine(data, flags)
        print(p_max * 1e-5)

        i += 1
        works.append(power_induced * 1e-3 * 2)
        effs.append(eff)
        flow.append(air_flow * 2)

    fig, ax = plt.subplots()
    ax.plot(cr_s, flow, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Mass flow of air [kg/s]')
    ax.set_title(f'Thermal efficiency vs air flow')

    fig, ax = plt.subplots()
    ax.plot(cr_s, works, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Shaft power induced [kW]')
    ax.set_title(f'Thermal efficiency vs induced power')

    fig, ax = plt.subplots()
    ax.plot(cr_s, effs, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Thermal efficiency vs compression ratio')
    plt.show()

    return


def sweep_sc(d):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    # chose parameter to investigate
    points = 12

    phi_scs = np.linspace((300/180)*np.pi, (380/180)*np.pi, points)
    works = []
    effs = []
    flow = []
    peaks = []
    T4s = []

    i = 1

    for phi_sc in phi_scs:
        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, d.throttle,
                d.cylinders]

        print(f'Data point {i} out of {points}. Start of combustion: {phi_sc*180/np.pi}')
        T4, work_piston, eff, air_flow, p_max, T_max, far, equ_trapped, power_induced = run_piston_engine(data, flags)

        i += 1
        works.append(power_induced * 1e-3 * 2)
        effs.append(eff)
        flow.append(air_flow * 2)
        peaks.append(p_max)
        T4s.append(T4)

    phi_scs = phi_scs * 180 / np.pi

    fig, ax = plt.subplots()
    ax.plot(phi_scs, flow, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Mass flow of air [kg/s]')
    ax.set_title(f'Air flow vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(phi_scs, works, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Shaft power induced [kW]')
    ax.set_title(f'Induced power vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(phi_scs, effs, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Thermal efficiency vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(phi_scs, peaks, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Peak pressure vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(phi_scs, T4s, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Outlet temp vs start of combustion')
    plt.show()

    return

def sweep_lift(d):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    # chose parameter to investigate
    points = 4

    lv_maxs = np.linspace(0.01, 0.03, points)
    works = []
    effs = []
    flow = []
    peaks = []
    T4s = []

    i = 1

    for lv_max in lv_maxs:
        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, d.throttle,
                d.cylinders]

        print(f'Data point {i} out of {points}. Lv max: {lv_max}')
        T4, work_piston, eff, air_flow, p_max, T_max, far, equ_trapped, power_induced = run_piston_engine(data, flags)

        i += 1
        works.append(power_induced * 1e-3 * 2)
        effs.append(eff)
        flow.append(air_flow * 2)
        peaks.append(p_max)
        T4s.append(T4)

    fig, ax = plt.subplots()
    ax.plot(lv_maxs, flow, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Mass flow of air [kg/s]')
    ax.set_title(f'Air flow vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(lv_maxs, works, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Shaft power induced [kW]')
    ax.set_title(f'Induced power vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(lv_maxs, effs, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Thermal efficiency vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(lv_maxs, peaks, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Peak pressure vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(lv_maxs, T4s, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Outlet temp vs start of combustion')
    plt.show()

    return

def sweep_throttle(d):
    # flags: plot, output, validation, sweep
    flags = ["sweep"]

    # chose parameter to investigate
    points = 10

    throttles = np.linspace(0.005, 0.05, points)
    works = []
    effs = []
    flow = []
    peaks = []
    T4s = []

    i = 1

    for throttle in throttles:
        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, throttle,
                d.cylinders]

        print(f'Data point {i} out of {points}.')
        T4, work_piston, eff, air_flow, p_max, T_max, far, equ_trapped, power_induced = run_piston_engine(data, flags)

        i += 1
        works.append(power_induced * 1e-3 * 2)
        effs.append(eff)
        flow.append(air_flow * 2)
        peaks.append(p_max)
        T4s.append(T4)

    fig, ax = plt.subplots()
    ax.plot(throttles, flow, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Mass flow of air [kg/s]')
    ax.set_title(f'Air flow vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(throttles, works, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Shaft power induced [kW]')
    ax.set_title(f'Induced power vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(throttles, effs, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Thermal efficiency vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(throttles, peaks, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Peak pressure vs start of combustion')

    fig, ax = plt.subplots()
    ax.plot(throttles, T4s, marker='o')
    ax.set_xlabel(f'Compression ratio [-]')
    ax.set_ylabel(r'Thermal efficiency [-]')
    ax.set_title(f'Outlet temp vs start of combustion')
    plt.show()

    return


def sweep_far_surrogate(d):
    import pickle
    # flags: plot, output, validation, sweep
    # load the surrogate model
    filename = '../piston_engine/surrogate_data/piston_surrogate.pkl'
    with open(filename, "rb") as f:
        t_list_load = pickle.load(f)

    # chose parameter to investigate
    points = 1000

    far_s = np.linspace(0.002923, 0.02923, points)
    works = []
    effs = []
    flow = []
    peaks = []
    T4s = []

    i = 1

    for far in far_s:
        print(f'Data point {i} out of {points}.')

        pin = d.p_in
        Tin = d.T_in
        cr = d.cr
        bore = d.d
        p_ratio = d.p_ratio

        # get the output of the surrogate
        piston_input = np.atleast_2d(np.array([pin, Tin, cr, bore, far, p_ratio]))
        air_flow = t_list_load[2].predict_values(piston_input)[0][0]
        induced_power = t_list_load[5].predict_values(piston_input)[0][0] * 1e3
        p_tdc = t_list_load[7].predict_values(piston_input)[0][0] * 1e5

        i += 1
        works.append(induced_power * 1e-3 * 2)

    fig, ax = plt.subplots()
    #ax.plot(far_s, works, marker='o')
    ax.plot(far_s, works)
    ax.set_xlabel(f'far [-]')
    ax.set_ylabel(r'Shaft power induced [kW]')
    ax.set_title(f'Induced power vs far')

    plt.show()

    return
