from timeit import default_timer as timer
import matplotlib.pyplot as plt
import importlib

import numpy as np

from piston_engine.engine import run_piston_engine  # import the piston engine function
from thermo import fuel_props

# import all the input variables

# input_file = "4T_HP"
# input_file = "4stroke"
#input_file = "4stroke_kaiser"
# input_file = "4stroke_hydrogen"
# input_file = "4stroke_hydrogen_bad_point"
#input_file = "H2_validation_italian.4stroke_hydrogen_validation_italian_08_v2"
#input_file = "validation.nasa_validation"
#input_file = "4stroke_hydrogen_sampling"
#input_file = "4stroke_sampling"
#input_file = "validation_twozone.two_zone_heider_alt"
input_file = "validation_twozone.nox_diesel_rakopolous"
# input_file = "validation_twozone.scania_d12"
#input_file = "validation_chalmers.case3"
# input_file = "validation_twozone.water_hydrogen"
# input_file = "validation_twozone.newcastle_h2_CI"
# input_file = "validation_twozone.newcastle_h2_HCCI"
#input_file = "4stroke_hydrogen_crashing_case"
#input_file = "4stroke_standard"

input_dir = "piston_engine.input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

# flags: plot_all, plot_essentials, plot_convergence, validation, output_all, output_power
# sweep, plot_details, plot_twozone, validate_twozone, validate_nox_diesel, fuel_mass, sweep_no_kth

# fuel_mass: specify mass of fuel instead of fuel air ratio

# to plot validation: first run validation case then run load

#flags = ["single"]
#flags = ['validation', 'fuel_mass', 'output_all', 'single', 'plot_convergence', 'plot_essentials', 'save']  # NASA validation case
#flags = ['validation_h2_performance']  # H2 performance validation
#flags = ['load']
#flags = ['validation', 'fuel_mass', 'output_all', 'single']  # NASA validation case no plots
#flags = ['plot_essentials', 'output', 'output_all', 'single', 'save']  # normal case
# flags = ['single', 'output_all', 'save']  # normal case no plots
# flags = ['single', 'output_all']  # normal case no plots
# flags = ['sweep']  # parametric study
# flags = ['optimise']  # optimisation
#flags = ['output', 'output_all', 'validate_twozone', 'save', 'single']  # validate two zone model (from book, Heider)
flags = ['sweep_no_greek']  # NO validation Rakoplpous
#flags = ["single", "validate_nox_diesel_late"]
# flags = ['sweep_no_kth']  # Scania validation
# flags = ['sweep_wiebe']
# flags = ['sweep_water_paper']
# flags = ['sweep_newcastle']
# flags = ['sweep_hcci']
# flags = ['fit_water_paper', 'single']
#flags = ['single', 'plot_twozone', 'plot_essentials'] # to look at the nox and twozone
#flags = ["load"]
# flags = ["single", "fit_newcastle"]
# flags = ["sweep_chalmers_h2"]
#flags = ["single", "validate_chalmers", 'plot_convergence']
#flags = ["single", "validate_chalmers"]
#flags = ["single", "validate_twozone"]

piston_input = {
    'p_in': d.p_in,
    'T_in': d.T_in,
    'p_ratio': d.p_ratio,
    'cycle': d.cycle,
    'cooling': d.cooling,
    'opposed': d.opposed,
    'cr': d.cr,
    'bore': d.d,
    'bsr': d.bsr,
    'v_mean': d.v_mean,
    'lms': d.lms,
    'Twalls': d.Twalls,
    'ch': d.ch,
    'valve_timings': d.valve_timings,
    'n_valve': d.n_valve,
    'lv_max': d.lv_max,
    'cd': d.cd,
    'eta_c': d.eta_c,
    'mf_tot': d.mf_tot,
    'wa': d.wa,
    'wm': d.wm,
    'm_wiebe': d.m_wiebe,
    'phi_sc': d.phi_sc,
    'phi_cd': d.phi_cd,
    'T_fuel': d.T_fuel,
    'p_fuel': d.p_fuel,
    'it': d.it,
    'wiebe_type': d.wiebe_type,
    'valve_type': d.valve_type,
    'far_goal': d.far_goal,
    'cylinders': d.cylinders,
    'fuel': d.fuel,
    'c1': d.c1,
    'c4': d.c4,
    'c5': d.c5,
    'premixed': d.premixed,
}


if "single" in flags:

    start = timer()
    pist_output = run_piston_engine(piston_input, flags)
    end = timer()
    print(f"Time: {end - start}")

    imep = pist_output["imep"]
    far = pist_output["far"]
    air_flow = pist_output["air_flow"]
    T4 = pist_output["T_out"]
    eta_th = pist_output["eta_th"]
    p_max = pist_output["peak pressure"]
    T_max = pist_output["peak temperature"]
    no = pist_output["no_ppm"]
    EI_nox = pist_output["EI_NO"]
    indicated_power = pist_output["indicated power"]

    print(f"IMEP: {imep * 1e-5} bar")



    # print(far / 0.02923)
    # print(far)
    # print(d.throttle - far)
    # print(p_tdc*1e-5)
    # 0.01124952812
    #print(f'Indicated power: {indicated_power * 1e-3} [kW]')
    # print(f'PE losses: {friction_loss * 1e-6} [MW]')
    # print(f'Aux losses: {aux_loss * 1e-6} [MW]')
    # print(f'Heat losses: {heat_loss * 1e-6} [MW]')
    # print(f'Pressure at TDC: {p_tdc*1e-5}')

    far_stoch, lhv = fuel_props(d.fuel)
    #print(f"Lambda trapped: {1 / equ_trapped} ")  # for H2
    print(f'Lambda: {far_stoch/far}')
    # print(f'airflow: {air_flow}')
    print(f"Fuel flow: {air_flow * far * 1e3} g/s")
    # print(f'mass flow out: {air_flow * (1 + far)}')
    # print(p_max)
    print(f"Outlet temperature: {T4}")
    print(f"Thermal efficiency: {eta_th}")
    # print(T_max)
    print(
        f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no}, EI_no: {EI_nox}, power: {indicated_power*1e-3} kW"
    )


elif "sweep" in flags:
    from piston_engine.src.misc.sweep import (
        sweep_pressure_ratio,
        sweep_valve_timings,
        sweep_cr,
        sweep_sc,
        sweep_lift,
        sweep_throttle,
        sweep_far_surrogate,
    )

    sweep_valve_timings(d=d)
    # sweep_cr(d=d)
    # sweep_sc(d=d)
    # sweep_lift(d=d)
    # sweep_throttle(d=d)
    # sweep_far_surrogate(d=d)

elif "optimise" in flags:
    from piston_engine.src.misc.optimisation import (
        optimisation_combust_start,
        optimisation_valve_timings,
        optimisation_combustion,
    )

    # optimisation_combust_start(d=d)
    optimisation_valve_timings(d=d)
    # optimisation_combustion(d=d)

elif "load" in flags:
    from piston_engine.src.misc.plot_output import plot_validation_nasa
    from piston_engine.src.misc.post_processing import validation_error
    from numpy import genfromtxt

    P = genfromtxt("piston_engine/simulation_data/P.csv", delimiter=",")
    T = genfromtxt("piston_engine/simulation_data/T.csv", delimiter=",")
    m = genfromtxt("piston_engine/simulation_data/m.csv", delimiter=",")
    equ = genfromtxt("piston_engine/simulation_data/equ.csv", delimiter=",")
    phi = genfromtxt("piston_engine/simulation_data/phi.csv", delimiter=",")
    plot_validation_nasa(phi, P, T, m, equ)
    validation_error(phi, P, T, m, equ)


elif "sweep_no_greek" in flags:
    from piston_engine.src.misc.sweep_no_validation import (
        sweep_no_diesel_greek_validation,
    )

    sweep_no_diesel_greek_validation(d, flags)


elif "sweep_no_kth" in flags:
    from piston_engine.src.misc.sweep_no_validation_kth import (
        sweep_no_diesel_kth_validation,
    )

    sweep_no_diesel_kth_validation(d, flags)

elif "sweep_wiebe" in flags:
    from piston_engine.src.misc.sweep_wiebe import sweep_wiebe

    sweep_wiebe(d, flags)

elif "sweep_water_paper" in flags:
    from piston_engine.src.misc.sweep_water_paper import sweep_equ

    sweep_equ(d, flags)

elif "sweep_newcastle" in flags:
    from piston_engine.src.misc.sweep_newcastle import sweep_equ

    sweep_equ(d, flags)

elif "sweep_hcci" in flags:
    from piston_engine.src.misc.sweep_hcci import sweep_equ

    sweep_equ(d, flags)


elif "sweep_chalmers_h2" in flags:
    from piston_engine.src.misc.sweep_chalmers_h2 import sweep_chalmers

    sweep_chalmers(d, flags)


else:
    print("What to you want to do?")
