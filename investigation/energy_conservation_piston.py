from piston_engine import engine
import importlib

input_file = "4stroke_hydrogen_sampling"

input_dir = "piston_engine.input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

flaggus = ['single', 'output_all']  # normal case no plots


far_goal = 0.02


data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
        d.v_mean, d.lms, d.Twalls, d.ch,
        d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
        d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_goal,
        d.cylinders, d.fuel, d.c1, d.c4, d.c5]


T4, work_piston, eta_th, air_flow, p_max, T_max, far_output, equ_trapped, induced_power, friction_loss, aux_loss, \
    heat_loss, p_tdc = engine.run_piston_engine(indata=data, flags=flaggus)


print(f'Diff in far: {far_output - far_goal}')