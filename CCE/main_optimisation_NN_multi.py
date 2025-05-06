from CCE.src import cce_propulsion_system, geared_turbofan_h2_recuperated
from CCE.src import auxiliaries
import importlib
from neural_network.src import load_ANN

from timeit import default_timer as timer

# Importing input parameters

#input_file = "MCR_H2_opt"
input_file = "MCR_H2_conventional"
input_dir = "input"
path = input_dir + "." + input_file

input_file_pist = "4stroke_hydrogen"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist

d = importlib.import_module(path)
d_p = importlib.import_module(path_pist)

flags = ["single", "print_output", "conventional"]  # normal case
#flags = ['single']
#flags = ['sweep']
#flags = ['optim']

p_in_dummy = 1
T_in_dummy = 1
pi_pe_dummy = 1
cr_dummy = 1
bore_dummy = 1


if "conventional" in flags:
    data = [
        d.Fn,
        d.dTisa,
        d.bpr,
        d.T4,
        d.fpr_outer,
        d.Fs_req,
        d.dp_intake,
        d.dp_bypass,
        d.M,
        d.eta_fan,
        d.eta_p_hpc,
        d.eta_p_lpc,
        d.eta_b,
        d.dPcomb,
        d.eta_s,
        d.eta_g,
        d.q_ngv,
        d.bpr_c,
        d.eta_hpt,
        d.eta_lpt,
        d.cfg_core,
        d.cfg_bypass,
        d.cd_nozzle,
        d.alt,
        d.fuel,
        d.OPR,
        d.PR,
        d.t_fuel,
        d.t_tank,
        d.power_offtake,
    ]

    (
        sfc, vel_ratio, F, m0
    ) = geared_turbofan_h2_recuperated.run_turbofan(data, flags)

    print(f"mass flow: {m0} [kg/s]")
    print(f"SFC: {sfc*1e6} [mg/Ns]")
    print(f"Thrust: {F*1e-3} [kN]]")
    print(f"Velocity ratio: {vel_ratio}")

elif "cce" in flags:

    data = [
        d.Fn,
        d.dTisa,
        d.bpr,
        d.T4,
        d.fpr_outer,
        d.Fs_req,
        d.dp_intake,
        d.dp_bypass,
        d.M,
        d.eta_inner_fan,
        d.eta_outer_fan,
        d.pi_hpc,
        d.eta_p_hpc,
        d.pi_ipc,
        d.eta_p_ipc,
        d.eta_b,
        d.dPcomb,
        d.eta_s,
        d.eta_g,
        d.q_ngv,
        d.bpr_c,
        d.eta_s_lpt,
        d.cfg_core,
        d.cfg_bypass,
        d.cd_nozzle,
        d.alt,
        d.fuel,
        d.pi_pe,
        d.surrogate,
        d.m0,
        d.cr,
        d.OPR,
        d.PR,
        d.bore,
        d.second_burner,
        d.t_fuel,
        d.t_tank,
        d.CCE,
        d.power_offtake,
    ]

    data_piston = [
        p_in_dummy,
        T_in_dummy,
        pi_pe_dummy,
        d_p.cycle,
        d_p.thermo,
        d_p.cooling,
        d_p.opposed,
        cr_dummy,
        bore_dummy,
        d_p.bsr,
        d_p.v_mean,
        d_p.lms,
        d_p.Twalls,
        d_p.ch,
        d_p.valve_timings,
        d_p.n_valve,
        d_p.lv_max,
        d_p.cd,
        d_p.eta_c,
        d_p.mf_tot,
        d_p.wa,
        d_p.wm,
        d_p.m_wiebe,
        d_p.phi_sc,
        d_p.phi_cd,
        d_p.T_fuel,
        d_p.p_fuel,
        d_p.it,
        d_p.wiebe_type,
        d_p.valve_type,
        d_p.throttle,
        d_p.cylinders,
        d_p.fuel,
        d_p.c1,
        d_p.c4,
        d_p.c5,
    ]

    # Load the trained model
    meta_model = load_ANN("../neural_network/models/H2_128_2.pth")
    meta_model.double()
    print(meta_model)

    if "single" in flags:
        start = timer()

        (
            sfc,
            v_ratio,
            thrust,
            m0,
            error,
            p_max,
            T_max,
            T_in_piston,
            T_out_piston,
            TET,
            far_piston,
            T35,
        ) = cce_propulsion_system.run_cce(data, data_piston, flags, meta_model)


        #sfc, v_ratio, thrust, m0, error, fpr, p_max, T_max, T_in_piston, T_out_piston, TET, far_piston, T35\
        #   = auxiliaries.run_cce_fpr(data, data_piston, flags, meta_model)

        if error:
            print("error in cce_propulsion_system")
            exit()

        end = timer()
        print(sfc * 1e6)
        print(far_piston)
        print(thrust / m0)
        print(m0)
        print(v_ratio)
        # print(fpr)
        print(f"Thrust: {thrust}. Required thrust: {d.Fn}")
        print(f"simulation time: {end - start}")
    elif "sweep" in flags:
        #parameter = "bpr"
        # parameter = 'fpr'
        #parameter = "pi_pe"

        # pressure split
        parameter = "PR"

        # parameter = 'v_ratio'
        auxiliaries.sweep(data, data_piston, meta_model, parameter)
    elif "optim" in flags:
        auxiliaries.global_optimisation(data, data_piston, flags, meta_model)
    else:
        print("No known flags")
