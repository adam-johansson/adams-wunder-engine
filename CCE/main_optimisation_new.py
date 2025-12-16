
import sys
sys.path.append("./../")


from CCE.src import cce_propulsion_system_specific, geared_turbofan_h2_recuperated, geared_turbofan_jetA, cce_propulsion_system_h2
from CCE.src import auxiliaries
import importlib
from neural_network.src import load_ANN

from timeit import default_timer as timer

# Importing input parameters
input_file_pist = "4stroke_jetA"
input_dir_pist = "input.piston"
path_pist = input_dir_pist + "." + input_file_pist

d_p = importlib.import_module(path_pist)

#flags = ["single", "print_output", "conventional"]  # normal case
#flags = ["single", "print_output", "cce"]  # normal case
flags = ["life_hack", "cce"]  # life hack version
#flags = ['single', "cce"] # for matching thrust
#flags = ['sweep']
#flags = ['optim', "cce"]


if "cce" in flags:
    input_file = "MR_TOC_jetA"
    input_dir = "input.cce_jetA"
    path = input_dir + "." + input_file
    d = importlib.import_module(path)

elif "conventional" in flags:
    input_file = "mid_range_TOC_jetA_conventional"
    input_dir = "input.conventional_jetA"
    path = input_dir + "." + input_file
    d = importlib.import_module(path)

if "conventional" in flags:
    data_dict = {
        "Fn": d.Fn,
        "dTisa": d.dTisa,
        "bpr": d.bpr,
        "T4": d.T4,
        "fpr_outer": d.fpr_outer,
        "Fs_req": d.Fs_req,
        "dp_intake": d.dp_intake,
        "dp_bypass": d.dp_bypass,
        "M": d.M,
        "eta_fan": d.eta_fan,
        "eta_p_hpc": d.eta_p_hpc,
        "eta_p_lpc": d.eta_p_lpc,
        "eta_b": d.eta_b,
        "dPcomb": d.dPcomb,
        "eta_s": d.eta_s,
        "eta_g": d.eta_g,
        "eta_hpt": d.eta_hpt,
        "eta_lpt": d.eta_lpt,
        "cfg_core": d.cfg_core,
        "cfg_bypass": d.cfg_bypass,
        "cd_nozzle": d.cd_nozzle,
        "alt": d.alt,
        "fuel": d.fuel,
        "OPR": d.OPR,
        "PR": d.PR,
        "t_fuel": d.t_fuel,
        "t_tank": d.t_tank,
        "power_offtake": d.power_offtake,
        "dp_rec": d.dp_rec,
        "dT_rec": d.dT_rec,
    }

    if d.fuel == "H2":
        (
            sfc, vel_ratio, F, m0
        ) = geared_turbofan_h2_recuperated.run_turbofan(data_dict, flags)
    else:
        (
            sfc, vel_ratio, F, m0
        ) = geared_turbofan_jetA.run_turbofan(data_dict, flags)

    print(f"mass flow: {m0} [kg/s]")
    print(f"SFC: {sfc*1e6} [mg/Ns]")
    print(f"Thrust: {F*1e-3} [kN]]")
    print(f"Velocity ratio: {vel_ratio}")

elif "cce" in flags:

    cce_input = {
        "Fn": d.Fn,
        "dTisa": d.dTisa,
        "bpr": d.bpr,
        "T4": d.T4,
        "fpr_outer": d.fpr_outer,
        "Fs_req": d.Fs_req,
        "dp_intake": d.dp_intake,
        "dp_bypass": d.dp_bypass,
        "M": d.M,
        "eta_fan": d.eta_fan,
        "eta_p_hpc": d.eta_p_hpc,
        "eta_p_lpc": d.eta_p_lpc,
        "eta_b": d.eta_b,
        "dPcomb": d.dPcomb,
        "eta_s": d.eta_s,
        "eta_g": d.eta_g,
        "eta_lpt": d.eta_lpt,
        "cfg_core": d.cfg_core,
        "cfg_bypass": d.cfg_bypass,
        "cd_nozzle": d.cd_nozzle,
        "alt": d.alt,
        "fuel": d.fuel,
        "OPR": d.OPR,
        "PR": d.PR,
        "t_fuel": d.t_fuel,
        "t_tank": d.t_tank,
        "power_offtake": d.power_offtake,
        "surrogate": d.surrogate,
        "second_burner": d.second_burner,
        "pi_pe": d.pi_pe,
        "cr": d.cr,
        "bore": d.bore,
        "far piston": d.far_piston,
        'effectiveness IC': d.eff_IC,
        "dp_inter_compressor": d.dp_inter_compressor,
        "intercooler": d.intercooler,
        "specific": d.specific,
        "v_mean": d.v_mean,
        "life_hack": d.life_hack,
        "start_of_combustion": d.start_of_combustion,
    }

    piston_input = {
        'p_in': d_p.p_in,
        'T_in': d_p.T_in,
        'p_ratio': d_p.p_ratio,
        'cycle': d_p.cycle,
        'cooling': d_p.cooling,
        'opposed': d_p.opposed,
        'cr': d_p.cr,
        'bore': d_p.d,
        'bsr': d_p.bsr,
        'v_mean': d_p.v_mean,
        'lms': d_p.lms,
        'Twalls': d_p.Twalls,
        'ch': d_p.ch,
        'valve_timings': d_p.valve_timings,
        'n_valve': d_p.n_valve,
        'lv_max': d_p.lv_max,
        'cd': d_p.cd,
        'eta_c': d_p.eta_c,
        'mf_tot': d_p.mf_tot,
        'wa': d_p.wa,
        'wm': d_p.wm,
        'm_wiebe': d_p.m_wiebe,
        'phi_sc': d_p.phi_sc,
        'phi_cd': d_p.phi_cd,
        'T_fuel': d_p.T_fuel,
        'p_fuel': d_p.p_fuel,
        'it': d_p.it,
        'wiebe_type': d_p.wiebe_type,
        'valve_type': d_p.valve_type,
        'far_goal': d_p.far_goal,
        'cylinders': d_p.cylinders,
        'fuel': d_p.fuel,
        'c1': d_p.c1,
        'c4': d_p.c4,
        'c5': d_p.c5,
        'premixed': d_p.premixed,
    }


    if "single" in flags:
        start = timer()

        if d.fuel == "jetA":

            # Load the trained model
            meta_model = load_ANN("meta_models/jetA_256_2.pth")
            meta_model.double()
            print(meta_model)

            dict = auxiliaries.run_cce_bpr(cce_input, piston_input, meta_model)
            print(dict["bpr"])
            cce_input["bpr"] = dict["bpr"]
            flags.append("print_output")
            output_dict = cce_propulsion_system_specific.run_cce(cce_input, piston_input, flags, meta_model)

        elif d.fuel == "H2":

            # Load the trained model
            meta_model = load_ANN("../neural_network/models/H2_128_2.pth")
            meta_model.double()
            print(meta_model)

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
            ) = cce_propulsion_system_h2.run_cce(cce_input, piston_input, flags, meta_model)


        end = timer()
    elif "sweep" in flags:
        #parameter = "bpr"
        # parameter = 'fpr'
        #parameter = "pi_pe"

        # pressure split
        parameter = "PR"
        meta_model = "placeholder"
        # parameter = 'v_ratio'
        data_dict = "placeholder"
        auxiliaries.sweep(data_dict, piston_input, meta_model, parameter)
    elif "optim" in flags:
        if d.fuel == "jetA":

            # Load the trained model
            meta_model = load_ANN("meta_models/jetA_128_2_pinn.pth")

        elif d.fuel == "H2":

            # Load the trained model
            meta_model = load_ANN("../neural_network/models/H2_128_2.pth")

        meta_model.double()
        print(meta_model)

        auxiliaries.global_optimisation(cce_input, piston_input, flags, meta_model)
        #auxiliaries.nsga_optimisation(cce_input, piston_input, flags, meta_model)


    elif "life_hack" in flags:
        meta_model = "placeholder"

        # run once to get specific power and linear output temperature from piston engine
        # this simulation run will quite after the piston engine simulations

        cce_input["life_hack"] = "Simulate"
        # bpr 15 will almost certainly work
        cce_input["bpr"] = 20
        output_dict = cce_propulsion_system_specific.run_cce(cce_input, piston_input, flags, meta_model)

        # input simulation data for bpr matching
        piston_input["k_m"] = output_dict["k_m"]
        piston_input["k0_T"] = output_dict["k0_T"]
        piston_input["k1_T"] = output_dict["k1_T"]
        piston_input["k0_H"] = output_dict["k0_H"]
        piston_input["k1_H"] = output_dict["k1_H"]
        piston_input["piston_specific_power"] = output_dict["piston_specific_power"]

        # no simulation just quick evaluations
        cce_input["life_hack"] = "Express"


        dict = auxiliaries.run_cce_bpr(cce_input, piston_input, meta_model)
        print(dict["bpr"], dict["bore_match"])
        if dict["error"] == True:
            print(f"No match because: {dict["error_type"]}")
        
        else:


            cce_input["bpr"] = dict["bpr"][0]
            cce_input["bore"] = dict["bore_match"]

            flags.append("print_output")
            cce_input["life_hack"] = "Simulate_final"
            print("Final simulation")
            output_dict = cce_propulsion_system_specific.run_cce(cce_input, piston_input, flags, meta_model)
    else:
        print("No known flags")
