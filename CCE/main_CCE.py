from CCE.src import cce_propulsion_system
import importlib


# Importing input parameters

input_file = "ToC_H2"
#input_file = 'ToC_Kaiser'
input_dir = "input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

# Use surrogate or real model
surrogate_status = True

data = [d.Fn, d.dTisa, d.bpr, d.TET, d.fpr_inner, d.fpr_outer, d.dp_intake, d.dp_bypass,
        d.M, d.eta_inner_fan, d.eta_outer_fan,
        d.pi_hpc, d.eta_p_hpc, d.pi_ipc, d.eta_p_ipc,
        d.eta_b, d.dPcomb, d.eta_s, d.eta_g, d.q_ngv,
        d.bpr_c, d.eta_s_lpt, d.cfg_core, d.cfg_bypass, d.cd_nozzle, d.alt, d.fuel, d.pi_pe, d.surrogate]


cce_propulsion_system.run_cce(data)
