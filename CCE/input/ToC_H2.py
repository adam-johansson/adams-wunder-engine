# Two shaft CCE piston fueled on hydrogen

alt = 10668  # [m] altitude
M = 0.78  # Flight Mach number
dTisa = 10  # [K] deviation from ISA

# gas properties
fuel = "H2"

# add decided specif thrust here 80 m/s. motivated by Anders and Carlos. mass wont be variable anymore
Fn = 25.0e3  # [N] #Net thrust requirement
m0 = 200.0  # intake mass flow [kg/s]  RIGHT NOW ITERATIVELY SOLVED FOR BASED ON Fn

bpr = 18.1  # Bypass ratio
bpr = 23.5

fpr_inner = 1.355  # Inner fan pressure ratio #TODO: adjust fpr for optimum jet speeds
fpr_outer = 1.355  # Outer fan pressure ratio

# piston stuff
cr = 8  # piston engine geometric compression ratio
pi_pe = 1.309071  # Piston engine pressure ratio

pi_ipc = 2.0  # IPC Pressure ratio funkar
pi_hpc = 8.0  # HPC pressure ratio funkar
# pi_ipc = 1.5  # IPC Pressure ratio funkar
# pi_hpc = 15.0  # HPC pressure ratio funkar

TET = 1225.78  # [K] Turbine entry temperature

dp_intake = 0.3 / 100  # intake pressure loss
dp_bypass = 1.0 / 100  # bypass duct pressure loss
dp_hx = 5.7 / 100  # heat exchanger cold site loss
# 34.8 kg/s bypass air in hx

eta_inner_fan = 89.55 / 100  # inner fan polytropic efficiency
eta_outer_fan = 94.47 / 100  # outer fan polytropic efficiency
eta_p_ipc = 90.53 / 100  # polytropic
eta_p_hpc = 89.823 / 100  # polytropic

eta_b = 99.99 / 100  # combustor efficiency (burner)
dPcomb = 4.0 / 100  # combustor pressure lost [%]

eta_s_lpt = 94.0 / 100  # isentropic efficiency lpt

eta_s = 99.0 / 100  # Shaft efficiency
eta_g = 99.0 / 100  # planetary gearbox efficiency

cfg_core = 98.0 / 100  # core thrust coefficient
cfg_bypass = 99.3 / 100  # bypass thrust coefficient
cd_nozzle = 99.5 / 100  # nozzle discharge coefficient

# cooling  TODO: decide cooling based on heat limits on the turbine
q_ngv = 0.87  # fraction of cooling from cooling first stator
bpr_c = 0.039  # cooling from fraction of core flow

surrogate = True
