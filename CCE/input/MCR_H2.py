# Two shaft CCE piston fueled on hydrogen

alt = 10058  # [m] altitude
M = 0.7  # Flight Mach number
dTisa = 0  # [K] deviation from ISA

# either H2 or jetA
fuel = 'H2'

# add decided specific thrust here 80 m/s. motivated by Anders and Carlos. mass wont be variable anymore
Fn = 4384 * 4.44822  # [N] #Net thrust requirement (lb_f * conversion to Newton)
m0 = 200.0  # intake mass flow [kg/s]  RIGHT NOW ITERATIVELY SOLVED FOR BASED ON Fn
Fs_req = 80  # specific thrust [m/s]


bpr = 22.39  # bypass ratio

fpr_outer = 1.2905  # Outer fan pressure ratio 1.3087

# piston stuff
cr = 6.905  # piston engine geometric compression ratio
pi_pe = 1.34  # Piston engine pressure ratio
bore = 0.112  # piston bore

OPR = 13.27  # overall pressure ratio (excluding pressure losses for now)
PR = 0.1239  # pressure split, with regard to the LPC

T35 = 800  # [K] Turbine entry temperature

dp_intake = 0.3/100  # intake pressure loss
dp_bypass = 1.0/100  # bypass duct pressure loss
dp_hx = 5.7/100  # heat exchanger cold site loss
# 34.8 kg/s bypass air in hx

eta_inner_fan = 89.55/100  # inner fan polytropic efficiency
eta_outer_fan = 94.47/100  # outer fan polytropic efficiency
eta_p_ipc = 89.0/100  # polytropic
eta_p_hpc = 90.0/100  # polytropic

eta_b = 99.99/100 #combustor efficiency (burner)
dPcomb = 4.0/100 #combustor pressure lost [%]

eta_s_lpt = 94.0/100  # isentropic efficiency lpt

eta_s = 99.0/100  # Shaft efficiency
eta_g = 99.0/100  # planetary gearbox efficiency

cfg_core = 98.0/100  # core thrust coefficient
cfg_bypass = 99.3/100  # bypass thrust coefficient
cd_nozzle = 99.5/100  # nozzle discharge coefficient

#cooling  TODO: decide cooling based on heat limits on the turbine
q_ngv = 0.87  # fraction of cooling from cooling first stator
bpr_c = 0.00  # cooling from fraction of core flow

surrogate = True
second_burner = False




# Not used anymore
pi_ipc = 2.0  # IPC Pressure ratio funkar
pi_hpc = 8.0  # HPC pressure ratio funkar