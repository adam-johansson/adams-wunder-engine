# Two shaft CCE piston fueled on hydrogen

alt = 0.0  # [m] altitude
M = 0.0  # Flight Mach number
dTisa = 15  # [K] deviation from ISA

# either H2 or jetA
fuel = "H2"

Fn = 108500.0  # [N] #Net thrust requirement

Fs_req = 219.6  # specific thrust [m/s]

# power offtake from HPT
power_offtake = 110*1e3

bpr = 16.4  # bypass ratio

fpr_outer = 1.325  # Outer fan pressure ratio 1.3087

OPR = 38.0  # overall pressure ratio (including losses)
PR = 0.2767  # pressure split, with regard to the LPC

T4 = 1828  # [K] Turbine entry temperature

dp_intake = 0.2 / 100  # intake pressure loss
dp_bypass = 0.0 / 100  # bypass duct pressure loss
dp_rec = 3.0 / 100  # recuperator pressure loss
dT_rec = 151


eta_fan = 92.6 / 100  # inner fan polytropic efficiency
eta_p_lpc = 89.4 / 100  # polytropic
eta_p_hpc = 90.1 / 100  # polytropic

eta_b = 99.9 / 100  # combustor efficiency (burner)
dPcomb = 2.9 / 100  # combustor pressure lost [%]

eta_hpt = 90.0 / 100  # isentropic efficiency hpt
eta_lpt = 94.3 / 100  # isentropic efficiency lpt

eta_s = 99.5 / 100  # Shaft efficiency
eta_g = 99.0 / 100  # planetary gearbox efficiency

cfg_core = 97.7 / 100  # core thrust coefficient
cfg_bypass = 99.5 / 100  # bypass thrust coefficient
cd_nozzle = 99.0 / 100  # nozzle discharge coefficient

# cooling  TODO: decide cooling based on heat limits on the turbine
q_ngv = 0.5  # fraction of cooling from cooling first stator
bpr_c = 0.19  # cooling from fraction of core flow

second_burner = False

# fuel temperature upon injection in piston engine and burner
t_fuel = 747
# fuel tank temperature
t_tank = 22
