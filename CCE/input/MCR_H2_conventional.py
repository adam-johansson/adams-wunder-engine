# Two shaft CCE piston fueled on hydrogen

alt = 10058.0  # [m] altitude
M = 0.70  # Flight Mach number
dTisa = 0  # [K] deviation from ISA

# either H2 or jetA
fuel = "H2"

Fn = 18015.0  # [N] #Net thrust requirement (lb_f * conversion to Newton) (4050 lbf)

Fs_req = 82.1  # specific thrust [m/s]

# power offtake from HPT
power_offtake = 110*1e3

bpr = 17.6  # bypass ratio

fpr_outer = 1.333  # Outer fan pressure ratio 1.3087

OPR = 41.12  # overall pressure ratio (including losses)
PR = 0.2767  # pressure split, with regard to the LPC

T4 = 1545  # [K] Turbine entry temperature

dp_intake = 0.2 / 100  # intake pressure loss
dp_bypass = 0.0 / 100  # bypass duct pressure loss
dp_hx = 0.0 / 100  # heat exchanger cold site loss
# 34.8 kg/s bypass air in hx

eta_fan = 94.75 / 100  # inner fan polytropic efficiency
eta_p_lpc = 89.0 / 100  # polytropic
eta_p_hpc = 90.0 / 100  # polytropic

eta_b = 99.9 / 100  # combustor efficiency (burner)
dPcomb = 3.0 / 100  # combustor pressure lost [%]

eta_hpt = 90.0 / 100  # isentropic efficiency hpt
eta_lpt = 94.0 / 100  # isentropic efficiency lpt

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
t_fuel = 636
# fuel tank temperature
t_tank = 20
