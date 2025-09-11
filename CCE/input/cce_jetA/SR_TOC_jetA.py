# Two shaft CCE piston fueled on hydrogen

alt = 10058.0  # [m] altitude
M = 0.70  # Flight Mach number
dTisa = 10  # [K] deviation from ISA

# either H2 or jetA
fuel = "jetA"

Fn = 23000  # [N] #Net thrust requirement

Fs_req = 105.9  # specific thrust [m/s]

# power offtake from HPT
power_offtake = 110*1e3

bpr = 14.0  # bypass ratio

fpr_outer = 1.325  # Outer fan pressure ratio 1.3087

OPR = 16  # overall pressure ratio (including losses)
PR = 0.15  # pressure split, with regard to the LPC

T4 = 1070  # [K] Turbine entry temperature

dp_intake = 0.2 / 100  # intake pressure loss
dp_bypass = 0.0 / 100  # bypass duct pressure loss


eta_fan = 91.7 / 100  # inner fan polytropic efficiency
eta_p_lpc = 88.2 / 100  # polytropic
eta_p_hpc = 90.0 / 100  # polytropic

eta_b = 99.9 / 100  # combustor efficiency (burner)
dPcomb = 2.0 / 100  # combustor pressure lost [%]

eta_hpt = 90.0 / 100  # isentropic efficiency hpt
eta_lpt = 94.0 / 100  # isentropic efficiency lpt

eta_s = 99.5 / 100  # Shaft efficiency
eta_g = 99.0 / 100  # planetary gearbox efficiency

cfg_core = 97.7 / 100  # core thrust coefficient
cfg_bypass = 99.5 / 100  # bypass thrust coefficient
cd_nozzle = 99.0 / 100  # nozzle discharge coefficient

# cooling  TODO: decide cooling based on heat limits on the turbine
q_ngv = 0.5  # fraction of cooling from cooling first stator
bpr_c = 0.0  # cooling from fraction of core flow

second_burner = False

# piston engine stuff
pi_pe = 1.0
cr = 10
bore = 0.14

surrogate = True


# fuel temperature upon injection in piston engine and burner
t_fuel = 300
# fuel tank temperature
t_tank = 300
