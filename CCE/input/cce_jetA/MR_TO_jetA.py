# Two shaft CCE piston fueled on hydrogen

alt = 0.0  # [m] altitude
M = 0.0  # Flight Mach number
dTisa = 0  # [K] deviation from ISA

# either H2 or jetA
fuel = "jetA"

Fn = 147.3e3  # [N] #Net thrust requirement 6900 lbs
 

Fs_req = 221.5  # specific thrust [m/s]

# power offtake from HPT
power_offtake = 150e3


# piston engine stuff
pi_pe = 1.0
cr = 10
far_piston = 0.035
v_mean = 15
start_of_combustion = 360

OPR = 15  # overall pressure ratio (including losses)
PR = 0.3  # pressure split, with regard to the LPC
eff_IC = 0.7

T4 = 1600  # [K] Turbine entry temperature (1200)

dp_intake = 0.2 / 100  # intake pressure loss
dp_bypass = 0.0 / 100  # bypass duct pressure loss

#dp_inter_compressor = 0.0115
dp_inter_compressor = 0.0


eta_fan = 92.7 / 100  # inner fan polytropic efficiency
eta_p_lpc = 90.3 / 100  # polytropic
eta_p_hpc = 90.0 / 100  # polytropic

eta_b = 99.9 / 100  # combustor efficiency (burner)
dPcomb = 2.9 / 100  # combustor pressure lost [%]

eta_hpt = 91.0 / 100  # isentropic efficiency hpt
eta_lpt = 94.3 / 100  # isentropic efficiency lpt

eta_s = 99.5 / 100  # Shaft efficiency
eta_g = 99.0 / 100  # planetary gearbox efficiency

cfg_core = 97.7 / 100  # core thrust coefficient
cfg_bypass = 99.5 / 100  # bypass thrust coefficient
cd_nozzle = 99.0 / 100  # nozzle discharge coefficient


second_burner = True
surrogate = False
intercooler = False
specific = False
life_hack = True

bore = 0.18 # not used


fpr_outer = 1.396  # Outer fan pressure ratio


bpr = 16.0  # bypass ratio (also not used)

# fuel temperature upon injection in piston engine and burner
t_fuel = 300
# fuel tank temperature
t_tank = 300
