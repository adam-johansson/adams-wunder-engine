# Two shaft CCE piston fueled on hydrogen

alt = 10058.0  # [m] altitude
M = 0.78  # Flight Mach number
dTisa = 10  # [K] deviation from ISA

# either H2 or jetA
fuel = "jetA"

Fn = 30693  # [N] #Net thrust requirement 6900 lbs
 

Fs_req = 95.6  # specific thrust [m/s]

# power offtake from HPT
power_offtake = 150e3


## OBS: OPR 18 and CR 10 worked

# piston engine stuff
pi_pe = 1.1
cr = 8  # 8
far_piston = 0.03
v_mean = 16
start_of_combustion = 362

OPR = 18  # overall pressure ratio (including losses) (16 + far 03)
PR = 0.3  # pressure split, with regard to the LPC
eff_IC = 0.7
ratio_IC = 0.0

T4 = 1200  # [K] Turbine entry temperature (1200)

dp_intake = 0.2 / 100  # intake pressure loss
dp_bypass = 0.0 / 100  # bypass duct pressure loss

#dp_inter_compressor = 0.0115
dp_inter_compressor = 0.0


eta_fan = 91.8 / 100  # inner fan polytropic efficiency
eta_p_lpc = 89.3 / 100  # polytropic
eta_p_hpc = 90.0 / 100  # polytropic

eta_b = 99.9 / 100  # combustor efficiency (burner)
dPcomb = 2.9 / 100  # combustor pressure lost [%]

eta_hpt = 91.0 / 100  # isentropic efficiency hpt
eta_lpt = 94.0 / 100  # isentropic efficiency lpt

eta_s = 99.5 / 100  # Shaft efficiency
eta_g = 99.0 / 100  # planetary gearbox efficiency

cfg_core = 97.7 / 100  # core thrust coefficient
cfg_bypass = 99.5 / 100  # bypass thrust coefficient
cd_nozzle = 99.0 / 100  # nozzle discharge coefficient


second_burner = True
surrogate = False
intercooler = True
specific = False
life_hack = True
piston_mode = "HCCI"

bore = 0.18 # not used


fpr_outer = 1.396  # Outer fan pressure ratio


bpr = 16.0  # bypass ratio (also not used)

# fuel temperature upon injection in piston engine and burner
t_fuel = 300
# fuel tank temperature
t_tank = 300
