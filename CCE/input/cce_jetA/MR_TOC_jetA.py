# Two shaft CCE piston fueled on hydrogen

alt = 10668  # [m] altitude
M = 0.78  # Flight Mach number
dTisa = 10  # [K] deviation from ISA

# either H2 or jetA
fuel = "jetA"

Fn = 27141 # [N]
 

Fs_req = 109.149038848  # specific thrust [m/s]

# power offtake from HPT
power_offtake = 150e3


## OBS: OPR 18 and CR 10 worked

# piston engine stuff
pi_pe = 1.1
cr = 8  # 8
far_piston = 0.0325  * (44 / 43)  # adjust for difference in LHV between JETA and SAF
v_mean = 14
start_of_combustion = 355

OPR = 24  # overall pressure ratio (including losses) (16 + far 03)
PR = 0.25  # pressure split, with regard to the LPC
eff_IC = 0.7
ratio_IC = 0.0

T4 = 1250  # [K] Turbine entry temperature (1200)

dp_intake = 0.00264968668  # intake pressure loss
dp_bypass = 0.00994454006  # bypass duct pressure loss

#dp_inter_compressor = 0.0115
dp_inter_compressor = 0.0


eta_fan = 0.918085501  # fan isentropic
eta_p_lpc = 0.892862994  # polytropic
eta_p_hpc = 0.8993560718463893  # polytropic #0.9075 after correction for reference engine

eta_b = 99.9 / 100  # combustor efficiency (burner)
dPcomb = 1 - 0.970624678   # combustor pressure lost [%]

eta_lpt = 0.9211330307501496  # polytropic efficiency lpt

eta_s = 99.5 / 100  # Shaft efficiency
eta_g = 99.5 / 100  # planetary gearbox efficiency

cfg_core = 99.0 / 100  # core thrust coefficient
cfg_bypass = 99.0 / 100  # bypass thrust coefficient
cd_nozzle = 95.0 / 100  # nozzle discharge coefficient


second_burner = True
surrogate = False
intercooler = True
specific = False
life_hack = True
piston_mode = "DI"

bore = 0.18 # not used

fpr_outer = 1.47  # Outer fan pressure ratio 

bpr = 16.0  # bypass ratio (also not used)

# fuel temperature upon injection in piston engine and burner
t_fuel = 300
# fuel tank temperature
t_tank = 270

LPT_eff_type = "poly"  # polytropic efficiency lpt
