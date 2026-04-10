
alt = 10668  # [m] altitude
M = 0.78  # Flight Mach number
dTisa = 10  # [K] deviation from ISA

# either H2 or jetA
fuel = "jetA"

Fn = 27141 # [N]

Fs_req = 109.149038848  # specific thrust [m/s]

# power offtake from HPT
power_offtake = 150*1e3

bpr = 12.0  # bypass ratio

fpr_outer = 1.47  # Outer fan pressure ratio 

OPR = 51.893210825  # overall pressure ratio (including losses)
PR = 0.28075641779 # pressure split, with regard to the LPC

T4 = 1704  # [K] Turbine entry temperature

dp_intake = 0.00264968668  # intake pressure loss
dp_bypass = 0.00994454006  # bypass duct pressure loss
dp_rec = 6.0 / 100  # recuperator pressure loss
dT_rec = 134


eta_fan = 0.918085501  # fan isentropic
eta_p_lpc = 0.892862994  # polytropic
eta_p_hpc = 0.9075  # polytropic #0.900486422

eta_b = 99.9 / 100  # combustor efficiency (burner)
dPcomb = 1 - 0.970624678   # combustor pressure lost [%]

#eta_hpt = 0.909849252  # isentropic efficiency hpt
#eta_lpt = 0.939893246  # isentropic efficiency lpt

eta_hpt = 0.8935325636828619  # polytropic efficiency hpt
eta_lpt = 0.9211330307501496  # polytropic efficiency lpt

eta_s = 99.5 / 100  # Shaft efficiency
eta_g = 99.5 / 100  # planetary gearbox efficiency

cfg_core = 99.0 / 100  # core thrust coefficient
cfg_bypass = 99.0 / 100  # bypass thrust coefficient
cd_nozzle = 95.0 / 100  # nozzle discharge coefficient


# fuel temperature upon injection in piston engine and burner
t_fuel = 300
# fuel tank temperature
t_tank = 270

HPT_eff_type = "poly"  # polytropic efficiency hpt
LPT_eff_type = "poly"  # polytropic efficiency lpt

