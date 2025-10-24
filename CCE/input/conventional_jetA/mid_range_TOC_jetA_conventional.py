
alt = 10058.0  # [m] altitude
M = 0.78  # Flight Mach number
dTisa = 10  # [K] deviation from ISA

# either H2 or jetA
fuel = "jetA"

Fn = 30693  # [N] #Net thrust requirement 6900 lbs

Fs_req = 95.6  # specific thrust [m/s]

# power offtake from HPT
power_offtake = 150*1e3

bpr = 14.0  # bypass ratio

fpr_outer = 1.396  # Outer fan pressure ratio 1.3087

OPR = 49.1  # overall pressure ratio (including losses)
PR = 0.2767  # pressure split, with regard to the LPC

T4 = 1714  # [K] Turbine entry temperature

dp_intake = 0.2 / 100  # intake pressure loss
dp_bypass = 0.0 / 100  # bypass duct pressure loss
dp_rec = 6.0 / 100  # recuperator pressure loss
dT_rec = 134


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


# fuel temperature upon injection in piston engine and burner
t_fuel = 300
# fuel tank temperature
t_tank = 270

