# Two shaft CCE piston

alt = 11278  # [m] altitude
M = 0.80  # Flight Mach number
dTisa = 0  # [K] deviation from ISA

# cce or normal operation
cce = True

#gas properties
gamma_c = 1.400  # cold
gamma_h = 1.333  # hot gas
cpa = 1005  # cp air
cpg = 1148  # cp for combustion products
LHV = 42.6e6  # lower heat value for diesel

Fn = 32.87e3  # [N] #Net thrust requirement

bpr = 36.5  # Bypass ratio

opr = 31.3  # Overall pressure ratio
pi_ipc = 3.1  # IPC Pressure ratio
pi_hpc = 6.2  # HPC pressure ratio
pi_pe = 1.46  # Piston engine pressure ratio
fpr = opr/(pi_pe*pi_hpc*pi_ipc)  # Fan pressure ratio

TET = 1345  # [K] Turbine entry temperature

eta_intake = 99.7/100  # intake efficiency
eta_p_fan = 90/100  # fan polytropic efficiency
eta_p_ipc = 91.0/100  # polytropic
eta_p_hpc = 89.0/100  # polytropic

eta_b = 99.99/100 #combustor efficiency (burner)
dPcomb = 4.0/100 #combustor pressure lost [%]

eta_p_hpt = 90.5/100  # High pressure turbine
eta_p_lpt = 91.5/100
effCold = 98/100
effHot = 99/100
eta_s = 99.0/100  # Shaft efficiency


#cooling
alpha = 0.6
bpr_c = 0.2

alpha = 0.0
bpr_c = 0.0
