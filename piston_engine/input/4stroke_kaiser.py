import numpy as np

cycle = "4T"

thermo = "NASA"

fuel = "jetA"

cooling = "Hohenberg"
cooling = "Woschni"

opposed = False
# cr = 9.6  # geometric compression ratio (Kaiser used 9.6 ToC and 6.4 TO and 18 in Cruise)
cr = 7.2

cylinders = 12  # V12

# piston
# d = 0.16915  # diameter
d = 0.173

v_mean = 18  # mean velocity from Kaiser's thesis
bsr = 1  # bore stroke ratio
# rpm = 4000  # revolutions per minute
lms = 1 / (
    2 * 1.7
)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)

# inlet and outlet conditions
p_in = 912.02e3 * 0.99  # inlet pressure
T_in = 708.827  # inlet temperature
p_ratio = (1170.14e3 / 0.99) / p_in  # pressure ratio after and before engine

# Heat transfer
Twall = 500  # Wall temperature / liner temp
Tpiston = 600
Thead = 600
Twalls = [Twall, Tpiston, Thead]


ch = 1.0  # multiplier to decrease heat transfer

# Inlet valve
phi_open_in = (730 / 180) * np.pi  # testar
phi_close_in = (900.0 / 180) * np.pi  # testar (900 verkningsgrad)

# optimal verkingsgrad : 730.0 900.0 520.0089031606961 735.0
# optimal massflöde : 727.0 910.7 500.0 726.2
# outlet valve
phi_open_out = (520.0 / 180) * np.pi  # (511 för bättre power output)
phi_close_out = (735 / 180) * np.pi  # testar
valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.04
cd = 0.8

eta_c = 0.999

throttle = 0.03

wiebe_type = "Double"

# This is for Kaisers wiebe function (double)
wa = 6.91  # funkar
wm = 1.4  # funkar
# wa = 6.82  # matchar power
# wm = 1.492  # matchar power

# this if for single wiebe function
m_wiebe = 1.0

phi_sc = (355 / 180) * np.pi  # angle at combustion start
phi_cd = (50 / 180) * np.pi  # angle related to combustion duration

T_fuel = 300
p_fuel = 2500e5

it = 20

mf_tot = 1.5e-4  # this is only for validation
