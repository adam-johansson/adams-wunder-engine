import numpy as np


# article: Effect of water injection and spark timing on the nitric oxide emission and combustion parameters of a hydrogen fuelled spark ignition engine

cycle = "4T"

fuel = "H2"

thermo = "NASA"

# cooling = "Hohenberg"
cooling = "Woschni"
# cooling = "H2"

# premixed or direct_injection or external?
premixed = True

opposed = False
cr = 17

cylinders = 1  # for sampling data use 1

# piston
d = 0.100  # diameter / bore
s = 0.105  # stroke
v_mean = (2200 / 60) * 2 * s  # rpm rpm = v_mean / (2 * s) * 60

bsr = d / s  # bore stroke ratio

lms = 1 / (2 * 1.7)  # connecting rod ratio

# inlet and outlet conditions
p_in = 1.01e5  # inlet pressure
T_in = 366  # inlet temperature

p_ratio = 0.95  # pressure ratio after and before engine

# Heat transfer
Twall = 373  # Wall temperature
Tpiston = 373
Thead = 373
Twalls = [Twall, Tpiston, Thead]


# COULD ARGUE FOR LOWER NUMBER FOR HCCI
ch = 1  # multiplier to decrease/increase heat transfer (THIS WAS 1.8 before)

# inlet valve
phi_open_in = (688.0 / 180) * np.pi
phi_close_in = (959.0 / 180) * np.pi

# outlet valve
phi_open_out = (469.0 / 180) * np.pi
phi_close_out = (752.0 / 180) * np.pi


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.1 * d
cd = 0.8

eta_c = 0.999

far_goal = 0.0291756 / 3

wiebe_type = "Single"


# this is for single wiebe function
m_wiebe = 0.3  # 4
phi_sc = (359 / 180) * np.pi  # angle at combustion start
phi_cd = (10 / 180) * np.pi  # given


T_fuel = 300
p_fuel = 50e5

it = 100

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function

# wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 11.0
wm = 0.6
