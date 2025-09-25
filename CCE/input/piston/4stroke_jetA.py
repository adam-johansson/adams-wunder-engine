import numpy as np

cycle = "4T"

thermo = "NASA"

fuel = "jetA"

cooling = "Hohenberg"

opposed = False


cylinders = 12  # V12

v_mean = 14  # mean velocity from Kaiser's thesis

bsr = 1  # bore stroke ratio
# rpm = 4000  # revolutions per minute
lms = 1 / (
    2 * 1.7
)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)


# Heat transfer
Twall = 500  # Liner temperature
Tpiston = 600
Thead = 600
Twalls = [Twall, Tpiston, Thead]


ch = 1.0  # multiplier to decrease heat transfer

# Inlet valve
phi_open_in = (719.0 / 180) * np.pi  # testar
phi_close_in = (913.1 / 180) * np.pi  # testar

# outlet valve
phi_open_out = (512.5 / 180) * np.pi  # testar
phi_close_out = (730.8 / 180) * np.pi  # testar

valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

cd = 0.8

eta_c = 0.999


wiebe_type = "Single"
# wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 6.91  # funkar
wm = 1.40  # funkar
# wa = 6.82  # matchar power
# wm = 1.492  # matchar power


# this if for single wiebe function
m_wiebe = 1.0

phi_sc = (345 / 180) * np.pi  # angle at combustion start  THIS WORKED WITH SINGLE #345
phi_cd = (
    55 / 180
) * np.pi  # angle related to combustion duration WORKED WITH SINGLE #55

# phi_sc = (345/180)*np.pi  # angle at combustion start
# phi_cd = (50/180)*np.pi  # angle related to combustion duration

# phi_sc = (362/180)*np.pi  # angle at combustion start
# phi_cd = (40/180)*np.pi  # angle related to combustion duration

T_fuel = 300
p_fuel = 2500e5

it = 40

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function

p_in = 999
T_in = 999
p_ratio = 999
d = 999
cr = 999
far_goal = 999
lv_max = 999
premixed = False
