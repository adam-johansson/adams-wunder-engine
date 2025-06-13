import numpy as np

cycle = "4T"

fuel = "H2"

thermo = "NASA"

#cooling = "Hohenberg"
cooling = "Woschni"

# premixed or direct_injection or external?
premixed = True

opposed = False
cr = 10.0

cylinders = 1  # for sampling data use 1

# piston
d = 0.082  # diameter / bore
s = 0.090  # stroke
v_mean = (2000 / 60) * 2 * s  # rpm rpm = v_mean / (2 * s) * 60 this is 1400 rpm
l_con = 0.1395  # rod length
bsr = d / s  # bore stroke ratio
rod_stroke_ratio = l_con / s
lms = 1 / (2 * rod_stroke_ratio)  # connecting rod ratio

# inlet and outlet conditions
p_in = 1.38e5  # inlet pressure
T_in = 273.15 + 28  # inlet temperature 28 celsius

p_ratio = 1.4 / 1.38  # pressure ratio after and before engine

# Heat transfer
Twall = 350  # Wall temperature
Tpiston = 350
Thead = 350
Twalls = [Twall, Tpiston, Thead]


ch = 1.4  # multiplier to decrease/increase heat transfer

# Inlet valve
phi_open_in = (716 / 180) * np.pi  #
phi_close_in = (938 / 180) * np.pi  #

# outlet valve
phi_open_out = (505 / 180) * np.pi  #
phi_close_out = (717 / 180) * np.pi  #


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

#valve lift
lv_max = d * 0.1
cd = 0.9

eta_c = 0.999

from thermo import fuel_props
far_s, _ = fuel_props(fuel)
# given for the standard cases
far_goal = far_s / 1.5


wiebe_type = "Single"
# wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 11.0
wm = 0.6


# this is for single wiebe function
m_wiebe = 3.0

phi_sc = (347 / 180) * np.pi
phi_cd = (28 / 180) * np.pi

T_fuel = 300
p_fuel = 50e5

it = 300

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function
