import numpy as np

cycle = "4T"

thermo = "NASA"

fuel = "jetA"

#cooling = "Hohenberg"
cooling = "Woschni"

premixed = False

opposed = False
cr = 15.5

cylinders = 1  # V12


# inlet and outlet conditions (1.5 bar)
p_in = 1.45e5  # inlet pressure
T_in = 310  # inlet temperature
p_ratio = 1.0  # pressure ratio after and before engine

# 0.035 works
far_goal = 0.035

# this is for single wiebe function
m_wiebe = 0.3

# 0.35 + 359 + 80 works best so far

phi_sc = (359 / 180) * np.pi  # angle at combustion start SORT OF GIVEN
phi_cd = (
    90 / 180
) * np.pi  # angle related to combustion duration WORKED WITH SINGLE #55



# Heat transfer
Twall = 450  # Liner temperature
Tpiston = 450
Thead = 450
Twalls = [Twall, Tpiston, Thead]

# piston
d = 0.172  # diameter  # approx 4 liter

# rpm = 1400  # revolutions per minute
v_mean = (1400 / 60) * 2 * d  # rpm rpm = v_mean / (2 * s) * 60 this is 1400 rpm
bsr = 1  # bore stroke ratio

lms = 1 / (
    2 * 1.5
)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)
# lms = 1 / (2 * 1.5)  # connecting rod ratio


ch = 1.0  # multiplier to decrease heat transfer

# Inlet valve
phi_open_in = (719.0 / 180) * np.pi  # testar
phi_close_in = (905 / 180) * np.pi  # VARYING

# outlet valve
phi_open_out = (490.0 / 180) * np.pi  # THIS IS GIVEN
phi_close_out = (730.8 / 180) * np.pi  # testar

valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

# valves per side
n_valve = 2
valve_type = "valve"

lv_max = 0.1 * d

# coefficeint of flow in valves
cd = 0.8

# 99.9 should be used
eta_c = 0.999


wiebe_type = "Single_mass"


# This is for Kaisers wiebe function (double)
wa = 6.91  # funkar
wm = 1.40  # funkar


T_fuel = 500
p_fuel = 2500e5

it = 300

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function
