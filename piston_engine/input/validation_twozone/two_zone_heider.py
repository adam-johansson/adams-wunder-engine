import numpy as np

cycle = "4T"

thermo = "NASA"

fuel = 'jetA'

#cooling = "Hohenberg"
cooling = "Woschni"

opposed = False
cr = 15.5

cylinders = 1  # V12

# piston
d = 0.37  # diameter  # 4 liter is it supposed to be (V * 4/pi)^(1/3)

#rpm = 1400  # revolutions per minute
v_mean = (1400 / 60 ) * 2 * d  # rpm rpm = v_mean / (2 * s) * 60 this is 1400 rpm
bsr = 1  # bore stroke ratio

#lms = 1 / (2 * 1.7)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)
lms = 1 / (2 * 1.5)  # connecting rod ratio

# inlet and outlet conditions (1.5 bar)
p_in = 1.5e5  # inlet pressure
T_in = 298  # inlet temperature
p_ratio = 0.95  # pressure ratio after and before engine

# Heat transfer
Twall = 400          # Liner temperature
Tpiston = 400
Thead = 400
Twalls = [Twall, Tpiston, Thead]


ch = 1.0  # multiplier to decrease heat transfer

# Inlet valve
phi_open_in = (719.0/180)*np.pi  # testar
phi_close_in = (905/180)*np.pi  # VARYING

# outlet valve
phi_open_out = (490.0/180)*np.pi  # THIS IS GIVEN
phi_close_out = (730.8/180)*np.pi  # testar

valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

# valves per side
n_valve = 2
valve_type = "valve"

lv_max = 0.1 * d

# coefficeint of flow in valves
cd = 0.8

# 99.9 should be used
eta_c = 0.999

# 0.05
far_goal = 0.035


wiebe_type = "Single_mass"
#wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 6.91  # funkar
wm = 1.40  # funkar



# this is for single wiebe function
m_wiebe = 0.35

# 0.35 + 359 + 80 works best so far

phi_sc = (359/180)*np.pi  # angle at combustion start SORT OF GIVEN
phi_cd = (80/180)*np.pi  # angle related to combustion duration WORKED WITH SINGLE #55
T_fuel = 500
p_fuel = 2500e5

it = 100

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function