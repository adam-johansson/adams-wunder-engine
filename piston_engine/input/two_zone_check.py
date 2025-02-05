import numpy as np

cycle = "4T"

thermo = "NASA"

fuel = 'jetA'

cooling = "Hohenberg"
#cooling = "Woschni"

opposed = False
#cr = 9.6  # geometric compression ratio (Kaiser used 9.6 ToC and 6.4 TO and 18 in Cruise)
cr = 15

cylinders = 1  # V12

# piston
d = 0.37  # diameter  # 4 liter is it supposed to be (V * 4/pi)^(1/3)
#d = 0.1699  # diameter
v_mean = (1400 / 60 ) * 2 * d  # rpm rpm = v_mean / (2 * s) * 60
bsr = 1  # bore stroke ratio
#rpm = 1400  # revolutions per minute
lms = 1 / (2 * 1.7)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)

# inlet and outlet conditions
p_in = 1.5e5  # inlet pressure
T_in = 350  # inlet temperature
p_ratio = 1.1  # pressure ratio after and before engine

# Heat transfer
Twall = 400          # Liner temperature
Tpiston = 400
Thead = 400
Twalls = [Twall, Tpiston, Thead]


ch = 1.0  # multiplier to decrease heat transfer

# Inlet valve
#phi_open_in = (724/180)*np.pi  # working
#phi_close_in = (920/180)*np.pi  # working
phi_open_in = (719.0/180)*np.pi  # testar
phi_close_in = (913.1/180)*np.pi  # testar

# outlet valve
#phi_open_out = (475/180)*np.pi  # working
#phi_close_out = (727/180)*np.pi  # working
phi_open_out = (512.5/180)*np.pi  # testar
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
far_goal = 0.04


wiebe_type = "Single_mass"
#wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 6.91  # funkar
wm = 1.40  # funkar
#wa = 6.82  # matchar power
#wm = 1.492  # matchar power



# this if for single wiebe function
m_wiebe = 1.0

phi_sc = (360/180)*np.pi  # angle at combustion start  THIS WORKED WITH SINGLE #345
phi_cd = (55/180)*np.pi  # angle related to combustion duration WORKED WITH SINGLE #55

#phi_sc = (345/180)*np.pi  # angle at combustion start
#phi_cd = (50/180)*np.pi  # angle related to combustion duration

#phi_sc = (362/180)*np.pi  # angle at combustion start
#phi_cd = (40/180)*np.pi  # angle related to combustion duration

T_fuel = 500
p_fuel = 2500e5

it = 100

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function