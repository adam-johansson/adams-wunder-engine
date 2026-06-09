import numpy as np

cycle = "4T"

fuel = 'jetA'

cooling = "Hohenberg"

premixed = False
opposed = False



cr = 0

cylinders = 1  # V12

# piston
d = 0  # diameter
#d = 0.1699  # diameter
v_mean = 0  # mean velocity from Kaiser's thesis
bsr = 1  # bore stroke ratio
# rpm = 4000  # revolutions per minute
lms = 0.28  # connecting rod ratio from Hanbuch Verbrennungsmotor sid 95

# inlet and outlet conditions
p_in = 0 # inlet pressure (kaiser had 8 bar cruise 26 bar take off)
T_in = 0  # inlet temperature (670 cruise 770 TO)
p_ratio = 0  # pressure ratio after and before engine

# Heat transfer
Twall = 500          # Liner temperature
Tpiston = 600
Thead = 600
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

n_valve = 2
valve_type = "valve"

lv_max = 0.015

cd = 0.8

# 99.9 should be used
eta_c = 0.999

far_goal = 999


wiebe_type = "Single_mass"
#wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 6.91  # funkar
wm = 1.40  # funkar
#wa = 6.82  # matchar power
#wm = 1.492  # matchar power



# this if for single wiebe function
# VALUES FROM NASAS TWOSTROKE PAPER
m_wiebe = 2.0

phi_sc = 99999 # angle at combustion start  THIS WORKED WITH SINGLE #345
phi_cd = (50/180)*np.pi  # angle related to combustion duration WORKED WITH SINGLE #55


T_fuel = 0
p_fuel = 2500e5

it = 300

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function