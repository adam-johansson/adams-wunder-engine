import numpy as np

cycle = "2T"

fuel = 'jetA'

cooling = "Hohenberg"

premixed = False
opposed = False
mode = "DI"

cr = 18.2

cylinders = 4  

# piston
d = 0.500  # diameter
bsr = 500 / 2200  # bore stroke ratio
rpm = 111.8 #rpm (this is not loaded to the model)
s = d/bsr #stroke (2200 mm)
v_mean = rpm * 2 * s /60 # mean velocity (input to the model)   approx 9 m/s

l_con = 2.885  #connecting rod length
lms = s/(2*l_con)  #connecting rod ratio

# inlet and outlet conditions
p_in = 3.65e5  # inlet pressure (kaiser had 8 bar cruise 26 bar take off)
T_in = 390  # inlet temperature (670 cruise 770 TO)
p_ratio = 0.97  # pressure ratio after and before engine

mf_tot = 0.034

# EGR
equ_in = 0.0


# Heat transfer
Twall = 450          # Liner temperature
Tpiston = 450
Thead = 450
Twalls = [Twall, Tpiston, Thead]


ch = 1.0  # multiplier to decrease heat transfer


# TIMINGS FROM LAMARIS PAPER 2009 this is for the
# Inlet valve
phi_open_in = (140 / 180) * np.pi
phi_close_in = (220 / 180) * np.pi

# outlet valve
phi_open_out = (113 / 180) * np.pi 
#phi_close_out = (220 / 180) * np.pi
phi_close_out = (266 / 180) * np.pi  #266


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"   # or change to port??

lv_max = 0.070 # from some power point online
#lv_max = 0.1

cd = 0.8

# 99.9 should be used
eta_c = 0.999

far_goal = 0.03

wiebe_type = "Single_mass"


# this is for single wiebe function
m_wiebe = 1.2

phi_sc = (363/180)*np.pi  # angle at combustion start  
phi_cd = (29/180)*np.pi  # angle related to combustion duration 

T_fuel = 300
p_fuel = 2500e5

it = 300


# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function