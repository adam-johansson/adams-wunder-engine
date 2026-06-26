import numpy as np

cycle = "4T"

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
rpm = 123 #rpm (this is not loaded to the model)
s = d/bsr #stroke (2200 mm)
v_mean = rpm * 2 * s /60 # mean velocity (input to the model)

l_con = 2.885  #connecting rod length
lms = s/(2*l_con)  #connecting rod ratio

# inlet and outlet conditions
p_in = 1.0e5  # inlet pressure (kaiser had 8 bar cruise 26 bar take off)
T_in = 300  # inlet temperature (670 cruise 770 TO)
p_ratio = 1.1  # pressure ratio after and before engine

# EGR
equ_in = 0.0


# Heat transfer
Twall = 400          # Liner temperature
Tpiston = 400
Thead = 400
Twalls = [Twall, Tpiston, Thead]


ch = 1.0  # multiplier to decrease heat transfer

# Inlet valve
phi_open_in = (719.0/180)*np.pi  # testar
phi_close_in = (913.1/180)*np.pi  # testar

# outlet valve
phi_open_out = (512.5/180)*np.pi  # testar
phi_close_out = (730.8/180)*np.pi  # testar

valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.070 # from some power point online

cd = 0.8

# 99.9 should be used
eta_c = 0.999

far_goal = 0.03

wiebe_type = "Single_mass"


# this if for single wiebe function
m_wiebe = 2.0

phi_sc = (355/180)*np.pi  # angle at combustion start  THIS WORKED WITH SINGLE #345
phi_cd = (55/180)*np.pi  # angle related to combustion duration WORKED WITH SINGLE #55


T_fuel = 300
p_fuel = 2500e5

it = 300

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function