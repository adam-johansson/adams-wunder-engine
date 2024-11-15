import numpy as np

cycle = "2T"
#cycle = "4T


# Choose between "Cantera" or "NASA"
#thermo_outdated = "CANTERA"
thermo = "NASA"
#thermo_outdated = "Coolprop"

#cooling = "Woschni"
cooling = "Hohenberg"

opposed = True
cr = 9.1712 #geometric compression ratio

#piston
d = 0.0787654  #diameter
s = 0.074676     #stroke
v_mean = 15.24 #mean velocity
l_con = 0.182067  #rod length

#inlet and outlet conditions
p_in = 10.342e5 # inlet pressure
T_in = 491.7 #inlet temperature
p_out = 9.4803e5 # outlet pressure

#Heat transfer
Twall = 811          #Wall temperature
Tpiston = 811
Thead = 811
#Twalls = [Twall, Tpiston, Thead]
Twalls = [Twall, Tpiston, Tpiston] #opposed piston

ch = 1 #multiplier to decrease heat transfer

#Inlet valve
phi_open_in = (125/180)*np.pi
phi_close_in= (235/180)*np.pi

#outlet valve
phi_open_out= (100/180)*np.pi
phi_close_out = (260/180)*np.pi

valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
#lv_max = 0.0144
#lv_max = 0.55*d/4
lv_max = 0.01235
cd = 0.8

eta_c = 0.99

#TODO: Change this to far
#mf_tot = 1.0375e-4 #for single wiebe
mf_tot = 9.98e-5 #for double wiebe
mf_tot = 9.6e-5


wa = 6
wm = 0.95

# this if for single wiebe function
#this is not implemented now
m_wiebe = 1.05
#m_wiebe = 0.7



phi_sc = (345/180)*np.pi # angle at combustion start
phi_cd = (55/180)*np.pi # angle related to combustion duration


T_fuel = 300
p_fuel = 2500e5