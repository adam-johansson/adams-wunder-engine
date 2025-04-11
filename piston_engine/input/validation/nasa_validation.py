import numpy as np

cycle = "2T"
thermo = "NASA"

fuel = 'jetA'

cooling = "Hohenberg"
#cooling = "Woschni"

premixed = False

opposed = True
cr = 9.1712  # geometric compression ratio

# piston
d = 0.0787654  # diameter (bore)
s = 0.074676  # stroke
bsr = d/s  # bore stroke ratio
v_mean = 15.24  # mean velocity
l_con = 0.182067  # rod length
rpm = v_mean / (2 * s) * 60  # revolutions per minute
lms = s / (2 * l_con)

# inlet and outlet conditions
p_in = 10.342e5  # inlet pressure
T_in = 491.7  # inlet temperature
p_out = 9.4803e5  # outlet pressure
p_ratio = p_out/p_in

#Heat transfer
Twall = 811          #Wall temperature
Tpiston = 811
Thead = 811

Twalls = [Twall, Tpiston, Tpiston] #opposed piston

ch = 1.0 #multiplier to decrease heat transfer (if Hohenberg)
#ch = 0.8 #if woschni

#Inlet valve
phi_open_in = (125/180)*np.pi
phi_close_in = (235/180)*np.pi

#outlet valve
phi_open_out = (100/180)*np.pi
phi_close_out = (260/180)*np.pi

valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

valve_type = "port"
n_valve = 2

# den används såklart inte för vi har ports
lv_max = 0.01235
# cd = 0.84 worked well
cd = 0.84

# this is assumed in the wiebe function
eta_c = 0.999

# fuel mass per cycle
mf_tot = 9.979e-5


wiebe_type = "Single_mass"

wa = 6
wm = 0.94

# this if for single wiebe function from NASA validation paper
m_wiebe = 1.0

phi_sc = (345/180)*np.pi  # angle at combustion start
phi_cd = (55/180)*np.pi  # angle related to combustion duration

T_fuel = 300
p_fuel = 2500e5

# used to be 30
it = 1000

far_goal = 0  # just for input so program is satisfied
cylinders = 1

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function
