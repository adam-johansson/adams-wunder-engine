import numpy as np

# THIS IS THE PAPER TYSK2 NOT ARGONNE
# H2 DIRECT INJECTION A HIGHLY PROMISING COMBUSTION CONCEPT
# this uses woschni model

cycle = "4T"

fuel = 'H2'

thermo = "NASA"

cooling = "Woschni"
#cooling = "H2"
#cooling = "Hohenberg"

opposed = False
cr = 10.5  # geometric compression ratio

cylinders = 1

# piston
d = 0.086  # diameter / bore (specified)
# s = 0.074676  # stroke
v_mean = 5.7333  # mean velocity 2000 rpm (specified in tysk2)
# l_con = 0.182067  # rod length
bsr = 0.086 / 0.086  # bore stroke ratio ( specified)

lms = 1 / (2 * 1.0)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor) (1.7 original)

# inlet and outlet conditions
p_in = 1.01325e5*0.99  # inlet pressure
T_in = 293.15  # inlet temperature ASSUMPTION 20 deg Celsius
p_ratio = 1.01 / 0.99  # pressure ratio after and before engine (assumption based on 1% loss in inlet and outlet)

# Heat transfer
Twall = 500          # Wall temperature
Tpiston = 500
Thead = 500
Twalls = [Twall, Tpiston, Thead]


ch = 1.8  # multiplier to increase heat transfer for hydrogen operation

# Inlet valve

# 1.7 lms + 465 open_out + 1/2.649 = 5.46 IMEP

# camshaft has half speed of engine (110 LSA means 220 angle between exhaust opening and intake opening)
open_out = 475  # 465 set bra ut NOT SPECIFIED
diff = 220  # SPECIFIED
phi_open_in = ((open_out + diff)/180)*np.pi  # working 725
phi_close_in = ((open_out + diff + 268)/180)*np.pi  # working 935

# outlet valve
phi_open_out = (open_out/180)*np.pi  # working 496
phi_close_out = ((open_out + 268)/180)*np.pi  # working 750

#phi_open_in = ((678)/180)*np.pi  # working 725
#phi_close_in = ((946)/180)*np.pi  # working 935

# outlet valve
#phi_open_out = (458/180)*np.pi  # working 496
#phi_close_out = ((716)/180)*np.pi  # working 750


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.0095  # this is specified in tysk2
cd = 0.7

#eta_c = 0.999
eta_c = 1.0

# 2.3 matchar total heat för 11 cd och m = 1.1
throttle = 0.0298 / 2.7

phi_sc = (362/180)*np.pi  # angle at combustion start
phi_cd = (12/180)*np.pi  # angle related to combustion duration

T_fuel = 300
p_fuel = 150e5

wiebe_type = "Single"

# This is for Kaisers wiebe function
wa = 11.0
wm = 0.6

# this is for single wiebe function
m_wiebe = 1.1  #0.6

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function

it = 40

mf_tot = 1.5e-4