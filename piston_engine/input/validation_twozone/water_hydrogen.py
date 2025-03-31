import numpy as np



# article: Effect of water injection and spark timing on the nitric oxide emission and combustion parameters of a hydrogen fuelled spark ignition engine

cycle = "4T"

fuel = 'H2'

thermo = "NASA"

#cooling = "Hohenberg"
cooling = "Woschni"

# premixed or direct_injection or external?
premixed = True

opposed = False
cr = 9

cylinders = 1  # for sampling data use 1

# piston
d = 0.085  # diameter / bore
s = 0.09  # stroke
v_mean = (2500 / 60 ) * 2 * s  # rpm rpm = v_mean / (2 * s) * 60 this is 1400 rpm
l_con = 0.1395  # rod length
bsr = d / s  # bore stroke ratio
rod_stroke_ratio = l_con / s
lms = 1 / (2 * rod_stroke_ratio)  # connecting rod ratio

# inlet and outlet conditions
p_in = 1.01e5  # inlet pressure
T_in = 298  # inlet temperature

p_ratio = 1.0  # pressure ratio after and before engine

# Heat transfer
Twall = 400          # Wall temperature
Tpiston = 400
Thead = 400
Twalls = [Twall, Tpiston, Thead]


ch = 3  # multiplier to decrease/increase heat transfer (THIS WAS 1.8 before)

# Inlet valve
#phi_open_in = (716/180)*np.pi  #
#phi_close_in = (938/180)*np.pi  #

# outlet valve
#phi_open_out = (535/180)*np.pi  #
#phi_close_out = (717/180)*np.pi  #

# inlet valve
phi_open_in = (718.0/180)*np.pi  # given
phi_close_in = (931.0/180)*np.pi  # given

# outlet valve
phi_open_out = (506.0/180)*np.pi  # given
phi_close_out = (734.0/180)*np.pi  # given


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.1 * d
cd = 0.8

eta_c = 0.999

far_goal = 0.0291755 * 0.9


wiebe_type = "Single"
#wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 11.0
wm = 0.6


# this is for single wiebe function
m_wiebe = 4.0  # from validation italian

phi_sc = (359/180)*np.pi  # angle at combustion start  from validation italian
phi_cd = (19.0/180)*np.pi  # angle related to combustion duration from validation italian

T_fuel = 300
p_fuel = 50e5

it = 100

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function