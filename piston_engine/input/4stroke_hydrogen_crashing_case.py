import numpy as np

cycle = "4T"

fuel = 'H2'

thermo = "NASA"

#cooling = "Hohenberg"
cooling = "Woschni"
#cooling = "H2"

premixed = False


#print(p*1e-5, T, cr, bore, far_goal, p_ratio, v_mean, fuel_t)
#10.87802882712856 254.86646473468676 8.696823708631488 0.17920054591095363 0.019652998085297933 1.2938272519363732 9.045152901698128 306.8975735624858


opposed = False
cr = 7.548272190148182

cylinders = 1  # for sampling data use 1

# piston
d = 0.16875509154376395   # diameter / bore
# s = 0.074676  # stroke
v_mean = 11.947611201257603   # mean velocity
# l_con = 0.182067  # rod length
bsr = 1.0  # bore stroke ratio
lms = 1 / (2 * 1.7)  # connecting rod ratio (from Kaiser, cite 147 Handbuch Verbrennungsmotor)

# inlet and outlet conditions
p_in = 5.586222153126521e5  # inlet pressure
T_in = 284.6943897271728  # inlet temperature
#T_in = 1000
p_ratio = 1.3943488292879005   # pressure ratio after and before engine

# Heat transfer
Twall = 500          # Wall temperature
Tpiston = 600
Thead = 600
Twalls = [Twall, Tpiston, Thead]


ch = 1.4  # multiplier to decrease/increase heat transfer

# Inlet valve
phi_open_in = (715/180)*np.pi
phi_close_in = (918/180)*np.pi

# outlet valve
phi_open_out = (515/180)*np.pi
phi_close_out = (729/180)*np.pi


valve_timings = [phi_open_in, phi_close_in, phi_open_out, phi_close_out]

n_valve = 2
valve_type = "valve"

lv_max = 0.1 * d
cd = 0.8

eta_c = 0.999

far_goal =  0.022023661416901992


wiebe_type = "Single"
#wiebe_type = "Double"
# This is for Kaisers wiebe function (double)
wa = 11.0
wm = 0.6


# this is for single wiebe function
m_wiebe = 1.75  # from validation italian

phi_sc = (361/180)*np.pi  # angle at combustion start  from validation italian
phi_cd = (35/180)*np.pi  # angle related to combustion duration from validation italian

T_fuel = 391.44193190319555
p_fuel = 300e5

it = 30

mf_tot = 1.5e-4

# double wiebe function
c1 = 2.0  # shape factor for diffusion burning
c4 = 0.3  # premixed / diffusion burning distribution
c5 = 2.0  # shape factor for premixed burning function