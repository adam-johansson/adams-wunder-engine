import matplotlib.pyplot as plt
from thermo import fuel_props
from piston_engine.engine import run_piston_engine
import importlib


import numpy as np
from neural_network.src import load_ANN

### LOAD THE MODELS

input_file_pist = "4stroke_sampling"
input_dir_pist = "piston_engine.input"
path_pist = input_dir_pist + "." + input_file_pist
d = importlib.import_module(path_pist)
piston_input = {
    'p_in': d.p_in,
    'T_in': d.T_in,
    'p_ratio': d.p_ratio,
    'cycle': d.cycle,
    'cooling': d.cooling,
    'opposed': d.opposed,
    'cr': d.cr,
    'bore': d.d,
    'bsr': d.bsr,
    'v_mean': d.v_mean,
    'lms': d.lms,
    'Twalls': d.Twalls,
    'ch': d.ch,
    'valve_timings': d.valve_timings,
    'n_valve': d.n_valve,
    'lv_max': d.lv_max,
    'cd': d.cd,
    'eta_c': d.eta_c,
    'mf_tot': d.mf_tot,
    'wa': d.wa,
    'wm': d.wm,
    'm_wiebe': d.m_wiebe,
    'phi_sc': d.phi_sc,
    'phi_cd': d.phi_cd,
    'T_fuel': d.T_fuel,
    'p_fuel': d.p_fuel,
    'it': d.it,
    'wiebe_type': d.wiebe_type,
    'valve_type': d.valve_type,
    'far_goal': d.far_goal,
    'cylinders': d.cylinders,
    'fuel': d.fuel,
    'c1': d.c1,
    'c4': d.c4,
    'c5': d.c5,
    'premixed': d.premixed,
}

# input
pin = 15e5
Tin = 500
cr = 9
p_ratio = 1.3
v_mean = 16
T_fuel = 400

far34 = 0.055

fuel_type = "jetA"

piston_input["p_in"] = pin  # fuel air ratio
piston_input["T_in"] = Tin  # fuel air ratio
piston_input["cr"] = cr  # fuel air ratio
piston_input["p_ratio"] = p_ratio  # fuel air ratio
piston_input["v_mean"] = v_mean  # fuel air ratio
piston_input["T_fuel"] = T_fuel  # fuel air ratio
piston_input["far_goal"] = far34  # fuel air ratio


num = 10
bores = np.linspace(0.10,0.20,num)

specific_powers = np.zeros(num)
m_ins = np.zeros(num)
heat_losses = np.zeros(num)
spec_heat_losses = np.zeros(num)
p_maxs = np.zeros(num)
T_maxs = np.zeros(num)
T_outs = np.zeros(num)
noxs = np.zeros(num)
effs = np.zeros(num)
powers = np.zeros(num)

displacements = np.zeros(num)

far_s, LHV = fuel_props(fuel_type)

flags =  ["sweep"]

i = 0

for bore in bores:
    print(i)
    piston_input["bore"] = bore  # fuel air ratio
    (
        T34,
        _,
        _,
        m_in,
        p_max,
        T_max,
        _,
        _,
        indicated_power,
        _,
        _,
        heat_loss,
        p_tdc,
        _,
        nox_ppm,
        _,
        EI_nox,
        _,
        nox_spec,
    ) = run_piston_engine(piston_input, flags)


    #specific fuel flow (since fuel_flow_tot = far34 * m_in)
    fuel_flow = far34 * m_in

    powers[i] = indicated_power
    specific_powers[i] = indicated_power / m_in
    m_ins[i] = m_in
    spec_heat_losses[i] = heat_loss / m_in
    heat_losses[i] = heat_loss
    p_maxs[i] = p_max
    T_maxs[i] = T_max
    T_outs[i] = T34
    noxs[i] = nox_ppm
    effs[i] = indicated_power /(fuel_flow * LHV)


    displacements[i] = bore * np.pi * (bore/2)**2
    i = i + 1



# get specific power and mass flow from mid bore
spec_power = specific_powers[5]
mass_flow = m_ins[5]
T_out_mid = T_outs[5]
heat_loss_mid = heat_losses[5]

# also get it from one more point
T_out_low = T_outs[0]
heat_loss_low = heat_losses[0]
bore_low = bores[0]

### assume mdot = k*bore**2
bore_mid = bores[5]
k = mass_flow/bore_mid**2

#print(spec_power, mass_flow, bore_mid, k)

#approximate mass flow and power
approx_m_ins = bores*bores*k
approx_powers = approx_m_ins * spec_power

approx_eff = spec_power / (far34 * LHV) * np.ones(num)

# try to model heat loss and T out
# assume T_out = c0 + c1*bore

c1_T = (T_out_mid - T_out_low) / (bore_mid - bore_low)
c0_T = T_out_mid - c1_T * bore_mid
approx_T_out = c0_T + c1_T * bores


# assume heat_loss = k0 + k1*bore**2
c1_T = (heat_loss_mid - heat_loss_low) / (bore_mid**2 - bore_low**2)
c0_T = heat_loss_mid - c1_T * bore_mid**2
approx_heat_loss = c0_T + c1_T * bores**2

param_name = "bore"

_, ax1 = plt.subplots()
ax1.plot(bores, specific_powers*1e-3)
ax1.set_xlabel(f"{param_name}")
ax1.set_ylabel(f"Specific power [kW/kg/s]")




_, ax2 = plt.subplots()
ax2.plot(bores, m_ins, label="True")
ax2.plot(bores, approx_m_ins, label="Approximation")
ax2.set_xlabel(f"{param_name}")
ax2.set_ylabel(f"massflow in [kg/s]")
ax2.legend()



_, ax3 = plt.subplots()
ax3.plot(bores, spec_heat_losses*1e-3)
ax3.set_xlabel(f"{param_name}")
ax3.set_ylabel(f"Specific heat loss [kW/kg/s]")


_, ax4 = plt.subplots()
ax4.plot(bores, p_maxs*1e-5)
ax4.set_xlabel(f"{param_name}")
ax4.set_ylabel(f"Max pressure [bar]")

_, ax5 = plt.subplots()
ax5.plot(bores, T_outs, label="True")
ax5.plot(bores, approx_T_out, label="Approx")
ax5.set_xlabel(f"{param_name}")
ax5.set_ylabel(f"Outlet temperature [K]")

_, ax6 = plt.subplots()
ax6.plot(bores, noxs)
ax6.set_xlabel(f"{param_name}")
ax6.set_ylabel(f"NOX [ppm]")



_, ax7 = plt.subplots()
ax7.plot(bores*bores, m_ins)
ax7.set_xlabel(f"bore**2")
ax7.set_ylabel(f"massflow in [kg/s]")
ax7.grid()


_, ax8 = plt.subplots()
ax8.plot(bores, effs*100, label="True")
ax8.plot(bores, approx_eff*100, label="Approx")


ax8.set_xlabel(f"{param_name}")
ax8.set_ylabel(f"Thermal efficiency [percent]")
ax8.legend()

_, ax9 = plt.subplots()
ax9.plot(bores, powers*1e-3, label="True")
ax9.plot(bores, approx_powers*1e-3, label="Approximation")
ax9.set_xlabel(f"{param_name}")
ax9.set_ylabel(f"Power [kW]")
ax9.legend()

_, ax10 = plt.subplots()
ax10.plot(bores, 100*(approx_powers - powers)/powers)
ax10.set_xlabel(f"{param_name}")
ax10.set_ylabel(f"Power percentage error")

_, ax11 = plt.subplots()
ax11.plot(bores, 100*(approx_m_ins - m_ins) / m_ins)
ax11.set_xlabel(f"{param_name}")
ax11.set_ylabel(f"massflow percentage error")

_, ax12 = plt.subplots()
ax12.plot(bores, heat_losses*1e-3, label="True")
ax12.plot(bores, approx_heat_loss*1e-3, label="Approximation")
ax12.set_xlabel(f"{param_name}")
ax12.set_ylabel(f"Heat losses [kW]")
ax12.legend()


_, ax13 = plt.subplots()
ax13.plot(bores, 100*(approx_heat_loss - heat_losses) / heat_losses)
ax13.set_xlabel(f"{param_name}")
ax13.set_ylabel(f"heatloss percentage error")

plt.show()



