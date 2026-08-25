import numpy as np
import matplotlib.pyplot as plt

param_name = "SC"
output_dir = f"./results/{param_name}"

textsize = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = textsize


def load(filename):
    data = np.loadtxt(f"{output_dir}/{filename}", skiprows=1)
    return data[:, 0], data[:, 1]


phi_sc, thermal_eff = load("thermal_eff.dat")
_, specific_nox = load("specific_nox.dat")
_, core_spec_power = load("core_spec_power.dat")
_, pmax = load("peak_pressure.dat")
_, Tmax = load("Tmax.dat")
_, Tmax2zone = load("Tmax2zone.dat")
_, T34 = load("T34.dat")
_, T35 = load("T35.dat")
_, NO_ppm_piston = load("NO_ppm_piston.dat")


def make_sweep_plot(x, y, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color='black', linewidth=2, marker='o', markersize=8)

    ax.set_xlabel(r"$\theta_{SC}$ [°]", fontsize=textsize)
    ax.set_ylabel(ylabel, fontsize=textsize)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    ax.tick_params(direction='out', color='black', labelsize=textsize,
                    top=True, right=True, which='both')
    ax.grid(True, color='lightgrey')

    fig.tight_layout()
    fig.savefig(f"{output_dir}/{filename}", dpi=300)
    return fig


fig1 = make_sweep_plot(phi_sc, thermal_eff, r"$\eta_{th}$ [%]", "thermal_eff_vs_phi_sc.pdf")
fig2 = make_sweep_plot(phi_sc, specific_nox, r"Thrust specific $\mathrm{NO_x}$ [mg/Ns]", "nox_vs_phi_sc.pdf")
fig3 = make_sweep_plot(phi_sc, core_spec_power, r"$\dot{W}_{\mathrm{core},V_d}$ [kW/litre]", "core_spec_power_vs_phi_sc.pdf")
fig4 = make_sweep_plot(phi_sc, pmax, r"$p_{max}$ [bar]", "pmax_vs_phi_sc.pdf")
fig5 = make_sweep_plot(phi_sc, Tmax, r"$T_{max}$ [K]", "Tmax_vs_phi_sc.pdf")
fig6 = make_sweep_plot(phi_sc, Tmax2zone, r"$T_{max,2zone}$ [K]", "Tmax2zone_vs_phi_sc.pdf")
fig7 = make_sweep_plot(phi_sc, T34, r"$T_{34}$ [K]", "T34_vs_phi_sc.pdf")
fig8 = make_sweep_plot(phi_sc, T35, r"$T_{35}$ [K]", "T35_vs_phi_sc.pdf")
fig9 = make_sweep_plot(phi_sc, NO_ppm_piston, "NO PM exhaust [ppm]", "NO_ppm_vs_phi_sc.pdf")

plt.show()