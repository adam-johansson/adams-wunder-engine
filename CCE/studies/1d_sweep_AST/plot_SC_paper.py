import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

param_name = "SC"
output_dir = f"./results/{param_name}"

textsize = 25
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = textsize


def load(filename):
    data = np.loadtxt(f"{output_dir}/{filename}", skiprows=1, max_rows=44)
    return data[:, 0], data[:, 1]


phi_sc, thermal_eff = load("thermal_eff.dat")
_, specific_nox = load("specific_nox.dat")
_, core_spec_power = load("core_spec_power.dat")
_, T35 = load("T35.dat")

# Loaded only to find where each constraint is crossed -- not plotted themselves
_, pmax = load("peak_pressure.dat")
_, T34 = load("T34.dat")
_, bores = load("bore.dat")

XLIM = (335, 380)
COLOR_LEFT = 'black'
COLOR_RIGHT = 'tab:red'


def find_threshold_crossing(x, y, threshold):
    """First x where y crosses threshold, linearly interpolated between samples."""
    x = np.asarray(x)
    y = np.asarray(y)
    diff = y - threshold
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return None
    i = sign_change[0]
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]
    frac = (threshold - y0) / (y1 - y0)
    return x0 + frac * (x1 - x0)


bore_x = find_threshold_crossing(phi_sc, bores, 200)
pmax_x = find_threshold_crossing(phi_sc, pmax, 150)
T34_x = find_threshold_crossing(phi_sc, T34, 1250)

# (x position, label, line color, infeasible-region side: 'left', 'right', or None)
constraint_lines = [
    (pmax_x, r"$p_{max} > 150$ bar", 'tab:blue', 'left'),
    (bore_x, r"$d > 200$ mm", 'tab:green', None),
    (T34_x, r"$T_{34} > 1250$ K", 'tab:purple', 'right'),
]


def style_axis(ax, color):
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    ax.tick_params(axis='y', colors=color, labelsize=textsize)
    ax.tick_params(axis='x', direction='out', color='black', labelsize=textsize, which='both')


# (x position, label, line color, infeasible-region side, linestyle)
constraint_lines = [
    (pmax_x, r"$p_{max} > 150$ bar", 'tab:blue', 'left', ':'),
    (bore_x, r"$d > 200$ mm", 'tab:green', None, '--'),
    (T34_x, r"$T_{34} > 1250$ K", 'tab:purple', 'right', '-.'),
]


def add_constraint_lines(ax, xlim):
    for x_pos, label, color, side, linestyle in constraint_lines:
        if x_pos is None:
            print(f"Warning: threshold for '{label}' never crossed in data range -- skipping.")
            continue
        ax.axvline(x_pos, color=color, linestyle=linestyle, linewidth=3.5, label=label, zorder=4)
        if side == 'left':
            ax.axvspan(xlim[0], x_pos, color=color, alpha=0.12, zorder=0.5)
        elif side == 'right':
            ax.axvspan(x_pos, xlim[1], color=color, alpha=0.12, zorder=0.5)


def make_legend_figure(filename, ncol=3):
    handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=3.5, label=label)
        for x_pos, label, color, _, linestyle in constraint_lines if x_pos is not None
    ]
    fig_leg = plt.figure(figsize=(8, 0.6))
    fig_leg.legend(handles=handles, loc='center', ncol=ncol,
                    frameon=True, framealpha=1, fontsize=textsize * 0.7)
    fig_leg.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close(fig_leg)


def make_twin_plot(x, y_left, y_right, ylabel_left, ylabel_right, filename):
    fig, ax = plt.subplots(figsize=(7, 5))

    add_constraint_lines(ax, XLIM)

    ax.plot(x, y_left, color=COLOR_LEFT, linewidth=2, marker='o', markersize=8, zorder=5)
    ax.set_xlabel(r"$\theta_{SC}$ [°]", fontsize=textsize)
    ax.set_ylabel(ylabel_left, fontsize=textsize, color=COLOR_LEFT)
    ax.set_xlim(*XLIM)
    style_axis(ax, COLOR_LEFT)
    ax.grid(True, color='lightgrey')

    ax_r = ax.twinx()
    ax_r.plot(x, y_right, color=COLOR_RIGHT, linewidth=2, linestyle='--',
              marker='^', markersize=9, markerfacecolor='none', markeredgewidth=1.5, zorder=5)
    ax_r.set_ylabel(ylabel_right, fontsize=textsize, color=COLOR_RIGHT)
    ax_r.set_xlim(*XLIM)
    style_axis(ax_r, COLOR_RIGHT)

    fig.tight_layout(pad=0.5)
    fig.savefig(f"{output_dir}/{filename}", dpi=300)
    return fig


make_legend_figure("constraint_legend.pdf")

fig1 = make_twin_plot(
    phi_sc, thermal_eff, T35,
    r"$\eta_{th}$ [%]", r"$T_{35}$ [K]",
    "thermal_eff_T35_vs_phi_sc.pdf",
)

fig2 = make_twin_plot(
    phi_sc, core_spec_power, specific_nox,
    r"$\dot{W}_{\mathrm{core},V_d}$ [kW/litre]", r"Thrust specific $\mathrm{NO_x}$ [mg/Ns]",
    "core_spec_power_nox_vs_phi_sc.pdf",
)

plt.show()
