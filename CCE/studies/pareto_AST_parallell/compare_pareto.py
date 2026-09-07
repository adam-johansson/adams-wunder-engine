import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- From your LaTeX preamble: \documentclass[]{article} (10pt default), geometry total={6in,8in} ---
TEXTWIDTH_PT = 433.62  # 6in * 72.27 pt/in
BASE_FONTSIZE = 12
FRACTION = 0.7         # figure will span 0.7\textwidth in the paper
RATIO = 0.75           # height/width; golden ratio (~0.62) was too flat, raise for taller


def set_size(width_pt, fraction=1.0, ratio=(5 ** 0.5 - 1) / 2):
    """Convert a LaTeX textwidth (in pt) to a matplotlib figsize (in inches)."""
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * ratio
    return (fig_width_in, fig_height_in)


# Make matplotlib's math/text rendering look like LaTeX's Computer Modern,
# and size everything relative to the paper's base font size.
plt.rcParams.update({
    "text.usetex": False,       # set True if a LaTeX installation is available for exact rendering
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["cmr10", "Computer Modern Roman"],
    "axes.unicode_minus": False,
    "font.size": BASE_FONTSIZE - 1,
    "axes.labelsize": BASE_FONTSIZE - 1,
    "xtick.labelsize": BASE_FONTSIZE - 2,
    "ytick.labelsize": BASE_FONTSIZE - 2,
    "legend.fontsize": BASE_FONTSIZE - 2,
    "legend.title_fontsize": BASE_FONTSIZE - 2,
})

seeds = [13, 15, 19]
labels = {
    13: "Default limits",
    15: r"$p_{max} < 200$ bar",
    19: r"$T_{34} < 1350$ K",
}
markers = {13: 'o', 15: 's', 19: '^'}
colors = {13: 'black', 15: 'tab:red', 19: 'tab:blue'}


def load_pareto(seed):
    df = pd.read_csv(f"optimisation_data/seed_{seed}/pareto_solutions.csv")
    df = df.sort_values('eta_th')
    x = df['eta_th'].values * 100
    y = df['specific_nox'].values
    return x, y


fig, ax = plt.subplots(figsize=set_size(TEXTWIDTH_PT, FRACTION, RATIO))

for seed in seeds:
    x, y = load_pareto(seed)
    ax.scatter(
        x, y,
        marker=markers[seed],
        color=colors[seed],
        s=18,
        linewidths=0.6,
        label=labels[seed],
        zorder=5,
    )

ax.set_xlabel(r'$\eta_{th}$ [%]')
ax.set_ylabel("Thrust specific NOx [mg/Ns]")
ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
ax.grid(True, color='lightgrey', linewidth=0.5)
fig.tight_layout(pad=0.3)
fig.savefig("pareto_fronts_comparison.pdf")
plt.show()