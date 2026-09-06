print(f"Starting")
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

pio.renderers.default = "browser"

print(f"hej")
seed = 21
output_dir = f"optimisation_data/seed_{seed}"

all_df = pd.read_csv(f"{output_dir}/all_evaluations.csv", delimiter=",")
pareto_df = pd.read_csv(f"{output_dir}/pareto_solutions.csv")
hv_df = pd.read_csv(f"{output_dir}/hypervolume.csv")

print(f"Data loaded")


# add pareto front for pmax = 200 bar
#pareto_df_200bar = pd.read_csv(f"optimisation_data/seed_14/pareto_solutions.csv")
#pareto_200bar_sorted = pareto_df_200bar.sort_values('eta_th')

# add pareto front for T34 = 1350 K
#pareto_df_1350K = pd.read_csv(f"optimisation_data/seed_8/pareto_solutions.csv")
#pareto_1350K_sorted = pareto_df_1350K.sort_values('eta_th')



# Filter data points with negative NOx 
all_df = all_df[(all_df['specific_nox'] > 0.0)]


# --- Pareto front plot ---
feasible = all_df[all_df['is_feasible']]
infeasible = all_df[~all_df['is_feasible'] & (all_df['eta_th'] != 0.0)]
pareto_sorted = pareto_df.sort_values('eta_th')


# Point a: lowest NOx
point_a = pareto_sorted.loc[pareto_sorted['specific_nox'].idxmin()]

# Point b: point closest to NOx = 1.20 (same as reference)
point_b = pareto_sorted.iloc[(pareto_sorted['specific_nox'] - 0.188).abs().argmin()]

# Point c: highest thermal efficiency
point_c = pareto_sorted.loc[pareto_sorted['eta_th'].idxmax()]



# Add all three labelled annotations
labelled_points = [
    (point_a, "A", 90, -140),
    (point_b, "B", 60, 40),
    (point_c, "C", 40, -80),
]




fig1 = go.Figure()
textsize = 18


for point, label, ax_off, ay_off in labelled_points:
    fig1.add_annotation(
        x=point['eta_th'] * 100,
        y=point['specific_nox'],
        ax=ax_off,
        ay=ay_off,
        text=label,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black",
        font=dict(size=textsize, family="Times New Roman", color="black"),
        align="left",
        bgcolor="white",
        #bordercolor="black",
        #borderwidth=1,
    )



# Infeasible points
fig1.add_trace(go.Scatter(
    x=infeasible['eta_th'] * 100,
    y=infeasible['specific_nox'],
    mode='markers',
    marker=dict(symbol='x', size=8, color='lightgrey', line=dict(width=1, color='lightgrey')),
    name='Infeasible',
))



# Feasible points coloured by core power per litre

fig1.add_trace(go.Scatter(
    x=feasible['eta_th'] * 100,
    y=feasible['specific_nox'],
    mode='markers',
    marker=dict(
        size=10,
        #color=feasible['piston_fuelsplit'],
        color=feasible['core_power_per_litre'],
        #color=feasible['split'],
        #color=feasible['far'],
        colorscale='Viridis',
        #cmin=feasible['piston_fuelsplit'].quantile(0.05),
        #cmax=feasible['piston_fuelsplit'].quantile(0.95),
        #cmin=feasible['core_power_per_litre'].quantile(0.05),
        #cmax=feasible['core_power_per_litre'].quantile(0.95),
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Piston fuel fraction [-]",
                font=dict(size=textsize, family="Times New Roman"),
                side="right",
            ),
            thickness=15,
            len=0.7,
            x=0.02,        # push inside the plot from the left
            y=0.98,        # top of the plot
            xanchor="left",
            yanchor="top",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        ),
        line=dict(width=0.5, color='black'),
    ),
    name='Feasible',
))

# Pareto front
fig1.add_trace(go.Scatter(
    x=pareto_sorted['eta_th'] * 100,
    y=pareto_sorted['specific_nox'],
    mode='markers+lines',
    marker=dict(symbol='square', size=12, color='red', line=dict(width=1, color='black')),
    line=dict(color='red', width=2),
    name='Pareto front',
))

# Pareto front for 200 bar
#fig1.add_trace(go.Scatter(
#    x=pareto_200bar_sorted['eta_th'] * 100,
#    y=pareto_200bar_sorted['specific_nox'],
#    mode='lines',
#    line=dict(color='black', width=2),
#    name='Pareto front 200 bar',
#))

# Pareto front for 1350 K
#fig1.add_trace(go.Scatter(
#    x=pareto_1350K_sorted['eta_th'] * 100,
#    y=pareto_1350K_sorted['specific_nox'],
#    mode='lines',
#    line=dict(color='blue', width=2),
#    name='Pareto front 1350 K',
#))

# Reference point
fig1.add_trace(go.Scatter(
    x=[49.4],
    y=[0.188],
    mode='markers',
    marker=dict(symbol='cross', size=16, color='yellow', line=dict(width=1, color='black')),
    name='Reference',
))





fig1.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=800, height=600,
    font=dict(family="Times New Roman", size=textsize, color="black"),
    xaxis=dict(
        title=dict(text="η<sub>th</sub> [%]", font=dict(size=textsize, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=textsize),
        dtick=2,  # tick every 2 units
    ),
    yaxis=dict(
        title=dict(text="Thrust specific NO<sub>x</sub> [mg/Ns]", font=dict(size=textsize, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=textsize),
    ),
    legend=dict(
    x=0.25,
    y=0.98,
    xanchor="left",
    yanchor="top",
    font=dict(size=textsize, family="Times New Roman"),
    bordercolor="black",
    borderwidth=1,
    bgcolor="white",
),
)

fig1.update_layout(

    margin=dict(l=20, r=20, t=20, b=20),
)
fig1.write_image(f"{output_dir}/pareto_plot_noburner.pdf", scale=2)
fig1.show()

