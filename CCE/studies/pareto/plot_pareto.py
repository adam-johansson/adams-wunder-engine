import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

all_df = pd.read_csv("optimisation_data/all_evaluations.csv")
pareto_df = pd.read_csv("optimisation_data/pareto_solutions.csv")
hv_df = pd.read_csv("optimisation_data/hypervolume.csv")

# --- Pareto front plot ---
feasible = all_df[all_df['is_feasible']]
infeasible = all_df[~all_df['is_feasible'] & (all_df['eta_th'] != 0.0)]
pareto_sorted = pareto_df.sort_values('eta_th')

fig1 = go.Figure()


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
        color=feasible['core_power_per_litre'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title=dict(text="Ẇ<sub>core,Vd</sub> [kW/litre]", font=dict(size=24, family="Times New Roman")),
            thickness=15, len=0.6,
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

fig1.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=800, height=600,
    font=dict(family="Times New Roman", size=24, color="black"),
    xaxis=dict(
        title=dict(text="η<sub>th</sub> [%]", font=dict(size=24, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=24),
    ),
    yaxis=dict(
        title=dict(text="Thrust specific NO<sub>x</sub> [mg/Ns]", font=dict(size=24, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=24),
    ),
    legend=dict(font=dict(size=20, family="Times New Roman"), bordercolor="black", borderwidth=1),
)
fig1.write_image("optimisation_data/pareto_plot.pdf", scale=2)
fig1.show()

# --- Hypervolume plot ---
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=hv_df['generation'],
    y=hv_df['hypervolume'],
    mode='lines+markers',
    line=dict(color='black', width=2),
    marker=dict(size=8, color='black'),
))

fig2.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=800, height=500,
    font=dict(family="Times New Roman", size=24, color="black"),
    xaxis=dict(
        title=dict(text="Generation [-]", font=dict(size=24, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=24),
    ),
    yaxis=dict(
        title=dict(text="Hypervolume [-]", font=dict(size=24, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=24),
    ),
)
fig2.write_image("optimisation_data/hypervolume.pdf", scale=2)
fig2.show()