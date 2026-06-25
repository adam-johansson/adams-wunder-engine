import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


seed = 1
output_dir = f"optimisation_data/seed_{seed}"

all_df = pd.read_csv(f"{output_dir}/all_evaluations.csv")
pareto_df = pd.read_csv(f"{output_dir}/pareto_solutions.csv")
hv_df = pd.read_csv(f"{output_dir}/hypervolume.csv")

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


textsize = 18

# Feasible points coloured by core power per litre

fig1.add_trace(go.Scatter(
    x=feasible['eta_th'] * 100,
    y=feasible['specific_nox'],
    mode='markers',
    marker=dict(
        size=10,
        color=feasible['piston_fuelsplit'],
        #color=feasible['core_power_per_litre'],
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

# Reference point
fig1.add_trace(go.Scatter(
    x=[49.4],
    y=[0.188],
    mode='markers',
    marker=dict(symbol='cross', size=16, color='blue', line=dict(width=1, color='black')),
    name='Reference',
))

# high efficiency point
fig1.add_annotation(
    x=55.90,
    y=1.086,
    ax=80,   # adjust these to position the text box
    ay=-80,
    text=(
        "OPR = 17.53<br>"
        "T<sub>4</sub> = 1035 K<br>"
        "Λ = 0.243<br>"
        "Π<sub>pe</sub> = 1.58<br>"
        "ϵ = 10.48<br>"
        "<i>f</i> = 3.90%<br>"
        "IC = 0.987"
        #"Ẇ<sub>core,V<sub>d</sub></sub> = 65.39kW/litre"
    ),
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="black",
    font=dict(size=textsize, family="Times New Roman", color="black"),
    align="left",
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
)

# low nox point
fig1.add_annotation(
    x=40.78,
    y=0.138,
    ax=90,   # adjust these to position the text box
    ay=-160,
    text=(
        "OPR = 10.0<br>"
        "T<sub>4</sub> = 1600 K<br>"
        "Λ = 0.50<br>"
        "Π<sub>pe</sub> = 0.98<br>"
        "ϵ = 4.14<br>"
        "<i>f</i> = 2.00%<br>"
        "IC = 0.996"
        #"Ẇ<sub>core,V<sub>d</sub></sub> = 63.43kW/litre"
    ),
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="black",
    font=dict(size=textsize, family="Times New Roman", color="black"),
    align="left",
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
)


# middle point
fig1.add_annotation(
    x=52.98,
    y=0.618,
    ax=120,   # adjust these to position the text box
    ay=+40,
    text=(
        "OPR = 17.11<br>"
        "T<sub>4</sub> = 1210 K<br>"
        "Λ = 0.367<br>"
        "Π<sub>pe</sub> = 1.58<br>"
        "ϵ = 11.32<br>"
        "<i>f</i> = 2.78%<br>"
        "IC = 0.994"
        #"Ẇ<sub>core,V<sub>d</sub></sub> = 69.46kW/litre"
    ),
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="black",
    font=dict(size=textsize, family="Times New Roman", color="black"),
    align="left",
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
)


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
fig1.write_image(f"{output_dir}/pareto_plot_EGR.pdf", scale=2)
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
    font=dict(family="Times New Roman", size=20, color="black"),
    xaxis=dict(
        title=dict(text="Generation [-]", font=dict(size=20, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=20),
    ),
    yaxis=dict(
        title=dict(text="Hypervolume [-]", font=dict(size=20, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=20),
    ),
)
fig2.write_image(f"{output_dir}/hypervolume_EGR.pdf", scale=2)
fig2.show()