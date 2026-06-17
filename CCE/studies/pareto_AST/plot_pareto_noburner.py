import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


seed = 8
output_dir = f"optimisation_data/seed_{seed}"

all_df = pd.read_csv(f"{output_dir}/all_evaluations.csv")
pareto_df = pd.read_csv(f"{output_dir}/pareto_solutions.csv")
hv_df = pd.read_csv(f"{output_dir}/hypervolume.csv")

# --- Pareto front plot ---
feasible = all_df[all_df['is_feasible']]
infeasible = all_df[~all_df['is_feasible'] & (all_df['eta_th'] != 0.0)]
pareto_sorted = pareto_df.sort_values('eta_th')

fig1 = go.Figure()

markersize1=12
markersize2=14
markersize3=16

# Infeasible points
fig1.add_trace(go.Scatter(
    x=infeasible['eta_th'] * 100,
    y=infeasible['specific_nox'],
    mode='markers',
    marker=dict(symbol='x', size=markersize1, color='lightgrey', line=dict(width=1, color='lightgrey')),
    name='Infeasible',
))


textsize = 28

# Feasible points coloured by core power per litre

fig1.add_trace(go.Scatter(
    x=feasible['eta_th'] * 100,
    y=feasible['specific_nox'],
    mode='markers',
    marker=dict(
        size=markersize2,
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
            orientation="h",
            title=dict(
                text="Ẇ<sub>core,V<sub>d</sub></sub> [kW/litre]",
                #text="Λ [-]",
                font=dict(size=textsize, family="Times New Roman"),
                side="bottom",
            ),
            thickness=15,
            len=0.78,
            x=0.20,        # push inside the plot from the left
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
    marker=dict(symbol='square', size=markersize3, color='red', line=dict(width=1, color='black')),
    line=dict(color='red', width=2),
    name='Pareto front',
))

"""
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
"""

fig1.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Times New Roman", size=textsize, color="black"),
    showlegend=False,
    xaxis=dict(
        range=[54.5, 56.3],
        title=dict(text="η<sub>th</sub> [%]", font=dict(size=textsize, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=textsize),
    ),
    yaxis=dict(
        range=[0.9, 1.25],
        title=dict(text="Thrust specific NO<sub>x</sub> [mg/Ns]", font=dict(size=textsize, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=textsize),
    ),
    #legend=dict(
    #x=0.53,
    #y=0.98,
    #xanchor="left",
    #yanchor="top",
    #font=dict(size=textsize, family="Times New Roman"),
    #bordercolor="black",
    #borderwidth=1,
    #bgcolor="white",
#),
)

fig1.update_layout(
    xaxis=dict(domain=[0.18, 1.0]),  # same fraction for both
    yaxis=dict(domain=[0.0, 1.0]),
)

fig1.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    #margin=dict(l=120, r=0, t=40, b=20),
)
fig1.write_image(f"{output_dir}/pareto_plot_noburner1.pdf", width=600, height=800, scale=2)
fig1.show()

"""
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
fig2.write_image(f"{output_dir}/hypervolume.pdf", scale=2)
fig2.show()
"""