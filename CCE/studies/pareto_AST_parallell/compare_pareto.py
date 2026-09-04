import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


seed1 = 11
seed2 = 2
seed3 = 13

output_dir1 = f"optimisation_data/seed_{seed1}"
output_dir2 = f"optimisation_data/seed_{seed2}"
output_dir3 = f"optimisation_data/seed_{seed3}"

pareto_df1 = pd.read_csv(f"{output_dir1}/pareto_solutions.csv")
pareto_df2 = pd.read_csv(f"{output_dir2}/pareto_solutions.csv")
pareto_df3 = pd.read_csv(f"{output_dir3}/pareto_solutions.csv")
hv_df1 = pd.read_csv(f"{output_dir1}/hypervolume.csv")
hv_df2 = pd.read_csv(f"{output_dir2}/hypervolume.csv")
hv_df3 = pd.read_csv(f"{output_dir3}/hypervolume.csv")



# --- Pareto front plot ---
pareto_sorted1 = pareto_df1.sort_values('eta_th')
pareto_sorted2 = pareto_df2.sort_values('eta_th')
pareto_sorted3 = pareto_df3.sort_values('eta_th')

fig1 = go.Figure()




textsize = 18

# pareto front
fig1.add_trace(go.Scatter(
    x=pareto_sorted1['eta_th'] * 100,
    y=pareto_sorted1['specific_nox'],
    mode='markers+lines',
    marker=dict(symbol='x', size=6, color='black', line=dict(width=1, color='black')),
    line=dict(color='black', width=2),
    name='Pareto front 1',
))

# pareto front
fig1.add_trace(go.Scatter(
    x=pareto_sorted2['eta_th'] * 100,
    y=pareto_sorted2['specific_nox'],
    mode='markers+lines',
    marker=dict(symbol='x', size=6, color='red', line=dict(width=1, color='black')),
    line=dict(color='red', width=2),
    name='Pareto front 2',
))
# pareto front
fig1.add_trace(go.Scatter(
    x=pareto_sorted3['eta_th'] * 100,
    y=pareto_sorted3['specific_nox'],
    mode='markers+lines',
    marker=dict(symbol='x', size=6, color='blue', line=dict(width=1, color='black')),
    line=dict(color='blue', width=2),
    name='Pareto front 3',
))

# Reference point
fig1.add_trace(go.Scatter(
    x=[49.4],
    y=[0.188],
    mode='markers',
    marker=dict(symbol='cross', size=16, color='blue', line=dict(width=1, color='black')),
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
fig1.write_image(f"pareto_fronts_convergence.pdf", scale=2)
fig1.show()

# --- Hypervolume plot ---
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=hv_df1['generation'],
    y=hv_df1['hypervolume'],
    mode='lines',
    line=dict(color='black', width=2),
    marker=dict(size=8, color='black'),
    name='Seed 1',
))

fig2.add_trace(go.Scatter(
    x=hv_df2['generation'],
    y=hv_df2['hypervolume'],
    mode='lines',
    line=dict(color='red', width=2),
    marker=dict(size=8, color='black'),
    name='Seed 2',
))

fig2.add_trace(go.Scatter(
    x=hv_df3['generation'],
    y=hv_df3['hypervolume'],
    mode='lines',
    line=dict(color='blue', width=2),
    marker=dict(size=8, color='black'),
    name='Seed 3',
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
fig2.write_image(f"hypervolume_convergence.pdf", scale=2)
fig2.show()