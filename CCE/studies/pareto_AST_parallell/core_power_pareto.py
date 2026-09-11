import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

textsize = 28
output_dir = "optimisation_data/seed_21"  # where the combined plot gets saved

colors = {21: "red", 31: "yellow", 33: "blue", 35: "green"}  
markers = {21: "square", 31: "cross", 33: "circle", 35: "diamond"} 
labels = {
    21: "Ẇ<sub>core,V<sub>d</sub></sub> unconstrained",
    31: "Ẇ<sub>core,V<sub>d</sub></sub> > 70 kW/litre",
    #32: "Ẇ<sub>core,V<sub>d</sub></sub> > 80 kW/litre",
    33: "Ẇ<sub>core,V<sub>d</sub></sub> > 90 kW/litre",
    35: "Ẇ<sub>core,V<sub>d</sub></sub> > 110 kW/litre",
}

fig1 = go.Figure()

for seed in [21, 31, 33, 35]:
    seed_dir = f"optimisation_data/seed_{seed}"
    pareto_df = pd.read_csv(f"{seed_dir}/pareto_solutions.csv")
    pareto_sorted = pareto_df.sort_values('eta_th')

    fig1.add_trace(go.Scatter(
        x=pareto_sorted['eta_th'] * 100,
        y=pareto_sorted['specific_nox'],
        mode='markers+lines',
        marker=dict(symbol=markers[seed], size=14, color=colors[seed], line=dict(width=1, color='black')),
        line=dict(color=colors[seed], width=2),
        name=labels[seed],
    ))

fig1.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Times New Roman", size=textsize, color="black"),
    showlegend=True,
    legend=dict(
        font=dict(size=textsize, family="Times New Roman"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
    ),
    xaxis=dict(
        range=[48.5, 57.0],
        title=dict(text="η<sub>th</sub> [%]", font=dict(size=textsize, family="Times New Roman")),
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True, tickfont=dict(size=textsize),
    ),
    yaxis=dict(
        range=[0.0, 1.6],
        side="right",
        title=dict(text=""),          # remove title
        showticklabels=False,         # remove tick labels
        showline=True, linecolor="black", linewidth=2,
        mirror="allticks", ticks="outside", tickcolor="black",
        gridcolor="lightgrey", showgrid=True,
    ),
)

fig1.update_layout(
    xaxis=dict(domain=[0.0, 0.82]),
    yaxis=dict(domain=[0.0, 1.0]),
)

fig1.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
)

fig1.write_image(f"{output_dir}/pareto_fronts_core_power_limit.png", width=600, height=800, scale=3)
fig1.show()