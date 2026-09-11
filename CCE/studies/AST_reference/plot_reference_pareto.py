import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

output_dir = "."

all_df = pd.read_csv(f"{output_dir}/all_evaluations.csv")
pareto_df = pd.read_csv(f"{output_dir}/pareto_solutions.csv")
hv_df = pd.read_csv(f"{output_dir}/hypervolume.csv")

pareto_sorted = pareto_df.sort_values('eta_th')

# --- Pareto front plot ---
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=all_df['eta_th'] * 100,
    y=all_df['specific_nox'],
    mode='markers',
    marker=dict(size=6, color='lightgrey'),
    name='All evaluations',
))

fig1.add_trace(go.Scatter(
    x=pareto_sorted['eta_th'] * 100,
    y=pareto_sorted['specific_nox'],
    mode='markers+lines',
    marker=dict(symbol='square', size=10, color='red', line=dict(width=1, color='black')),
    line=dict(color='red', width=2),
    name='Pareto front',
))

fig1.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(title="η_th [%]", showline=True, linecolor="black", gridcolor="lightgrey"),
    yaxis=dict(title="Thrust specific NOx [mg/Ns]", showline=True, linecolor="black", gridcolor="lightgrey"),
)

fig1.write_image(f"{output_dir}/pareto_plot.png", scale=4)
fig1.show()
