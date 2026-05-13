import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

textsize2 = 36

fig_legend = go.Figure()

# Dummy trace for the colorbar (horizontal)
fig_legend.add_trace(go.Contourcarpet(
    a=[0, 1],
    b=[0, 1],
    z=[[48, 49], [53, 54]],
    contours=dict(start=48, end=54, size=1.0),
    showscale=True,
    colorbar=dict(
        orientation="h",        # horizontal colorbar
        x=1.0,
        y=0.75,
        xanchor="center",
        len=0.8,
        thickness=20,
        title=dict(
            text="η<sub>th</sub> [%]",
            side="bottom",
            font=dict(size=textsize2, family="Times New Roman"),
        ),
        tickfont=dict(size=textsize2, family="Times New Roman"),
        tickvals=[48, 49, 50, 51, 52, 53, 54],
    ),
    visible=True,
))

# Dummy trace for the baseline marker legend entry
fig_legend.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(symbol="cross", size=16, color="black", line=dict(width=3, color="black")),
    name="Baseline",
    showlegend=True,
))

fig_legend.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=550,
    height=200,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    legend=dict(
        x=0.5,
        y=0.15,
        xanchor="center",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=textsize2, family="Times New Roman", color="black"),
        orientation="h",
    ),
    font=dict(family="Times New Roman", size=textsize2, color="black"),
)

fig_legend.write_image("legend_colorbar.pdf", width=550, height=200, scale=2)
fig_legend.show()