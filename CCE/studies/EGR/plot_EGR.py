import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

data = np.loadtxt("./results/EGR/thermal_eff.dat")

OPR = data[1:,0]
T4 = data[0,1:]

eff = data[1:,1:]
nox = np.loadtxt("./results/EGR/specific_nox.dat")[1:,1:]
power = np.loadtxt("./results/EGR/core_spec_power.dat")[1:,1:]
pmax = np.loadtxt("./results/EGR/peak_pressure.dat")[1:,1:]
bore = np.loadtxt("./results/EGR/bore.dat")[1:,1:]
Tout = np.loadtxt("./results/EGR/Tout_piston.dat")[1:,1:]

failed_mask = (eff == 0.0)

eff[eff == 0] = np.nan
nox[nox == 0] = np.nan
power[power == 0] = np.nan
pmax[pmax == 0] = np.nan
bore[bore == 0] = np.nan


# convert from g/s to mg/Ns NOX
#F = 30.693*1e3
#nox = nox*1e3 / F

fig = go.Figure()

# Carpet grid
fig.add_trace(go.Carpet(
    a=T4,
    b=OPR,
    x=power,
    y=nox,
    aaxis = dict(
    #ticksuffix = 'K',
    smoothing = 1,
    minorgridcolor = 'black',
    gridcolor = 'black',
    color = 'black',
    title=dict(text=""), 
    #title=dict(
    #            text="T4 (K)",
    #            font=dict(
    #                size=36,
    #                family="Times New Roman",
    #                color="black",
    #                weight=1000,
    #            )
    #        ),
    tickfont=dict(
    size=28
    )
    ),
    baxis = dict(
    smoothing = 1,
    #minorgridcount = 0.5,
    #tickmode='array',
    #tickvals=OPR[::2],                          # every second OPR value
    #icktext=[str(int(v)) for v in OPR[::2]],  # corresponding labels
    minorgridcolor = 'black',
    gridcolor = 'black',
    color = 'black',
    title=dict(text=""), 
    #title=dict(
    #            text="OPR (-)",
    #            font=dict(
    #                size=36,
    #                family="Times New Roman",
    #                color="black",
    #                weight=1000,
    #            )
    #        ),
    tickfont=dict(
    size=28
    )
)))

# Efficiency contours
fig.add_trace(go.Contourcarpet(
    a=T4,
    b=OPR,
    z=eff,
    contours=dict(
        #coloring="lines",
        showlabels=False,
        #start=48.5,
        #end=53.5,
        #size=0.5,
        #labelformat=".0f%%",   # <-- adds % symbol after the value

    ),
    #line=dict(color="black"),
    showscale=True,
    line = dict(
    width = 2,
    smoothing = 0
    ),
    colorbar=dict(
        #len=0.4,
        #x=0.75,        # horizontal position (0=left edge, 1=right edge of plot)
        #y=0.80,        # vertical position
        xref="paper",
        yref="paper",
        bgcolor="white",      # white background so it doesn't blend with contours
        bordercolor="black",
        borderwidth=1,
        thickness=15,         # thinner colorbar to save space
        title=dict(
            text="η<sub>th</sub> (%)",
            side="top",
            font=dict(size=26, family="Times New Roman"),
        ),
    )
))

# ── helper ────────────────────────────────────────────────────────────────────
def limit_line(a, b, z, threshold, color="black", dash="dot", width=3):
    """Returns (fill_trace, line_trace) for a carpet limit."""
    # Transparent fill — keeps the contour machinery but hides the shading
    fill = go.Contourcarpet(
        a=a, b=b, z=z,
        contours=dict(
            start=threshold,
            end=threshold * 10,   # well above any real value
            size=threshold * 10,
            coloring="fill",
            showlabels=False,
        ),
        colorscale=[
            [0,    "rgba(0,0,0,0)"],
            [1,    "rgba(0,0,0,0)"],   # fully transparent everywhere
        ],
        showscale=False,
        line=dict(width=0),            # hide the line on this trace
    )

    # Line-only trace at exactly the threshold
    line = go.Contourcarpet(
        a=a, b=b, z=z,
        contours=dict(
            start=threshold,
            end=threshold,
            size=0,                    # single contour level
            coloring="none",           # no fill at all
            showlabels=False,
        ),
        showscale=False,
        line=dict(
            width=width,
            color=color,
            dash=dash,
        ),
    )
    return fill, line


# ── bore limit ────────────────────────────────────────────────────────────────
bore_fill, bore_line = limit_line(T4, OPR, bore, threshold=200)
fig.add_trace(bore_fill)
fig.add_trace(bore_line)

# ── Tout limit ────────────────────────────────────────────────────────────────
tout_fill, tout_line = limit_line(T4, OPR, Tout, threshold=1250)
fig.add_trace(tout_fill)
fig.add_trace(tout_line)

# ── pmax limit ────────────────────────────────────────────────────────────────
pmax_fill, pmax_line = limit_line(T4, OPR, pmax, threshold=150)
fig.add_trace(pmax_fill)
fig.add_trace(pmax_line)

"""
# add bore lim text
fig.add_annotation(
    x=60,      # arrowhead x (pointing to the shaded region)
    y=1.15,     # arrowhead y
    ax=100,      # text box x (in data coordinates)
    ay=1.2,    # text box y (in data coordinates)
    text="d > 200 mm",
    showarrow=True,
    arrowhead=2,    # arrow head style 1-8
    arrowsize=1,    # relative size of arrowhead
    arrowwidth=2,   # line width
    arrowcolor="black",
    axref="x",      # ax is in data coordinates
    ayref="y",      # ay is in data coordinates
    font=dict(size=28, family="Times New Roman", color="black"),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
)

# add pmax text
fig.add_annotation(
    x=130,      # arrowhead x (pointing to the shaded region)
    y=0.8,     # arrowhead y
    ax=130,      # text box x (in data coordinates)
    ay=0.45,    # text box y (in data coordinates)
    text="p<sub>max</sub> > 150 bar",
    showarrow=True,
    arrowhead=2,    # arrow head style 1-8
    arrowsize=1,    # relative size of arrowhead
    arrowwidth=2,   # line width
    arrowcolor="black",
    axref="x",      # ax is in data coordinates
    ayref="y",      # ay is in data coordinates
    font=dict(size=28, family="Times New Roman", color="black"),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
)




fig.add_annotation(
    x=57,        # negative x puts it outside the plot to the left
    y=0.57,
    #xref="paper",
    #yref="paper",
    text="Λ [-]",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", weight=700, color="black"),
    textangle=57,  # rotate to follow axis
)

fig.add_annotation(
    x=45,        # negative x puts it outside the plot to the left
    y=1.2,
    #xref="paper",
    #yref="paper",
    text="OPR [-]",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", weight=700, color="black"),
    textangle=-70,  # rotate to follow axis
)
"""
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    autosize=False,
    width=550,
    height=650,
    font=dict(family="Times New Roman", size=28),
    margin=dict(l=80, r=20, t=20, b=80),  # generous margins for saved file
    xaxis=dict(
        #pythrange=[37, 150],
        gridcolor="lightgrey",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        title=dict(
            text="Ẇ<sub>core,V<sub>d</sub></sub> [kW/litre]",
            font=dict(size=28, family="Times New Roman",color="black"),
        ),
        tickfont=dict(size=28),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror="allticks", 
        ticks="outside",
    ),
    yaxis=dict(
        #range=[0.401, 1.27],
        gridcolor="lightgrey",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        title=dict(
            text="Thrust specific NO<sub>x</sub> [mg/Ns]",
            font=dict(size=28, family="Times New Roman",color="black"),
        ),
        tickfont=dict(size=28),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror="allticks", 
        ticks="outside",
    ),
)

"""
# Then after building the figure, add the crosses
T4_grid, OPR_grid = np.meshgrid(T4, OPR)
failed_a = T4_grid[failed_mask]
failed_b = OPR_grid[failed_mask]


fig.add_trace(go.Scattercarpet(
    a=failed_a,
    b=failed_b,
    mode="markers",
    marker=dict(
        symbol="x",
        size=16,
        color="black",
        line=dict(width=2, color="black"),
    ),
    name="No convergence",
    showlegend=False,
))
"""
"""
fig.add_annotation(
    x=0.75,
    y=0.10,
    xref="paper",
    yref="paper",
    text="✕  T35 > T4",
    showarrow=False,
    font=dict(size=24, family="Times New Roman", color="black", weight=400),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=6,
    align="left",
    xanchor="right",
    yanchor="bottom",
)
"""

"""
fig.add_trace(go.Scatter(
    x=[80.40],
    y=[1.10],
    mode="markers",
    marker=dict(
        symbol="cross",
        size=16,
        color="black",
        line=dict(width=3, color="black"),
    ),
    name="Baseline",
    showlegend=True,
))

# add legend
fig.update_layout(
    legend=dict(
        x=0.025,           # position inside plot
        y=0.975,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=28, family="Times New Roman", color="black"),
    )
)
"""


fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=90, pad=10),
)

fig.update_layout(
    font=dict(family="Times New Roman", size=28, color="black"),
)

fig.write_image("EGR_plot.pdf", width=550, height=850, scale=2)
fig.show()