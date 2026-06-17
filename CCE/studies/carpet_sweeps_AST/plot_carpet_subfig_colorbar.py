import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

data = np.loadtxt("./results/OPR_T4/thermal_eff.dat")

OPR = data[1:,0]
T4 = data[0,1:]

eff = data[1:,1:]
nox = np.loadtxt("./results/OPR_T4/specific_nox.dat")[1:,1:]
power = np.loadtxt("./results/OPR_T4/core_spec_power.dat")[1:,1:]
pmax = np.loadtxt("./results/OPR_T4/peak_pressure.dat")[1:,1:]
bore = np.loadtxt("./results/OPR_T4/bore.dat")[1:,1:]
Tout = np.loadtxt("./results/OPR_T4/Tout_piston.dat")[1:,1:]

# detta hade nog funkat
failed_mask = (eff == 0.0)

eff[eff == 0] = np.nan
nox[nox == 0] = np.nan
power[power == 0] = np.nan
pmax[pmax == 0] = np.nan
bore[bore == 0] = np.nan


# convert from g/s to mg/Ns NOX
#F = 30.693*1e3
#nox = nox*1e3 / F


textsize1 = 32
textsize2 = 36

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
    size=textsize1
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
        start=48,
        end=54,
        size=0.5,
        labelformat=".0f%%",   # <-- adds % symbol after the value
    ),
    #line=dict(color="black"),
    showscale=True,
    line = dict(
    width = 4,
    smoothing = 0
    ),
    colorbar=dict(
        orientation="h",        # horizontal colorbar
        len=0.7,  
        #len=0.4,
        #x=0.75,        # horizontal position (0=left edge, 1=right edge of plot)
        x=0.0,
        y=0.5,        # vertical position
        xanchor="left",
        xref="paper",
        yref="paper",
        bgcolor="white",      # white background so it doesn't blend with contours
        bordercolor="black",
        borderwidth=1,
        thickness=20,         # thinner colorbar to save space
        title=dict(
            text="η<sub>th</sub> (%)",
            side="right",
            font=dict(size=40, family="Times New Roman"),
        ),
    )
))


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





fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    autosize=False,
    width=550,
    height=650,
    font=dict(family="Times New Roman", size=28),
    margin=dict(l=80, r=20, t=20, b=80),  # generous margins for saved file
    xaxis=dict(
        range=[51, 109],
        gridcolor="lightgrey",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        tickfont=dict(size=textsize2),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror="allticks", 
        ticks="outside",
    ),
    yaxis=dict(
        range=[0.62, 1.19],
        gridcolor="lightgrey",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        title=dict(
            text="Thrust specific NO<sub>x</sub> [mg/Ns]",
            font=dict(size=textsize2, family="Times New Roman",color="black"),
        ),
        tickfont=dict(size=textsize2),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=False, 
        ticks="outside",
    ),
)

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
        size=32,
        color="black",
        line=dict(width=3, color="black"),
    ),
    name="No convergence",
    showlegend=False,
))

# add legend
fig.update_layout(
    legend=dict(
        x=0.001,           # position inside plot
        y=0.995,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=40, family="Times New Roman", color="black"),
    )
)

#fig.update_layout(
#    margin=dict(l=0, r=0, t=0, b=90, pad=10),
#)
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0, pad=10),
)

fig.update_layout(
    font=dict(family="Times New Roman", size=40, color="black"),
)

fig.write_image("legend_plot.pdf", width=2000, height=800, scale=2)

fig.show()