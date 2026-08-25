import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

data = np.loadtxt("./results/OPR_PI/thermal_eff.dat")

OPR = data[1:,0]
T4 = data[0,1:]

eff = data[1:,1:]
nox = np.loadtxt("./results/OPR_PI/specific_nox.dat")[1:,1:]
power = np.loadtxt("./results/OPR_PI/core_spec_power.dat")[1:,1:]
pmax = np.loadtxt("./results/OPR_PI/peak_pressure.dat")[1:,1:]
bore = np.loadtxt("./results/OPR_PI/bore.dat")[1:,1:]
Tout = np.loadtxt("./results/OPR_PI/Tout_piston.dat")[1:,1:]

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
        start=51.5,
        end=54.5,
        size=0.5,
        labelformat=".0f%%",   # <-- adds % symbol after the value

    ),
    #line=dict(color="black"),
    showscale=True,
    line = dict(
    width = 2,
    smoothing = 0
    ),
    colorbar=dict(
        len=0.4,
        x=0.05,        # horizontal position (0=left edge, 1=right edge of plot)
        y=0.2,        # vertical position
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


#bore lim
fig.add_trace(go.Contourcarpet(
    a=T4,
    b=OPR,
    z=bore,
    contours=dict(
        start=200,
        end=300,
        size=200,
        coloring="fill",
        showlabels=False,
    ),
    colorscale=[
        [0, "rgba(0,0,0,0)"],
        [0.01, "rgba(0,0,0,0)"],
        [0.01, "rgba(128,128,128,1.0)"],  # pink, semi-transparent
        [1, "rgba(128,128,128,1.0)"],
    ],
    showscale=False,
    line=dict(
        width=3,
        color="black",
        dash="dot",    
    ),
))

#Tout lim
fig.add_trace(go.Contourcarpet(
    a=T4,
    b=OPR,
    z=Tout,
    contours=dict(
        start=1250,
        end=2000,        # set this above your actual pmax maximum
        size=2000,       # large size so only one band is drawn
        coloring="fill",
        showlabels=False,
    ),
    colorscale=[
        [0, "rgba(0,0,0,0)"],        # below 150: transparent
        [0.01, "rgba(0,0,0,0)"],     # keep transparent up to threshold
        [0.01, "rgba(128,128,128,1.0)"], # above 150: shaded red (adjust alpha as needed)
        [1, "rgba(128,128,128,1.0)"],
    ],
    showscale=False,
    # Keep the boundary line
    line=dict(
        width=3,
        color="black",
        dash="dot",
    ),
))

#pmax lim
fig.add_trace(go.Contourcarpet(
    a=T4,
    b=OPR,
    z=pmax,
    contours=dict(
        start=150,
        end=300,        # set this above your actual pmax maximum
        size=200,       # large size so only one band is drawn
        coloring="fill",
        showlabels=False,
    ),
    colorscale=[
        [0, "rgba(0,0,0,0)"],        # below 150: transparent
        [0.01, "rgba(0,0,0,0)"],     # keep transparent up to threshold
        [0.01, "rgba(128,128,128,1.0)"], # above 150: shaded red (adjust alpha as needed)
        [1, "rgba(128,128,128,1.0)"],
    ],
    showscale=False,
    # Keep the boundary line
    line=dict(
        width=3,
        color="black",
        dash="dot",
    ),
))

"""
# bore lim text
fig.add_annotation(
    x=55,  # specifioc power
    y=1.20,  # NOx
    text="d > 200 mm",
    showarrow=False,
    font=dict(
        size=28,
        family="Times New Roman",
        color="black",
    ),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    opacity=1.0,
)
"""

# add bore lim text
fig.add_annotation(
    x=58,      # arrowhead x (pointing to the shaded region)
    y=1.07,     # arrowhead y
    ax=72,      # text box x (in data coordinates)
    ay=0.88,    # text box y (in data coordinates)
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


# add T out lim text
fig.add_annotation(
    x=65,      # arrowhead x (pointing to the shaded region)
    y=1.155,     # arrowhead y
    ax=85,      # text box x (in data coordinates)
    ay=1.17,    # text box y (in data coordinates)
    text="T<sub>34</sub> > 1250 K",
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
    x=85,  # specifioc power
    y=1.125,  # NOx
    text="p<sub>max</sub> > 150 bar",
    showarrow=False,
    font=dict(
        size=28,
        family="Times New Roman",
        color="black",
    ),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    opacity=1.0,
)

#add thermal efficiency text
"""
fig.add_annotation(
    x=90,  # position near one of the efficiency contours
    y=0.9,
    text="η<sub>th</sub> [%]",
    showarrow=False,
    font=dict(
        size=24, 
        family="Times New Roman", 
        color="black"
    ),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    opacity=1.0,
)
"""

fig.add_annotation(
    x=91,        # negative x puts it outside the plot to the left
    y=0.98,
    #xref="paper",
    #yref="paper",
    text="OPR [-]",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", weight=700, color="black"),
    textangle=-75,  # rotate to follow axis
)

fig.add_annotation(
    x=55,        # negative x puts it outside the plot to the left
    y=1.025,
    #xref="paper",
    #yref="paper",
    text="Π<sub>p</sub> [-]",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", weight=700, color="black"),
    textangle=50,  # rotate to follow axis
)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    autosize=False,
    width=550,
    height=650,
    font=dict(family="Times New Roman", size=28),
    margin=dict(l=80, r=20, t=20, b=80),  # generous margins for saved file
    xaxis=dict(
        range=[50, 95],
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
        range=[0.87, 1.19],
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



fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=90, pad=10),
)

fig.update_layout(
    font=dict(family="Times New Roman", size=28, color="black"),
)

fig.write_image("OPR_PI_plot.pdf", width=550, height=850, scale=2)
fig.show()