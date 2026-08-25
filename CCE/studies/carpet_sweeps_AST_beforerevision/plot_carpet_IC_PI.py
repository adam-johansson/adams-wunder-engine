import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

data = np.loadtxt("./results/intercooling_PI/thermal_eff.dat")

OPR = data[1:,0]
T4 = data[0,1:]

eff = data[1:,1:]
nox = np.loadtxt("./results/intercooling_PI/specific_nox.dat")[1:,1:]
power = np.loadtxt("./results/intercooling_PI/core_spec_power.dat")[1:,1:]
pmax = np.loadtxt("./results/intercooling_PI/peak_pressure.dat")[1:,1:]
bore = np.loadtxt("./results/intercooling_PI/bore.dat")[1:,1:]
Tout = np.loadtxt("./results/intercooling_PI/Tout_piston.dat")[1:,1:]

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
    showticklabels='none',
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
    showticklabels='none',
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
        start=52,
        end=55,
        size=1.0,
        #labelformat=".0f",   # <-- adds % symbol after the value

    ),
    #colorscale="Viridis",
    #line=dict(color="black"),
    showscale=True,
    line = dict(
    width = 2,
    smoothing = 0
    ),
    colorbar=dict(
        len=0.4,
        x=0.1,        # horizontal position (0=left edge, 1=right edge of plot)
        y=0.25,        # vertical position
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
        [0, "rgba(0,0,0,0)"],        # below 150: transparent
        [0.01, "rgba(0,0,0,0)"],     # keep transparent up to threshold
        [0.01, "rgba(128,128,128,1.0)"], # above 150: shaded red (adjust alpha as needed)
        [1, "rgba(128,128,128,1.0)"],
    ],
    showscale=False,
    line=dict(
        width=5,
        color="black",
        dash="dot",        # dotted boundary line
    ),
))

"""
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
"""

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
    x=105,        # negative x puts it outside the plot to the left
    y=0.88,
    #xref="paper",
    #yref="paper",
    text="IC ratio [-]",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", weight=700, color="black"),
    textangle=55,  # rotate to follow axis
)

fig.add_annotation(
    x=77,        # negative x puts it outside the plot to the left
    y=1.05,
    #xref="paper",
    #yref="paper",
    text="PE pressure ratio [-]",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", weight=700, color="black"),
    textangle=45,  # rotate to follow axis
)

# PE ratio labels
fig.add_annotation(
    x=85,        # negative x puts it outside the plot to the left
    y=0.97,
    #xref="paper",
    #yref="paper",
    text="0.9",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", color="black"),
    textangle=40,  # rotate to follow axis
)


fig.add_annotation(
    x=72,        # negative x puts it outside the plot to the left
    y=1.03,
    #xref="paper",
    #yref="paper",
    text="1.2",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", color="black"),
    textangle=45,  # rotate to follow axis
)

fig.add_annotation(
    x=60,        # negative x puts it outside the plot to the left
    y=1.1,
    #xref="paper",
    #yref="paper",
    text="1.5",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", color="black"),
    textangle=45,  # rotate to follow axis
)


# IC RATIO LABELS
fig.add_annotation(
    x=87,        # negative x puts it outside the plot to the left
    y=0.95,
    #xref="paper",
    #yref="paper",
    text="0",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", color="black"),
    textangle=55,  # rotate to follow axis
)


fig.add_annotation(
    x=92,        # negative x puts it outside the plot to the left
    y=0.91,
    #xref="paper",
    #yref="paper",
    text="0.25",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", color="black"),
    textangle=55,  # rotate to follow axis
)

fig.add_annotation(
    x=102,        # negative x puts it outside the plot to the left
    y=0.85,
    #xref="paper",
    #yref="paper",
    text="0.5",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", color="black"),
    textangle=55,  # rotate to follow axis
)

fig.add_annotation(
    x=110,        # negative x puts it outside the plot to the left
    y=0.8,
    #xref="paper",
    #yref="paper",
    text="0.75",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", color="black"),
    textangle=55,  # rotate to follow axis
)

fig.add_annotation(
    x=117,        # negative x puts it outside the plot to the left
    y=0.76,
    #xref="paper",
    #yref="paper",
    text="1.0",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", color="black"),
    textangle=55,  # rotate to follow axis
)



fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    autosize=False,
    width=950,
    height=550,
    font=dict(family="Times New Roman", size=28),
    margin=dict(l=80, r=20, t=20, b=80),  # generous margins for saved file
    xaxis=dict(
        range=[54, 119],
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
    range=[0.9, 1.41],
    gridcolor="lightgrey",
    gridwidth=1,
    showgrid=True,
    zeroline=False,
    tickformat=".1f",       # one decimal place
    dtick=0.1,              # gridline every 0.1, so showing every other means 0.2 below
    title=dict(
        text="Thrust specific NO<sub>x</sub> [mg/Ns]",
        font=dict(size=28, family="Times New Roman", color="black"),
    ),
    tickfont=dict(size=28),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror="allticks", 
        ticks="outside",
    ),
)

# add bore lim text
fig.add_annotation(
    x=65,      # arrowhead x (pointing to the shaded region)
    y=1.03,     # arrowhead y
    ax=100,      # text box x (in data coordinates)
    ay=1.04,    # text box y (in data coordinates)
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

#fig.update_layout(
#    margin=dict(l=120, r=40, t=40, b=120, pad=10),
#)

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=90, pad=10),
)

fig.update_layout(
    font=dict(family="Times New Roman", size=28, color="black"),
)

fig.write_image("IC_PI_plot.pdf", width=550, height=650, scale=2)
fig.show()