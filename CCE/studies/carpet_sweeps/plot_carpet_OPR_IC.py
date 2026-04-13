import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

data = np.loadtxt("./results/OPR_IC/thermal_eff.dat")

OPR = data[1:,0]
T4 = data[0,1:]

eff = data[1:,1:]
nox = np.loadtxt("./results/OPR_IC/specific_nox.dat")[1:,1:]
power = np.loadtxt("./results/OPR_IC/core_spec_power.dat")[1:,1:]
pmax = np.loadtxt("./results/OPR_IC/peak_pressure.dat")[1:,1:]
bore = np.loadtxt("./results/OPR_IC/bore.dat")[1:,1:]
Tout = np.loadtxt("./results/OPR_IC/Tout_piston.dat")[1:,1:]

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
        start=49,
        end=53.5,
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
        x=0.75,        # horizontal position (0=left edge, 1=right edge of plot)
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



# add pmax text
fig.add_annotation(
    x=101,  # specifioc power
    y=1.08,  # NOx
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
    x=64,        # negative x puts it outside the plot to the left
    y=1.0,
    #xref="paper",
    #yref="paper",
    text="OPR [-]",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", weight=700, color="black"),
    textangle=-75,  # rotate to follow axis
)

fig.add_annotation(
    x=59,        # negative x puts it outside the plot to the left
    y=0.65,
    #xref="paper",
    #yref="paper",
    text="IC ratio [-]",
    showarrow=False,
    font=dict(size=28, family="Times New Roman", weight=700, color="black"),
    textangle=68,  # rotate to follow axis
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
        range=[55, 114],
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
        range=[0.57, 1.16],
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

fig.write_image("OPR_IC_plot.pdf", width=550, height=850, scale=2)
fig.show()