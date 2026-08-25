import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

data = np.loadtxt("./results/OPR_CR/thermal_eff.dat")

OPR = data[1:,0]
T4 = data[0,1:]

eff = data[1:,1:]
nox = np.loadtxt("./results/OPR_CR/specific_nox.dat")[1:,1:]
power = np.loadtxt("./results/OPR_CR/core_spec_power.dat")[1:,1:]
pmax = np.loadtxt("./results/OPR_CR/peak_pressure.dat")[1:,1:]
bore = np.loadtxt("./results/OPR_CR/bore.dat")[1:,1:]

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
textsize2 = 40

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
    size=textsize1
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
        start=48.0,
        end=54.0,
        #start=50.5,
        #end=53.5,
        size=0.5,
        labelformat=".0f%%",   # <-- adds % symbol after the value

    ),
    #line=dict(color="black"),
    showscale=False,
    line = dict(
    width = 2,
    smoothing = 0
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
        width=3,
        color="black",
        dash="dot",        # dotted boundary line
    ),
))

# bore lim text
fig.add_annotation(
    x=57,  # specifioc power
    y=0.90,  # NOx
    text="d > 200 mm",
    showarrow=False,
    font=dict(
        size=textsize1,
        family="Times New Roman",
        color="black",
    ),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    opacity=1.0,
)


# add pmax text
fig.add_annotation(
    x=80,
    y=0.875,
    ax=82,  # specifioc power
    ay=0.7,  # NOx
    text="p<sub>max</sub> > 150 bar",
    showarrow=True,
    arrowhead=2,    # arrow head style 1-8
    arrowsize=1,    # relative size of arrowhead
    arrowwidth=2,   # line width
    arrowcolor="black",
    axref="x",      # ax is in data coordinates
    ayref="y",      # ay is in data coordinates
    font=dict(
        size=textsize1,
        family="Times New Roman",
        color="black",
    ),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    opacity=1.0,
)



fig.add_annotation(
    x=57,        # negative x puts it outside the plot to the left
    y=1.1,
    #xref="paper",
    #yref="paper",
    text="OPR [-]",
    showarrow=False,
    font=dict(size=textsize2, family="Times New Roman", weight=700, color="black"),
    textangle=-65,  # rotate to follow axis
)

fig.add_annotation(
    x=57,        # negative x puts it outside the plot to the left
    y=0.7,
    #xref="paper",
    #yref="paper",
    text="ϵ [-]",
    showarrow=False,
    font=dict(size=textsize2, family="Times New Roman", weight=700, color="black"),
    textangle=60,  # rotate to follow axis
)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    autosize=False,
    width=550,
    height=650,
    font=dict(family="Times New Roman", size=textsize1),
    margin=dict(l=80, r=20, t=20, b=80),  # generous margins for saved file
    xaxis=dict(
        range=[49, 93],
        gridcolor="lightgrey",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        tickfont=dict(size=textsize1),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror="allticks", 
        ticks="outside",
    ),
    yaxis=dict(
        range=[0.61, 1.26],
        gridcolor="lightgrey",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        tickfont=dict(size=textsize1),
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
    showlegend=False,
))




fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0, pad=10),
)

fig.update_layout(
    font=dict(family="Times New Roman", size=textsize1, color="black"),
)

fig.write_image("OPR_CR_plot_subfig.pdf", width=550, height=800, scale=2)
fig.show()