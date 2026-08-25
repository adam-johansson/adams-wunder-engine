import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

param_name = "CR_PR"

data = np.loadtxt(f"./results/{param_name}/thermal_eff.dat")

cr = data[1:,0]
pr = data[0,1:]

eff = data[1:,1:]
nox = np.loadtxt(f"./results/{param_name}/m_NOx.dat")[1:,1:]
power = np.loadtxt(f"./results/{param_name}/core_spec_power.dat")[1:,1:]
pmax = np.loadtxt(f"./results/{param_name}/peak_pressure.dat")[1:,1:]

eff[eff == 0] = np.nan
nox[nox == 0] = np.nan
power[power == 0] = np.nan
pmax[pmax == 0] = np.nan

fig = go.Figure()

# Carpet grid
fig.add_trace(go.Carpet(
    a=pr,
    b=cr,
    x=power,
    y=nox,
    cheaterslope = 1,
    aaxis = dict(
    smoothing = 1,
    #minorgridcount = 1,
    minorgridcolor = 'black',
    gridcolor = 'black',
    color = 'black',
    title=dict(
                text="Pressure split",
                font=dict(
                    size=18,
                    family="Arial",
                    color="black",
                    weight=1000,
                )
            ),
    tickfont=dict(
    size=14
    )
    ),
    baxis = dict(
    smoothing = 1,
    #minorgridcount = 1,
    minorgridcolor = 'black',
    gridcolor = 'black',
    color = 'black',
    title=dict(
                text="Compression ratio",
                font=dict(
                    size=18,
                    family="Arial",
                    color="black",
                    weight=1000,
                )
            ),
    tickfont=dict(
    size=14
    )
)))

# Efficiency contours
fig.add_trace(go.Contourcarpet(
    a=pr,
    b=cr,
    z=eff,
    contours=dict(
        #coloring="lines",
        showlabels=True,
        start=48,
        end=53,
        size=1.0
    ),
    #line=dict(color="black"),
    #showscale=False,
    line = dict(
    width = 2,
    smoothing = 0
    ),
    colorbar = dict(
       len = 0.4,
        y = 0.25
    )
))

# Show limit on peak pressure
fig.add_trace(go.Contourcarpet(
    a=pr,
    b=cr,
    z=pmax,
    contours=dict(
        start=150,
        end=150,
        size=1,
        coloring="lines",
        showlabels=True
    ),
    colorscale=[
    [0, "black"],
    [1, "black"]
    ],
    line=dict(
        width=6
    ),
    showscale=False
))

fig.update_layout(
    yaxis_title="NOx [g/s]",
    xaxis_title="Specific core power [kW/litre]"
)

fig.show()