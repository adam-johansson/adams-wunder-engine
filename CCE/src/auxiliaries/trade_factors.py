import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def load_trade_factors():

    # load the trade factors excel sheet into data framce
    df    = pd.read_excel("Non_Linear_Trade_Factors.xlsx", sheet_name="MR_ff")

    # convert to 2D grid
    pivot = df.pivot(index="Fuel flow factor [-]",
                    columns="Engine weight factor [-]",
                    values="Design mission top-of-climb thrust [N]")

    # Extract the axes and the grid values
    # fuel factor
    ff_axis  = pivot.index.to_numpy(dtype=float)    # [0.7, 0.8, 0.9, 1.0, 1.05]

    # weight factor
    wf_axis  = pivot.columns.to_numpy(dtype=float)  # [0.9, 1.0, 1.33, 1.66, 2.0]

    # thrust values
    grid     = pivot.to_numpy(dtype=float)          # shape (5, 5)

    # MAYBE LOAD ENERGY BLOCK AND STUFF LATER
    
    # create the interpolator
    interp_thrust = RegularGridInterpolator((ff_axis, wf_axis), grid)


    return interp_thrust