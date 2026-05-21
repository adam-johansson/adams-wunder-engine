
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def load_trade_factors():

    # load the trade factors excel sheet into data framce
    df    = pd.read_excel("trade_factors/Non_Linear_Trade_Factors.xlsx", sheet_name="MR_ff")

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
    grid_thrust     = pivot.to_numpy(dtype=float)          # shape (5, 5)

    # MAYBE LOAD ENERGY BLOCK AND STUFF LATER
    
    # create the interpolator
    interp_thrust = RegularGridInterpolator((ff_axis, wf_axis), grid_thrust)

    # DO THE SAME FOR BLOCK ENERGY
    # convert to 2D grid
    pivot = df.pivot(index="Fuel flow factor [-]",
                    columns="Engine weight factor [-]",
                    values="Design mission block energy/PAX/NM [MJ/NM]")

    # Extract the axes and the grid values
    # fuel factor
    ff_axis  = pivot.index.to_numpy(dtype=float)    # [0.7, 0.8, 0.9, 1.0, 1.05]

    # weight factor
    wf_axis  = pivot.columns.to_numpy(dtype=float)  # [0.9, 1.0, 1.33, 1.66, 2.0]

    # thrust values
    grid_energy     = pivot.to_numpy(dtype=float)          # shape (5, 5)

    # MAYBE LOAD ENERGY BLOCK AND STUFF LATER
    
    # create the interpolator
    interp_energy = RegularGridInterpolator((ff_axis, wf_axis), grid_energy)


    return interp_thrust, interp_energy




def thrust_requirement_function(sfc,weight,trade_factor_interpolator):
    
    # convert SFC to SAF from jet A
    sfc = sfc * (43/44)


    # this is ToC SFC
    ff_factor = sfc / (14.04920699 * 1e-6)
    wf_factor = weight / 3605.95223204919

    
    print(f"SFC: {sfc*1e6} ff_factor: {ff_factor}")




    F_goal = float(trade_factor_interpolator([[ff_factor, wf_factor]])[0])

    print(f"F goal: {F_goal*1e-3} kN")




    return F_goal



def get_block_energy(sfc,weight,trade_factor_interpolator):
    
    # convert SFC to SAF from jet A
    sfc = sfc * (43/44)


    # this is ToC SFC
    ff_factor = sfc / (14.04920699 * 1e-6)
    wf_factor = weight / 3605.95223204919
   
    print(f"SFC: {sfc*1e6} ff_factor: {ff_factor}")

    block_energy = float(trade_factor_interpolator([[ff_factor, wf_factor]])[0])

    return block_energy
