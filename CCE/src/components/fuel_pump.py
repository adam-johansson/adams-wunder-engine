# this function needs to be done more elaborate


def fuel_pump(p_tdc, fuel_type, fuel_flow):

    # fuel pump power requirement
    if fuel_type == "jetA":
        fuel_rail_pressure = 2500 * 1e5  # 2500 bar common rail pressure
        dens_fuel = 820  # density of kerosene
    elif fuel_type == "H2":
        # fuel pressure twice the maximum pressure before combustion (at TDC)
        fuel_rail_pressure = 2 * p_tdc
        dens_fuel = 71  # density of liquid hydrogen (from Coolprop, hottest density before becoming gas at 1bar)
    fuel_store_pressure = 1e5  # 1 bar fuel store pressure
    dp_fuel = fuel_rail_pressure - fuel_store_pressure
    P_fuelpump = fuel_flow * dp_fuel / dens_fuel

    return P_fuelpump
