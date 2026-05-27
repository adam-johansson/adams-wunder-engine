import numpy as np

from thermo import fuel_props, mixture


def calculate_powerplant_weight(p_array, T_array, f_array, m_array):


    areas = area_calculator(p_array, T_array, f_array, m_array)


    
    hx_weight = intercooler + oil_cooler 

    turbo_weight = fan + lpc + hpc + lpt 

    power_transfer_weight = + hp_shaft + lpt_shaft + fan_gearbox + piston_gearbox

    piston_module_weight = piston + cylinder + engine_block + topplock + crankshaft + conrod + camshaft

    misc_weight = nacelle + nozzle + secondary_burner 

    pump_weights = fuel_pump + oil_pump 

    total_weight = hx_weight + turbo_weight + power_transfer_weight + piston_module_weight + misc_weight + pump_weights



    return total_weight




def area_calculator(p, T, f, m, fuel_type, Mach_corenozzle):
            
    # function that calculates the flow path areas at each station

    #INPUT: arrays of p, T, far and m at each station. the stations are:
    # a    Ambient static conditions (depending on altitude and ISA temperature deviation)
    # 0    Stagnation properties before the inlet (depending on ambient and flight Mach number). No losses.
    # 2    Inlet conditions. After inlet pressure loss
    # 13   Bypass stream. After the fan.
    # 14   After the cold side of the intercooler.
    # 15   After piston module oil cooler.
    # 21   Core stream. After fan.
    # 25   After LPC.
    # 26   After intercooler. 
    # 3    After HPC.
    # 31   Piston duct intake loss. (Add losses here later MAYBE 5%)
    # 32   Right before the piston module. Mass flow going into the piston engine. 
    # 34   Piston module exhaust.
    # 35   After mixing piston circumventing flow.
    # 4    Secondary combustor outlet.
    # 41   After NGV cooling.
    # 42   After power extraction in the rotor.
    # 5    After rotor cooling.
    # 8    Total properties at nozzle outlet.

    # CREATE ARRAY OF ALL MACH NUMBERS FOR AREA CALCULATIONS
    Mach_ambient = 1.0 # dummy value. No area is calculated here
    Mach_0 = 0.78  # flight mach number. Calculate the area of air being sucked into the engine
    Mach_2 = 0.603  # fan inlet axial Mach number. (source: DT2)
    Mach_13 = 0.5  #fan exit axial Mach number. source: Schaber 2000
    # Mach_LPC_in = 0.452 # booster entrance Mach number from DT2
    # Mach_25 = 0.422 # booster exit Mach number from DT2
    Mach_25 = 0.35  #  LPC exit axial Mach number. Source: Schaber 2000
    Mach_26 = 0.25  # intercooler exit Mach number. DUMMY VALUE. NEED SOURCE
    #Mach_HPC_in = 0.482 HPC inlet axial Mach number from DT2
    #Mach_3 = 0.263 HPC exit Mach number from DT2
    Mach_3 = 0.254 # HPC exit mach number Source: Rolt 2017
    #Mach_3 = 0.20 # HPC Exit. Source: Schaber 2000
    Mach_31 = Mach_3  # after piston duct... but currently I dont have any loss there...
    Mach_32 = 0.5 # piston module inlet. DUMMY VALUE FROM GOOGLE LLM
    Mach_34 = 0.5 # piston module outlet. DUMMY VALUE FROM GOOGLE LLM
    Mach_35 = 0.5 # constant pressure burner inlet Mach number. DUMMY VALUE
    Mach_avg_combustor = 0.06 # average combustor Mach number from DT2
    Mach_4 = 0.12  # constant pressure burner outlet. Source: Schaber 2000
    #Mach_LPT_in =##..... check DT2 but should I USE HPT, IPT or LPT values... DT2 says geared turbofans have higher mach numbers than stated in the document
    Mach_41 = 1.0  # inside LPT. DUMMY VALUE. NO AREA CALCUALTED HERE
    Mach_42 = 1.0 # inside LPT. DUMMY VALUE. NO AREA CALCULATED HERE
    Mach_5 = 0.43 # LPT exit axial Mach number. Source: Schaber 2000  (Samce soucr gives 0.47 as exit from HPT)
    Mach_8 = Mach_corenozzle  # core nozzle exit Mach number. from cycle calculations

    #

    # EIS = 2021 ????
    # hub_tip_ratio = nu
    nu_fan1 = 44.29/(98.94 + np.exp(0.01850 * EIS - 33.31)) # fan leading edge. FROM DT2
    hade_angle_fan = 15 # 15 degree hade angle on fan
    fan_tip2 = fan_tip1 * 0.98 # DT2





    hub_tip_ratio = 0.925
    
    Mach = (
        Mach_ambient,
        Mach_0,
        Mach_2,   # stream before fan
        Mach_13,  #bypass stream after fan
        T14,
        T15,
        Mach_13, # core stream after fan
        Mach_25,   # LPC exit
        Mach_26,
        Mach_3,
        Mach_31,
        Mach_31,
        Mach_34,
        Mach_35,
        Mach_4,     # after secondary burner
        Mach_41,    # after ngv cooling
        Mach_42,    # after rotor expansion
        Mach_5,  #after rotor cooling
        Mach_8,
    )



    areas = np.zeros(len(p))

    # fuel props
    far_s, _= fuel_props(fuel_type)

    for i in range(len(p)):
        _, _, _, _, R, gamma, _, _ = mixture(T[i], p[i], equivalence_ratio=f[i]/far_s, fuel_type=fuel_type)

        areas[i] = Q_function(p[i], T[i], gamma, R, m[i], Mach[i])






    # last stage hub and tip
    r_tip_HPC2 = np.sqrt(A3 / (np.pi * (1-hub_tip_ratio**2) ) )
    r_hub_HPC2 = r_tip_HPC2 * hub_tip_ratio

    last_blade_height = r_tip_HPC2 - r_hub_HPC2




    return areas 


def Q_function(p, T, gamma, R, m, Mach):
        

        area = m / ( np.sqrt(gamma)*Mach*p*(1/np.sqrt(R*T))*(1+0.5*(gamma-1)*Mach**2)**( -0.5*(gamma+1)/(gamma-1) ))


        return area