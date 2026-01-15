'''
Program to observe effect of variable temperature on ignition delay

Chemical reaction
CH4 + 2(O2 + 3.76 N2) <-> CO2 + 2H2O + 7.52N2
'''

# Importing libraries
import sys
sys.path.append("./../")

import numpy as np 
import matplotlib.pyplot as plt 
import cantera as ct 
from thermo import molar_fractions




# Total time to run simulation
simulation_time = 1 #seconds
step_time = 1e-3 #seconds i.e. milisecond
# Calculating number of timesteps
time_steps = int(simulation_time/step_time)


# Burned gas equi-ratio before start of combustion
equ_sc = 0.0

fuel_type = "jetA"

xi_N2_0, xi_O2_0, xi_CO2_0, xi_H2O_0, _, _ = molar_fractions(equ_sc, fuel_type)


# Set gas composition
#composition = (
#    f"CO2:{xi_CO2_0}, H2O:{xi_H2O_0}, O2:{xi_O2_0}, N2:{xi_N2_0}"
#)

species = {
        S.name: S for S in ct.Species.list_from_file("nDodecane_Reitz.yaml")
    }

gas = ct.Solution(thermo="ideal-gas", species=species.values())


# Array for auto ignition delay
Auto_ig_time = []



# Initial temperature in K
temp = np.linspace(1000,1450,5)


# Initial pressure in Pa
P = 5*101325
equ_combustion = 1.0




# Loop for varying values of initial temperature
for T in temp:

    print(f"Temperature: {T}")
    # Creating gas object
    # methane
    #gas = ct.Solution('gri30.yaml')
    gas = ct.Solution('nDodecane_Reitz.yaml', 'nDodecane_IG')

    # Providing initial conditions to gas object

    gas.TP = T, P


    gas.set_equivalence_ratio(
        equ_combustion,
        "c12h26",
        f"O2:{xi_O2_0}, N2:{xi_N2_0}, CO2:{xi_CO2_0}, H2O:{xi_H2O_0}",
    )

    #gas.set_equivalence_ratio(
    #equ_combustion,
    #"CH4",
    #f"O2:{xi_O2_0}, N2:{xi_N2_0}, CO2:{xi_CO2_0}, H2O:{xi_H2O_0}",
    #)

    # Reactor
    r = ct.IdealGasReactor(gas)

    # Reactor network
    sim = ct.ReactorNet([r])

    # Initial time
    time = 0

    # Prepearing solution array for each state
    states = ct.SolutionArray(gas, extra = ['time_ms'])

    loop_counter = 0

    # Time loop (Time integration for states)
    for n in range(time_steps):
        time += step_time
        
        # Advancing time
        sim.advance(time)
        states.append(r.thermo.state, time_ms = time/step_time)

        
        # Loop to calculate ignition delay
        # Time required to ignition delay = Time for acheiving_(initial temprature + 400K)
        if ((states.T[n] >= (T + 400)) and loop_counter == 0):

            Auto_ig_time.append(states.time_ms[n])

            loop_counter = 1

    # Plotting
    plt.figure(3)
    plt.plot(states.time_ms, states.T)
    plt.legend(['Temperature = ' + str(int(T)) + ' k' for T in temp])
    plt.show()

    #plt.figure(3)
    #plt.plot(states.time_ms, states.P)
    #plt.legend(['Temperature = ' + str(int(T)) + ' k' for T in temp])
    #plt.show()

plt.figure(3)
plt.grid('on')
plt.xlabel('Time in mili-second')
plt.ylabel('Auto ignition temperature in K')
plt.title('Variation in Auto-ignition temperature for variable initial temperature')

plt.figure(4)
plt.plot(temp, Auto_ig_time,'-o',color='purple')
plt.grid('on')
plt.xlabel('Initial temerature in K')
plt.ylabel('Ignition delay in mili-seconds')
plt.title('Ignition delay for variable initial temperature')

plt.show()

# Array for auto ignition delay
Auto_ig_time = []

pressure = np.linspace(1e5,10e5,10)

T = 1200
equ_combustion = 1.0

for P in pressure:

    print(f"Pressure: {P*1e-5} bar")
    # Creating gas object
    #gas = ct.Solution('gri30.yaml')


    # Reactant species dictionary
    #species_dict = {'CH4':1, 'O2':2, 'N2':7.52}

    # Providing initial conditions to gas object
    #gas.TPX = T, P, species_dict

    gas.TP = T, P


    gas.set_equivalence_ratio(
        equ_combustion,
        "c12h26:1",
        f"O2:{xi_O2_0}, N2:{xi_N2_0}, CO2:{xi_CO2_0}, H2O:{xi_H2O_0}",
    )

    # Reactor
    r = ct.IdealGasReactor(gas)

    # Reactor network
    sim = ct.ReactorNet([r])

    # Initial time
    time = 0

    # Prepearing solution array for each state
    states = ct.SolutionArray(gas, extra = ['time_ms'])

    loop_counter = 0

    # Time loop (Time integration for states)
    for n in range(time_steps):
        time += step_time
        
        # Advancing time
        sim.advance(time)
        states.append(r.thermo.state, time_ms = time/step_time)

        
        # Loop to calculate ignition delay
        # Time required to ignition delay = Time for acheiving_(initial temprature + 400K)
        if ((states.T[n] >= (T + 400)) and loop_counter == 0):

            Auto_ig_time.append(states.time_ms[n])

            loop_counter = 1

    # Plotting
    plt.figure(3)
    plt.plot(states.time_ms, states.T)
    plt.legend(['Temperature = ' + str(int(P*1e-5)) + ' bar' for P in pressure])
    #plt.show()

    #plt.figure(3)
    #plt.plot(states.time_ms, states.P)
    #plt.legend(['Temperature = ' + str(int(T)) + ' k' for T in temp])
    #plt.show()

plt.figure(3)
plt.grid('on')
plt.xlabel('Time in mili-second')
plt.ylabel('Auto ignition temperature in K')
plt.title('Variation in Auto-ignition temperature for variable initial pressure')

plt.figure(4)
plt.plot(pressure*1e-5, Auto_ig_time,'-o',color='purple')
plt.grid('on')
plt.xlabel('Initial pressure in bar')
plt.ylabel('Ignition delay in mili-seconds')
plt.title('Ignition delay for variable initial pressure')

plt.show()



# Array for auto ignition delay
Auto_ig_time = []

equs = np.linspace(0.2,1.0,10)

T = 1200
P = 5e5

for equ_combustion in equs:

    print(f"Equ: {equ_combustion}")


    gas.TP = T, P


    gas.set_equivalence_ratio(
        equ_combustion,
        "c12h26:1",
        f"O2:{xi_O2_0}, N2:{xi_N2_0}, CO2:{xi_CO2_0}, H2O:{xi_H2O_0}",
    )

    # Reactor
    r = ct.IdealGasReactor(gas)

    # Reactor network
    sim = ct.ReactorNet([r])

    # Initial time
    time = 0

    # Prepearing solution array for each state
    states = ct.SolutionArray(gas, extra = ['time_ms'])

    loop_counter = 0

    # Time loop (Time integration for states)
    for n in range(time_steps):
        time += step_time
        
        # Advancing time
        sim.advance(time)
        states.append(r.thermo.state, time_ms = time/step_time)

        
        # Loop to calculate ignition delay
        # Time required to ignition delay = Time for acheiving_(initial temprature + 400K)
        if ((states.T[n] >= (T + 400)) and loop_counter == 0):

            Auto_ig_time.append(states.time_ms[n])

            loop_counter = 1

    # Plotting
    plt.figure(3)
    plt.plot(states.time_ms, states.T)
    plt.legend(['Temperature = ' + str(int(P*1e-5)) + ' bar' for P in pressure])
    #plt.show()

    #plt.figure(3)
    #plt.plot(states.time_ms, states.P)
    #plt.legend(['Temperature = ' + str(int(T)) + ' k' for T in temp])
    #plt.show()

plt.figure(3)
plt.grid('on')
plt.xlabel('Time in mili-second')
plt.ylabel('Auto ignition temperature in K')
plt.title('Variation in Auto-ignition temperature for variable initial pressure')

plt.figure(4)
plt.plot(pressure*1e-5, Auto_ig_time,'-o',color='purple')
plt.grid('on')
plt.xlabel('Initial pressure in bar')
plt.ylabel('Ignition delay in mili-seconds')
plt.title('Ignition delay for variable initial pressure')

plt.show()