import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

from thermo import polynomials


# Entropy, enthalpy and entropy looks good for all reference materials (N2, Ar, O2, H2) but enthalpy bad for CO2 and H2O

# Check if all 12 species are available in cantera. Compare with NASA polynomials
p = 5e5
gas = ct.Solution("gri30.yaml")

#gas.TPX = 298, 1e5, "CO2:1"

#print(gas.standard_enthalpies_RT)

n = 1000
T = np.linspace(200,4000, n)

n_species = 12
cp_ct = np.zeros([n, n_species])
h_ct = np.zeros([n, n_species])
s_ct = np.zeros([n, n_species])
poly_data = np.zeros([n, n_species, 4])

i = 0
for t in T:
    # Set reactants state
    gas.TPX = t, p, "O2:1"
    cp_ct[i, 0] = gas.cp_mass
    h_ct[i, 0] = gas.enthalpy_mass
    s_ct[i, 0] = gas.entropy_mass

    gas.TPX = t, p, "N2:1"
    cp_ct[i, 1] = gas.cp_mass
    h_ct[i, 1] = gas.enthalpy_mass
    s_ct[i, 1] = gas.entropy_mass

    gas.TPX = t, p, "CO2:1"
    cp_ct[i, 2] = gas.cp_mass
    h_ct[i, 2] = gas.enthalpy_mass
    s_ct[i, 2] = gas.entropy_mass

    gas.TPX = t, p, "Ar:1"
    cp_ct[i, 3] = gas.cp_mass
    h_ct[i, 3] = gas.enthalpy_mass
    s_ct[i, 3] = gas.entropy_mass

    gas.TPX = t, p, "H2O:1"
    cp_ct[i, 4] = gas.cp_mass
    h_ct[i, 4] = gas.enthalpy_mass
    s_ct[i, 4] = gas.entropy_mass

    gas.TPX = t, p, "H2:1"
    cp_ct[i, 5] = gas.cp_mass
    h_ct[i, 5] = gas.enthalpy_mass
    s_ct[i, 5] = gas.entropy_mass

    gas.TPX = t, p, "CO:1"
    cp_ct[i, 6] = gas.cp_mass
    h_ct[i, 6] = gas.enthalpy_mass
    s_ct[i, 6] = gas.entropy_mass

    gas.TPX = t, p, "OH:1"
    cp_ct[i, 7] = gas.cp_mass
    h_ct[i, 7] = gas.enthalpy_mass
    s_ct[i, 7] = gas.entropy_mass

    gas.TPX = t, p, "O:1"
    cp_ct[i, 8] = gas.cp_mass
    h_ct[i, 8] = gas.enthalpy_mass
    s_ct[i, 8] = gas.entropy_mass

    gas.TPX = t, p, "H:1"
    cp_ct[i, 9] = gas.cp_mass
    h_ct[i, 9] = gas.enthalpy_mass
    s_ct[i, 9] = gas.entropy_mass

    gas.TPX = t, p, "NO:1"
    cp_ct[i, 10] = gas.cp_mass
    h_ct[i, 10] = gas.enthalpy_mass
    s_ct[i, 10] = gas.entropy_mass

    gas.TPX = t, p, "N:1"
    cp_ct[i, 11] = gas.cp_mass
    h_ct[i, 11] = gas.enthalpy_mass
    s_ct[i, 11] = gas.entropy_mass

    poly_data[i, 0, :] = polynomials.O2(t, p)
    poly_data[i, 1, :] = polynomials.N2(t, p)
    poly_data[i, 2, :] = polynomials.CO2(t, p)
    poly_data[i, 3, :] = polynomials.Ar(t, p)
    poly_data[i, 4, :] = polynomials.H2O(t, p)
    poly_data[i, 5, :] = polynomials.H2(t, p)
    poly_data[i, 6, :] = polynomials.CO(t, p)
    poly_data[i, 7, :] = polynomials.OH(t, p)
    poly_data[i, 8, :] = polynomials.O(t, p)
    poly_data[i, 9, :] = polynomials.H(t, p)
    poly_data[i, 10, :] = polynomials.NO(t, p)
    poly_data[i, 11, :] = polynomials.N(t, p)

    i = i + 1


figure, axis = plt.subplots(3, 1)
axis[0].plot(T, cp_ct[:,0], label="Cantera")
axis[0].plot(T, poly_data[:,0,0], label="Polynomials")

axis[1].plot(T, h_ct[:,0], label="Cantera")
axis[1].plot(T, poly_data[:,0,1], label="Polynomials")
axis[2].plot(T, s_ct[:,0], label="Cantera")
axis[2].plot(T, poly_data[:,0,2], label="Polynomials")
figure.suptitle("O2")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
axis[0].plot(T, cp_ct[:,1], label="Cantera")
axis[0].plot(T, poly_data[:,1,0], label="Polynomials")
axis[1].plot(T, h_ct[:,1], label="Cantera")
axis[1].plot(T, poly_data[:,1,1], label="Polynomials")
axis[2].plot(T, s_ct[:,1], label="Cantera")
axis[2].plot(T, poly_data[:,1,2], label="Polynomials")
figure.suptitle("N2")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 2
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("CO2")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 3
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("Ar")
axis[0].set_ylim([400, 600])
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 4
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("H2O")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 5
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("H2")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 6
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("CO")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 7
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("OH")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 8
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("O")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 9
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("H")
axis[0].set_ylim([15000, 25000])
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 10
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("NO")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()

figure, axis = plt.subplots(3, 1)
species = 11
axis[0].plot(T, cp_ct[:,species], label="Cantera")
axis[0].plot(T, poly_data[:,species,0], label="Polynomials")
axis[1].plot(T, h_ct[:,species], label="Cantera")
axis[1].plot(T, poly_data[:,species,1], label="Polynomials")
axis[2].plot(T, s_ct[:,species], label="Cantera")
axis[2].plot(T, poly_data[:,species,2], label="Polynomials")
figure.suptitle("N")
axis[0].set_xlabel("Temperature [K]")
axis[0].set_ylabel("Specific heat capacity Cp [kJ/(kg K)]")
axis[1].set_ylabel("Specific enthalpy h [kJ/kg]")
axis[2].set_ylabel("Specific entropy [kJ/(kg K)]")
axis[0].legend()


plt.show()