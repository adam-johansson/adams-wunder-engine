import matplotlib.pyplot as plt

from neural_network.src import load_ANN
import numpy as np

from matplotlib import cm



folder = "jetA"
# Load the trained model
hidden_dim = 128
layers = 2
model = load_ANN(f"../models/{folder}_{hidden_dim}_{layers}_pinn.pth")
model = model.double()
print(model)


p_ratio = 1.0
cr = 10
bore = 0.15
v_mean = 10
T_fuel = 400
far = 0.04


num = 25
Tins = np.linspace(300,700,num)
pins = np.linspace(1e5,5e5,num)

Tins, pins = np.meshgrid(Tins, pins)

powers = np.zeros([num,num])
noxs = np.zeros([num,num])



# make this 2d later maybe
for i in range(num):
    for j in range(num):

        Tin = Tins[i, j]
        pin = pins[i, j]

        piston_input = np.atleast_2d(
            np.array([pin, Tin, p_ratio, cr, bore, v_mean, T_fuel, far])
        )

        # use meta model to get outputs from the piston egnine
        output = model.inference(piston_input)[0]


        indicated_power = output[0]
        heat_loss = output[1]
        nox_ppm = output[2]
        p_tdc = output[3]
        p_max = output[4]
        T_max = output[5]
        T34 = output[6]
        air_flow = output[7]

        fuel_flow = air_flow * far

        out_flow = air_flow + fuel_flow

        # calculate EI NOX (convert from ppm to fraction and from kg to g)
        nox_gram = nox_ppm * out_flow * 1e-3
        EI_nox = nox_gram / fuel_flow

        powers[i,j] = indicated_power
        noxs[i,j] = EI_nox



fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})


surf1 = ax1.plot_surface(Tins, pins*1e-5, powers*1e-3, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig1.colorbar(surf1, shrink=0.5, aspect=5)

fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
surf2 = ax2.plot_surface(Tins, pins*1e-5, noxs, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig2.colorbar(surf2, shrink=0.5, aspect=5)


plt.show()

powers = powers.flatten()*1e-3
noxs = noxs.flatten()
Tins = Tins.flatten()
pins = pins.flatten()*1e-5

#power_data = np.concatenate((Tins, pins, powers), axis=1)
#nox_data = np.concatenate((Tins, pins, noxs), axis=1)

power_data = np.vstack((Tins, pins, powers))
nox_data = np.vstack((Tins, pins, noxs))

power_data = np.transpose(power_data)
nox_data = np.transpose(nox_data)

# save the data to csv-file
np.savetxt("../validation_output_data/smooth/power.dat", power_data, fmt="%.5f")
np.savetxt("../validation_output_data/smooth/nox.dat", nox_data, fmt="%.5f")

