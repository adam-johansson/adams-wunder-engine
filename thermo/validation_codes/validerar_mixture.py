import numpy as np

from thermo import mixture

import matplotlib.pyplot as plt

import CEA_Wrap as cea

num = 40

R_univ = 8.3144626  # J mol^-1 K^-1

cea_val = False

T = 900
p = 1e5
# equ = 0.5
# equ2 = 0.5


data = np.zeros((num, num, 8))
data_cea = np.zeros((num, num, 7))

var = np.linspace(0, 1, num)
var2 = np.linspace(0, 1, num)

i = 0
for equ2 in var:
    j = 0
    for equ in var2:

        if equ > equ2:
            data[i, j, :] = np.zeros((1, 8)) * np.inf
            data_cea[i, j, :] = np.zeros((1, 7)) * np.inf

        else:
            h, u, cp, cv, R, gamma, s, M = mixture(
                T, p, equ, fuel_type="H2", pure_fuel=True, fuel_equ_ratio=equ2
            )
            # h, u, cp, cv, R, gamma, s, M = mixture(T, p, equ, fuel_type="jetA")

            data[i, j, :] = np.array([h * 1e-3, u * 1e-3, cp, cv, R, gamma, s, M * 1e3])

            # print((h - u) * 1e-3)

            # cea as reference
            air = cea.Oxidizer("Air", temp=T)
            h2 = cea.Fuel("H2", temp=T)
            jetA = cea.Fuel("Jet-A(g)", temp=T)

            if cea_val:
                if equ > 0:
                    burning = cea.TPProblem(
                        pressure=p * 1e-5,
                        pressure_units="bar",
                        materials=[air, jetA],
                        phi=equ,
                        temperature=T,
                    )

                    exhaust = burning.run()
                    # check temperature
                    # print(exhaust.t)

                    h_cea = exhaust.h
                    cp_cea = exhaust.cp
                    gamma_cea = exhaust.gammas
                    M_cea = exhaust.mw
                    R_cea = R_univ / M_cea
                    cv_cea = cp_cea - R_cea
                    u_cea = h_cea - R_cea * T

                    # print((h_cea - u_cea) * 1e-3)

                    data_cea[i, j, :] = np.array(
                        [
                            h_cea,
                            u_cea,
                            cp_cea * 1e3,
                            cv_cea * 1e3,
                            R_cea * 1e3,
                            gamma_cea,
                            M_cea,
                        ]
                    )

                else:
                    data_cea[i, j, :] = np.zeros((1, 7)) * np.inf

        j += 1
    i += 1
    # print(i)

label = ["h", "u", "cp", "cv", "R", "gamma", "s", "M"]

for k in range(0, 8):
    plt.figure()
    cs = plt.contourf(var, var2, data[:, :, k], 100)
    plt.xlabel("Combustion equ")
    plt.ylabel("Pure fuel equ")
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel(label[k])

if cea_val:
    label_cea = ["h CEA", "u CEA", "cp CEA", "cv CEA", "R CEA", "gamma CEA", "M CEA"]

    for k in range(0, 7):
        plt.figure()
        cs = plt.contourf(var, var2, data_cea[:, :, k], 20)
        plt.xlabel("Combustion equ")
        plt.ylabel("Pure fuel equ")
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel(label_cea[k])

    label_diff = [
        "h diff",
        "u diff",
        "cp diff",
        "cv diff",
        "R diff",
        "gamma diff",
        "M diff",
    ]
    for k in range(0, 7):
        plt.figure()
        if k == 6:
            cs = plt.contourf(var, var2, data_cea[:, :, k] - data[:, :, k + 1], 20)
        else:
            cs = plt.contourf(var, var2, data_cea[:, :, k] - data[:, :, k], 20)
        plt.xlabel("Combustion equ")
        plt.ylabel("Pure fuel equ")
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel(label_diff[k])

plt.show()
