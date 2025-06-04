import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("../../optimisation_data/pareto_solutions.csv")


F = np.array([df["sfc"], df["EI_nox"]]).T
X = np.array([df["bpr"], df["opr"], df["split"], df["cr"], df["bore"]]).T

# convert to mm
X[:, 4] = X[:, 4] * 1000

# Choose which variable to color by (e.g., bpr is column 0)
color_by = X[:, 4]  # split

fs = 20

plt.figure(figsize=(8, 6))
sc = plt.scatter(F[:, 0], F[:, 1], c=color_by, cmap='viridis', edgecolors='k', s=256)
#sc = plt.scatter(F[:, 0], F[:, 1], edgecolors='k')
cbar = plt.colorbar(sc)
cbar.set_label('Piston bore (mm)', size=fs)
cbar.ax.tick_params(labelsize=fs)  # Set tick label size
plt.xlabel("SFC [mg/Ns]", size=fs)
plt.ylabel("EI_NOx [g/kg]", size=fs)
#plt.title("Pareto front colored by piston engine size")
plt.grid(True)
plt.tight_layout()

plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.show()
