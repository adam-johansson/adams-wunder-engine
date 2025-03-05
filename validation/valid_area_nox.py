import matplotlib.pyplot as plt

# first data point is Heider
# second to seventh is Rakoplpoplus
# eight to +++ Scania


p_peak = [85, 85.6, 89.4, 92.3, 72.6, 74.8, 76.5, 61.3, 80.98, 97.2, 62.2, 98.7, 127.2]
T_peak = [1740, 1850, 2005, 2146, 1773, 1924, 2064, 1218, 1253, 1268, 2411, 2247, 2194]
nox = [860, 938, 1268, 1620, 552, 758, 1001, 220, 270, 276, 1651, 1747, 1570]

# Example case TAKE OFF (p_in 15 bar, T_in 770, far: 0.5)
p_ex = [279.7]
T_ex = [2775]
NO_ex = [2608] # EI_NO 54.8

# Example case TAKE OFF (p_in 8 bar, T_in 670, far: 0.4)
p_ex2 = [162]
T_ex2 = [2675]
NO_ex2 = [3059] # EI_NO 54.8


fig, ax = plt.subplots()

ax.scatter(p_peak, T_peak, label="Validation")
ax.scatter(p_ex, T_ex, label="Take off", s=64, marker="D")
ax.scatter(p_ex2, T_ex2, label="Cruise", s=64, marker="D")
ax.set_xlabel("Peak pressure (bar)")
ax.set_ylabel("Peak temperature (K)")
ax.set_xlim(50,300)
ax.set_ylim(1000,3000)
plt.legend()
plt.show()
