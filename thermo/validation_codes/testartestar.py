import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(10, 1, 100)

y = np.exp(x-1.25)

equ = 1/x
plt.plot(equ,y)
plt.show()