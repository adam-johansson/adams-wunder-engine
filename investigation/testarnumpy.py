import numpy as np


arrays = []
for i in range(10):
    arrays.append(np.linspace(0, 25, 10))


hej = [item[-1] for item in arrays]

print(hej)
