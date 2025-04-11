
import numpy as np

x = np.array(([1, 2, 3, 6], [8, 9, 10, 11]))

print(x)

b = np.array([5, 9])

print(b)

print(x * b[:, np.newaxis])

