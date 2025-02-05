import numpy as np

tid = np.array([10, 20, 30, 40, 50])
tryck = np.array([100, 200, 300, 400, 500])

print(np.nonzero(tid == 30))


print(tryck[np.nonzero(tid == 30)])