import numpy as np


def burner_old(p1, t1, TET, dP, eta):
    p2 = p1 * (1 - dP)
    t2 = TET

    FAR1 = 0.10118 + 2.00376e-05 * (700 - t1)
    FAR2 = 3.7078e-03 - 5.2368e-06 * (700 - t1) - 5.2632e-06 * t2
    FAR3 = 8.889e-08 * abs(t2 - 950)
    FAR = (FAR1 - np.sqrt(FAR1**2 + FAR2) - FAR3) / eta

    return p2, t2, FAR
