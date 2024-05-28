import matplotlib.pyplot as plt
import numpy as np

from CCE.src import surrogate

model = surrogate.load_nn()

pin = 5e5
Tin = 700
cr = 10
bore = 0.11
p_ratio = 1.2
v_mean = 18


nr = 10000

far_s = np.linspace(0.02923/3.0, 0.02923/1.0, nr)
outputs = np.zeros((nr, 7))
i = 0
for far in far_s:
    input_pist = np.atleast_2d(np.array([pin, Tin, cr, bore, far, p_ratio, v_mean]))
    output = surrogate.nn_output(input_pist, model)
    outputs[i, :] = output
    i += 1

plt.plot(far_s, outputs[:, 6])
plt.show()

# air flow is beheaving weirldy (output[1])
# power a bit weird but not so much [4]
# p_tdc reall weird [6]

