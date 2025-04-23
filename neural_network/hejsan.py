from neural_network.src import load_ANN
import numpy as np

# Load the trained model
meta_model = load_ANN("./models/straight_2048_0.pth")
print(meta_model)


p_in = 5e5
t_in = 500
cr = 15
bore = 0.15
far = 0.02923 / 3
p_ratio = 1.2
v_mean = 15
fuel_t = 400


piston_input_final = np.atleast_2d(
    np.array([p_in, t_in, cr, bore, far, p_ratio, v_mean, fuel_t])
)

output = meta_model.inference(piston_input_final)[0]

print(output)
