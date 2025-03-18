import torch
import numpy as np


# this is how the torchscript model works:
# import the model, scale the input data manually (min-max scaling)
# the min and max values for all input and output (scaling parameters) have to be inputed manually. this should not be a problem.

# dont forget to re-scale the output



# load the torch script model
filename = f'./models/torchscript/jetA_ts_model.pt'
model = torch.jit.load(filename)

# have some indata
p_in = 5e5
t_in = 500
cr = 8
bore = 0.15
far = 0.02923 / 2.0
p_ratio = 1.2
v_mean = 12
fuel_t = 400

input = np.array([p_in, t_in, cr, bore, far, p_ratio, v_mean, fuel_t])

# scale it
scaler = "standard"
if scaler == "minmax":
     # these are the scaling parameters (min and max for all inputs)
     x_min = np.array([2.00343672e05, 2.50068037e+02, 6.00002527e+00, 1.00014748e-01, 9.74383607e-03, 9.00003960e-01, 8.00015928e+00, 3.00005560e+02])
     x_max = np.array( [3.25632844e+06, 9.99922868e+02, 1.19989449e+01, 1.99996440e-01,
      2.65705351e-02, 1.49997712e+00, 1.49996845e+01, 4.99992481e+02])

     # these are the scaling parameters for the outputs
     y_min = np.array([7.81078607e+02, 2.33389061e-01, 2.66579849e-02, 4.78214558e+01,
                       1.74430253e+03, 2.20274307e+01, 1.20396391e+01, 2.42695641e+01])

     y_max = np.array([1.90411452e+03, 4.85670320e-01, 3.03203792e+00, 9.91808934e+02,
                       3.32192301e+03, 2.90508945e+03, 9.08735380e+02, 3.62908486e+02])

     # scaling
     input = (input - x_min) / (x_max - x_min)

     # convert to tensor
     input = torch.tensor(input, dtype=torch.float32)

     # predict output with the neural network
     output = model(input)

     # scale output back to normal

     # do the scaling
     output = y_min + (y_max - y_min) * output.detach().numpy()

elif scaler == "standard":
     xmean = np.array([1.25817612e+06, 5.57495785e+02, 8.30769279e+00, 1.49890287e-01, 1.78328629e-02, 1.19983675e+00, 1.15013082e+01, 4.00024383e+02])
     xstd = np.array([6.05470142e+05, 1.87831466e+02, 1.65984271e+00, 2.88064466e-02, 4.85724103e-03, 1.74127412e-01, 2.02525853e+00, 5.77058886e+01])
     ymean = np.array([1.29204155e+03, 3.70445518e-01, 4.20836533e-01, 3.54460586e+02, 2.54721198e+03, 3.83262972e+02, 1.82407342e+02, 1.98306337e+02])
     ystd = np.array([2.03904906e+02, 3.45304683e-02, 2.88537503e-01, 1.51654223e+02, 3.41338920e+02, 2.90507033e+02, 1.11875666e+02, 7.63159581e+01])

     # scaling
     input = (input - xmean) / xstd

     # convert to tensor
     input = torch.tensor(input, dtype=torch.float32)

     # predict output with the neural network
     output = model(input)

     # scale output back to normal

     # do the scaling
     output = ymean + ystd * output.detach().numpy()

print(output)