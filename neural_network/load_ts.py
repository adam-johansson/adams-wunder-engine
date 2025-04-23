import torch
import numpy as np


# this is how the torchscript model works:
# import the model, scale the input data manually (min-max scaling)
# the min and max values for all input and output (scaling parameters) have to be inputed manually. this should not be a problem.

# dont forget to re-scale the output


# load the torch script model
filename = f"./models/torchscript/jetA_ts_model.pt"
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
    x_min = np.array(
        [
            2.00343672e05,
            2.50068037e02,
            6.00002527e00,
            1.00014748e-01,
            9.74383607e-03,
            9.00003960e-01,
            8.00015928e00,
            3.00005560e02,
        ]
    )
    x_max = np.array(
        [
            3.25632844e06,
            9.99922868e02,
            1.19989449e01,
            1.99996440e-01,
            2.65705351e-02,
            1.49997712e00,
            1.49996845e01,
            4.99992481e02,
        ]
    )

    # these are the scaling parameters for the outputs
    y_min = np.array(
        [
            7.81078607e02,
            2.33389061e-01,
            2.66579849e-02,
            4.78214558e01,
            1.74430253e03,
            2.20274307e01,
            1.20396391e01,
            2.42695641e01,
        ]
    )

    y_max = np.array(
        [
            1.90411452e03,
            4.85670320e-01,
            3.03203792e00,
            9.91808934e02,
            3.32192301e03,
            2.90508945e03,
            9.08735380e02,
            3.62908486e02,
        ]
    )

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
    xmean = np.array(
        [
            1.25817612e06,
            5.57495785e02,
            8.30769279e00,
            1.49890287e-01,
            1.78328629e-02,
            1.19983675e00,
            1.15013082e01,
            4.00024383e02,
        ]
    )
    xstd = np.array(
        [
            6.05470142e05,
            1.87831466e02,
            1.65984271e00,
            2.88064466e-02,
            4.85724103e-03,
            1.74127412e-01,
            2.02525853e00,
            5.77058886e01,
        ]
    )
    ymean = np.array(
        [
            1.29204155e03,
            3.70445518e-01,
            4.20836533e-01,
            3.54460586e02,
            2.54721198e03,
            3.83262972e02,
            1.82407342e02,
            1.98306337e02,
        ]
    )
    ystd = np.array(
        [
            2.03904906e02,
            3.45304683e-02,
            2.88537503e-01,
            1.51654223e02,
            3.41338920e02,
            2.90507033e02,
            1.11875666e02,
            7.63159581e01,
        ]
    )

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
