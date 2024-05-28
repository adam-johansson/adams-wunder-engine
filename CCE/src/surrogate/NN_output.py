from torch import tensor
from torch import float32


def nn_output(piston_input, meta_model):

    # unpack meta_model
    model = meta_model[0]

    x_scaler = meta_model[1]

    y_scaler = meta_model[2]

    # scale input
    piston_input = x_scaler.transform(piston_input)

    # convert to PyTorch tensor
    piston_input = tensor(piston_input, dtype=float32)

    # get output
    y = model(piston_input)

    # get back to actual numbers
    y_real = y_scaler.inverse_transform(y.detach().numpy())

    Tout = y_real[0,0]
    air_flow = y_real[0,2]
    p_max = y_real[0,3]
    T_max = y_real[0,4]
    indicated_power = y_real[0,5]
    heat_loss = y_real[0,6]
    p_tdc = y_real[0,7]



    return Tout, air_flow, p_max, T_max, indicated_power, heat_loss, p_tdc
