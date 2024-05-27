from torch import tensor
from torch import float32


def nn_output_old(piston_input, meta_model):

    # unpack meta_model
    model0 = meta_model[0]
    model2 = meta_model[1]
    model3 = meta_model[2]
    model4 = meta_model[3]
    model5 = meta_model[4]
    model6 = meta_model[5]
    model7 = meta_model[6]

    x_scaler = meta_model[7]

    y0_scaler = meta_model[8]
    y2_scaler = meta_model[9]
    y3_scaler = meta_model[10]
    y4_scaler = meta_model[11]
    y5_scaler = meta_model[12]
    y6_scaler = meta_model[13]
    y7_scaler = meta_model[14]


    # scale input
    piston_input = x_scaler.transform(piston_input)

    # convert to PyTorch tensor
    piston_input = tensor(piston_input, dtype=float32)

    # get output
    y0 = model0(piston_input)
    y2 = model2(piston_input)
    y3 = model3(piston_input)
    y4 = model4(piston_input)
    y5 = model5(piston_input)
    y6 = model6(piston_input)
    y7 = model7(piston_input)


    # rescale output
    Tout = y0_scaler.inverse_transform(y0.detach().numpy())[0][0]
    air_flow = y2_scaler.inverse_transform(y2.detach().numpy())[0][0]
    p_max = y3_scaler.inverse_transform(y3.detach().numpy())[0][0]
    T_max = y4_scaler.inverse_transform(y4.detach().numpy())[0][0]
    indicated_power = y5_scaler.inverse_transform(y5.detach().numpy())[0][0]
    heat_loss = y6_scaler.inverse_transform(y6.detach().numpy())[0][0]
    p_tdc = y7_scaler.inverse_transform(y7.detach().numpy())[0][0]

    return Tout, air_flow, p_max, T_max, indicated_power, heat_loss, p_tdc
