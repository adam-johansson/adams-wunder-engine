import torch


def clean_folder(folder):
    # function that cleans all files in a folder
    import os
    import shutil

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return


def checkpoint(model, optimizer, x_scaler, y_scaler, filename):
    # function that saves both the weights of the ANN and the momentum of the optimizer
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'in_features': model.input_to_hidden.in_features,
        'out_features': model.hidden_to_output.out_features,
        'hidden_features': model.hidden[0].in_features,
        'layers': len(model.hidden),
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
    }, filename)
    return


def save(model, filename):
    # function that saves the weights of the ANN
    torch.save({
        'model': model.state_dict(),
        'in_features': model.input_to_hidden.in_features,
        'out_features': model.hidden_to_output.out_features,
        'hidden_features': model.input_to_hidden.out_features,
        'layers': len(model.hidden),
    }, filename)
    return


def save_inference(model, filename, x_std, x_mean, y_std, y_mean):
    # function that saves the weights of the ANN and the scaling params
    torch.save({
        'model': model.state_dict(),
        'in_features': model.input_to_hidden.in_features,
        'out_features': model.hidden_to_output.out_features,
        'hidden_features': model.input_to_hidden.out_features,
        'layers': len(model.hidden),
        'x_std': x_std,
        'x_mean': x_mean,
        'y_std': y_std,
        'y_mean': y_mean,
    }, filename)
    return


def resume(model, optimizer, filename):
    # loads both network weights and optimizer momentum
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return


def load(filename):
    from neural_network.src import NET_narrowing, NET_straight
    # loads the saved model
    save_file = torch.load(filename)
    # retrieves the dimensions fo the network
    input_dim = save_file['in_features']
    output_dim = save_file['out_features']
    if save_file['hidden_features']:
        hidden_dim = save_file['hidden_features']
    layers = save_file['layers']
    # creates a model with the corresponding dimensions
    if "narrowing" in filename:
        model = NET_narrowing(layers, input_dim, hidden_dim, output_dim)
    elif "straight" in filename:
        model = NET_straight(layers, input_dim, hidden_dim, output_dim)
    # loads the weights from the saved file
    model.load_state_dict(save_file['model'])

    return model


def load_inference(filename):
    from neural_network.src import InferenceModel, InferenceModelStraight
    # loads the saved model
    checkpoint = torch.load(filename)
    # retrieves the dimensions fo the network
    input_dim = checkpoint['in_features']
    output_dim = checkpoint['out_features']
    hidden_dim = checkpoint['hidden_features']
    layers = checkpoint['layers']
    weights = checkpoint['model']
    x_scaler = checkpoint['x_scaler']
    y_scaler = checkpoint['y_scaler']
    # creates a model with the corresponding dimension
    model_inference = InferenceModel(layers, input_dim, hidden_dim, output_dim, weights, x_scaler, y_scaler)


    return model_inference


def load_ANN(filename):
    # loads the straight neural network in attack mode
    from neural_network.src import InferenceModelStraight
    # loads the saved model
    model = torch.load(filename)
    # retrieves the dimensions fo the network
    input_dim = model['in_features']
    output_dim = model['out_features']
    hidden_dim = model['hidden_features']
    layers = model['layers']
    weights = model['model']
    x_std = model['x_std']
    x_mean = model['x_mean']
    y_std = model['y_std']
    y_mean = model['y_mean']
    # creates a model with the corresponding dimension
    model_inference = InferenceModelStraight(layers, input_dim, hidden_dim, output_dim, weights, x_std, x_mean, y_std, y_mean)

    return model_inference

