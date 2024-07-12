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


def checkpoint(model, optimizer, filename):
    # function that saves both the weights of the ANN and the momentum of the optimizer
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'in_features': model.input_to_hidden.in_features,
        'out_features': model.hidden_to_output.out_features,
        'hidden_features': model.hidden[0].in_features,
        'layers': len(model.hidden),
    }, filename)

    return


def resume(model, optimizer, filename):
    # loads both network weights and optimizer momentum
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return


def load(filename):
    from neural_network.src import NET_narrowing
    # loads the saved model
    checkpoint = torch.load(filename)
    # retrieves the dimensions fo the network
    input_dim = checkpoint['in_features']
    output_dim = checkpoint['out_features']
    hidden_dim = checkpoint['hidden_features']
    layers = checkpoint['layers']
    # creates a model with the corresponding dimensions
    model = NET_narrowing(layers, input_dim, hidden_dim, output_dim)
    # loads the weights from the saved file
    model.load_state_dict(checkpoint['model'])

    return model




