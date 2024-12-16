"""Load a PyTorch model and convert it to TorchScript."""

import os
import sys
from typing import Optional

import numpy as np

from neural_network.src import load_ANN

# Add a module import with your model here:
# This example assumes the model architecture is in an adjacent module `my_ml_model.py`
import torch


def script_to_torchscript(
    model: torch.nn.Module, filename: Optional[str] = "scripted_model.pt"
) -> None:
    """
    Save PyTorch model to TorchScript using scripting.

    Parameters
    ----------
    model : torch.NN.Module
        a PyTorch model
    filename : str
        name of file to save to
    """
    # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
    scripted_model = torch.jit.script(model)
    # print(scripted_model.code)
    scripted_model.save(filename)


def trace_to_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filename: Optional[str] = "traced_model.pt",
) -> None:
    """
    Save PyTorch model to TorchScript using tracing.

    Parameters
    ----------
    model : torch.NN.Module
        a PyTorch model
    dummy_input : torch.Tensor
        appropriate size Tensor to act as input to model
    filename : str
        name of file to save to
    """
    # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
    traced_model = torch.jit.trace(model, dummy_input)
    # traced_model.save(filename)
    frozen_model = torch.jit.freeze(traced_model)
    ## print(frozen_model.graph)
    ## print(frozen_model.code)
    frozen_model.save(filename)


def load_torchscript(filename: Optional[str] = "h2_ts_model.pt") -> torch.nn.Module:
    """
    Load a TorchScript from file.

    Parameters
    ----------
    filename : str
        name of file containing TorchScript model
    """
    model = torch.jit.load(filename)

    return model


if __name__ == "__main__":
    # =====================================================
    # Load model and prepare for saving
    # =====================================================

    # FPTLIB-TODO
    # Load a pre-trained PyTorch model
    # Insert code here to load your model as `trained_model`.
    # This example assumes my_ml_model has a method `initialize` to load
    # architecture, weights, and place in inference mode
    folder = "H2"
    hidden_dim = 32
    layers = 1

    trained_model = load_ANN(f'./models/{folder}_{hidden_dim}_{layers}.pth')


    # Switch off specific layers/parts of the model that behave
    # differently during training and inference.
    # This may have been done by the user already, so just make sure here.
    trained_model.eval()

    # =====================================================
    # Prepare dummy input and check model runs
    # =====================================================

    # Generate a dummy input Tensor `dummy_input` to the model of appropriate size.
    # This example assumes two inputs of size (512x40) and (512x1)
    p_in = 5e5
    t_in = 500
    cr = 8
    bore = 0.15
    far = 0.02923 / 2.0
    p_ratio = 1.2
    v_mean = 12
    fuel_t = 400


    trained_model_dummy_input_1 = np.array([p_in, t_in, cr, bore, far, p_ratio, v_mean, fuel_t])
    trained_model_dummy_input_2 = np.array([p_in, t_in, cr, bore, far, p_ratio, v_mean, fuel_t])
    #trained_model_dummy_input_2 = torch.ones((512, 1), dtype=torch.float64)


    # Uncomment the following lines to save for inference on GPU (rather than CPU):
    # device = torch.device('cuda')
    # trained_model = trained_model.to(device)
    # trained_model.eval()
    # trained_model_dummy_input_1 = trained_model_dummy_input_1.to(device)
    # trained_model_dummy_input_2 = trained_model_dummy_input_2.to(device)

    # FPTLIB-TODO
    # Run model for dummy inputs
    # If something isn't working This will generate an error
    trained_model_dummy_outputs = trained_model.inference(trained_model_dummy_input_1)

    # =====================================================
    # Save model
    # =====================================================

    # FPTLIB-TODO
    # Set the name of the file you want to save the torchscript model to:
    saved_ts_filename = "models/torchscript/h2_ts_model_new.pt"
    # A filepath may also be provided. To do this, pass the filepath as an argument to
    # this script when it is run from the command line, i.e. `./pt2ts.py path/to/model`.

    # FPTLIB-TODO
    # Save the PyTorch model using either scripting (recommended if possible) or tracing
    # -----------
    # Scripting
    # -----------
    script_to_torchscript(trained_model, filename=saved_ts_filename)

    # -----------
    # Tracing
    # -----------
    # trace_to_torchscript(
    #     trained_model, trained_model_dummy_input, filename=saved_ts_filename
    # )

    # =====================================================
    # Check model saved OK
    # =====================================================

    # Load torchscript and run model as a test
    # FPTLIB-TODO
    # Scale inputs as above and, if required, move inputs and mode to GPU
    #trained_model_dummy_input_1 = 2.0 * trained_model_dummy_input_1
    #trained_model_dummy_input_2 = 2.0 * trained_model_dummy_input_2
    trained_model_testing_outputs = trained_model.inference(trained_model_dummy_input_1)


    #scale input
    if trained_model.scaler == "minmax":
        x_max = trained_model.x_max
        x_min = trained_model.x_min

        trained_model_dummy_input_1 = (trained_model_dummy_input_1 - x_min) / (x_max - x_min)

    elif trained_model.scaler == "standard":
        x_mean = trained_model.x_mean
        x_std = trained_model.x_std

        trained_model_dummy_input_1 = (trained_model_dummy_input_1 - x_mean) / x_std
    # convert to tensor
    trained_model_dummy_input_1_tensor = torch.tensor(trained_model_dummy_input_1, dtype=torch.float32)

    ts_model = load_torchscript(filename=saved_ts_filename)
    ts_model_outputs = ts_model(trained_model_dummy_input_1_tensor)

    # scale output
    if trained_model.scaler == "minmax":
        y_max = trained_model.y_max
        y_min = trained_model.y_min

        print(f"xmin: {x_min}")
        print(f"xmax: {x_max}")
        print(f"ymin: {y_min}")
        print(f"ymax: {y_max}")
        ts_model_outputs = y_min + (y_max - y_min) * ts_model_outputs.detach().numpy()

    elif trained_model == "standard":
        y_mean = trained_model.y_mean
        y_std = trained_model.y_std

        print(f"xmean: {x_mean}")
        print(f"xstd: {x_std}")
        print(f"ymean: {y_mean}")
        print(f"ystd: {y_std}")
        ts_model_outputs = y_mean + y_std * ts_model_outputs.detach().numpy()

    print(f"Output normal model: {trained_model_testing_outputs}")

    print(f"Output torch script model: {ts_model_outputs}")


    print(trained_model_testing_outputs - ts_model_outputs)

    #if not isinstance(ts_model_outputs, tuple):
    #    ts_model_outputs = (ts_model_outputs,)
    #if not isinstance(trained_model_testing_outputs, tuple):
    #    trained_model_testing_outputs = (trained_model_testing_outputs,)
    #for ts_output, output in zip(ts_model_outputs, trained_model_testing_outputs):
    #    if torch.all(ts_output.eq(output)):
    #        print("Saved TorchScript model working as expected in a basic test.")
    #       print("Users should perform further validation as appropriate.")
    #    else:
    #        model_error = (
    #            "Saved Torchscript model is not performing as expected.\n"
    #            "Consider using scripting if you used tracing, or investigate further."
    #        )
    #        raise RuntimeError(model_error)

    # Check that the model file is created
    filepath = os.path.dirname(__file__) if len(sys.argv) == 1 else sys.argv[1]
    if not os.path.exists(os.path.join(filepath, saved_ts_filename)):
        torchscript_file_error = (
            f"Saved TorchScript file {os.path.join(filepath, saved_ts_filename)} "
            "cannot be found."
        )
        raise FileNotFoundError(torchscript_file_error)