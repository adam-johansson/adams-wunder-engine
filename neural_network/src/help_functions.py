import numpy as np


def scale_data(scaler, X_train, y_train, X_test, y_test):

    if scaler == "standard":
        # normalise the data, fit the normalisation on training data (mean 0 std 1)
        X_mean = np.mean(X_train, 0)
        X_std = np.std(X_train, 0)

        y_mean = np.mean(y_train, 0)
        y_std = np.std(y_train, 0)

        # normalise the training data
        X_train = (X_train - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std

        # normalise the test data
        X_test = (X_test - X_mean) / X_std
        y_test = (y_test - y_mean) / y_std

        x1 = X_mean
        x2 = X_std
        y1 = y_mean
        y2 = y_std

    elif scaler == "minmax":
        # scale between 0 and 1
        x_min = X_train.min(axis=0)
        x_max = X_train.max(axis=0)

        y_min = y_train.min(axis=0)
        y_max = y_train.max(axis=0)

        X_train = (X_train - x_min) / (x_max - x_min)
        y_train = (y_train - y_min) / (y_max - y_min)

        X_test = (X_test - x_min) / (x_max - x_min)
        y_test = (y_test - y_min) / (y_max - y_min)

        x1 = x_min
        x2 = x_max
        y1 = y_min
        y2 = y_max

    else:
        print(f"Unknown scaler: {scaler}")

    return X_train, y_train, X_test, y_test, x1, x2, y1, y2
