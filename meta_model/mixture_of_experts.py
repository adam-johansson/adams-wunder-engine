from smt.applications import MOE
from smt.utils.misc import compute_rms_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_status = True


xt = np.genfromtxt('../piston_engine/surrogate_data/xt.csv', delimiter=',')
xval = np.genfromtxt('../piston_engine/surrogate_data/xval.csv', delimiter=',')
yt = np.genfromtxt('../piston_engine/surrogate_data/yt.csv', delimiter=',')
yval = np.genfromtxt('../piston_engine/surrogate_data/yval.csv', delimiter=',')

# Only 1 cluster is considered and the surrogate models_old are compared on a test set from the training database
moe = MOE(n_clusters=1)
moe = MOE(n_clusters=1, xtest=xval, ytest=yval[:, 0])

# to choose some restrictions of the available models_old use allow option
moe = MOE(n_clusters=1, allow=['KRG', 'LS', 'QP', 'KPLSK', 'RBF', 'IDW', 'GEKPLS'])
#allow=["KRG", "LS", "QP"]

print("MOE enabled experts: ", moe.enabled_experts)

moe.set_training_values(xt, yt[:, 0])
moe.train()

print('Best model found with MOE', moe._experts[0].name)
if (moe._experts[0].name == "Kriging") or (moe._experts[0].name == "KPLS") or (moe._experts[0].name == "KPLSK"):
    print('Correlation parameter of this model:', moe._experts[0].options['corr'])
    print('Regression parameter of this model:', moe._experts[0].options['poly'])

# Prediction of the validation points
y = moe.predict_values(xval)
print('MOE + 1 cluster,  err: ' + str(compute_rms_error(moe, xval, yval[:, 0])))
if plot_status:
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    fig = plt.figure()
    plt.plot(yval[:, 0], yval[:, 0], '-', label='$y_{true}$')
    plt.plot(yval[:, 0], y, 'r.', label='$\hat{y}$')

    plt.xlabel('$y_{true}$')
    plt.ylabel('$\hat{y}$')

    plt.legend(loc='upper left')
    plt.title('MOE 1 cluster with ' + str(moe._experts[0].name) + ': validation of the prediction model')

if plot_status:
    plt.show()

# to add a validation set (xtest,ytest) in order to compare different surrogate models_old