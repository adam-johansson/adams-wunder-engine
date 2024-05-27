from __future__ import print_function, division
import numpy as np
from scipy import linalg
from smt.utils.misc import compute_rms_error

from smt.problems import Sphere, NdimRobotArm, Rosenbrock
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, MGP

#to ignore warning messages
import warnings
warnings.filterwarnings("ignore")

try:
    from smt.surrogate_models import IDW, RBF, RMTC, RMTB
    compiled_available = True
except:
    compiled_available = False

try:
    import matplotlib.pyplot as plt
    plot_status = True
except:
    plot_status = False

import scipy.interpolate

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


########### Initialization of the problem, construction of the training and validation points

ndim = 2
ndoe = 20 #int(10*ndim)
# Define the function
fun = Rosenbrock(ndim=ndim)

# Construction of the DOE
# in order to have the always same LHS points, random_state=1
sampling = LHS(xlimits=fun.xlimits, criterion='ese', random_state=1)
xt = sampling(ndoe)
# Compute the outputs
yt = fun(xt)

# Construction of the validation points
ntest = 200 #500
sampling = LHS(xlimits=fun.xlimits, criterion='ese', random_state=1)
xtest = sampling(ntest)
ytest = fun(xtest)

#To visualize the DOE points
fig = plt.figure(figsize=(10, 10))
plt.scatter(xt[:,0],xt[:,1],marker = 'x',c='b',s=200,label='Training points')
plt.scatter(xtest[:,0],xtest[:,1],marker = '.',c='k', s=200, label='Validation points')
plt.title('DOE')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

# To plot the Rosenbrock function
x = np.linspace(-2,2,50)
res = []
for x0 in x:
    for x1 in x:
        res.append(fun(np.array([[x0,x1]])))
res = np.array(res)
res = res.reshape((50,50)).T
X,Y = np.meshgrid(x,x)
fig = plt.figure()
ax =  fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, res, cmap=cm.viridis,
                       linewidth=0, antialiased=False,alpha=0.5)

ax.scatter(xt[:,0],xt[:,1],yt,zdir='z',marker = 'x',c='b',s=200,label='Training point')
ax.scatter(xtest[:,0],xtest[:,1],ytest,zdir='z',marker = '.',c='k',s=200,label='Validation point')

plt.title('Rosenbrock function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

########### The LS model

# Initialization of the model
t = LS(print_prediction=False)

# Add the DOE
t.set_training_values(xt, yt[:, 0])

# Train the model
t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('LS,  err: ' + str(compute_rms_error(t, xtest, ytest)))


# Plot prediction/true values
if plot_status:
    fig = plt.figure()
    plt.plot(ytest, ytest, '-', label='$y_{true}$')
    plt.plot(ytest, y, 'r.', label='$\hat{y}$')

    plt.xlabel('$y_{true}$')
    plt.ylabel('$\hat{y}$')

    plt.legend(loc='upper left')
    plt.title('LS model: validation of the prediction model')
    plt.show()



########### The QP model

t = QP(print_prediction=False)
t.set_training_values(xt, yt[:, 0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('QP,  err: ' + str(compute_rms_error(t, xtest, ytest)))

# Plot prediction/true values
if plot_status:
    fig = plt.figure()
    plt.plot(ytest, ytest, '-', label='$y_{true}$')
    plt.plot(ytest, y, 'r.', label='$\hat{y}$')

    plt.xlabel('$y_{true}$')
    plt.ylabel('$\hat{y}$')

    plt.legend(loc='upper left')
    plt.title('QP model: validation of the prediction model')
    plt.show()


########### The Kriging model

# The variable 'theta0' is a list of length ndim.
t = KRG(theta0=[1e-2] * ndim, print_prediction=False)
t.set_training_values(xt, yt[:, 0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('Kriging,  err: ' + str(compute_rms_error(t, xtest, ytest)))
if plot_status:
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    fig = plt.figure()
    plt.plot(ytest, ytest, '-', label='$y_{true}$')
    plt.plot(ytest, y, 'r.', label='$\hat{y}$')

    plt.xlabel('$y_{true}$')
    plt.ylabel('$\hat{y}$')

    plt.legend(loc='upper left')
    plt.title('Kriging model: validation of the prediction model')
    plt.show()


# Value of theta
print("theta values", t.optimal_theta)

# estimated variance for the validation points
s2 = t.predict_variances(xtest)
# plot with the associated interval confidence
yerr = 2 * 3 * np.sqrt(s2)  # in order to use +/- 3 x standard deviation: 99% confidence interval estimation
if plot_status:
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    fig = plt.figure()
    plt.plot(ytest, ytest, '-', label='$y_{true}$')
    plt.plot(ytest, y, 'r.', label='$\hat{y}$')
    plt.errorbar(np.squeeze(ytest), np.squeeze(y), yerr=np.squeeze(yerr), fmt='none', capsize=5, ecolor='lightgray',
                 elinewidth=1, capthick=0.5, label='confidence estimate 99%')
    plt.xlabel('$y_{true}$')
    plt.ylabel('$\hat{y}$')

    plt.legend(loc='upper left')
    plt.title('Kriging model: validation of the prediction model with the estimate of confidence')

if plot_status:
    plt.show()

# estimated derivative for the validation points
dydx1 = t.predict_derivatives(xtest, 0) #derivative according to the x1
dydx2 = t.predict_derivatives(xtest, 1) #derivative according to the x2

# estimated variance derivative for the validation points
dsigmadx1 = t.predict_variance_derivatives(xtest, 0) # variance derivative according to the x1
dsigmadx2 = t.predict_variance_derivatives(xtest, 1) # variance derivative according to the x2

#squared exponential by default
t1 = KRG(theta0=[1e-2]*ndim,print_prediction = False, corr='squar_exp')
t1.set_training_values(xt,yt[:,0])
t1.train()
# Prediction of the validation points
y1 = t1.predict_values(xtest)

#absolute exponential
t2 = KRG(theta0=[1e-2]*ndim,print_prediction = False, corr='abs_exp')
t2.set_training_values(xt,yt[:,0])
t2.train()
# Prediction of the validation points
y2 = t2.predict_values(xtest)

#matern32
t3 = KRG(theta0=[1e-2]*ndim,print_prediction = False, corr='matern32')
t3.set_training_values(xt,yt[:,0])
t3.train()
# Prediction of the validation points
y3 = t3.predict_values(xtest)

#matern52
t4 = KRG(theta0=[1e-2]*ndim,print_prediction = False, corr='matern52')
t4.set_training_values(xt,yt[:,0])
t4.train()
# Prediction of the validation points
y4 = t4.predict_values(xtest)



print('\n')
print('Comparison of errors')
print('Kriging squared exponential,  err: '+ str(compute_rms_error(t1,xtest,ytest)))
print('Kriging absolute exponential,  err: '+ str(compute_rms_error(t2,xtest,ytest)))
print('Kriging matern32,  err: '+ str(compute_rms_error(t3,xtest,ytest)))
print('Kriging matern52,  err: '+ str(compute_rms_error(t4,xtest,ytest)))

#squared exponential + constant term by default
t1 = KRG(theta0=[1e-2]*ndim,print_prediction = False, corr='squar_exp', poly='constant')
t1.set_training_values(xt,yt[:,0])
t1.train()
# Prediction of the validation points
y1 = t1.predict_values(xtest)

#squared exponential + linear term
t2 = KRG(theta0=[1e-2]*ndim,print_prediction = False, corr='squar_exp', poly='linear')
t2.set_training_values(xt,yt[:,0])
t2.train()
# Prediction of the validation points
y2 = t2.predict_values(xtest)

#squared exponential + quadratic term
t2 = KRG(theta0=[1e-2]*ndim,print_prediction = False, corr='squar_exp', poly='quadratic')
t2.set_training_values(xt,yt[:,0])
t2.train()
# Prediction of the validation points
y2 = t2.predict_values(xtest)


print('\n')
print('Comparison of errors')
print('Kriging squared exponential + constant term,  err: '+ str(compute_rms_error(t1,xtest,ytest)))
print('Kriging squared exponential + linear term,  err: '+ str(compute_rms_error(t2,xtest,ytest)))
print('Kriging squared exponential + quadratic term,  err: '+ str(compute_rms_error(t3,xtest,ytest)))

# Plot the surrogate model in 3D
x = np.linspace(-2, 2, 50)
resSM = []
varSM = []
for x0 in x:
    for x1 in x:
        resSM.append(t.predict_values(np.array([[x0, x1]])))
        varSM.append(t.predict_variances(np.array([[x0, x1]])))

resSM = np.array(resSM)
resSM = resSM.reshape((50, 50)).T
varSM = np.array(varSM)
varSM = varSM.reshape((50, 50)).T
X, Y = np.meshgrid(x, x)

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(xt[:, 0], xt[:, 1], yt, zdir='z', marker='x', c='b', s=200, label='DOE')
surf = ax.plot_surface(X, Y, resSM, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.5)
ax.scatter(xtest[:, 0], xtest[:, 1], ytest, zdir='z', marker='.', c='g', s=100, label='Validation')
ax.scatter(xtest[:, 0], xtest[:, 1], y, zdir='z', marker='x', c='r', s=100, label='Prediction')
plt.legend()
plt.title('Rosenbrock function with the DOE points and predicted values')

plt.show()

# Plot the surrogate with 99% confidence by using the estimated variance information
fig = plt.figure(figsize=(15, 10))
ax =  fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, resSM, cmap=cm.viridis,
                       linewidth=0, antialiased=False,alpha=0.5)
surf = ax.plot_surface(X, Y, resSM+3*np.sqrt(varSM), color='r',  cmap=cm.cool,
                       linewidth=0, antialiased=False,alpha=0.2)
surf = ax.plot_surface(X, Y, resSM-3*np.sqrt(varSM), color='r',  cmap=cm.cool,
                       linewidth=0, antialiased=False,alpha=0.2)


ax.scatter(xt[:,0],xt[:,1],yt,zdir='z',marker = 'x',c='b',s=200,label='Training point')
#ax.scatter(xtest[:,0],xtest[:,1],ytest,zdir='z',marker = '.',c='k',s=200,label='Validation point')

plt.title(' Rosenbrock Surrogate Model with the 99% confidence interval ')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

# Plot of the variance
fig = plt.figure(figsize=(15, 10))
ax =  fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, varSM, cmap=cm.viridis,
                       linewidth=0, antialiased=False, alpha=0.5)

plt.title('Rosenbrock surrogate model error')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()