# imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy
from keras.optimizers import Adam
import keras
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# import data  IN FUTURE JUST IMPORT ONE CSV FILE
xt = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/xt_cleaned.csv', index_col=0)
xval = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/xval_cleaned.csv', index_col=0)
yt = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/yt_cleaned.csv', index_col=0)
yval = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/yval_cleaned.csv', index_col=0)

# convert to numpy arrays
X1 = pd.DataFrame.to_numpy(xt)
X2 = pd.DataFrame.to_numpy(xval)
Y1 = pd.DataFrame.to_numpy(yt)
Y2 = pd.DataFrame.to_numpy(yval)

# Create model
model = Sequential()
model.add(Dense(128, activation="relu", input_dim=7))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))

# Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
# Typically ReLu-based activation are used but since it is performed regression, it is needed a linear activation.
model.add(Dense(1, activation="linear"))

# Compile model: The model is initialized with the Adam optimizer and then it is compiled.
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-3, decay=1e-3 / 200))

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# Fit the model
history = model.fit(X1, Y1, validation_data=(X2, Y2), epochs=1000, batch_size=100, verbose=2, callbacks=[es])

# Calculate predictions
PredTestSet = model.predict(X1)
PredValSet = model.predict(X2)

# Save predictions
numpy.savetxt("trainresults.csv", PredTestSet, delimiter=",")
numpy.savetxt("valresults.csv", PredValSet, delimiter=",")

# Plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()