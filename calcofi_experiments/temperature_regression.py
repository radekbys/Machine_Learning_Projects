# in this experiment calcofi dataset is used to train a neural network, 
# which will guess ocean water temperature based on depth, salinity and oxygen in the water

import pandas as pd
import keras
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
from save_model_and_score import save
from pathlib import Path

# reading data
data = pd.read_csv("~/Datasets/CALCOFI/bottle.csv")
data = data.loc[0:, ["Depthm", "Salnty", "T_degC", "Oxy_µmol/Kg"]]
data= data.dropna()
X = data.loc[0:, ["Depthm", "Salnty",  "Oxy_µmol/Kg"]]
y= data.loc[0:, ["T_degC"]]

# Split data into training and test sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Initialize MinMaxScaler
scaler =sklearn.preprocessing.MinMaxScaler()

# Fit the scaler on the training data and transform it
X_train = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test = scaler.transform(X_test)

# Convert data to tensor
X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

#creating model
model = keras.Sequential(
    [
        tf.keras.layers.Dense(3, activation="relu", input_shape=(3,)),
        tf.keras.layers.Dense(1),
    ]
)

optimizer = keras.optimizers.RMSprop(0.001)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae", "mse"]
)

#number of epochs
EPOCHS = 20

# Train the model using GPU if available
with tf.device("/GPU:0"):
    model.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=16
    )
    
score = model.evaluate(X_test, y_test)
score = f"mean_absolute_error={score[1]}, mean_square_error={score[2]}, epochs={EPOCHS}"
save(model, score, Path.home() / "Models", 'temperature_regression')