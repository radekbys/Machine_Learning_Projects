import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import mlflow
import tensorflow as tf
import keras

# configure mlflow for saving experiments
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name="keras_diabetes_test_experiment")
mlflow.autolog()


# Preparation of data
diabetes_dataset = datasets.load_diabetes(as_frame=True, scaled=True)
X = diabetes_dataset["data"]
y_raw = diabetes_dataset["target"]

y_array = []
for glucose in y_raw:
    if glucose < 54:
        y_array.append(0)
    elif glucose < 70:
        y_array.append(1)
    elif glucose < 140:
        y_array.append(2)
    elif glucose < 200:
        y_array.append(3)
    else:
        y_array.append(4)

y = pd.Series(y_array)
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(value=y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(value=y_test)


# creating model and training using nvidia GPU
model = keras.Sequential(
    [
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(5),
    ]
)
model.compile(
    optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"]
)
with tf.device("/GPU:0"):
    model.fit(
        X_train, y_train, epochs=60, batch_size=2, validation_data=(X_test, y_test)
    )
