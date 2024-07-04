import pandas as pd
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

from save_model_and_score import save

# Preparation of data
diabetes_dataset = datasets.load_diabetes(as_frame=True, scaled=True)
X = diabetes_dataset["data"]
y_raw = diabetes_dataset["target"]

# Create labels for glucose levels
y_array = []
# for glucose in y_raw:
#     if glucose < 54:
#         y_array.append("severe hypoglycemia")
#     elif glucose < 70:
#         y_array.append("hypoglycemia")
#     elif glucose < 140:
#         y_array.append("proper glucose level")
#     elif glucose < 200:
#         y_array.append("low glucose tolerance")
#     else:
#         y_array.append("diabetes")

for glucose in y_raw:
    if glucose < 200:
        y_array.append("not_diabetes")
    else:
        y_array.append("diabetes")

# Convert labels to one-hot encoded format
y = pd.Series(y_array)
y = pd.get_dummies(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Convert data to tensors
X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

# Define the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(2, activation="sigmoid")
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Train the model using GPU if available
with tf.device("/GPU:0"):
    model.fit(
        X_train, y_train, epochs=50, batch_size=4
    )
score = model.evaluate(X_test, y_test)
score = f"accuracy:{score[1] * 100:.2f}%, loss:{score[0] * 100:.2f}%"

save(model, score, '/mnt/e/saved_tensorflow', 'keras_diabetes_50e4b')
