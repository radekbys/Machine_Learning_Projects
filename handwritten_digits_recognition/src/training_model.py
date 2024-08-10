from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import keras
from save_model_and_score import save
from sklearn.preprocessing import MinMaxScaler

# load data from csv into X y sets
images_data = pd.read_csv(
    Path.home() / "Datasets" / "Handwritten_numbers_prepared_data.csv"
)
X = images_data.iloc[:, :-10]
y = images_data.iloc[:, -10:]


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform it
X_train = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test = scaler.transform(X_test)

# Convert data to tensors
X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

#creating model
model = keras.Sequential(
    [
        tf.keras.layers.Dense(504, activation="relu"),
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dense(10, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.F1Score(average='macro', threshold=0.5, name="f1_score")]
)

#number of epochs
EPOCHS = 50

# Train the model using GPU if available
with tf.device("/GPU:0"):
    model.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=12
    )
    
score = model.evaluate(X_test, y_test)
score = f"f1_score:{score[1]}, loss:{score[0]}"

save(model, score, Path.home() / "Models", 'handwritten_digit_recognition')