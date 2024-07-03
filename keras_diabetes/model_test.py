import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import mlflow
from mlflow import pyfunc
import tensorflow as tf
import keras

mlflow.set_tracking_uri("http://localhost:5000")

diabetes_dataset = datasets.load_diabetes(as_frame=True, scaled=True)
X = diabetes_dataset["data"]
# X = X.astype('float32')

logged_model = 'runs:/7b63b3e167da4f1ca4679f0d126f6d65/model'
model = mlflow.pyfunc.load_model(logged_model)

print(model.predict(X))
