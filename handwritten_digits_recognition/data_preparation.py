# Before starting the project input data needs to be reduced, and converted to file format

import os
from pathlib import Path

import cv2
import pandas as pd

array_of_converted_images = []

for subdir, dirs, files in os.walk(Path.home()/'Datasets/Handwritten_numbers'):
    for file in files:
        # extracting filepath and digit represented by the image
        file_path = os.path.join(subdir, file)
        represented_digit = subdir[-1]

        # loading image, converting to grayscale and reducing its size from 90x140 to 18x28
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (18, 28))

        # converting image to a single row in dataframe
        image_dictionary = {'represented_digit': represented_digit}
        rows = image.shape[0]
        columns = image.shape[1]

        for row in range(rows):
            for col in range(columns):
                col_name = f"[x={col}, y={row}]"
                image_dictionary[col_name] = image[row, col]

        array_of_converted_images.append(image_dictionary)

df = pd.DataFrame.from_records(array_of_converted_images)  # converting data to dataframe
dummies = pd.get_dummies(df['represented_digit'], dtype='int')  # preparing categories
df.drop('represented_digit', axis=1, inplace=True)  # dropping redundant column
df = df.join(dummies)  # adding class columns
df = df.sample(frac=1)  # shuffling data

# saving data to csv
df.to_csv(Path.home()/'Datasets/Handwritten_numbers_prepared_data.csv', index=False)
