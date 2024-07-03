import pandas as pd

array1 = [0, 0, 1, 1]
array2 = [1, 1, 0, 0]

dataframe = pd.DataFrame()
dataframe.insert(loc=0, column="arr1", value=array1, allow_duplicates=True)
dataframe.insert(loc=1, column="arr2", value=array2, allow_duplicates=True)

print(dataframe)
