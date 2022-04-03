import pandas as pd
import numpy as np

"""
df_1 = pd.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
df_2 = pd.DataFrame([1, 2, 3], ['a', 'b', 'c'])
df_3 = pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c']], columns=['a', 'b'])
df_4 = pd.DataFrame(np.array([[1, 'a'], [2, 'b'], [3, 'c']]))
df_4.columns = ['a', 'b']

print(df_1)
print(df_2)
print(df_3)
print(df_4)

print(df_1.describe())
"""

# Splitting data into train/test
from sklearn.model_selection import train_test_split

df = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])

print("Source-DF: \n", df)

df_train, df_test = train_test_split(
    df,
    test_size=0.2,  # 20% goes to test
    random_state=0  # makes splitting data repeatable
)
print("Train: \n", df_train)
print("Test: \n", df_test)
