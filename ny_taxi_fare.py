import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

train_data_path = 'new-york-city-taxi-fare-prediction/train.csv'
test_data_path = 'new-york-city-taxi-fare-prediction/test.csv'

df_train=pd.read_csv(train_data_path)

df_train.head()

len(df_train)

df_train.describe()
df_train.dtypes
# this will take a while, because .to_datetime tries to be smart
df_train['pickup_datetime']=pd.to_datetime(df_train['pickup_datetime'])

# we create a smaller samle of the train data (10%)
df_train_sample=df_train.sample(frac=0.1)
