import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import LinearRegression


train_data_path = 'new-york-city-taxi-fare-prediction/train.csv'
test_data_path = 'new-york-city-taxi-fare-prediction/test.csv'

df_train=pd.read_csv(train_data_path, nrows=100000)

df_train.head()

len(df_train)

df_train.describe()
df_train.dtypes
# this will take a while, because .to_datetime tries to be smart
df_train['pickup_datetime'] = df_train['pickup_datetime'].str.slice(0,-4)
df_train['pickup_datetime']=pd.to_datetime(df_train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

df_y=df_train[['fare_amount']]
df_x=df_train.copy()

# have to drop timestamp as the linear regression doesn't fit timestamps
df_x.drop(columns=['fare_amount','key','pickup_datetime'], inplace=True)
df_x.head()

reg=LinearRegression().fit(df_x,df_y)

#how well does this line fit the data?
reg.score(df_x,df_y)
#ans: shitly

#now take the test data and take timestamp off
df_test=pd.read_csv(test_data_path)
df_x_test=df_test.copy()
df_x_test.drop(columns=['fare_amount','key','pickup_datetime'], inplace=True)
