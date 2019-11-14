import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import math

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


train_data_path = 'new-york-city-taxi-fare-prediction/train.csv'
test_data_path = 'new-york-city-taxi-fare-prediction/test.csv'

df_all_data=pd.read_csv(train_data_path, nrows=100000)


df_all_data.describe()
df_all_data.dtypes
# this will take a while, because .to_datetime tries to be smart
df_all_data['pickup_datetime'] = df_all_data['pickup_datetime'].str.slice(0,-4)
df_all_data['pickup_datetime']=pd.to_datetime(df_all_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

df_train, df_test = train_test_split(df_train, test_size=0.2)

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
df_x_test=df_test.copy()
df_x_test.drop(columns=['fare_amount','key','pickup_datetime'], inplace=True)
df_x_test.head()
df_y_test=df_test[['fare_amount']]

reg.score(df_x_test,df_y_test)




df_all_data.head()

# 100000 sample passanger count vs fare.
df_all_data.plot.scatter('passenger_count','fare_amount')
# no obvious relationship, might add it later

# First we will look at relationship between hour of pickup and fare_amount
df_all_data['pickup_time']=pd.DatetimeIndex(df_all_data['pickup_datetime']).hour
df_all_data.plot.scatter('pickup_time','fare_amount')

# check for day of the week and fare_amount
df_all_data['pickup_wkday']=pd.DatetimeIndex(df_all_data['pickup_datetime']).weekday
df_all_data.plot.scatter('pickup_wkday','fare_amount')

# ok neither of those gives us very good or clear relationships.
# Most important one will probably be distance travelled
# we re going to simplify the distance and ignore the curvature of the earth. lol.

df_all_data['distance']=np.sqrt((df_all_data['dropoff_longitude']-df_all_data['pickup_longitude'])**2
                                +(df_all_data['dropoff_latitude']-df_all_data['pickup_latitude'])**2)

df_all_data.head()

df_all_data.plot.scatter('distance','fare_amount')
# next thing - clean up the data!!

# next we will plot dropoff vs pickup long/lat on the same scatter plot and
#create clusters to create regions #coolfeatures
