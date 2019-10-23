import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

df_train=pd.read_csv('new-york-city-taxi-fare-prediction/train.csv')

df_train.head()

df_train.dtypes
