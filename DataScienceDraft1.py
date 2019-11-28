import numpy as np
import sklearn as sk
import pandas as pd

# via https://towardsdatascience.com/logistic-regression-a-simplified-approach-using-python-c4bc81a87c31
# We are importing the following libraries

import matplotlib.pyplot as plt
import seaborn as sns



# %matplotlib inline


# Change Unnamed first column to ID


# Line 16 just a way to view the columns in terminal to visualize...
pd.set_option('display.max_columns', 15)

# Read data from file 'filename.csv'
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later)
data = pd.read_csv("african_crises.csv")
print(data.head())
# data1 = pd.read_csv("C:\Users\Austen\PycharmProjects\DataScienceProject\data.csv")

# print(data.isnull())  # Helps us see which data points are empty

# After looking at the data we can see that Region 2 is not an effective category
# As the majority of values appear to be null
# https://stackoverflow.com/questions/26597116/seaborn-plots-not-showing-up

sns.heatmap(data.isnull())
plt.show()

# Looking at the heatmap chart created, we can see that designation, region_2,
# taster name, and taster twitter handle are probably not useful datapoints to
# continue the analysis

# Removing taster_twitter_handle, region_2
data.drop('taster_twitter_handle', axis=1, inplace=True)
data.drop('region_2', axis=1, inplace=True)

# Data Check:
sns.heatmap(data.isnull())
plt.show()

# However, perhaps there is some value in continuing to look at the taster name,
# to uncover any possible biases. Let's fill in the null values with a text string - 'unknown'


""" This is for the taster name change to unknown
def impute_taster(cols):
    taster_name = cols[0]
    if pd.isnull(taster_name):
        return str("Unknown")
    else:
        return taster_name


data['taster_name'] = data[['taster_name']].apply(impute_taster(9), axis=1)
"""


def impute_price(cols):
    price = cols[0]
    if pd.isnull(price):
        return 0
    else:
        return price


# data = data.fillna(method='ffill') # This one forward fills all the columns.
# You can also apply to specific columns as below
data[['price', 'taster_name', 'designation', 'region_1']] = data[
    ['price', 'taster_name', 'designation', 'region_1']].fillna(method='ffill')

# data['price'] = data[['price']].apply(impute_price(5), axis=1)


sns.heatmap(data.isnull())
plt.show()

# Preview the first 5 lines of the loaded data
# print(data.head(1))

# df_csv = pd.read_csv("african_crises.csv", index=False)

# Load only 3 rows
# df_csv = pd.read_csv('african_crises.csv', nrows=3)

# print(df_csv)

data.info()

# Dummy variable for New World Old World
# Dummy for White Wine / Red Wine - Austen to do
# Price 4 - 3300
# 4 - 20, 20 - 50, 50 - 300, 300 - 1000, 1000+
# 50 or less, 50+
# 4 - 15, 16- 22, 23 - 32, 33 - 50, 50+

csv_input = pd.read_csv('african_crises.csv', error_bad_lines=False)
csv_input['New/Old World'] = csv_input['country']
csv_input['Price4_15'] = csv_input['price']
csv_input['Price16_22'] = csv_input['price']
csv_input['Price23_32'] = csv_input['price']
csv_input['Price33_50'] = csv_input['price']
csv_input['Price51_'] = csv_input['price']
csv_input.to_csv('wineanalysis.csv', index=False)

data = pd.read_csv("wineanalysis.csv")
print(data.head())
# data = data.fillna(method='ffill') # This one forward fills all the columns.
# You can also apply to specific columns as below
data[['New/Old World', 'Price4_15', 'Price16_22', 'Price23_32',
      'Price33_50', 'Price51_']] = data[['New/Old World', 'Price4_15',
                                         'Price16_22', 'Price23_32', 'Price33_50',
                                         'Price51_']].fillna(method='ffill')

data.loc[data.Price4_15 < 16, 'Price4_15'] = 1
data.loc[data.Price4_15 >= 16, 'Price4_15'] = 0

data.loc[data.Price16_22 < 23, 'Price16_22'] = 1
data.loc[data.Price16_22 < 16, 'Price16_22'] = 0
data.loc[data.Price16_22 > 23, 'Price16_22'] = 0

data.loc[data.Price23_32 < 33, 'Price23_32'] = 1
data.loc[data.Price23_32 < 23, 'Price23_32'] = 0
data.loc[data.Price23_32 > 33, 'Price23_32'] = 0

data.loc[data.Price33_50 < 51, 'Price33_50'] = 1
data.loc[data.Price33_50 < 33, 'Price33_50'] = 0
data.loc[data.Price33_50 > 51, 'Price33_50'] = 0

data.loc[data.Price51_ > 50, 'Price51_'] = 1
data.loc[data.Price51_ < 51, 'Price51_'] = 0

print(data.head())

sns.heatmap(data.isnull())
plt.show()

sns.set()

data['price'].replace('', np.nan, inplace=True)
data.dropna(subset=['price'], inplace=True)

import statsmodels
from statsmodels.discrete.discrete_model import Probit


statsmodels.discrete.discrete_model.Probit(data['points'], data['country', 'price', 'province', 'variety'])
result_3 = statsmodels.discrete.discrete_model.Probit
print(result_3.summary())


print("This is the pivot.")
# https://pbpython.com/pandas-pivot-table-explained.html


wine = pd.pivot_table(data, index=['Price4_15', 'Price16_22', 'Price23_32', 'Price33_50', 'Price51_'], values=["price"],
                      columns=["country"], aggfunc={'price': 'count'}, fill_value=0)

# pd.Series.nunique
print(wine.columns)

print(wine)

f, ax = plt.subplots(figsize=(18, 12))
sns.heatmap(wine, annot=True, fmt="d", linewidths=.5, ax=ax, cbar=False, cmap="YlGnBu")
plt.show()

# Load the example flights dataset and conver to long-form
# wine = data.pivot(index='country', columns=['Price4_15', 'Price16_22', 'Price23_32', 'Price33_50', 'Price51'],
#                  values='price')

# Draw a heatmap with the numeric values in each cell
# f, ax = plt.subplots(figsize=(9, 6))
# sns.heatmap(wine, annot=True, fmt="d", linewidths=.5, ax=ax)

sns.heatmap(data.isnull())
plt.show()

#print(data['price'].describe())




