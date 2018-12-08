"""
###############################################################################
GETTING STARTED WITH PANDAS
###############################################################################
##################################
Content:
    1. Pandas data structures
    2. Essential functionality
    3. Summarizing and computing descriptive statistics
    4. Reading and writing data in text format
##################################
"""

print(__doc__)

#-----------------------------------------------------------------------------#

# 1. Pandas data structures
# Again, we import using the standard convention:
import pandas as pd
import numpy as np

# A Series is a one-dimensional array-like object containing a sequence of values
obj = pd.Series([4, 7, -5, 3])
# It has an index!

# Get the values only:
obj.values
type(obj.values)

# Custom index:
obj2 = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'd'])

"""
Compared with NumPy arrays, you can use labels in the index when selecting single
values or a set of values:
"""
obj2[['c', 'a', 'd']]
obj2[[2, 0, 3]]

# We can filter using NumPy operations:
obj2[obj2 > 0]

# We can also use other common NumPy operations:
obj2 * 2

np.exp(obj2)

# data contained in a Python dict: you can create a Series from it
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)


"""
A DataFrame represents a rectangular table of data and contains an ordered collec‐
tion of columns, each of which can be a different value type (numeric, string,
boolean, etc.). The DataFrame has both a row and column index; it can be thought of
as a dict of Series all sharing the same index. 
"""

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002, 2003],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}

df = pd.DataFrame(data)
df.head()
pd.DataFrame(data, columns=['year', 'state', 'pop'])

# Column access:
df.year
df['year']

# Row access:
df.iloc[1]
df.loc[1]

# Adding columns
df['debt'] = 16.5
val = pd.Series([-1.2, -1.5, -1.7], index=[1, 3, 5])
df['debt'] = val

df['debt'] = df['debt'].fillna(df['debt'].mean())
df.fillna(0)

df['eastern'] = df['state'] == 'Ohio'
df['eastern'] = df['eastern'] *  1

# Delete column
del df['eastern']

# Transpose the data frame
df.T

# Again, we might only want the values
df.values

#-----------------------------------------------------------------------------#

# 2. Essential functionality

# indexing and reindexing

obj = pd.DataFrame(np.array([[2, 0, 4, 5], [4, 8, 3, 9]]), index=['a', 'b'])
obj.columns = ['y', 'x1', 'x2', 'x3']


# selection and filtering
obj[['x1', 'x2', 'x3']].values
obj.iloc[0]
obj.loc['b']

obj.index = np.arange(1,3)

print(obj.head())

obj[obj['x1'] > 0]

obj.iloc[:, 1:]
obj.loc[:, 'x1':]

# Creating dummies
data = [['a', 'yes', 3.45], ['b', 'no', 5.65], ['c', 'yes', 9.87]]

df = pd.DataFrame(data, columns = ['x1', 'x2', 'y'], index = [1, 2, 3])

dummy_x1 = pd.get_dummies(df['x1'])
dummy_x2 = pd.get_dummies(df['x2'])
df['x1_a'] = dummy_x1.iloc[:, 0]
df['x1_b'] = dummy_x1.iloc[:, 1]
df['x2_yes'] = dummy_x2.iloc[:, 1]

# Here we get dummies for all possibilities
df['x1_a'] = df['x1'].apply(lambda val: 1 if val == 'a' else 0)
df['x1_b'] = df['x1'].apply(lambda val: 1 if val == 'b' else 0)


#-----------------------------------------------------------------------------#

# 3. Summarizing and computing descriptive statistics

obj.describe()

#-----------------------------------------------------------------------------#

# 4. Reading and writing data in text format

# Reading csv directly from the web

path_to_data = 'https://raw.githubusercontent.com/jeshan49/eemp2/master/Advertising2.csv'

df = pd.read_csv(path_to_data)

df = df.set_index('Unnamed: 0')

# otherwise, you can download the file, and add it to your working directory
# check your working directory

pwd

df2 = pd.read_csv('data/Advertising.csv').set_index('Unnamed: 0')

df.to_csv('data/my_data.csv')

# There are plenty of files that can be imported, but we will use csv files
