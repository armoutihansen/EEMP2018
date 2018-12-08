"""
###############################################################################
PLOTTING AND VISUALIZATION
###############################################################################
##################################
Content:
    1. Matplotlib
    2. Plotting with pandas and seaborn
##################################
"""

print(__doc__)

#-----------------------------------------------------------------------------#

# 1. Matplotlib
# Again, we import using the standard convention:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Plotting a line
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))

# Scatter-plot points
x = np.random.normal(size=500)
y = np.random.normal(size=500)
plt.scatter(x, y)


# Histogram
np.random.seed(0)

# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(437)
num_bins = 50
plt.hist(x, num_bins, normed=1)
plt.title('histogram of sampling 437 realizations')




#-----------------------------------------------------------------------------#

# 2. Plotting with pandas and seaborn
df = sns.load_dataset("iris")

df[df['species'] == 'setosa'].plot(x='sepal_length', y='sepal_width', kind='scatter')

# Scatterplot Matrix

sns.set(style="ticks")
sns.pairplot(df, hue="species")

# Linear regression with marginal distributions

tips = sns.load_dataset("tips")

sns.regplot("total_bill", "tip", data=tips, color="m")
plt.title('Regression plot of total bill on tips')
plt.xlabel('total bill')
plt.ylabel('tips')


sns.jointplot("total_bill", "tip", data=tips, kind="reg", color="m")

path_to_data = 'https://raw.githubusercontent.com/jeshan49/eemp2/master/Advertising2.csv'
df = pd.read_csv(path_to_data).set_index('Unnamed: 0')

sns.regplot("TV", "Sales", data=df, color="m")
plt.title('Regression plot of TV on Sales')
plt.xlabel('TV')
plt.ylabel('Sales')

