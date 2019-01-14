# Linear Model Selection and Regularization #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import cross_val_score, KFold  
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA,
from sklearn.cross_decomposition import PLSRegression 

# Let us first import the data

path_to_data = 'https://raw.githubusercontent.com/jeshan49/eemp2/master/Credit.csv'


# Let us define the dummy variables



# Summary statistics



# Let us define y and X





##########################################################################
################# Here we perform best subset selection ##################
##########################################################################

# First, let us define a function which takes in a list of indices
# and then returns a generator with all possible indices

def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

list_of_indices = [i for i in range(X.shape[1])]

# We will store all of this indices in a list an remove the empty member
list_of_all = [i for i in powerset(list_of_indices) if len(i)>0]
# Next, we'll sort the list based on the indices length
list_of_all.sort(key=len)

# Next, we will create two lists that will contain:
#   (i) the best r2 for each number of variables
#   (ii) the indices of the best model for each number of variables
list_of_best_r2 = [0 for i in range(11)]
list_of_best_models = [0 for i in range(11)]

# Now, we will create a for loop which gets the r2 of each possible model
# and stores the best one for each number of variables
for spec in list_of_all:
    length = len(spec)
    print(length)
    reg = LinearRegression()
    reg.fit(X[:, spec], y)
    r_2 = reg.score(X[:, spec], y)
    if r_2 > list_of_best_r2[length-1]:
        list_of_best_r2[length-1] = r_2
        list_of_best_models[length-1] = spec
        
for i in range(len(list_of_best_models)):
    print('Best {} parameter model is {}'.format(i+1, [col for col in data.drop(['Balance'], axis=1).columns[list_of_best_models[i]]]))
    
# Finally, we'll do 5-fold CV to determine which of the best models are the winner
kfold = KFold(n_splits=5, random_state=181)
cv_scores = []
cv_stds = []

for spec in list_of_best_models:
    reg = LinearRegression()
    cv = cross_val_score(reg, X[:, spec], y, cv=kfold, scoring='neg_mean_squared_error')
    cv_scores.append(-1 * cv.mean())
    cv_stds.append(cv.std())
    
plt.errorbar([i for i in range(1,12)], cv_scores, yerr=cv_stds, color='red')
plt.xticks(list_of_indices)
plt.xlabel('number of predictors')
plt.ylabel('MSE')

print('Best model from best subset selection is {}'.format([col for col in data.drop(['Balance'], axis=1).columns[list_of_best_models[np.argmin(cv_scores)]]]))


##########################################################################
################### Here we perform Ridge regression #####################
##########################################################################




# Let's do Ridge regression for many alphas



# Select alpha with CV:



# Alternatively:




##########################################################################
################### Here we perform Lasso regression #####################
##########################################################################




# Let's do Lasso regression for many alphas


# Select alpha with CV:



# Alternatively:




##########################################################################
############ Here we perform Principal Component Regression ##############
##########################################################################



# Loadings (z_ij)


# Principal components (Z_i)


# Explained variance:


# Displaying the first two Princiapl components:


##########################################################################
################ Here we perform Partial Least Squares ###################
##########################################################################



