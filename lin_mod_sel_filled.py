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
from sklearn.preprocessing import StandardScaler

# Let us first import the data

path_to_data = 'https://raw.githubusercontent.com/jeshan49/eemp2/master/Credit.csv'

data = pd.read_csv(path_to_data, index_col=0)

print(data.describe())

print(data.info())

np.unique(data['Education'])

np.unique(data['Gender'])

# Let us define the dummy variables

data = pd.get_dummies(data, drop_first=True)

# Summary statistics

pd.scatter_matrix(data[['Balance', 'Limit', 'Rating', 'Income', 'Student_Yes', 'Cards', 'Gender_Male']], alpha=0.2)

corr = data.corr()

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap=sns.diverging_palette(250, 10, as_cmap=True),
        linewidth=.5,
        vmin=-1,
        vmax=1,
        annot=True)


data.describe()

# Let us define y and X

y = data['Balance'].values

X = data.drop(['Balance'], axis=1).values



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

ridge = Ridge()

ridge.fit(scale(X), y)

ridge.coef_

ridge.intercept_


# Let's do Ridge regression for many alphas

alphas = np.logspace(-2, 5, 200)

coefs = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(scale(X),y)
    coefs.append(ridge.coef_)
    
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# Select alpha with CV:

ridge = RidgeCV(alphas, scoring='neg_mean_squared_error', cv=kfold)

ridge.fit(scale(X), y)

ridge.alpha_

# Alternatively:

cv_scores = []
cv_stds = []
for a in alphas:
    ridge = Ridge(alpha=a)
    cv = cross_val_score(ridge, scale(X), y, cv=kfold, scoring='neg_mean_squared_error')
    cv_scores.append(-1*cv.mean())
    cv_stds.append(cv.std())

plt.errorbar(alphas, cv_scores, yerr=cv_stds, color='red')
plt.xscale('log')

print(alphas[np.argmin(cv_scores)])

ridge = Ridge(alpha=alphas[np.argmin(cv_scores)])
ridge.fit(scale(X), y)
pd.Series(ridge.coef_, index=data.drop(['Balance'], axis=1).columns)


##########################################################################
################### Here we perform Lasso regression #####################
##########################################################################

lasso = Lasso()

lasso.fit(scale(X), y)

lasso.coef_

lasso.intercept_



# Let's do Lasso regression for many alphas
alphas = np.logspace(-2, 5, 200)

coefs = []
for a in alphas:
    lasso = Lasso(alpha=a)
    lasso.fit(scale(X),y)
    coefs.append(lasso.coef_)
    
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# Select alpha with CV:

lasso = LassoCV(alphas=alphas, cv=kfold)

lasso.fit(scale(X), y)

lasso.alpha_

# Alternatively:

cv_scores = []
cv_stds = []
for a in alphas:
    lasso = Lasso(alpha=a)
    cv = cross_val_score(lasso, scale(X), y, cv=kfold, scoring='neg_mean_squared_error')
    cv_scores.append(-1*cv.mean())
    cv_stds.append(cv.std())

plt.errorbar(alphas, cv_scores, yerr=cv_stds, color='red')
plt.xscale('log')


##########################################################################
############ Here we perform Principal Component Regression ##############
##########################################################################

pca = PCA()

X_reduced = pca.fit_transform(scale(X))

# Loadings (\phi_ij)
print(pca.components_.shape)
pd.DataFrame(pca.components_, index=data.drop(['Balance'], axis=1).columns, columns=[i for i in range(1,12)])

# Principal component scores (z_ij)
pd.DataFrame(X_reduced).loc[:5,:]

# Explained variance:
plt.plot(np.arange(1,12), pca.explained_variance_ratio_*100, label='explained variance')
plt.plot(np.arange(1,12), [np.sum(pca.explained_variance_ratio_[:i])*100 for i in range(1,12)], label='cumulative explained variance')
plt.xlabel('# of components')
plt.ylabel('variance')
plt.xticks(np.arange(1,12))
plt.legend()

# Displaying the first two Princiapl components:
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')

plt.scatter(X_reduced[:, 0],scale(X[:,0]))
plt.xlabel('1st Principal Component')
plt.ylabel('X_1')

reg = LinearRegression()
cv_scores = []

for i in np.arange(1, 12):
    score = -1*cross_val_score(reg, X_reduced[:,:i], y, cv=kfold, scoring='neg_mean_squared_error').mean()
    cv_scores.append(score)

plt.plot([i for i in range(1, 12)], cv_scores, '-o')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')

print('number of components: {}'.format(np.argmin(cv_scores)+1))

##########################################################################
################ Here we perform Partial Least Squares ###################
##########################################################################

cv_scores = []

for i in np.arange(1, 12):
    pls = PLSRegression(n_components=i)
    score = -1*cross_val_score(pls, scale(X), y, cv=kfold, scoring='neg_mean_squared_error').mean()
    cv_scores.append(score)

plt.plot([i for i in range(1,12)], cv_scores, '-o')
plt.xlabel('Number of components in regression')
plt.ylabel('MSE')

print('number of components: {}'.format(np.argmin(cv_scores)+1))


