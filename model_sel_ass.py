"""
Model selection and assessment
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold   
from sklearn.model_selection import RepeatedKFold



# Let us generate the data:

rng = np.random.RandomState(181)

X_pop = rng.uniform(size=(100000, 20))

np.mean(X_pop, axis=0)

mask = (X_pop[:,0]>=0.5)

y_pop = 1*mask

np.sum(y_pop)

X_train, X_test, y_train, y_test = train_test_split(X_pop, y_pop,
                                                    test_size=0.999, random_state=1)

nbs = np.arange(1, 51, 2)

# Let's start with regression

reg_train_mse = []
reg_test_mse = []

for n in nbs:
    knnreg = KNeighborsRegressor(n_neighbors=n)
    knnreg.fit(X_train, y_train)
    y_hat_train = knnreg.predict(X_train)
    y_hat_test = knnreg.predict(X_test)
    train_mse = MSE(y_train, y_hat_train)
    reg_train_mse.append(train_mse)
    test_mse = MSE(y_test, y_hat_test)
    reg_test_mse.append(test_mse)
    print(n)
    
# Let us plot the train and test errors:
    
plt.plot(nbs, reg_train_mse, marker='.', color='blue', label='Train MSE')
plt.plot(nbs, reg_test_mse, marker='.', color='orange', label='Test MSE')
plt.xticks(nbs)
plt.xlabel('# of neighbors')
plt.ylabel('MSE')
plt.title('kNN regression')
plt.legend()

print('Best model is with {} number of neighbors'.format(nbs[np.argmin(reg_test_mse)]),
      '\n', 'with Err_Tr: {}'.format(np.min(reg_test_mse)))


# Let's see how close we get to select the best model with CV:

## Before, let us see how CV is done with sklearn

knnreg = KNeighborsRegressor(n_neighbors=3)
cv = cross_val_score(knnreg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
-1*cv.mean()

## Compare our score to Err_Tr

print('CV MSE: {}'.format(-1*cv.mean()), '\n',
      'Err_Tr: {}'.format(reg_test_mse[1]),'\n',
      'difference: {}'.format(-1*cv.mean() - reg_test_mse[1]))

# Now let's do CV for model selection:

parameters = {'n_neighbors':nbs}
knneg = KNeighborsRegressor()

cv_5 = GridSearchCV(knnreg, parameters, cv=5, scoring='neg_mean_squared_error')
cv_5.fit(X_train, y_train)

cv_5.cv_results_['mean_test_score']

cv_5_mean = -1*cv_5.cv_results_['mean_test_score']
cv_5_std = cv_5.cv_results_['std_test_score']

cv_5.best_estimator_

# Let us try to plug these into our plot

plt.plot(nbs, reg_train_mse, marker='.', color='blue', label='err')
plt.plot(nbs, reg_test_mse, marker='.', color='orange', label='Err_Tr')
plt.errorbar(nbs, cv_5_mean, yerr=cv_5_std, color='red')
plt.legend()
plt.xticks(nbs)

# Now, let's try LOOCV and 10-fold CV:

loocv = GridSearchCV(knnreg, parameters, cv=X_train.shape[0], scoring='neg_mean_squared_error')
loocv.fit(X_train, y_train)

loocv_mean = -1*loocv.cv_results_['mean_test_score']
loocv_std = loocv.cv_results_['std_test_score']

loocv.best_estimator_

# Let us try to plug these into our plot

plt.plot(nbs, reg_train_mse, marker='.', color='blue', label='err')
plt.plot(nbs, reg_test_mse, marker='.', color='orange', label='Err_Tr')
plt.errorbar(nbs, loocv_mean, yerr=loocv_std, color='red')
plt.legend()
plt.xticks(nbs)

cv_10 = GridSearchCV(knnreg, parameters, cv=10, scoring='neg_mean_squared_error')
cv_10.fit(X_train, y_train)

cv_10_mean = -1*cv_10.cv_results_['mean_test_score']
cv_10_std = cv_10.cv_results_['std_test_score']

cv_10.best_estimator_

# Let us try to plug these into our plot

plt.plot(nbs, reg_train_mse, marker='.', color='blue', label='err')
plt.plot(nbs, reg_test_mse, marker='.', color='orange', label='Err_Tr')
plt.plot(nbs, cv_10_mean, color='red')
plt.legend()
plt.xticks(nbs)

rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=181)
r_10_cv = GridSearchCV(knnreg, parameters, cv=rkf, scoring='neg_mean_squared_error')
r_10_cv.fit(X_train, y_train)

r_10_mean = -1*r_10_cv.cv_results_['mean_test_score']
r_10_std = r_10_cv.cv_results_['std_test_score']

r_10_cv.best_estimator_

# Let us try to plug these into our plot

plt.plot(nbs, reg_train_mse, marker='.', color='blue', label='err')
plt.plot(nbs, reg_test_mse, marker='.', color='orange', label='Err_Tr')
plt.plot(nbs, r_10_mean, color='red')
plt.legend()
plt.xticks(nbs)

### Now we'll work with the kNN classifier ###
""" Try on your own, but notice the following changes compared to regression:
# 1. We cannot and should no longer use MSE
    Insted, directly after using the fit method, use train_loss = 1-clf.score(X_train, y_train).
    Thus, you do not need to specify y_hat anymore.
    Here: clf refers to classifier, and should be changed if you name it something else
    We use -1 times the score, as we want the MIS-classification rate.
    Remember also to calculate the test misclassification rate
# 2. When you do CV, you do no longer need to specify a scoring parameter
    The classification rate is returned, and you can get the MIS-classification rate 
    by multiplying with -1
"""


clf_train_l = []
clf_test_l = []

nbs = np.arange(1, 51, 2)
for n in nbs:
    knnclf = KNeighborsClassifier(n_neighbors=n)
    knnclf.fit(X_train, y_train)
    train_l = knnclf.score(X_train, y_train)
    test_l = knnclf.score(X_test, y_test)
    clf_train_l.append(1 - train_l)
    clf_test_l.append(1 - test_l)
    print(n)
    
# Let us plot the train and test errors:
    
plt.plot(nbs, clf_train_l, marker='.', color='blue', label='err')
plt.plot(nbs, clf_test_l, marker='.', color='orange', label='Err_Tr')
plt.legend()
plt.xticks(nbs)

# Best model:

print('Best model is with {} number of neighbors'.format(nbs[np.argmin(clf_test_l)]),
      '\n', 'with Err_Tr: {}'.format(np.min(clf_test_l)))


# Let's see how close we get to select the best model with CV:

## Before, let us see how CV is done with sklearn
knnclf = KNeighborsClassifier(n_neighbors=3)
cv = cross_val_score(knnclf, X_train, y_train, cv=5)
## Compare our score to Err_Tr
print('CV mean misclassification rate: {}'.format(1-cv.mean()), '\n',
      'Err_Tr: {}'.format(clf_test_l[1]),'\n',
      'difference: {}'.format(1-cv.mean() - clf_test_l[1]))

# Now let's do CV for model selection:

parameters = {'n_neighbors':nbs}
knnclf = KNeighborsClassifier()

cv_5 = GridSearchCV(knnclf, parameters, cv=5)
cv_5.fit(X_train, y_train)

cv_5.cv_results_['mean_test_score']

cv_5_mean = 1 - cv_5.cv_results_['mean_test_score']
cv_5_std = cv_5.cv_results_['std_test_score']

cv_10 = GridSearchCV(knnclf, parameters, cv=5)
cv_10.fit(X_train, y_train)

cv_10.cv_results_['mean_test_score']

cv_10_mean = 1 - cv_10.cv_results_['mean_test_score']
cv_10_std = cv_10.cv_results_['std_test_score']

# Let us try to plug these into our plot

plt.plot(nbs, clf_train_l, marker='.', color='blue', label='err')
plt.plot(nbs, clf_test_l, marker='.', color='orange', label='Err_Tr')
plt.errorbar(nbs, cv_5_mean, yerr=cv_5_std, color='red')
plt.legend()
plt.xticks(nbs)

print('Best model is with {} number of neighbors'.format(nbs[np.argmin(clf_test_l)]),
      '\n', 'Best model with CV is with {} number of neighbors'.format(nbs[np.argmin(cv_5_mean)]))
      
# Now let us perform 5-fold nested CV

inner_cv_5 = KFold(10)
inner_cv_5_score = GridSearchCV(knnclf, parameters, cv=inner_cv_5)
inner_cv_5_score.fit(X_train, y_train)
outer_cv_5 = KFold(10)
nested_cv_5 = cross_val_score(inner_cv_5_score, X_train, y_train, cv=outer_cv_5)

nested_cv_5 = cross_val_score(cv_5, X_train, y_train, cv=outer_cv_5)

1-nested_cv_5.mean()
inner_cv_5_score.cv_results_['mean_test_score']
cv_5_mean = 1 - inner_cv_5_score.cv_results_['mean_test_score']
cv_5_std = inner_cv_5_score.cv_results_['std_test_score']

# Finally, we'll do repeated CV

rkf = RepeatedKFold(n_splits=10, n_repeats=100)
mod = GridSearchCV(knnclf, parameters, cv=rkf)
mod.fit(X_train, y_train)
mod.best_estimator_
mod.cv_results_['mean_test_score']
mod.cv_results_['std_test_score']
mod_cv = 1 - mod.cv_results_['mean_test_score']
mod_cv_std = mod.cv_results_['std_test_score']