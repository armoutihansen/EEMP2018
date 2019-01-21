"""
Decision Trees and Ensemble Learning with Scikit-Learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz


pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, auc
from sklearn.model_selection import GridSearchCV 


# We’ll start by using classification trees to analyze the Carseats data set

path_to_carseats = 'https://raw.githubusercontent.com/jeshan49/eemp2/master/Carseats.csv'

df = pd.read_csv(path_to_carseats, index_col=0)

print(df.describe())

print(df.info())

pd.scatter_matrix(df, alpha=0.2)

"""
We use the ifelse() function to create a variable, called High, which takes
on a value of Yes if the Sales variable exceeds 8, and takes on a value of No
otherwise. We’ll append this onto our dataFrame using the .map() function:
"""

df['High'] = df['Sales'].map(lambda x: 1 if x>8 else 0)
df = pd.get_dummies(df, drop_first=True)

print(df.info())

# Let us define X and y

X = df.drop(['High', 'Sales'], axis=1)
y = df['High']

# We first split the observations into a training set and a test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=181)

# We now use the DecisionTreeClassifier() function to fit a classification tree
# in order to predict High using all variables but Sales

clf = DecisionTreeClassifier(max_depth=6)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)


##########################################################################
# We use the export graphviz() function to export the tree
# structure to a temporary .dot file, and the graphviz.Source() function to display the image:
##########################################################################

export_graphviz(clf,
                out_file='dec_tree.dot',
                feature_names=X_train.columns,
                class_names=['No','Yes'],
                rounded=True,
                filled=True)

with open('dec_tree.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph).view()
##########################################################################
##########################################################################


# Finally, let’s evaluate the tree’s performance on the test data. The predict() function can be used for
# this purpose. We can then build a confusion matrix

pred = clf.predict(X_test)
cm = pd.DataFrame(confusion_matrix(y_test, pred).T, index=['No', 'Yes'], columns=['No', 'Yes'])
print(cm)
# 86+59/200 = 0.725


##########################################################################
############ Here we construct the ROC curve for the tree ################
##########################################################################
y_score = clf.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='orange',
         label='ROC curve (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for our Decision Tree')
plt.legend(loc="lower right")
##########################################################################
##########################################################################


"""
Now let’s try fitting a regression tree to the Boston data set. First, we create a
training set, and fit the tree to the training data using medv (median home value) as our response
"""

path_to_boston = 'https://raw.githubusercontent.com/jeshan49/eemp2/master/Boston.csv'

df2 = pd.read_csv(path_to_boston, index_col=0)

print(df2.describe())

print(df2.info())

pd.scatter_matrix(df2, alpha=0.2)

X2 = df2.drop(['medv'], axis=1)
y2 = df2['medv']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.5, random_state=181)

# Pruning not supported. Choosing max depth 4)
reg = DecisionTreeRegressor(max_depth=2)
reg.fit(X2_train, y2_train)

##########################################################################
############ Let’s take a look at the tree: ##############################
##########################################################################
export_graphviz(reg,
                out_file='reg_tree.dot',
                feature_names=X2_train.columns,
                class_names=['No','Yes'],
                rounded=True,
                filled=True)

with open('reg_tree.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph).view()
##########################################################################
##########################################################################

#Now let’s see how it does on the test data:
pred = reg.predict(X2_test)
plt.scatter(pred, y2_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

mean_squared_error(y2_test, pred)

# The test set MSE associated with the regression tree is 29.2. 

# Just to see how well a linear model is doing in comparison

linreg = LinearRegression()

linreg.fit(X2_train, y2_train)

pred = linreg.predict(X2_test)
plt.scatter(pred, y2_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')
mean_squared_error(y2_test, pred)


# Let’s see if we can improve on this result using bagging and random forests

reg2 = BaggingRegressor(DecisionTreeRegressor(), n_estimators=500, random_state=181)

reg2.fit(X2_train, y2_train)

# How well does this bagged model perform on the test set?
pred = reg2.predict(X2_test)
plt.scatter(pred, y2_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')
mean_squared_error(y2_test, pred)

# The test set MSE associated with the bagged regression tree is 25.6.

# We can grow a random forest in exactly the same way, except that we’ll use a smaller value of the
# max features argument. Here we’ll use max features = 6

reg3 = RandomForestRegressor(max_features=6, n_estimators=500, random_state=181) 
reg3.fit(X2_train, y2_train)

# How well does this random forest perform on the test set?
pred = reg3.predict(X2_test)
plt.scatter(pred, y2_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')
mean_squared_error(y2_test, pred)

# The test set MSE associated with the random forest is 23.7.

# Using the feature importances attribute of the RandomForestRegressor, we can view the importance
# of each variable:

reg3.feature_importances_

(pd.Series(reg3.feature_importances_*100, index=X2_train.columns).nlargest(13).plot(kind='barh'))   

# The results indicate that across all of the trees considered in the random forest, the wealth level of the
# community (lstat) and the house size (rm) are by far the two most important variables.

# Now let's try to find the optimal max features, n_estimators, max depth by CV:

reg_tree = RandomForestRegressor()
param_grid = {
        'n_estimators': [200,300,400,500],
        'max_features': [6, 'sqrt', 'log2'],
        'max_depth':[3,4,5,6]
        }

CV_rf = GridSearchCV(estimator=reg_tree, param_grid=param_grid, cv= 5, n_jobs=-1)

CV_rf.fit(X2_train, y2_train)

CV_rf.best_params_

pred = CV_rf.predict(X2_test)
plt.scatter(pred, y2_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')
mean_squared_error(y2_test, pred)

"""
Now we’ll use the GradientBoostingRegressor package to fit boosted regression trees to the
Boston data set
"""

reg4 = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=2, random_state=181)
reg4.fit(X2_train, y2_train)

pred = reg4.predict(X2_test)
plt.scatter(pred, y2_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')
mean_squared_error(y2_test, pred)

# The test set MSE associated with the gradient boosting is 24.0.

# Let us look at feature importance again

(pd.Series(reg4.feature_importances_*100, index=X2_train.columns).nlargest(13).plot(kind='barh'))   


# Now let's try to find the optimal n_estimators, max depth and learning rate by CV:

gr_boo = GradientBoostingRegressor()
param_grid = {
        'n_estimators': [200,300,400,500],
        'learning_rate': np.arange(1, 1000, 100)/1000,
        'max_depth':[3,4,5,6]
        }


CV_gr = GridSearchCV(estimator=gr_boo, param_grid=param_grid, cv= 5, n_jobs=-1)

CV_gr.fit(X2_train, y2_train)

CV_gr.best_params_

pred = CV_gr.predict(X2_test)
plt.scatter(pred, y2_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')
mean_squared_error(y2_test, pred)

# The test set MSE associated with the optimized gradient boosting is 21.3.
